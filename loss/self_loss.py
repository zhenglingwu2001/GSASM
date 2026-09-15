import torch
import torch.nn.functional as F
from collections import namedtuple
from typing import Callable, Optional
from cupy.cuda import function
from pynvrtc.compiler import Program

_apply_disparity_func_pos: Optional[Callable] = None
_apply_disparity_func_neg: Optional[Callable] = None


def _build_cuda_kernels():
    global _apply_disparity_func_pos
    global _apply_disparity_func_neg

    _apply_disparity_pos_kernel = """
    extern "C" {
        __global__ void apply_disparity_pos(
        float *dst, const float *src, const int *disp, int h, int w, int c, int total_l) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= total_l)
                return;
            int dbase = (i/h/c*h+i%h)*w;
            for (int j = w - 1; j >=0; j--) {
                int idx = j + disp[dbase+j];
                if (idx < w)
                    dst[i*w+idx] = src[i*w+j];
            }
        }
        __global__ void apply_disparity_neg(
        float *dst, const float *src, const int *disp, int h, int w, int c, int total_l) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= total_l)
                return;
            int dbase = (i/h/c*h+i%h)*w;
            for (int j = 0; j < w; j++) {
                int idx = j + disp[dbase+j];
                if (idx > -1)
                    dst[i*w+idx] = src[i*w+j];
            }
        }
    }
    """
    program = Program(_apply_disparity_pos_kernel, "apply_disparity.cu")
    m = function.Module()
    m.load(bytes(program.compile().encode()))
    _apply_disparity_func_pos = m.get_function("apply_disparity_pos")
    _apply_disparity_func_neg = m.get_function("apply_disparity_neg")


def apply_disparity_cu(img: torch.Tensor, disp: torch.Tensor):
    """
    Apply disparity using jit cuda ops.

    :param img: tensor needed warping. (N, C, H, W)
    :param disp: (N, H, W) or (N, 1, H, W)
    :return:
    """

    # load kernel if haven't
    if _apply_disparity_func_neg is None or _apply_disparity_func_neg is None:
        _build_cuda_kernels()

    # tensor check
    assert img.is_contiguous() and disp.is_contiguous()
    assert img.device.type == disp.device.type == "cuda"
    assert disp.dtype == torch.int

    if torch.all(disp >= 0):
        warp_fn = _apply_disparity_func_pos
    else:
        assert torch.all(disp <= 0)
        warp_fn = _apply_disparity_func_neg

    # send data to cuda ops
    stream = namedtuple("Stream", ["ptr"])
    s = stream(ptr=torch.cuda.current_stream().cuda_stream)

    ret = torch.zeros_like(img)

    b, c, h, w = img.shape
    total_l = b * c * h
    grid_size = total_l // 512 + 1
    warp_fn(
        stream=s,
        grid=(grid_size, 1, 1),
        block=(512, 1, 1),
        args=[ret.data_ptr(), img.data_ptr(), disp.data_ptr(), h, w, c, total_l],
    )

    return ret


def apply_disparity(img, disp):
    batch_size, _, height, width = img.size()
    disp = disp / width

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = (
        torch.linspace(0, 1, height)
        .repeat(batch_size, width, 1)
        .transpose(1, 2)
        .type_as(img)
    )

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(
        img, 2 * flow_field - 1, mode="bilinear", padding_mode="zeros"
    )

    return output

def get_reproj_error_patch(input_L, input_R, pred_disp_l, mask=None, ps=5):
    assert ps % 2 == 1
    bs, c, h, w = input_L.shape
    unfold_func = torch.nn.Unfold(
        kernel_size=(ps, ps), stride=1, padding=(ps - 1) // 2, dilation=1
    )
    fold_func = torch.nn.Fold(
        output_size=(h + ps - 1, w + ps - 1), kernel_size=(ps, ps)
    )
    input_L = unfold_func(input_L)
    input_R = unfold_func(input_R)
    input_L = input_L.reshape(bs, c * ps * ps, h, w)
    input_R = input_R.reshape(bs, c * ps * ps, h, w)
    input_L_warped = apply_disparity(input_R, -pred_disp_l)
    if mask is not None:
        _, new_c, _, _ = input_L.shape
        mask = mask.repeat(1, new_c, 1, 1)
    else:
        mask = torch.ones_like(input_L_warped).type(torch.bool)
    reprojection_loss = F.mse_loss(input_L_warped[mask], input_L[mask])

    input_L_warped = input_L_warped.reshape(bs, c * ps * ps, h * w)
    input_L_warped = fold_func(input_L_warped)
    if ps > 1:
        input_L_warped = input_L_warped[
            :, :, ((ps - 1) // 2): -((ps - 1) // 2), ((ps - 1) // 2): -((ps - 1) // 2)
        ]
    mask = mask[:, :c, :, :]
    return reprojection_loss, input_L_warped, mask.type(torch.int)
