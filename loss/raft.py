import torch
import torch.nn as nn
from loss.self_loss import get_reproj_error_patch, apply_disparity_cu

def SSIM(x, y, md=3):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    refl = nn.ReflectionPad2d(md)

    x = refl(x)
    y = refl(y)
    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)

    base_grid = torch.stack([x_base, y_base], 1)
    return base_grid


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad(disp, image, alpha, order=1):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(disp)
    if order == 2:
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        dx, dy = dx2, dy2

    loss_x = weights_x[:, :, :, 1:] * dx[:, :, :, 1:].abs()
    loss_y = weights_y[:, :, 1:, :] * dy[:, :, 1:, :].abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def loss_smooth(disp, im1_scaled):
    func_smooth = smooth_grad
    loss = []
    loss += [func_smooth(disp, im1_scaled, 1, order=1)]
    return sum([l.mean() for l in loss])


def disp_warp(x, disp, r2l=False, pad='border', mode='bilinear', device='cuda'):
    B, _, H, W = x.size()
    offset = -1
    if r2l:
        offset = 1

    base_grid = mesh_grid(B, H, W).type_as(x)
    v_grid = norm_grid(base_grid + torch.cat((offset * disp, torch.zeros_like(disp)), 1))
    x_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    mask = torch.autograd.Variable(torch.ones(x_recons.size())).to(device)
    mask = nn.functional.grid_sample(mask, v_grid)
    return x_recons, mask


def photometric_loss(im1_scaled, im1_recons):
    loss = []
    loss += [0.15 * (im1_scaled - im1_recons).abs().mean(1, True)]
    loss += [0.85 * SSIM(im1_recons, im1_scaled).mean(1, True)]
    return sum([l for l in loss])


def trinocular_loss(disp, im1, im2, im3, uncertainty):
    im2_recons_from_1, mask_12 = disp_warp(im1, disp, r2l=True)
    im2_recons_from_3, mask_23 = disp_warp(im3, disp, r2l=False)

    photometric_loss_12 = photometric_loss(im2, mask_12 * im2_recons_from_1)
    photometric_loss_23 = photometric_loss(im2, mask_23 * im2_recons_from_3)
    loss_warp, _ = torch.min(torch.cat((photometric_loss_12, photometric_loss_23), dim=1), dim=1)

    photometric_loss_1 = photometric_loss(im2, im1)
    photometric_loss_3 = photometric_loss(im2, im3)
    loss_2, _ = torch.min(torch.cat((photometric_loss_1, photometric_loss_3), dim=1), dim=1)

    automask = loss_warp < loss_2
    loss = (loss_warp * uncertainty)[automask]

    return loss.mean()


def binocular_loss(disp, im1, im2, uncertainty):
    im1_recons, _ = disp_warp(im2, disp, r2l=False)

    loss_warp = photometric_loss(im1, im1_recons).squeeze()
    loss_2 = photometric_loss(im2, im1).squeeze()

    automask = loss_warp < loss_2
    loss = (loss_warp * uncertainty)[automask]

    return loss.mean()


def image_loss(disp, im1, im2, im3, uncertainty, trinocular=True):
    if trinocular:
        return trinocular_loss(disp, im1, im2, im3, uncertainty)
    else:
        return binocular_loss(disp, im2, im3, uncertainty)


def ns_loss(pred_disps, target_disp, conf, im0, im1, im2, trinocular_loss=True, alpha_disp_loss=1.0,
            alpha_photometric=0.1, conf_threshold=0.5):
    target_disp = -target_disp
    conf = conf * (target_disp < 0).float()

    n_predictions = len(pred_disps)
    loss_gamma = 0.9
    target_disp = target_disp.unsqueeze(1)

    disp_loss = 0.0
    photometric_loss = 0.0

    for i in range(n_predictions):
        assert not torch.isnan(pred_disps[i]).any() and not torch.isinf(pred_disps[i]).any()

        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

        disp_diff = torch.abs(pred_disps[i] - target_disp)
        disp_loss += i_weight * (disp_diff * conf.unsqueeze(1)).mean()

        if alpha_photometric != 0.:
            photometric_loss += i_weight * image_loss(pred_disps[i], im0, im1, im2, 1 - conf, trinocular_loss)
        else:
            photometric_loss = torch.zeros_like(disp_loss)

    loss = alpha_disp_loss * disp_loss + alpha_photometric * photometric_loss

    epe = torch.sum((pred_disps[-1] - target_disp) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[(conf > conf_threshold).view(-1)]

    metrics = {
        'disp_err': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return loss, metrics

def total_loss(item, pred_disps, dynamic_weights, disp_gt_l=None, isSuper=True, isSelf=False):
    loss = 0
    metrics = {}
    if dynamic_weights is not None:
        super_ratio, self_ratio = dynamic_weights.get_weights()
    else:
        super_ratio = torch.tensor(0.01, device=disp_gt_l.device if disp_gt_l is not None else 'cpu')
        self_ratio = torch.tensor(2.0, device=disp_gt_l.device if disp_gt_l is not None else 'cpu')
        
    if isSuper:
        super_loss, error = ns_loss(pred_disps, disp_gt_l, conf=item["conf"], im0=item["img_L"],
                                    im1=item["img_L"], im2=item["img_R"], trinocular_loss=False)
        loss += super_ratio * super_loss
        for key in error.keys():
            metrics[key] = error[key]
        metrics["super_loss"] = super_loss.item()
        metrics["super_ratio"] = super_ratio.item()
    if isSelf:
        if disp_gt_l != None:
            cu_disp = apply_disparity_cu(disp_gt_l.unsqueeze(1), disp_gt_l.unsqueeze(1).type(torch.int))
            Mask = (cu_disp < 192) * (cu_disp > 0)
        else:
            Mask = None
        reproj_loss, _, _ = get_reproj_error_patch(
            input_L=item['img_L_reproj'],
            input_R=item['img_R_reproj'],
            pred_disp_l=-pred_disps[-1],
            mask=Mask, ps=11)
        metrics["reproj_loss"] = reproj_loss.item()
        metrics["self_ratio"] = self_ratio.item()
        loss += self_ratio * reproj_loss

    if dynamic_weights is not None and isSuper and isSelf and disp_gt_l is not None:
        dynamic_weights.update(super_loss.item(), reproj_loss.item())

    return loss, metrics


if __name__ == "__main__":
    # 生成随机数据
    B, C, H, W = 4, 3, 256, 256
    pred_disps = [torch.rand((B, 1, H, W)).cuda(), torch.rand((B, 1, H, W)).cuda(), torch.rand((B, 1, H, W)).cuda(), torch.rand((B, 1, H, W)).cuda()]
    target_disp = -torch.rand((B, H, W)).cuda() * 10
    conf = torch.ones_like(target_disp)
    im0 = torch.rand((B, C, H, W)).cuda()
    im1 = torch.rand((B, C, H, W)).cuda()
    im2 = torch.rand((B, C, H, W)).cuda()

    # 调用 ns_loss 计算损失和指标
    loss, metrics = ns_loss(pred_disps, target_disp, conf, im1, im2,
                            alpha_disp_loss=1.0,
                            alpha_photometric=0.1, conf_threshold=0.5)

    print("Loss:", loss.item())
    print("Metrics:", metrics)
