import copy
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def disp_svg(image, path):
    colormap = plt.get_cmap('jet')
    norm_disp_map = (image - np.min(image)) / (np.max(image) - np.min(image))
    # 将归一化后的视差图应用 colormap
    color_disp_map = colormap(norm_disp_map)
    # 显示彩色化的视差图
    plt.imshow(color_disp_map)
    plt.colorbar()
    plt.title('Colored Disparity Map')
    plt.savefig(path, format='svg', bbox_inches='tight')


def gen_error_colormap_disp():
    cols = np.array(
        [
            [0, 0.00001, 0, 0, 0],
            [0.00001, 0.1875 / 3.0, 49, 54, 149],
            [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
            [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
            [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
            [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
            [3 / 3.0, 6 / 3.0, 254, 224, 144],
            [6 / 3.0, 12 / 3.0, 253, 174, 97],
            [12 / 3.0, 24 / 3.0, 244, 109, 67],
            [24 / 3.0, 48 / 3.0, 215, 48, 39],
            [48 / 3.0, np.inf, 165, 0, 38],
        ],
        dtype=np.float32,
    )
    cols[:, 2:5] /= 255.0
    return cols


def disp_error_img(disp_pred_tensor, disp_gt_tensor, mask, abs_thres=3.0, rel_thres=0.05):
    D_gt_np = disp_gt_tensor.squeeze(0).detach().cpu().numpy()
    D_est_np = disp_pred_tensor.squeeze(0).detach().cpu().numpy()
    mask = mask.squeeze(0).detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = D_gt_np > 0
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(
        error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres
    )
    # get colormap
    cols = gen_error_colormap_disp()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[
            i, 2:
        ]

    error_image[np.logical_not(mask)] = 0.0

    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance : (i + 1) * distance, :] = cols[i, 2:]
    return error_image[0]  # [H, W, 3]


def gen_error_colormap_depth():
    cols = np.array(
        [
            [0, 0.00001, 0, 0, 0],
            [0.00001, 2000.0 / (2 ** 10), 49, 54, 149],
            [2000.0 / (2 ** 10), 2000.0 / (2 ** 9), 69, 117, 180],
            [2000.0 / (2 ** 9), 2000.0 / (2 ** 8), 116, 173, 209],
            [2000.0 / (2 ** 8), 2000.0 / (2 ** 7), 171, 217, 233],
            [2000.0 / (2 ** 7), 2000.0 / (2 ** 6), 224, 243, 248],
            [2000.0 / (2 ** 6), 2000.0 / (2 ** 5), 254, 224, 144],
            [2000.0 / (2 ** 5), 2000.0 / (2 ** 4), 253, 174, 97],
            [2000.0 / (2 ** 4), 2000.0 / (2 ** 3), 244, 109, 67],
            [2000.0 / (2 ** 3), 2000.0 / (2 ** 2), 215, 48, 39],
            [2000.0 / (2 ** 2), np.inf, 165, 0, 38],
        ],
        dtype=np.float32,
    )
    cols[:, 2:5] /= 255.0
    return cols


def depth_error_img(D_est_tensor, D_gt_tensor, mask, abs_thres=1.0):
    D_gt_np = D_gt_tensor.squeeze(0).detach().cpu().numpy()
    D_est_np = D_est_tensor.squeeze(0).detach().cpu().numpy()
    mask = mask.squeeze(0).detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = (D_gt_np > 0) & (D_gt_np < 1250)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = error[mask] / abs_thres
    # get colormap
    cols = gen_error_colormap_depth()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[
            i, 2:
        ]

    error_image[np.logical_not(mask)] = 0.0
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance : (i + 1) * distance, :] = cols[i, 2:]
    return error_image[0]  # [H, W, 3]




