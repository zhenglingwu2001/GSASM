import torch
import torch.nn.functional as F
from loss.self_loss import get_reproj_error_patch, apply_disparity_cu

def stereo_net_loss(left_img, right_img, pred_disp, disp_gt,
                    lambda_reconstruction=1.0, lambda_smoothness=1.0):
    reconstructed_left = warp_image(right_img, pred_disp)
    loss_reconstruction = F.l1_loss(reconstructed_left, left_img)

    loss_smoothness = smoothness_loss(pred_disp, left_img)

    loss_disp = F.l1_loss(pred_disp, disp_gt)

    total_loss = (lambda_reconstruction * loss_reconstruction +
                  lambda_smoothness * loss_smoothness +
                  loss_disp)

    return total_loss

def warp_image(img, disp):
    B, C, H, W = img.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if img.is_cuda:
        grid = grid.cuda()
    vgrid = grid + disp

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / (W - 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / (H - 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(img, vgrid, mode='bilinear', padding_mode='border')
    return output

def smoothness_loss(disp, img):
    def gradient(x):
        D_dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        D_dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return D_dx, D_dy

    disp_dx, disp_dy = gradient(disp)
    img_dx, img_dy = gradient(img)

    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True))

    smoothness_x = disp_dx * weights_x
    smoothness_y = disp_dy * weights_y

    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))


def robust_loss(x: torch.Tensor, alpha: float, c: float) -> torch.Tensor:
    return (abs(alpha - 2) / alpha) * (torch.pow(torch.pow(x / c, 2) / abs(alpha - 2) + 1, alpha / 2) - 1)
def stereonet_loss(pred_left_disp, disp_gt, mask):
    total_loss = 0
    scale_weights = [0.25, 0.5, 0.8, 1.0]
    for i, disp_pred_left in enumerate(pred_left_disp):
        disp_pred_left_rescaled = F.interpolate(disp_pred_left, size=(disp_gt.size(2), disp_gt.size(3)),
                                                mode='bilinear', align_corners=True)
        loss_left = torch.mean(robust_loss(disp_gt[mask] - disp_pred_left_rescaled[mask], alpha=1, c=2))
        total_loss += loss_left * scale_weights[i]
    return total_loss

def total_loss(item, pred_disps, dynamic_weights, disp_gt_l=None, isSuper=True, isSelf=False):
    loss = 0
    metrics = {}
    if dynamic_weights is not None:
        super_ratio, self_ratio = dynamic_weights.get_weights()
    else:
        super_ratio = torch.tensor(0.01, device=disp_gt_l.device if disp_gt_l is not None else 'cpu')
        self_ratio = torch.tensor(2.0, device=disp_gt_l.device if disp_gt_l is not None else 'cpu')

    if isSuper:
        super_loss = stereonet_loss(pred_left_disp=pred_disps, disp_gt=disp_gt_l, mask=disp_gt_l > 0)
        """
        super_loss = stereo_net_loss(left_img=item["img_L"], right_img=item["img_R"],
                                     pred_disp=pred_disps[-1], disp_gt=disp_gt_l,
                                     lambda_reconstruction=1.0, lambda_smoothness=1.0)
        """
        loss += super_ratio * super_loss
        metrics["super_loss"] = super_loss.item()
        metrics["super_ratio"] = super_ratio.item()
    if isSelf:
        if disp_gt_l != None:
            cu_disp = apply_disparity_cu(disp_gt_l, disp_gt_l.type(torch.int))
            Mask = (cu_disp < 192) * (cu_disp > 0)
        else:
            Mask = None
        reproj_loss, _, _ = get_reproj_error_patch(
            input_L=item['img_L_reproj'],
            input_R=item['img_R_reproj'],
            pred_disp_l=pred_disps[-1],
            mask=Mask, ps=11)
        metrics["reproj_loss"] = reproj_loss.item()
        metrics["self_ratio"] = self_ratio.item()
        loss += self_ratio * reproj_loss

    if disp_gt_l != None:
        epe = torch.sum((pred_disps[-1] - disp_gt_l) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[(disp_gt_l > 0).view(-1)]
        metrics["disp_err"] = epe.mean().item()
        metrics["1px"] = (epe < 1).float().mean().item()
        metrics["3px"] = (epe < 3).float().mean().item()
        metrics["5px"] = (epe < 5).float().mean().item()

    if dynamic_weights is not None and isSuper and isSelf and disp_gt_l is not None:
        dynamic_weights.update(super_loss.item(), reproj_loss.item())

    return loss, metrics

if __name__ == "__main__":
    left_img = torch.randn((1, 3, 256, 512)).cuda()
    right_img = torch.randn((1, 3, 256, 512)).cuda()
    disp_left = torch.randn((1, 2, 256, 512)).cuda() 
    target_left = torch.randn((1, 2, 256, 512)).cuda()

    loss = stereo_net_loss(left_img, right_img, disp_left, target_left,
                           lambda_reconstruction=1.0, lambda_smoothness=1.0)
    print(loss)
