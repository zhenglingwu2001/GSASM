import torch
import torch.nn.functional as F
from loss.self_loss import get_reproj_error_patch, apply_disparity_cu

def psmnet_disp(pred_disp, disp_gt_l, mask):
    pred_disp3, pred_disp2, pred_disp1 = pred_disp
    loss_disp = (0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt_l[mask], reduction="mean")
                 + 0.7 * F.smooth_l1_loss(pred_disp2[mask], disp_gt_l[mask], reduction="mean")
                 + F.smooth_l1_loss(pred_disp3[mask], disp_gt_l[mask], reduction="mean"))
    return loss_disp

class NormalizedDynamicWeights:
    def __init__(self, init_super=0.01, init_self=2.0, alpha=0.1, normalize_mode="softmax"):
        """
        Normalised Dynamic Weight Manager
        :param init_super: Supervised loss init weight
        :param init_self: Self loss init weight
        :param alpha: rate(0~1)
        :param normalize_mode: Normalize ("softmax" or "l1")
        """
        self.raw_super = torch.tensor(init_super, requires_grad=False)  # Init weight
        self.raw_self = torch.tensor(init_self, requires_grad=False)
        self.alpha = alpha  # Weight update rate
        self.normalize_mode = normalize_mode
        self.prev_super_loss = None  # Supervised loss from the previous round
        self.prev_self_loss = None  # Self loss from the previous round

    def _normalize(self):
        if self.normalize_mode == "softmax":
            # softmax
            exp_super = torch.exp(self.raw_super)
            exp_self = torch.exp(self.raw_self)
            sum_exp = exp_super + exp_self
            return exp_super / sum_exp, exp_self / sum_exp
        elif self.normalize_mode == "l1":
            # L1
            sum_weights = self.raw_super + self.raw_self
            return self.raw_super / sum_weights, self.raw_self / sum_weights
        else:
            raise ValueError("The normalisation method supports 'softmax' or 'l1'")

    def get_weights(self):
        """Obtain the normalised weights"""
        return self._normalize()

    def update(self, current_super_loss, current_self_loss):
        if self.prev_super_loss is None or self.prev_self_loss is None:
            self.prev_super_loss = current_super_loss
            self.prev_self_loss = current_self_loss
            return

        super_change = current_super_loss / (self.prev_super_loss + 1e-8)
        self_change = current_self_loss / (self.prev_self_loss + 1e-8)

        self.raw_super *= (1 + self.alpha * (super_change - 1))
        self.raw_self *= (1 + self.alpha * (self_change - 1))

        self.raw_super = torch.clamp(self.raw_super, 1e-4, 100.0)
        self.raw_self = torch.clamp(self.raw_self, 1e-4, 100.0)

        self.prev_super_loss = current_super_loss
        self.prev_self_loss = current_self_loss

import torch
import torch.nn.functional as F
from loss.self_loss import get_reproj_error_patch, apply_disparity_cu

def psmnet_disp(pred_disp, disp_gt_l, mask):
    pred_disp3, pred_disp2, pred_disp1 = pred_disp
    loss_disp = (0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt_l[mask], reduction="mean")
                 + 0.7 * F.smooth_l1_loss(pred_disp2[mask], disp_gt_l[mask], reduction="mean")
                 + F.smooth_l1_loss(pred_disp3[mask], disp_gt_l[mask], reduction="mean"))
    return loss_disp

def total_loss(item, pred_disps, dynamic_weights, disp_gt_l=None, isSuper=True, isSelf=False):
    loss = 0
    metrics = {}
    if dynamic_weights is not None:
        super_ratio, self_ratio = dynamic_weights.get_weights() 
    else:
        super_ratio = torch.tensor(0.01, device=disp_gt_l.device if disp_gt_l is not None else 'cpu')
        self_ratio = torch.tensor(2.0, device=disp_gt_l.device if disp_gt_l is not None else 'cpu')

    if isSuper and disp_gt_l is not None:
        mask = disp_gt_l > 0
        super_loss = psmnet_disp(pred_disps, disp_gt_l, mask)
        loss += super_ratio * super_loss
        metrics["super_loss"] = super_loss.item()
        metrics["super_ratio"] = super_ratio.item()

    if isSelf:
        if disp_gt_l is not None:
            cu_disp = apply_disparity_cu(disp_gt_l, disp_gt_l.type(torch.int))
            Mask = (cu_disp < 192) * (cu_disp > 0)
        else:
            Mask = None
        reproj_loss, _, _ = get_reproj_error_patch(
            input_L=item['img_L_reproj'],
            input_R=item['img_R_reproj'],
            pred_disp_l=pred_disps[0],
            mask=Mask, ps=11)
        metrics["reproj_loss"] = reproj_loss.item()
        metrics["self_ratio"] = self_ratio.item()
        loss += self_ratio * reproj_loss

    if disp_gt_l is not None:
        epe = torch.sum((pred_disps[0] - disp_gt_l)** 2, dim=1).sqrt()
        epe = epe.view(-1)[(disp_gt_l > 0).view(-1)]
        metrics["disp_err"] = epe.mean().item()
        metrics["1px"] = (epe < 1).float().mean().item()
        metrics["3px"] = (epe < 3).float().mean().item()
        metrics["5px"] = (epe < 5).float().mean().item()

    if dynamic_weights is not None and isSuper and isSelf and disp_gt_l is not None:
        dynamic_weights.update(super_loss.item(), reproj_loss.item())

    return loss, metrics