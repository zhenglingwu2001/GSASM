import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import numpy as np
from Core.Basic_Framework import Train_Base

from loss.psmnet import total_loss as psmnet_loss
from loss.raft import total_loss as raft_loss
from loss.stereonet import total_loss as stereonet_loss

class NormalizedDynamicWeights:
    def __init__(self, init_super=0.01, init_self=2.0, alpha=0.1, normalize_mode="softmax"):
        self.raw_super = torch.tensor(init_super, requires_grad=False)
        self.raw_self = torch.tensor(init_self, requires_grad=False)
        self.alpha = alpha
        self.normalize_mode = normalize_mode
        self.prev_super_loss = None
        self.prev_self_loss = None

    def _normalize(self):
        if self.normalize_mode == "softmax":
            exp_super = torch.exp(self.raw_super)
            exp_self = torch.exp(self.raw_self)
            sum_exp = exp_super + exp_self
            return exp_super / sum_exp, exp_self / sum_exp
        elif self.normalize_mode == "l1":
            sum_weights = self.raw_super + self.raw_self
            return self.raw_super / sum_weights, self.raw_self / sum_weights
        else:
            raise ValueError("The normalisation method supports 'softmax' or 'l1'")

    def get_weights(self):
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
        self.raw_super = torch.clamp(self.raw_super, 1e-3, 10.0)
        self.raw_self = torch.clamp(self.raw_self, 1e-3, 10.0)
        self.prev_super_loss = current_super_loss
        self.prev_self_loss = current_self_loss

def model_loss(item, pred_disps, backbone, is_super, is_self, dynamic_weights=None):
    if is_super == False:
        disp_gt = None
    else:
        disp_gt = item["disp_gt"]

    if backbone == "psmnet":
        loss, metrics = psmnet_loss(item, pred_disps, dynamic_weights, disp_gt,
                                    isSuper=is_super, isSelf=is_self)
    if backbone == "raft":
        loss, metrics = raft_loss(item, pred_disps, dynamic_weights, disp_gt,
                                  isSuper=is_super, isSelf=is_self)
    if backbone == "stereonet":
        loss, metrics = stereonet_loss(item, pred_disps, dynamic_weights, disp_gt,
                                       isSuper=is_super, isSelf=is_self)
    return loss, metrics


def val_metrics(pred_disps, disp_gt_l, backbone):
    metrics = {}
    if backbone == "psmnet":
        pred_disps = pred_disps[0]

    if backbone == "stereonet":
        pred_disps = pred_disps[-1]

    if backbone == "raft":
        pred_disps = -pred_disps[-1]
        disp_gt_l = disp_gt_l.unsqueeze(1)

    epe = torch.sum((pred_disps - disp_gt_l) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[(disp_gt_l > 0).view(-1)]
    metrics["disp_err"] = epe.mean().item()
    metrics["1px"] = (epe < 1).float().mean().item()
    metrics["3px"] = (epe < 3).float().mean().item()
    metrics["5px"] = (epe < 5).float().mean().item()
    return metrics


# TODO main train
class Train(Train_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_weights = NormalizedDynamicWeights(
            init_super=0.01, 
            init_self=2.0, 
            alpha=0.1, 
            normalize_mode="softmax"
        )

    def get_data(self, data, data_type):
        if data_type == "self":
            data["img_L"] = data["img_L"].to(self.device)
            data["img_R"] = data["img_R"].to(self.device)
            data["img_L_reproj"] = data["img_L_reproj"].to(self.device)
            data["img_R_reproj"] = data["img_R_reproj"].to(self.device)
        else:
            # psmnet and stereonet
            data["img_L"] = data["img_L"].to(self.device)
            data["img_R"] = data["img_R"].to(self.device)
            data["img_L_reproj"] = data["img_L_reproj"].to(self.device)
            data["img_R_reproj"] = data["img_R_reproj"].to(self.device)
            data['disp_gt'] = data['disp_gt'].to(self.device)
            if self.opt.backbone == "raft":
                data['conf'] = data['conf'].to(self.device)
        return data

    # sim data: super + self
    def train_sim_batch(self, data, use_dynamic=True):
        item = self.get_data(data["sim"], "super")
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            pred_disps = self.input_to_model(item)
            dw = self.dynamic_weights if use_dynamic else None   # ← 新增开关
            loss, metrics = model_loss(
                item, pred_disps, self.opt.backbone,
                is_super=True, is_self=True,
                dynamic_weights=dw
            )
        self.scaler_update(loss)
        return loss, metrics

    # d435 data: super + self
    # same
    def train_d435_batch(self, data):
        item = self.get_data(data["sim"], "super")
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            pred_disps = self.input_to_model(item)
            loss, metrics = model_loss(
                item, pred_disps, self.opt.backbone,
                is_super=True, is_self=True,
                dynamic_weights=self.dynamic_weights
            )
        self.scaler_update(loss)
        return loss, metrics

    # sim self super, real self super
    def train_self_batch(self, data):
        # metrics: disp_err, 1px, 3px, 5px, self_loss
        item = self.get_data(data["sim"], "self")
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            pred_disps = self.input_to_model(item)
            sim_loss, sim_metrics = model_loss(item, pred_disps, self.opt.backbone,
                                               is_super=False, is_self=True)
        self.scaler_update(sim_loss)
        # real
        item = self.get_data(data["real"], "self")
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            pred_disps = self.input_to_model(item)
            real_loss, real_metrics = model_loss(item, pred_disps, self.opt.backbone,
                                                 is_super=False, is_self=True)
        self.scaler_update(real_loss)
        return sim_loss, sim_metrics, real_loss, real_metrics

    # real data: super
    def train_real_batch(self, data):
        item = self.get_data(data["real"], "super")
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            pred_disps = self.input_to_model(item)
            real_loss, real_metrics = model_loss(item, pred_disps, self.opt.backbone,
                                                 is_super=True, is_self=False)
        self.scaler_update(real_loss)
        return real_loss, real_metrics
    
    def train_real_ss_batch(self, data):
        item = self.get_data(data["real"], "super")
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            pred_disps = self.input_to_model(item)
            real_loss, real_metrics = model_loss(
                item, pred_disps, self.opt.backbone,
                is_super=True, is_self=True,
                dynamic_weights=self.dynamic_weights
            )
        self.scaler_update(real_loss)
        return real_loss, real_metrics

    # sim super + sim self super, real self super
    def train_baseline_batch(self, data):
        sim_loss, sim_metrics = self.train_sim_batch(data, use_dynamic=False)

        item = self.get_data(data["real"], "self")
        self.optimizer.zero_grad()
        with autocast(enabled=True):
            pred_disps = self.input_to_model(item)
            real_loss, real_metrics = model_loss(item, pred_disps, self.opt.backbone,
                                                 is_super=False, is_self=True)
        self.scaler_update(real_loss)
        return sim_loss + real_loss, sim_metrics, real_metrics

    def sim_eval(self):
        metrics = {'disp_err': 0, '1px': 0, '3px': 0, '5px': 0}
        with torch.no_grad():
            loop = tqdm(self.val_loader, total=len(self.val_loader))
            for i_batch, data in enumerate(loop):
                item = self.get_data(data["sim"], "super")
                pred_disps = self.input_to_model(item, False)
                error = val_metrics(pred_disps, item["disp_gt"], self.opt.backbone)
                for key in metrics.keys():
                    metrics[key] += error[key]

                loop.set_description('val: ')
                loop.set_postfix(disp_err=error["disp_err"])

            for key in metrics.keys():
                metrics[key] /= len(self.val_loader)
            self.writer.add_scalar("val/disp_err", metrics["disp_err"], self.val_step)
            self.writer.add_scalar("val/1px", metrics["1px"], self.val_step)
            self.writer.add_scalar("val/3px", metrics["3px"], self.val_step)
            self.writer.add_scalar("val/5px", metrics["5px"], self.val_step)
            print(metrics["disp_err"])
            print(metrics["1px"])
            print(metrics["3px"])
            print(metrics["5px"])
            self.val_step = self.val_step + 1

            if metrics["disp_err"] < self.best_error:
                self.best_error = metrics["disp_err"]
                self.save_ckpt(os.path.join(self.param_path, "best_model.pth"),
                               self.val_step, metrics=metrics)