import os
import gc
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter


class Train_Base:
    def __init__(self, opt, device, model, optimizer, scheduler,
                 scaler, train_loader, val_loader=None):
        self.opt = opt
        self.device = device

        self.model = model
        self.optimizer = optimizer

        self.scaler = scaler
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.writer = SummaryWriter(log_dir=f"./runs/{self.opt.backbone}/{self.opt.datatype}")
        self.param_path = None
        self.latest_path = None
        self.pred_model = None
        self.init_path()

        self.metrics = {"disp_err": 0, "1px": 0, "3px": 0, "5px": 0}
        self.push_num = opt.push_num

        self.total_step = self.init_ckpt(self.param_path, self.latest_path, self.opt.datatype)
        self.best_error = np.inf
        self.val_step = 1

    def init_path(self):
        self.param_path = os.path.join(self.opt.root, "checkpoints", self.opt.backbone, self.opt.datatype)
        self.latest_path = os.path.join(self.param_path, f"model_latest.pth")
        self.pred_model = os.path.join(self.opt.root, "pred_model", self.opt.backbone, "model_latest.pth")

    def scaler_update(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def save_ckpt(self, model_path, total_step, metrics=None):
        checkpoints = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_step": total_step,
            "metrics": metrics
        }
        if hasattr(self, 'dynamic_weights') and self.dynamic_weights is not None:
            checkpoints["dynamic_weights"] = {
                "raw_super": self.dynamic_weights.raw_super,
                "raw_self": self.dynamic_weights.raw_self,
                "prev_super_loss": self.dynamic_weights.prev_super_loss,
                "prev_self_loss": self.dynamic_weights.prev_self_loss
            }
        torch.save(checkpoints, model_path)

    def init_ckpt(self, param_path, latest_path, datatype):
        os.makedirs(param_path, exist_ok=True)
        if os.path.exists(latest_path):
            dict_param = torch.load(latest_path)
            self.model.load_state_dict(dict_param["model"])
            self.optimizer.load_state_dict(dict_param["optimizer"])
            total_step = dict_param.get("total_step")
            print(f"************ recover train {datatype} model param !!! ************")
            return total_step
        else:
            print(f"************ training {datatype} new model !!! ************")
            return 1

    def push_infor(self, metrics, total_step):
        for key in self.metrics.keys():
            self.metrics[key] += metrics[key]
        if total_step % self.push_num == 0:
            for key in self.metrics.keys():
                self.metrics[key] /= self.push_num
                print(f"{key}: {self.metrics[key]}")
            self.metrics = {"disp_err": 0, "1px": 0, "3px": 0, "5px": 0}

    def push(self, metrics, total_step, Type="sim"):
        for key in metrics.keys():
            self.writer.add_scalar(f"train/{Type}_{key}", metrics[key], total_step)

    def update_lr(self):
        lr = self.scheduler.get_lr(self.total_step)
        self.optimizer.param_groups[0]["lr"] = lr
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.total_step)

    def input_to_model(self, data, isTrain=True):
        if self.opt.backbone == "psmnet":
            pred_disps = self.model(data["img_L"], data["img_R"])
        elif self.opt.backbone == "raft":
            if isTrain:
                pred_disps = self.model(data["img_L"], data["img_R"], self.opt.train_iters)
            else:
                pred_disps = self.model(data["img_L"], data["img_R"], self.opt.valid_iters)
        else:  # StereoNet
            inputs = torch.cat([data["img_L"], data["img_R"]], dim=1)
            pred_disps = self.model(inputs)
        return pred_disps



    def self_data(self, data):
        raise NotImplementedError

    def read_sim_data(self, data):
        raise NotImplementedError

    def read_real_data(self, data):
        raise NotImplementedError

    def train_sim_batch(self, data):
        raise NotImplementedError

    def train_d435_batch(self, data):
        raise NotImplementedError

    def train_self_batch(self, data):
        raise NotImplementedError

    def train_real_batch(self, data):
        raise NotImplementedError

    def train_baseline_batch(self, data):
        raise NotImplementedError

    def sim_eval(self):
        raise NotImplementedError

    def fit(self):
        global_num_step = True
        while global_num_step:
            self.model.train()

            loop = tqdm(self.train_loader, total=len(self.train_loader))
            for i_batch, data in enumerate(loop):
                self.update_lr()

                # super + self super
                if self.opt.datatype == "sim":
                    total_loss, metrics = self.train_sim_batch(data)

                    self.push_infor(metrics, self.total_step)
                    self.push(metrics, self.total_step, "sim")
                    loop.set_description(f'Epoch train: [{self.total_step}/{self.opt.num_steps}]')
                    loop.set_postfix(sim_err=metrics["disp_err"])

                if self.opt.datatype == "d435":
                    total_loss, metrics = self.train_d435_batch(data)

                    self.push_infor(metrics, self.total_step)
                    self.push(metrics, self.total_step, "d435")
                    loop.set_description(f'Epoch train: [{self.total_step}/{self.opt.num_steps}]')
                    loop.set_postfix(sim_err=metrics["disp_err"])

                # sim super + sim self super, real self super
                if self.opt.datatype == "baseline":
                    total_loss, sim_metrics, real_metrics = self.train_baseline_batch(data)

                    self.push_infor(sim_metrics, self.total_step)
                    self.push(sim_metrics, self.total_step, "sim")
                    loop.set_description(f'Epoch train: [{self.total_step}/{self.opt.num_steps}]')
                    loop.set_postfix(sim_err=sim_metrics["disp_err"])

                # sim super + sim self super, real super + real self super
                if self.opt.datatype == "ours":
                    sim_total_loss, sim_metrics = self.train_sim_batch(data)
                    real_total_loss, real_metrics = self.train_real_batch(data)

                    self.push_infor(real_metrics, self.total_step)
                    self.push(sim_metrics, self.total_step, "sim")
                    self.push(real_metrics, self.total_step, "real")
                    loop.set_description(f'Epoch train: [{self.total_step}/{self.opt.num_steps}]')
                    loop.set_postfix(sim_err=sim_metrics["disp_err"], real_err=real_metrics["disp_err"])

                # real super + real self super
                if self.opt.datatype == "real":
                    total_loss, metrics = self.train_real_batch(data)

                    self.push_infor(metrics, self.total_step)
                    self.push(metrics, self.total_step, "real")
                    loop.set_description(f'Epoch train: [{self.total_step}/{self.opt.num_steps}]')
                    loop.set_postfix(real_err=metrics["disp_err"])

                # sim self super, real self super
                if self.opt.datatype == "self":
                    sim_reproj, sim_metrics, real_reproj, real_metrics = self.train_self_batch(data)
                    loop.set_description(f'Epoch train: [{self.total_step}/{self.opt.num_steps}]')
                    loop.set_postfix(sim_loss=sim_reproj.item(), real_loss=real_reproj.item())

                if self.total_step % self.opt.save_step == 0:
                    torch.cuda.empty_cache()
                    if self.opt.datatype != "real":
                        self.model.eval()
                        self.sim_eval()
                        self.model.train()

                    self.save_ckpt(os.path.join(self.param_path, f"model_{self.total_step}.pth"),
                                   self.total_step, metrics=None)
                    self.save_ckpt(self.latest_path, self.total_step, metrics=None)

                if self.total_step >= self.opt.num_steps:
                    global_num_step = False
                    self.save_ckpt(self.latest_path, self.total_step, metrics=None)
                    break
                self.total_step += 1
            gc.collect()

