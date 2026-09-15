import torch
import random
import argparse
import numpy as np
from torch.cuda.amp import GradScaler
from configs import cfg
from init_build import build_dataloader, build_model_optim
from Core.Train_Framework import Train

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datatype", type=str, default="ours")
    parser.add_argument("--backbone", type=str, default="psmnet")
    parser.add_argument("--test", type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_args()
    cfg.backbone = opt.backbone
    cfg.datatype = opt.datatype


    set_random_seed(cfg.seed)
    # TODO GPU_ID
    torch.cuda.set_device(opt.gpu_id)
    cfg.device = torch.device(f"cuda:{opt.gpu_id}")
    # Load dataloader
    train_loader, val_loader = build_dataloader(cfg)
    model, optimizer, scheduler = build_model_optim(cfg)
    scaler = GradScaler()

    train = Train(opt=cfg, device=cfg.device, model=model, optimizer=optimizer,
                  scheduler=scheduler, scaler=scaler, train_loader=train_loader,
                  val_loader=val_loader)
    train.fit()



