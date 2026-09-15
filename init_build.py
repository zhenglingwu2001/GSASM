import torch
from utils.lr_policy import WarmUp_LR, WarmUp_StepLR
from datasets.IR_data import IR_loader, DataLoader


def build_dataloader(cfg):
    # baseline and ours 446.31  0.055
    # 436.1351, 0.055
    # d435 424.845  0.05

    train_dataset = IR_loader(sim_path=cfg.sim_train, real_path=cfg.real_train, task=cfg.datatype,
                              d435_path=cfg.d435_train, train=True, backbone=cfg.backbone)
    val_dataset = IR_loader(sim_path=cfg.sim_val, real_path=cfg.real_train, task=cfg.datatype,
                            d435_path=cfg.sim_val, train=False,  backbone=cfg.backbone)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,  num_workers=2,
                            pin_memory=True, drop_last=False)

    return train_loader, val_loader


def build_model_optim(cfg):
    if cfg.backbone == "psmnet":
        from model.psmnet.psmnet import Adapter_PSMNet
        model = Adapter_PSMNet().to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=1.0e-5, betas=(0.9, 0.999))
    if cfg.backbone == "raft":
        from model.raft.raft_stereo import RAFTStereo
        model = RAFTStereo(cfg).to(cfg.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=1.0e-5, eps=1.0e-8)
    if cfg.backbone == "stereonet":
        from model.stereonet.model import StereoNet
        model = StereoNet(in_channels=cfg.in_channels,
                          k_downsampling_layers=cfg.k_downsampling_layers,
                          k_refinement_layers=cfg.k_refinement_layers,
                          candidate_disparities=cfg.candidate_disparities,
                          mask=cfg.mask).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=1.0e-5, betas=(0.9, 0.999))

    if cfg.lr_scheduler != "StepLR":
        lr_policy = WarmUp_LR(base_lr=cfg.lr, total_steps=cfg.num_steps + 100,
                              warmup_ratio=0.01, Types=cfg.lr_scheduler, min_lr=1.0e-7)
    else:
        lr_policy = WarmUp_StepLR(base_lr=cfg.lr, total_steps=cfg.num_steps + 100,
                                  warmup_ratio=0.01, step_size=6, gamma=0.5)

    return model, optimizer, lr_policy
