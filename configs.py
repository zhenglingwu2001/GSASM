import os
from easydict import EasyDict as edit

cfg = edit()
# command config
cfg.backbone = None
cfg.datatype = None
cfg.device = None
cfg.test_model_path = None

# train
cfg.lr = 2.0e-4
cfg.wdecay = 1.0e-5
cfg.batch_size = 4
cfg.save_step = 2000
cfg.num_steps = 10000
cfg.seed = 10
cfg.lr_scheduler = "StepLR"  # StepLR or ploy, linear, cos
cfg.push_num = 200

# dataset root
cfg.root = "/mnt/home/wuzhengling/NS/Nerf_Stereo_open"
cfg.data_path = "/mnt/home/wuzhengling/NS/Nerf_Stereo_open/data"
cfg.pred_train_data = os.path.join(cfg.data_path, "pred_train.txt")
cfg.sim_train = os.path.join(cfg.data_path, "sim_train.txt")
cfg.sim_val = os.path.join(cfg.data_path, "sim_val.txt")
cfg.real_train = os.path.join(cfg.data_path, "real_train.txt")
cfg.d435_train = os.path.join(cfg.data_path, "d435_train.txt")

# raft data param
cfg.conf_threshold = 0.5
cfg.disp_threshold = 20.0
cfg.alpha_disp_loss = 1.0
cfg.alpha_photometric = 0.1

# TODO Model params
# PSMNet
cfg.max_disp = 192
# Raft
cfg.corr_implementation = "reg"   # "reg", "alt", "reg_cuda", "alt_cuda"
cfg.context_norm = "batch"        # 'group', 'batch', 'instance', 'none'
cfg.shared_backbone = True        # "use a single backbone for the context and feature encoders"
cfg.corr_levels = 4
cfg.corr_radius = 4               # width of the correlation pyramid
cfg.n_downsample = 2
cfg.slow_fast_gru = True
cfg.n_gru_layers = 3
cfg.hidden_dims = [128] * 3
cfg.mixed_precision = True        # use mixed precision
cfg.train_iters = 22
cfg.valid_iters = 32
# StereoNet
cfg.in_channels = 3
cfg.k_downsampling_layers = 3
cfg.k_refinement_layers = 3
cfg.candidate_disparities = 256
cfg.mask = True

# psmnet data_augmentation
cfg.bright_min = 0.4
cfg.bright_max = 1.4
cfg.contrast_min = 0.8
cfg.contrast_max = 1.2
cfg.guassian_min = 0.1
cfg.guassian_max = 2.0
cfg.guassian_kernel = 9

