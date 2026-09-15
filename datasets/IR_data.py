import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets.utils import *
"""
data typical
    sim data
        img_L, img_R, depth
        img_L -> pattern_left
        img_R -> pattern_right
    real data
        img_L, img_R, depth, AO, pattern_left, pattern_right

# baseline and ours 446.31  0.055
# 436.1351, 0.055
# d435 424.845  0.05
    
"""
# TODO define dataloader

class IR_loader(Dataset):
    def __init__(self, sim_path, real_path, d435_path, train=True, task="baseline",
                 backbone="raft", conf_threshold=0.5, disp_threshold=20.0):
        self.train = train
        self.tasks = task  # baseline, our, d435
        self.backbone = backbone

        self.data_aug = data_augmentation(True, True, backbone) if train else data_augmentation(False, False, backbone)
        self.sim_crop_size = (256, 256) if train else (640, 360)
        self.real_crop_size = (256, 512)

        self.sim_data = None
        self.len_sim = 0
        self.real_data = None
        self.len_real = 0
        self.d435_data = None
        self.len_d435 = 0

        self.sim_data = read_txt(sim_path)
        self.len_sim = len(self.sim_data)
        self.real_data = read_txt(real_path)
        self.len_real = len(self.real_data)
        self.d435_data = read_txt(d435_path)
        self.len_d435 = len(self.d435_data)

        self.fx_s = 446.31
        self.b_s = 0.055
        self.fx_d = 424.845
        self.b_d = 0.05
        self.conf_threshold = conf_threshold
        self.disp_threshold = disp_threshold

    def __len__(self):
        if self.tasks == "d435":
            return self.len_d435
        else: # baseline, our
            return self.len_sim

    def __getitem__(self, index):
        item = {"sim": {}, "real": {}}

        if self.tasks != "d435":
            # TODO sim
            left, right, left_mask, right_mask, disp = read_sim_data(self.sim_data[index], self.fx_s,
                                                                     self.b_s, isNorm=False)
            left, right, left_mask, right_mask, disp = image_crop_size([
                left, right, left_mask, right_mask, disp], self.sim_crop_size, isTrain=self.train)
            img_L_rgb, img_R_rgb = self.ir_process(left, right)

            item["sim"]["img_L"] = torch.as_tensor(img_L_rgb, dtype=torch.float32)
            item["sim"]["img_R"] = torch.as_tensor(img_R_rgb, dtype=torch.float32)
            item["sim"]["img_L_reproj"] = torch.as_tensor(left_mask, dtype=torch.float32).unsqueeze(0)
            item["sim"]["img_R_reproj"] = torch.as_tensor(right_mask, dtype=torch.float32).unsqueeze(0)
            if self.backbone == "raft":
                item["sim"]["disp_gt"] = torch.as_tensor(disp, dtype=torch.float32)
            else:
                item["sim"]["disp_gt"] = torch.as_tensor(disp, dtype=torch.float32).unsqueeze(0)
            item["sim"]["conf"] = torch.as_tensor(disp > 0, dtype=torch.float32)


            # TODO real
            left, right, left_mask, right_mask, disp, AO = read_real_data(
                self.real_data[index % self.len_real], self.conf_threshold, self.disp_threshold, isNorm=False)
            left, right, left_mask, right_mask, disp, AO = image_crop_size(
                [left, right, left_mask, right_mask, disp, AO], self.real_crop_size, isTrain=self.train)
            img_L_rgb, img_R_rgb = self.ir_process(left, right)

            item["real"]["img_L"] = torch.as_tensor(img_L_rgb, dtype=torch.float32)
            item["real"]["img_R"] = torch.as_tensor(img_R_rgb, dtype=torch.float32)
            item["real"]["img_L_reproj"] = torch.as_tensor(left_mask, dtype=torch.float32).unsqueeze(0)
            item["real"]["img_R_reproj"] = torch.as_tensor(right_mask, dtype=torch.float32).unsqueeze(0)
            if self.tasks == "baseline":
                return item


            if self.backbone == "raft":
                item["real"]["disp_gt"] = torch.as_tensor(disp, dtype=torch.float32)
                item["real"]["conf"] = torch.as_tensor(AO, dtype=torch.float32)
            else:
                item["real"]["disp_gt"] = torch.as_tensor(disp, dtype=torch.float32).unsqueeze(0)
                item["real"]["conf"] = torch.as_tensor(AO, dtype=torch.float32).unsqueeze(0)
            if self.tasks == "ours":
                return item

        else:
            # TODO d435
            if self.train == True:
                left, right, left_mask, right_mask, disp = read_d435_data(self.d435_data[index % self.len_d435],
                                                                          fx=self.fx_d, b=self.b_d, isNorm=False)
            else:
                left, right, left_mask, right_mask, disp = read_d435_data(self.d435_data[index % self.len_d435],
                                                                          fx=self.fx_s, b=self.b_s, isNorm=False)
            left, right, left_mask, right_mask, disp = image_crop_size([
                left, right, left_mask, right_mask, disp], self.sim_crop_size, isTrain=self.train)
            img_L_rgb, img_R_rgb = self.ir_process(left, right)

            item["sim"]["img_L"] = torch.as_tensor(img_L_rgb, dtype=torch.float32)
            item["sim"]["img_R"] = torch.as_tensor(img_R_rgb, dtype=torch.float32)
            item["sim"]["img_L_reproj"] = torch.as_tensor(left_mask, dtype=torch.float32).unsqueeze(0)
            item["sim"]["img_R_reproj"] = torch.as_tensor(right_mask, dtype=torch.float32).unsqueeze(0)
            if self.backbone == "raft":
                item["sim"]["disp_gt"] = torch.as_tensor(disp, dtype=torch.float32)
                item["sim"]["conf"] = torch.as_tensor(disp > 0, dtype=torch.float32)
            else:
                item["sim"]["disp_gt"] = torch.as_tensor(disp, dtype=torch.float32).unsqueeze(0)
                item["sim"]["conf"] = torch.as_tensor(disp > 0, dtype=torch.float32)
            return item

    def ir_process(self, left, right):
        img_L_rgb = np.repeat(left[:, :, None], 3, axis=-1)
        img_R_rgb = np.repeat(right[:, :, None], 3, axis=-1)

        if self.train:
            img_L_rgb = self.data_aug(img_L_rgb).type(torch.FloatTensor)
            img_R_rgb = self.data_aug(img_R_rgb).type(torch.FloatTensor)
            if self.backbone == "raft":
                img_L_rgb = img_L_rgb * 255
                img_L_rgb = img_L_rgb.byte()
                img_R_rgb = img_R_rgb * 255
                img_R_rgb = img_R_rgb.byte()
        else:
            if self.backbone == "stereonet" or self.backbone == "psmnet":
                img_L_rgb = self.data_aug(img_L_rgb).type(torch.FloatTensor)
                img_R_rgb = self.data_aug(img_R_rgb).type(torch.FloatTensor)
            else:
                img_L_rgb = np.transpose(img_L_rgb, (2, 0, 1))
                img_R_rgb = np.transpose(img_R_rgb, (2, 0, 1))
        return img_L_rgb, img_R_rgb

