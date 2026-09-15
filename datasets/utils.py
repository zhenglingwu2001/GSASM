import cv2
import random
import numpy as np
from torchvision import transforms

def read_txt(root):
    data = []
    with open(root, 'r') as f:
        for file in f.readlines():
            file = file.strip()
            data.append(file)
    random.shuffle(data)
    return data

def get_ir_pattern(img, ks=8):
    h, w = img.shape
    hs = int(h // ks)
    ws = int(w // ks)
    diff = (img - np.min(img)) / (np.max(img) - np.min(img))
    diff_avg = cv2.resize(diff, (ws, hs), interpolation=cv2.INTER_AREA)
    diff_avg = cv2.resize(diff_avg, (w, h), interpolation=cv2.INTER_AREA)
    ir = np.zeros_like(diff)
    diff2 = diff - diff_avg
    ir[diff2 > 0.005] = 1
    return ir

# PSMNet
def data_augmentation(gaussian_blur=False, color_jitter=False, backbone="stereonet"):
    transform_list = [transforms.ToTensor()]
    if gaussian_blur:
        gaussian_sig = random.uniform(0.1, 2.0)
        transform_list += [transforms.GaussianBlur(kernel_size=9,
                                                   sigma=gaussian_sig)]
    if color_jitter:
        bright = random.uniform(0.4, 1.4)
        contrast = random.uniform(0.8, 1.2)
        transform_list += [transforms.ColorJitter(brightness=[bright, bright],
                                                  contrast=[contrast, contrast])]
    # Normalization
    if backbone == "stereonet" or backbone == "psmnet":
        transform_list += [
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]
    custom_augmentation = transforms.Compose(transform_list)
    return custom_augmentation

def image_crop_size(data, crop_size, isTrain=True):
    image = []
    th, tw = crop_size
    if isTrain:
        h, w = data[0].shape[0:2]
        x = random.randint(0, h - th)
        y = random.randint(0, w - tw)
        for i in range(len(data)):
            image.append(data[i][x: (x + th), y: (y + tw)])
    else:
        for i in range(len(data)):
            image.append(cv2.resize(data[i], dsize=crop_size))
    return image

# isNorm区分raft和其他模型
def read_sim_data(path, s_fx, s_b, isNorm=True):
    parts = path.strip().split()
    
    left, right, depth = parts[0], parts[1], parts[2]
    disp_info = parts[3] if len(parts) > 3 and parts[3] else "default"
    
    if isNorm:
        left_img = cv2.imread(left, -1) / 255
        right_img = cv2.imread(right, -1) / 255
        left_mask = get_ir_pattern(left_img)
        right_mask = get_ir_pattern(right_img)
    else:
        left_img = cv2.imread(left, -1)
        right_img = cv2.imread(right, -1)
        left_mask = get_ir_pattern(left_img / 255)
        right_mask = get_ir_pattern(right_img / 255)

    depth_map = np.load(depth).astype(np.float32)

    if disp_info != "default":
        disp_map = np.load(disp_info).astype(np.float32)
    else:
        disp_map = np.where(depth_map > 0, (s_fx * s_b) / depth_map, 0).astype(np.float32)
    
    return [left_img, right_img, left_mask, right_mask, disp_map]


def read_real_data(path, conf_threshold, disp_threshold, isNorm=True):
    paths = path.strip().split()
    has_ao = len(paths) == 7
    
    left, right, left_mask, right_mask, depth, disp, *ao_rest = paths
    AO_path = ao_rest[0] if ao_rest else None
    
    if isNorm:
        left_img = cv2.imread(left, -1) / 255
        right_img = cv2.imread(right, -1) / 255
    else:
        left_img = cv2.imread(left, -1)
        right_img = cv2.imread(right, -1)

    if left_mask == "None" or right_mask == "None":
        left_img_for_mask = left_img if isNorm else left_img / 255
        right_img_for_mask = right_img if isNorm else right_img / 255
        left_mask = get_ir_pattern(left_img_for_mask)
        right_mask = get_ir_pattern(right_img_for_mask)
    else:
        left_mask = np.load(left_mask).astype(np.float32)
        right_mask = np.load(right_mask).astype(np.float32)

    disp = np.load(disp).astype(np.float32)
    
    AO = (1 - np.load(AO_path).astype(np.float32)) if has_ao else np.ones_like(disp)
    if has_ao:
        AO[AO < conf_threshold] = 0
        disp = disp * AO
        disp[disp < disp_threshold] = 0
    AO[disp == 0] = 0
    
    return [left_img, right_img, left_mask, right_mask, disp, AO]

def read_d435_data(path, fx, b, isNorm=True):
    left, right, depth = path.split(" ")
    if isNorm:
        left = cv2.imread(left, -1) / 255
        right = cv2.imread(right, -1) / 255
        left_mask = get_ir_pattern(left)
        right_mask = get_ir_pattern(right)
    else:
        left = cv2.imread(left, -1)
        right = cv2.imread(right, -1)
        left_mask = get_ir_pattern(left / 255)
        right_mask = get_ir_pattern(right / 255)
    depth = np.load(depth).astype(np.float32)
    disp = np.zeros_like(depth)
    disp[depth > 0] = (fx * b) / depth[depth > 0]
    return [left, right, left_mask, right_mask, disp]

