import os
import cv2
import glob
import numpy as np
from PIL import Image


def get_smoothed_ir_pattern(diff: np.array, ks=9, threshold=0.005):
    diff = np.abs(diff)
    diff_avg = cv2.blur(diff, (ks, ks))
    ir = np.zeros_like(diff)
    ir[diff - diff_avg > threshold] = 1
    return ir

def from_6_pattern_get_mask(data):
    img_0 = np.array(Image.open(data[0]).convert(mode="L"))
    img_1 = np.array(Image.open(data[1]).convert(mode="L"))
    img_2 = np.array(Image.open(data[2]).convert(mode="L"))
    img_3 = np.array(Image.open(data[3]).convert(mode="L"))
    img_4 = np.array(Image.open(data[4]).convert(mode="L"))
    img_5 = np.array(Image.open(data[5]).convert(mode="L"))
    img_6 = np.array(Image.open(data[6]).convert(mode="L"))
    img_temp = np.concatenate((img_0[:, :, None], img_1[:, :, None],
                               img_2[:, :, None], img_3[:, :, None],
                               img_4[:, :, None], img_5[:, :, None],
                               img_6[:, :, None]), axis=-1)

    h, w, d = img_temp.shape
    x = np.linspace(0, d - 1, num=d, dtype=int).reshape(1, 1, -1)
    x = np.repeat(x, h, axis=0)
    x = np.repeat(x, w, axis=1)  # [H, W, D]
    x_avg = np.average(x, axis=-1).reshape(h, w, 1)

    y = img_temp  # [H, W, D]
    y_avg = np.average(y, axis=-1).reshape(h, w, 1)

    numerator = np.sum((y - y_avg) * (x - x_avg), axis=-1)
    denominator = np.sum((x - x_avg) ** 2, axis=-1)  # [H, W]
    slope = numerator / denominator  # [H, W]
    slope = slope[:, :, None]
    intercept = y_avg - slope * x_avg
    img_temp_fit = slope * x + intercept

    # Get IR pattern
    diff = (img_temp_fit[:, :, -1] - img_temp_fit[:, :, 0]) / 255
    diff = np.abs(diff)
    # Normalize to [0,1]
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    pattern = get_smoothed_ir_pattern(diff, ks=11, threshold=0.005)
    return pattern


def get_ir_pattern(img, ks=11):
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



# TODO real data
root = "/media/data/stereo/real_data"
file_name = os.listdir(root)
for i in range(len(file_name)):
    os.makedirs(os.path.join(root, file_name[i], "left_pattern"), exist_ok=True)
    os.makedirs(os.path.join(root, file_name[i], "right_pattern"), exist_ok=True)
    left_path = glob.glob(os.path.join(root, file_name[i], "left_ir", "0/*.png"))
    print(file_name[i])
    for j in range(len(left_path)):
        img_id = left_path[j].split("/")[-1].split(".")[0]
        f0 = left_path[j]
        f1 = left_path[j].replace("left_ir/0", "left_ir/1")
        f2 = left_path[j].replace("left_ir/0", "left_ir/2")
        f3 = left_path[j].replace("left_ir/0", "left_ir/3")
        f4 = left_path[j].replace("left_ir/0", "left_ir/4")
        f5 = left_path[j].replace("left_ir/0", "left_ir/5")
        f6 = left_path[j].replace("left_ir/0", "left_ir/6")
        left_pattern = from_6_pattern_get_mask([f0, f1, f2, f3, f4, f5, f6])

        r0 = left_path[j].replace("left", "right")
        r1 = r0.replace("right_ir/0", "right_ir/1")
        r2 = r0.replace("right_ir/0", "right_ir/2")
        r3 = r0.replace("right_ir/0", "right_ir/3")
        r4 = r0.replace("right_ir/0", "right_ir/4")
        r5 = r0.replace("right_ir/0", "right_ir/5")
        r6 = r0.replace("right_ir/0", "right_ir/6")
        right_pattern = from_6_pattern_get_mask([r0, r1, r2, r3, r4, r5, r6])

        np.save(os.path.join(root, file_name[i], "left_pattern", img_id + ".npy"), left_pattern)
        np.save(os.path.join(root, file_name[i], "right_pattern", img_id + ".npy"), right_pattern)







