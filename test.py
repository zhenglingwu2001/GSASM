import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from configs import cfg
from init_build import build_dataloader, build_model_optim
from Core.Train_Framework import val_metrics


def get_args():
    parser = argparse.ArgumentParser(description="Stereo matching forward test")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datatype", type=str, default="ours")
    parser.add_argument("--backbone", type=str, default="psmnet")
    parser.add_argument("--weight", type=str, required=True,
                        help="path to checkpoint .pth file")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    parser.add_argument("--save_vis", action="store_true",
                        help="save disparity visualizations (png + npy)")
    parser.add_argument("--test_list", type=str, default=None,
                    help="path to validation txt file (overrides cfg.sim_val)")
    return parser.parse_args()


def load_weight(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"[INFO] Loaded model state_dict from: {weight_path}")
        if checkpoint.get("total_step") is not None:
            print(f"       checkpoint step: {checkpoint['total_step']}")
        if checkpoint.get("metrics"):
            print(f"       checkpoint metrics: {checkpoint['metrics']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded raw state_dict from: {weight_path}")
    return model


def input_to_model(model, data, backbone, cfg):
    if backbone == "psmnet":
        pred_disps = model(data["img_L"], data["img_R"])
    elif backbone == "raft":
        pred_disps = model(data["img_L"], data["img_R"], cfg.valid_iters)
    else:  # stereonet
        inputs = torch.cat([data["img_L"], data["img_R"]], dim=1)
        pred_disps = model(inputs)
    return pred_disps


def move_to_device(data, device, backbone):
    data["img_L"] = data["img_L"].to(device)
    data["img_R"] = data["img_R"].to(device)
    data["img_L_reproj"] = data["img_L_reproj"].to(device)
    data["img_R_reproj"] = data["img_R_reproj"].to(device)
    data["disp_gt"] = data["disp_gt"].to(device)
    if backbone == "raft":
        data["conf"] = data["conf"].to(device)
    return data


# ---------------------------------------------------------------------------
# Padding / cropping helpers (fix input size not divisible by 16)
# ---------------------------------------------------------------------------

def pad_to_multiple(item, multiple=16):
    """Pad img_L / img_R so H, W are divisible by multiple.
    Returns (padded_item, original_hw)."""
    h, w = item["img_L"].shape[2], item["img_L"].shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return item, (h, w)
    # F.pad order: (left, right, top, bottom)
    item["img_L"] = F.pad(item["img_L"], (0, pad_w, 0, pad_h),
                           mode="constant", value=0)
    item["img_R"] = F.pad(item["img_R"], (0, pad_w, 0, pad_h),
                           mode="constant", value=0)
    return item, (h, w)


def crop_preds(pred_disps, orig_hw):
    """Crop every disparity map back to original (H, W)."""
    h, w = orig_hw
    out = []
    for p in pred_disps:
        if p.dim() == 4:       # (B, C, H, W)
            out.append(p[:, :, :h, :w])
        elif p.dim() == 3:     # (B, H, W)
            out.append(p[:, :h, :w])
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def process_image(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img  # RGB


def process_disparity(disp_tensor, vmin, vmax):
    disp = disp_tensor.numpy()
    disp_norm = (disp - vmin) / (vmax - vmin + 1e-8)
    disp_norm = np.clip(disp_norm, 0, 1)
    disp_color = plt.cm.viridis(disp_norm)[:, :, :3]
    return (disp_color * 255).astype(np.uint8)  # RGB


def extract_pred(pred_disps, backbone, idx):
    if backbone == "psmnet":
        return pred_disps[0][idx].detach().cpu()
    elif backbone == "raft":
        return -pred_disps[-1][idx].detach().cpu()
    elif backbone == "stereonet":
        return pred_disps[-1][idx].detach().cpu()
    else:
        return pred_disps[idx].detach().cpu()


def save_sample_vis(item, pred_disps, backbone, sample_dir, idx):
    img_L = item["img_L"][idx].detach().cpu()
    img_R = item["img_R"][idx].detach().cpu()
    disp_gt = item["disp_gt"][idx].detach().cpu()
    pred_disp = extract_pred(pred_disps, backbone, idx)

    if disp_gt.dim() == 3:
        disp_gt = disp_gt[0]
    if pred_disp.dim() == 3:
        pred_disp = pred_disp[0]

    disp_diff = torch.abs(pred_disp - disp_gt)

    img_L = process_image(img_L)
    img_R = process_image(img_R)

    valid_mask = disp_gt > 0
    if valid_mask.any():
        gt_min = disp_gt[valid_mask].min().item()
        gt_max = disp_gt[valid_mask].max().item()
    else:
        gt_min = pred_disp.min().item()
        gt_max = pred_disp.max().item()

    disp_gt_color = process_disparity(disp_gt, gt_min, gt_max)
    pred_disp_color = process_disparity(pred_disp, gt_min, gt_max)
    diff_max = disp_diff.max().item() if disp_diff.numel() > 0 else 1.0
    disp_diff_color = process_disparity(disp_diff, 0, diff_max)

    cv2.imwrite(os.path.join(sample_dir, "left_img.png"),
                cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sample_dir, "right_img.png"),
                cv2.cvtColor(img_R, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sample_dir, "gt_disp.png"),
                cv2.cvtColor(disp_gt_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sample_dir, "pred_disp.png"),
                cv2.cvtColor(pred_disp_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sample_dir, "disp_diff.png"),
                cv2.cvtColor(disp_diff_color, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(sample_dir, "pred_disp.npy"), pred_disp.numpy())
    np.save(os.path.join(sample_dir, "gt_disp.npy"), disp_gt.numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(disp_gt_color)
    ax1.set_title("Ground Truth")
    ax1.axis("off")
    ax2.imshow(pred_disp_color)
    ax2.set_title("Predicted")
    ax2.axis("off")
    plt.savefig(os.path.join(sample_dir, "disp_gt_vs_pred.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    opt = get_args()
    cfg.backbone = opt.backbone
    cfg.datatype = opt.datatype

    if opt.test_list is not None:
        cfg.sim_val = opt.test_list
        print(f"[INFO] Override val_list: {cfg.sim_val}")

    torch.cuda.set_device(opt.gpu_id)
    cfg.device = torch.device(f"cuda:{opt.gpu_id}")

    _, val_loader = build_dataloader(cfg)
    model, _, _ = build_model_optim(cfg)

    model = load_weight(model, opt.weight, cfg.device)
    model.eval()

    output_dir = os.path.join(opt.output_dir, f"{opt.backbone}_{opt.datatype}")
    os.makedirs(output_dir, exist_ok=True)

    metrics_sum = {"disp_err": 0.0, "1px": 0.0, "3px": 0.0, "5px": 0.0}
    num_batches = 0
    global_sample_idx = 0

    print(f"\n[INFO] Start testing on validation set "
          f"(backbone={opt.backbone}, datatype={opt.datatype})")
    print(f"[INFO] Save visualization: {opt.save_vis}\n")

    with torch.no_grad():
        loop = tqdm(val_loader, total=len(val_loader))
        for i_batch, data in enumerate(loop):
            item = move_to_device(data["sim"], cfg.device, cfg.backbone)

            # pad input to be divisible by 16 (PSMNet requirement)
            item, orig_hw = pad_to_multiple(item, multiple=16)

            pred_disps = input_to_model(model, item, cfg.backbone, cfg)

            # crop predictions back to original size
            pred_disps = crop_preds(pred_disps, orig_hw)

            error = val_metrics(pred_disps, item["disp_gt"], cfg.backbone)
            for key in metrics_sum:
                metrics_sum[key] += error[key]
            num_batches += 1

            loop.set_description("test: ")
            loop.set_postfix(disp_err=error["disp_err"])

            if opt.save_vis:
                batch_size = item["img_L"].shape[0]
                for idx in range(batch_size):
                    sample_dir = os.path.join(
                        output_dir, f"sample_{global_sample_idx:04d}")
                    os.makedirs(sample_dir, exist_ok=True)
                    save_sample_vis(item, pred_disps, cfg.backbone,
                                    sample_dir, idx)
                    global_sample_idx += 1

    for key in metrics_sum:
        metrics_sum[key] /= num_batches

    print("\n" + "=" * 55)
    print("  Test Results (averaged over validation set)")
    print("=" * 55)
    print(f"  EPE (disp_err) : {metrics_sum['disp_err']:.4f}")
    print(f"  1px             : {metrics_sum['1px']:.4f}")
    print(f"  3px             : {metrics_sum['3px']:.4f}")
    print(f"  5px             : {metrics_sum['5px']:.4f}")
    print("=" * 55)

    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"backbone: {opt.backbone}\n")
        f.write(f"datatype: {opt.datatype}\n")
        f.write(f"weight: {opt.weight}\n")
        f.write(f"num_batches: {num_batches}\n")
        for key, val in metrics_sum.items():
            f.write(f"{key}: {val:.6f}\n")
    print(f"\n[INFO] Metrics saved to: {metrics_path}")
    if opt.save_vis:
        print(f"[INFO] Visualizations saved to: {output_dir} "
              f"({global_sample_idx} samples)")


if __name__ == "__main__":
    main()
