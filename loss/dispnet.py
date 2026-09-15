def dispnet_disp(disp_ests, disp_gt, mask):
    scale = [0, 1, 2, 3, 4, 5, 6]
    weights = [1, 1, 1, 0.8, 0.6, 0.4, 0.2]
    all_losses = []
    for disp_est, weight, s in zip(disp_ests, weights, scale):
        if s != 0:
            dgt = F.interpolate(disp_gt, scale_factor=1 / (2 ** s))
            m = F.interpolate(mask.float(), scale_factor=1 / (2 ** s)).byte()
        else:
            dgt = disp_gt
            m = mask
        all_losses.append(
            weight
            * F.smooth_l1_loss(disp_est[m], dgt[m], size_average=True, reduction="mean")
        )
    return sum(all_losses)
