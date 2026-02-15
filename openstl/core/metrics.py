# Copyright (c) CAIRI AI Lab. All rights reserved
# Clean + stable metrics (equivalent intent, faster, never crashes on small images)

import cv2
import numpy as np
import torch

try:
    import lpips as _lpips_pkg
except Exception:
    _lpips_pkg = None

try:
    from skimage.metrics import structural_similarity as _sk_ssim
except Exception:
    _sk_ssim = None


# ----------------------------
# helpers
# ----------------------------
def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rescale(x):
    # NOTE: this is unchanged from your original
    x = _as_numpy(x)
    return (x - x.max()) / (x.max() - x.min() + 1e-12) * 2 - 1


def _threshold(x, y, t):
    t = np.greater_equal(x, t).astype(np.float32)
    p = np.greater_equal(y, t).astype(np.float32)
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    t = np.where(is_nan, np.zeros_like(t, dtype=np.float32), t)
    p = np.where(is_nan, np.zeros_like(p, dtype=np.float32), p)
    return t, p


# ----------------------------
# basic errors (vectorized)
# ----------------------------
def MAE(pred, true, spatial_norm=False):
    pred = _as_numpy(pred)
    true = _as_numpy(true)
    if not spatial_norm:
        return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return np.mean(np.abs(pred - true) / max(norm, 1), axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    pred = _as_numpy(pred)
    true = _as_numpy(true)
    if not spatial_norm:
        return np.mean((pred - true) ** 2, axis=(0, 1)).sum()
    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return np.mean(((pred - true) ** 2) / max(norm, 1), axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    pred = _as_numpy(pred)
    true = _as_numpy(true)
    if not spatial_norm:
        return float(np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)).sum()))
    norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
    return float(np.sqrt(np.mean(((pred - true) ** 2) / max(norm, 1), axis=(0, 1)).sum()))


# ----------------------------
# psnr / snr (fast, vectorized)
# ----------------------------
def PSNR(pred, true, min_max_norm=True):
    # kept for backward compatibility with callers; vectorized version used in metric()
    pred = _as_numpy(pred).astype(np.float32)
    true = _as_numpy(true).astype(np.float32)
    mse = float(np.mean((pred - true) ** 2))
    if mse <= 0:
        return float("inf")
    if min_max_norm:
        return float(20.0 * np.log10(1.0 / np.sqrt(mse)))
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def SNR(pred, true):
    pred = _as_numpy(pred).astype(np.float32)
    true = _as_numpy(true).astype(np.float32)
    signal = float(np.mean(true ** 2))
    noise = float(np.mean((true - pred) ** 2))
    noise = max(noise, 1e-12)
    return float(10.0 * np.log10(signal / noise))


def _psnr_btchw_mean(pred_btchw, true_btchw, data_range=1.0):
    diff = pred_btchw.astype(np.float32) - true_btchw.astype(np.float32)
    mse = np.mean(diff * diff, axis=(2, 3, 4))  # (B,T)
    mse = np.maximum(mse, 1e-12)
    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return float(np.mean(psnr))


def _snr_btchw_mean(pred_btchw, true_btchw):
    true_f = true_btchw.astype(np.float32)
    diff = true_btchw.astype(np.float32) - pred_btchw.astype(np.float32)
    signal = np.mean(true_f * true_f, axis=(2, 3, 4))  # (B,T)
    noise = np.mean(diff * diff, axis=(2, 3, 4))       # (B,T)
    noise = np.maximum(noise, 1e-12)
    snr = 10.0 * np.log10(signal / noise)
    return float(np.mean(snr))


# ----------------------------
# SSIM (stable)
# ----------------------------
def _pick_win_size(h, w, preferred=7, min_allowed=3):
    m = int(min(h, w))
    if m < min_allowed:
        return None
    win = min(preferred, m)
    if win % 2 == 0:
        win -= 1
    if win < min_allowed:
        return None
    return win


def _ssim_btchw_mean(pred_btchw, true_btchw, data_range=1.0, preferred_win=7):
    """
    pred/true: (B,T,C,H,W), float numpy
    returns mean SSIM over B*T. Never throws; returns nan if not computable.
    """
    if _sk_ssim is None:
        return np.nan

    B, T, C, H, W = pred_btchw.shape
    win = _pick_win_size(H, W, preferred=preferred_win)
    if win is None:
        return np.nan

    pred_n = pred_btchw.transpose(0, 1, 3, 4, 2).reshape(B * T, H, W, C)
    true_n = true_btchw.transpose(0, 1, 3, 4, 2).reshape(B * T, H, W, C)

    s = 0.0
    n = pred_n.shape[0]
    valid = 0
    for i in range(n):
        try:
            s += _sk_ssim(
                pred_n[i],
                true_n[i],
                data_range=data_range,
                channel_axis=-1,   # correct replacement for multichannel=True
                win_size=win,
                gaussian_weights=False,
            )
            valid += 1
        except Exception:
            # skip frames that still fail
            continue

    return float(s / valid) if valid > 0 else np.nan


# ----------------------------
# OpenCV SSIM (kept as in your file; optional)
# ----------------------------
def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = _as_numpy(pred).astype(np.float64)
    img2 = _as_numpy(true).astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


# ----------------------------
# SEVIR metrics (unchanged)
# ----------------------------
def POD(hits, misses, eps=1e-6):
    pod = (hits + eps) / (hits + misses + eps)
    return float(np.mean(pod))


def SUCR(hits, fas, eps=1e-6):
    sucr = (hits + eps) / (hits + fas + eps)
    return float(np.mean(sucr))


def CSI(hits, fas, misses, eps=1e-6):
    csi = (hits + eps) / (hits + misses + fas + eps)
    return float(np.mean(csi))


def sevir_metrics(pred, true, threshold):
    pred = pred.transpose(1, 0, 2, 3, 4)
    true = true.transpose(1, 0, 2, 3, 4)
    hits, fas, misses = [], [], []
    for i in range(pred.shape[0]):
        t, p = _threshold(pred[i], true[i], threshold)
        hits.append(np.sum(t * p))
        fas.append(np.sum((1 - t) * p))
        misses.append(np.sum(t * (1 - p)))
    return np.array(hits), np.array(fas), np.array(misses)


# ----------------------------
# LPIPS (guarded)
# ----------------------------
class LPIPS(torch.nn.Module):
    """
    Learned Perceptual Image Patch Similarity, LPIPS.
    If lpips package is missing, this will raise at init.
    """

    def __init__(self, net="alex", use_gpu=True):
        super().__init__()
        if _lpips_pkg is None:
            raise RuntimeError("lpips is not installed")
        assert net in ["alex", "squeeze", "vgg"]
        self.use_gpu = bool(use_gpu and torch.cuda.is_available())
        self.loss_fn = _lpips_pkg.LPIPS(net=net)
        if self.use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        img1 = _lpips_pkg.im2tensor(img1 * 255)
        img2 = _lpips_pkg.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


# ----------------------------
# MAIN API (drop-in replacement)
# ----------------------------
def metric(
    pred,
    true,
    mean=None,
    std=None,
    metrics=("mae", "mse"),
    clip_range=(0, 1),
    channel_names=None,
    spatial_norm=False,
    return_log=True,
    threshold=74.0,
):
    """
    pred/true expected shape: (B, T, C, H, W)
    Accepts torch.Tensor or np.ndarray
    """

    pred = _as_numpy(pred)
    true = _as_numpy(true)

    # de-normalize if provided
    if mean is not None and std is not None:
        mean = _as_numpy(mean)
        std = _as_numpy(std)
        pred = pred * std + mean
        true = true * std + mean

    eval_res = {}
    eval_log = ""

    allowed = {"mae", "mse", "rmse", "ssim", "psnr", "snr", "lpips", "pod", "sucr", "csi"}
    metrics = list(metrics) if isinstance(metrics, (list, tuple)) else [metrics]
    invalid = set(metrics) - allowed
    if invalid:
        raise ValueError(f"metric {invalid} is not supported.")

    # channel grouping support (kept consistent with your code)
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    # ---- error metrics (same behavior as original)
    if "mse" in metrics:
        if channel_names is None:
            eval_res["mse"] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.0
            for i, c_name in enumerate(channel_names):
                v = MSE(
                    pred[:, :, i * c_width : (i + 1) * c_width, ...],
                    true[:, :, i * c_width : (i + 1) * c_width, ...],
                    spatial_norm,
                )
                eval_res[f"mse_{str(c_name)}"] = v
                mse_sum += v
            eval_res["mse"] = mse_sum / c_group

    if "mae" in metrics:
        if channel_names is None:
            eval_res["mae"] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.0
            for i, c_name in enumerate(channel_names):
                v = MAE(
                    pred[:, :, i * c_width : (i + 1) * c_width, ...],
                    true[:, :, i * c_width : (i + 1) * c_width, ...],
                    spatial_norm,
                )
                eval_res[f"mae_{str(c_name)}"] = v
                mae_sum += v
            eval_res["mae"] = mae_sum / c_group

    if "rmse" in metrics:
        if channel_names is None:
            eval_res["rmse"] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.0
            for i, c_name in enumerate(channel_names):
                v = RMSE(
                    pred[:, :, i * c_width : (i + 1) * c_width, ...],
                    true[:, :, i * c_width : (i + 1) * c_width, ...],
                    spatial_norm,
                )
                eval_res[f"rmse_{str(c_name)}"] = v
                rmse_sum += v
            eval_res["rmse"] = rmse_sum / c_group

    # ---- sevir threshold metrics (same behavior)
    if "pod" in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res["pod"] = POD(hits, misses)
        eval_res["sucr"] = SUCR(hits, fas)
        eval_res["csi"] = CSI(hits, fas, misses)

    # ---- clip once (prevents overflow and matches original intent)
    lo, hi = float(clip_range[0]), float(clip_range[1])
    pred = np.clip(pred, lo, hi)
    true = np.clip(true, lo, hi)
    data_range = hi - lo if hi > lo else 1.0

    # ---- ssim/psnr/snr (fast + stable)
    if "ssim" in metrics:
        eval_res["ssim"] = _ssim_btchw_mean(pred, true, data_range=data_range, preferred_win=7)

    if "psnr" in metrics:
        eval_res["psnr"] = _psnr_btchw_mean(pred, true, data_range=data_range)

    if "snr" in metrics:
        eval_res["snr"] = _snr_btchw_mean(pred, true)

    # ---- lpips (kept optional; slow; does not crash if missing)
    if "lpips" in metrics:
        if _lpips_pkg is None:
            eval_res["lpips"] = np.nan
        else:
            cal_lpips = LPIPS(net="alex", use_gpu=False)
            pred_nhwc = pred.transpose(0, 1, 3, 4, 2)
            true_nhwc = true.transpose(0, 1, 3, 4, 2)
            lp = 0.0
            n = pred.shape[0] * pred.shape[1]
            for b in range(pred.shape[0]):
                for t in range(pred.shape[1]):
                    lp += cal_lpips(pred_nhwc[b, t], true_nhwc[b, t])
            eval_res["lpips"] = float(lp / max(n, 1))

    if return_log:
        for k, v in eval_res.items():
            eval_log += ("" if len(eval_log) == 0 else ", ") + f"{k}:{v}"

    return eval_res, eval_log
