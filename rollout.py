import os
import os.path as osp
import json
import csv
import numpy as np
import torch
import imageio.v2 as imageio

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser, load_config, update_config, get_dataset

# optional SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None


# -------------------------
# helpers
# -------------------------
def pick_device(args):
    if str(args.device).lower() == "cpu":
        return torch.device("cpu")
    gid = args.gpus[0] if isinstance(args.gpus, (list, tuple)) else int(args.gpus)
    return torch.device(f"cuda:{gid}")


def load_plain_state_dict_with_model_prefix(method, ckpt_path):
    # your .pth is direct params, and you said you need "model." prefix
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = {f"model.{k}": v for k, v in sd.items()}
    method.load_state_dict(sd, strict=False)


def get_one_batch_at_index(loader, index):
    for i, batch in enumerate(loader):
        if i == index:
            return batch
    raise IndexError(f"index={index} out of range")


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
        return batch[0], batch[1]
    if isinstance(batch, dict):
        for kx, ky in [("inputs", "trues"), ("x", "y"), ("input", "true")]:
            if kx in batch and ky in batch:
                return batch[kx], batch[ky]
        vals = list(batch.values())
        if len(vals) >= 2:
            return vals[0], vals[1]
    raise ValueError(f"Unsupported batch type: {type(batch)}")


def call_model(method, x, chunk_len):
    method.eval()
    with torch.no_grad():
        try:
            out = method(x)
        except TypeError:
            B, _, C, H, W = x.shape
            dummy_y = torch.zeros((B, chunk_len, C, H, W), device=x.device, dtype=x.dtype)
            out = method(x, dummy_y)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out

def to_uint8_frame(frame):
    """
    frame: (C,H,W) or (H,W,C) or (H,W)
    returns: uint8 (H,W,3) always
    """
    x = frame

    # if (C,H,W) convert to (H,W,C) for C in {1,3}
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))  # (H,W,C)

    # squeeze grayscale
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]  # (H,W)

    # float handling: FORCE [0,1] before scaling
    if np.issubdtype(x.dtype, np.floating):
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        # clamp predicted values (very important)
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
    else:
        x = np.clip(x, 0, 255).astype(np.uint8)

    # force RGB
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected frame shape after conversion: {x.shape}")

    return x

def save_strip_10(seq_TCHW, out_path_png, chunk=10):
    """
    Saves ONE strip png: horizontally concatenates up to chunk frames.
    seq_TCHW: numpy (T,C,H,W) or (T,H,W,C) or (T,H,W)
    """
    T = seq_TCHW.shape[0]
    frames = []
    for t in range(T):
        frames.append(to_uint8_frame(seq_TCHW[t]))
    strip = np.concatenate(frames, axis=1)  # concat width
    os.makedirs(osp.dirname(out_path_png), exist_ok=True)
    imageio.imwrite(out_path_png, strip)


def save_strips_for_full_rollout(pred_TCHW, true_TCHW, out_dir, chunk=10, also_error=True):
    """
    For 50 frames, saves:
      pred_000-009.png ... pred_040-049.png
      true_000-009.png ... true_040-049.png
      err_000-009.png  ... err_040-049.png   (optional)
    """
    os.makedirs(out_dir, exist_ok=True)
    T = pred_TCHW.shape[0]
    assert true_TCHW.shape[0] == T

    # error visualization uses abs diff scaled to 0..1 then 0..255
    if also_error:
        # ensure float for error computation
        p = pred_TCHW.astype(np.float32)
        g = true_TCHW.astype(np.float32)
        err = np.abs(p - g)
        # normalize per-sequence for visibility (avoid all black)
        mx = float(err.max()) if err.size else 1.0
        if mx > 0:
            err = err / mx
        err = err.astype(np.float32)

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        pred_chunk = pred_TCHW[start:end]
        true_chunk = true_TCHW[start:end]

        save_strip_10(pred_chunk, osp.join(out_dir, f"pred_{start:03d}-{end-1:03d}.png"))
        save_strip_10(true_chunk, osp.join(out_dir, f"true_{start:03d}-{end-1:03d}.png"))

        if also_error:
            err_chunk = err[start:end]
            save_strip_10(err_chunk, osp.join(out_dir, f"err_{start:03d}-{end-1:03d}.png"))


def compute_metrics_framewise(pred_BTCHW, true_BTCHW, data_range=1.0, do_ssim=False):
    """
    pred/true: torch CPU (B,T,C,H,W)
    returns dict with per-frame arrays + averages
    """
    diff = pred_BTCHW - true_BTCHW
    mse_t = (diff * diff).mean(dim=(0, 2, 3, 4)).numpy()
    mae_t = diff.abs().mean(dim=(0, 2, 3, 4)).numpy()

    psnr_t = np.zeros_like(mse_t, dtype=np.float64)
    for i, m in enumerate(mse_t):
        if float(m) == 0.0:
            psnr_t[i] = float("inf")
        else:
            psnr_t[i] = 10.0 * np.log10((data_range * data_range) / float(m))

    out = {
        "mse_t": mse_t.astype(float).tolist(),
        "mae_t": mae_t.astype(float).tolist(),
        "psnr_t": psnr_t.astype(float).tolist(),
        "mse": float(np.mean(mse_t)),
        "mae": float(np.mean(mae_t)),
        "psnr": float(np.mean(psnr_t[np.isfinite(psnr_t)])) if np.any(np.isfinite(psnr_t)) else float("inf"),
    }

    if do_ssim and sk_ssim is not None:
        pred0 = pred_BTCHW[0].numpy()
        true0 = true_BTCHW[0].numpy()
        T, C, H, W = pred0.shape
        ssim_t = np.zeros((T,), dtype=np.float64)
        for t in range(T):
            p = pred0[t]
            g = true0[t]
            if C == 1:
                ssim_t[t] = sk_ssim(g[0], p[0], data_range=data_range)
            else:
                s = 0.0
                for c in range(C):
                    s += sk_ssim(g[c], p[c], data_range=data_range)
                ssim_t[t] = s / C
        out["ssim_t"] = ssim_t.astype(float).tolist()
        out["ssim"] = float(np.mean(ssim_t))
    return out


def save_metrics_json_csv(metrics, out_dir, base_name):
    os.makedirs(out_dir, exist_ok=True)

    # json
    with open(osp.join(out_dir, f"{base_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # csv
    T = len(metrics["mse_t"])
    has_ssim = "ssim_t" in metrics
    with open(osp.join(out_dir, f"{base_name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["t", "mse", "mae", "psnr"] + (["ssim"] if has_ssim else [])
        w.writerow(header)
        for t in range(T):
            row = [t, metrics["mse_t"][t], metrics["mae_t"][t], metrics["psnr_t"][t]]
            if has_ssim:
                row.append(metrics["ssim_t"][t])
            w.writerow(row)


def make_gif_gt_vs_pred(prev_TCHW, true_TCHW, pred_TCHW, out_gif, fps=6):
    """
    One GIF over full timeline (pre + aft):
      left: GT (prev then true)
      right: Pred (prev then pred)
    Always RGB and uint8.
    """
    os.makedirs(osp.dirname(out_gif), exist_ok=True)

    pre = prev_TCHW.shape[0]
    aft = true_TCHW.shape[0]
    assert pred_TCHW.shape[0] == aft

    H = to_uint8_frame(prev_TCHW[0]).shape[0]
    sep = np.full((H, 4, 3), 255, dtype=np.uint8)

    frames = []
    for i in range(pre + aft):
        if i < pre:
            gt = to_uint8_frame(prev_TCHW[i])
            pr = to_uint8_frame(prev_TCHW[i])
        else:
            gt = to_uint8_frame(true_TCHW[i - pre])
            pr = to_uint8_frame(pred_TCHW[i - pre])
        frames.append(np.concatenate([gt, sep, pr], axis=1))

    if not out_gif.endswith(".gif"):
        out_gif += ".gif"
    imageio.mimsave(out_gif, frames, duration=1.0 / float(fps))


def make_chunk_gif(true_chunk_TCHW, pred_chunk_TCHW, out_gif, fps=6):
    """
    Chunk-only GIF (10 frames): left true, right pred.
    """
    os.makedirs(osp.dirname(out_gif), exist_ok=True)

    T = true_chunk_TCHW.shape[0]
    assert pred_chunk_TCHW.shape[0] == T

    H = to_uint8_frame(true_chunk_TCHW[0]).shape[0]
    sep = np.full((H, 4, 3), 255, dtype=np.uint8)

    frames = []
    for t in range(T):
        gt = to_uint8_frame(true_chunk_TCHW[t])
        pr = to_uint8_frame(pred_chunk_TCHW[t])
        frames.append(np.concatenate([gt, sep, pr], axis=1))

    if not out_gif.endswith(".gif"):
        out_gif += ".gif"
    imageio.mimsave(out_gif, frames, duration=1.0 / float(fps))


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--rollout_aft", type=int, default=50)
    parser.add_argument("--rollout_chunk", type=int, default=10)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--gif_fps", type=int, default=6)
    parser.add_argument("--save_error_strips", action="store_true", default=True)
    parser.add_argument("--ssim", action="store_true", default=False)
    args = parser.parse_args()

    # load config
    cfg_path = osp.join("./configs", args.dataname, f"{args.method}.py") if args.config_file is None else args.config_file
    loaded_cfg = load_config(cfg_path)
    config = update_config(args.__dict__, loaded_cfg,
                           exclude_keys=["val_batch_size", "drop_path", "warmup_epoch"])
    defaults = default_parser()
    for k in defaults.keys():
        if config.get(k, None) is None:
            config[k] = defaults[k]
    args.metrics = ['mse', 'mae', 'ssim', 'psnr']
    args.in_shape = [10, 3, 64, 64]
    # build experiment/model
    exp = BaseExperiment(args)
    device = pick_device(args)
    exp.method.to(device)

    if not args.ckpt_path:
        raise ValueError("Pass --ckpt_path")

    load_plain_state_dict_with_model_prefix(exp.method, args.ckpt_path)

    # build LONG test loader to get GT for full rollout (10 -> rollout_aft)
    long_cfg = dict(exp.config)
    long_cfg["batch_size"] = 1
    long_cfg["val_batch_size"] = 1
    long_cfg["num_workers"] = 4
    long_cfg["pre_seq_length"] = exp.config["pre_seq_length"]
    long_cfg["aft_seq_length"] = args.rollout_aft
    long_cfg["total_length"] = args.rollout_aft + exp.config["pre_seq_length"]

    _, _, test_loader_long = get_dataset(args.dataname, long_cfg)
    batch = get_one_batch_at_index(test_loader_long, args.index)
    x, y = unpack_batch(batch)  # x: (1,pre,C,H,W), y:(1,aft,C,H,W)
    print(x.shape, y.shape)

    x = x.to(device).float()
    y = y.to(device).float()

    pre_len = x.shape[1]
    total_aft = y.shape[1]
    chunk = int(args.rollout_chunk)

    # autoregressive rollout
    preds = []
    cur = x
    produced = 0
    while produced < total_aft:
        out = call_model(exp.method, cur, chunk_len=chunk)  # (1,chunk,C,H,W)
        out = out[:, :chunk]
        step = min(chunk, total_aft - produced)
        preds.append(out[:, :step])
        produced += step
        cur = torch.cat([cur, out], dim=1)[:, -pre_len:]

    pred = torch.cat(preds, dim=1)  # (1,aft,C,H,W)

    # output dir structure (very explicit)
    out_dir = args.out_dir or osp.join("work_dirs", args.ex_name, "rollout_package")
    arr_dir = osp.join(out_dir, "arrays")
    strips_dir = osp.join(out_dir, "strips_full")
    metrics_dir = osp.join(out_dir, "metrics")
    gifs_dir = osp.join(out_dir, "gifs")
    chunk_dir = osp.join(out_dir, "chunks")

    for d in [arr_dir, strips_dir, metrics_dir, gifs_dir, chunk_dir]:
        os.makedirs(d, exist_ok=True)

    # save arrays
    np.save(osp.join(arr_dir, "inputs.npy"), x.detach().cpu().numpy().astype(np.float32))
    np.save(osp.join(arr_dir, "trues.npy"),  y.detach().cpu().numpy().astype(np.float32))
    np.save(osp.join(arr_dir, "preds.npy"),  pred.detach().cpu().numpy().astype(np.float32))

    # full framewise metrics (50 frames)
    metrics_full = compute_metrics_framewise(pred.detach().cpu(), y.detach().cpu(), data_range=1.0, do_ssim=args.ssim)
    save_metrics_json_csv(metrics_full, metrics_dir, "framewise_full_50")

    # full strips (pred + true + error) => should create 5 files each for 50 frames
    pred0 = pred[0].detach().cpu().numpy()  # (50,C,H,W)
    true0 = y[0].detach().cpu().numpy()
    save_strips_for_full_rollout(pred0, true0, strips_dir, chunk=chunk, also_error=args.save_error_strips)

    # full GIF (pre+50) GT vs Pred
    prev0 = x[0].detach().cpu().numpy()
    make_gif_gt_vs_pred(prev0, true0, pred0, osp.join(gifs_dir, f"full_gt_vs_pred_pre{pre_len}_aft{total_aft}"), fps=args.gif_fps)

    # per-chunk metrics + per-chunk gifs
    # chunk k covers [k*chunk : (k+1)*chunk - 1]
    pred_cpu = pred.detach().cpu()
    y_cpu = y.detach().cpu()

    num_chunks = (total_aft + chunk - 1) // chunk
    for k in range(num_chunks):
        start = k * chunk
        end = min((k + 1) * chunk, total_aft)

        pred_k = pred_cpu[:, start:end]  # (1, Tk, C,H,W)
        true_k = y_cpu[:, start:end]

        metrics_k = compute_metrics_framewise(pred_k, true_k, data_range=1.0, do_ssim=args.ssim)
        save_metrics_json_csv(metrics_k, metrics_dir, f"framewise_chunk{k}_{start:03d}-{end-1:03d}")

        # chunk gif (10 frames) GT vs Pred
        pred_k0 = pred_k[0].numpy()
        true_k0 = true_k[0].numpy()
        chunk_k_dir = osp.join(chunk_dir, f"chunk{k}_{start:03d}-{end-1:03d}")
        os.makedirs(chunk_k_dir, exist_ok=True)

        make_chunk_gif(true_k0, pred_k0, osp.join(chunk_k_dir, "gt_vs_pred"), fps=args.gif_fps)

    # print confirmations (so you can see if it's actually writing)
    print("[OK] Saved arrays:", arr_dir)
    print("[OK] Saved full strips:", strips_dir)
    print("[OK] Saved metrics:", metrics_dir)
    print("[OK] Saved full gif:", osp.join(gifs_dir, f"full_gt_vs_pred_pre{pre_len}_aft{total_aft}.gif"))
    print("[OK] Saved chunk gifs under:", chunk_dir)
    print("[OK] Example expected files:")
    print("     ", osp.join(strips_dir, "pred_000-009.png"))
    print("     ", osp.join(strips_dir, "pred_010-019.png"))
    print("     ", osp.join(strips_dir, "pred_020-029.png"))
    print("     ", osp.join(strips_dir, "pred_030-039.png"))
    print("     ", osp.join(strips_dir, "pred_040-049.png"))
