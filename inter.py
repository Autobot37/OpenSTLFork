import os
import os.path as osp
import json
import math
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from torch import nn
import imageio.v2 as imageio

# OpenSTL helpers
from openstl.utils import create_parser, default_parser, load_config, update_config
from openstl.datasets import dataset_parameters, load_data

# Your model class (must match the exact code you pasted / OpenSTL version)
from openstl.models.simvp_model import SimVP_Model  # in OpenSTL this exists

# optional SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None


# -------------------------
# checkpoint utilities
# -------------------------
def _strip_prefixes(sd: Dict[str, torch.Tensor], prefixes=("model.", "module.", "net.", "method.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out


def load_any_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only = False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        # assume it's already a state_dict
        sd = obj
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")

    # strip common prefixes
    sd = _strip_prefixes(sd)

    # sometimes lightning stores keys like "model.model.enc...."
    # strip multiple "model." already handled above; this is extra safety:
    sd = _strip_prefixes(sd, prefixes=("model.",))

    return sd


def sub_state(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Take keys starting with prefix and strip that prefix once."""
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


# -------------------------
# visualization utilities (always write uint8 RGB)
# -------------------------
def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """
    frame can be:
      (C,H,W) or (H,W,C) or (H,W)
    returns uint8 (H,W,3)
    """
    x = frame

    if x.ndim == 3 and x.shape[0] in (1, 3):          # (C,H,W) -> (H,W,C)
        x = np.transpose(x, (1, 2, 0))

    if x.ndim == 3 and x.shape[-1] == 1:              # (H,W,1) -> (H,W)
        x = x[..., 0]

    # floats: assume either [0,1] or [0,255]-ish
    if np.issubdtype(x.dtype, np.floating):
        mx = float(np.nanmax(x)) if x.size else 0.0
        if mx <= 1.0 + 1e-6:
            x = x * 255.0
        x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)

    x = np.clip(x, 0, 255).astype(np.uint8)

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected frame shape for RGB conversion: {x.shape}")

    return x


def save_strip(seq_TCHW: np.ndarray, out_png: str) -> None:
    """
    seq_TCHW: (T,C,H,W) or (T,H,W,C) or (T,H,W)
    Writes one horizontal strip png.
    """
    os.makedirs(osp.dirname(out_png), exist_ok=True)
    frames = [to_uint8_rgb(seq_TCHW[t]) for t in range(seq_TCHW.shape[0])]
    strip = np.concatenate(frames, axis=1)
    imageio.imwrite(out_png, strip)


def save_gif_gt_vs_pred(true_TCHW: np.ndarray, pred_TCHW: np.ndarray, out_gif: str, fps: int = 6) -> None:
    """
    left=GT, right=Pred for T frames.
    """
    os.makedirs(osp.dirname(out_gif), exist_ok=True)
    T = true_TCHW.shape[0]
    assert pred_TCHW.shape[0] == T

    sep = np.full((to_uint8_rgb(true_TCHW[0]).shape[0], 4, 3), 255, dtype=np.uint8)
    frames = []
    for t in range(T):
        gt = to_uint8_rgb(true_TCHW[t])
        pr = to_uint8_rgb(pred_TCHW[t])
        frames.append(np.concatenate([gt, sep, pr], axis=1))

    if not out_gif.endswith(".gif"):
        out_gif += ".gif"
    imageio.mimsave(out_gif, frames, duration=1.0 / float(fps))


# -------------------------
# metrics (dataset-level + framewise)
# -------------------------
@dataclass
class MetricAcc:
    count: int
    mse_sum: float
    mae_sum: float
    psnr_sum: float
    psnr_count: int
    # per-frame accumulators
    mse_t_sum: Optional[np.ndarray] = None
    mae_t_sum: Optional[np.ndarray] = None
    psnr_t_sum: Optional[np.ndarray] = None
    psnr_t_count: Optional[np.ndarray] = None
    ssim_sum: float = 0.0
    ssim_count: int = 0
    ssim_t_sum: Optional[np.ndarray] = None
    ssim_t_count: Optional[np.ndarray] = None


def psnr_from_mse(mse: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    out = np.empty_like(mse, dtype=np.float64)
    for i, m in enumerate(mse):
        if float(m) == 0.0:
            out[i] = float("inf")
        else:
            out[i] = 10.0 * math.log10((data_range * data_range) / float(m))
    return out


def update_metrics(acc: MetricAcc, pred: torch.Tensor, true: torch.Tensor, data_range: float = 1.0, do_ssim: bool = False) -> None:
    """
    pred,true: (B,T,C,H,W), float
    """
    pred = pred.detach().cpu()
    true = true.detach().cpu()

    B, T, C, H, W = pred.shape
    diff = pred - true

    mse_t = (diff * diff).mean(dim=(0, 2, 3, 4)).numpy()  # (T,)
    mae_t = diff.abs().mean(dim=(0, 2, 3, 4)).numpy()     # (T,)
    psnr_t = psnr_from_mse(mse_t, data_range=data_range)

    mse = float(mse_t.mean())
    mae = float(mae_t.mean())
    finite = np.isfinite(psnr_t)
    psnr = float(psnr_t[finite].mean()) if finite.any() else float("inf")

    acc.count += 1
    acc.mse_sum += mse
    acc.mae_sum += mae

    if np.isfinite(psnr):
        acc.psnr_sum += psnr
        acc.psnr_count += 1

    if acc.mse_t_sum is None:
        acc.mse_t_sum = np.zeros((T,), dtype=np.float64)
        acc.mae_t_sum = np.zeros((T,), dtype=np.float64)
        acc.psnr_t_sum = np.zeros((T,), dtype=np.float64)
        acc.psnr_t_count = np.zeros((T,), dtype=np.int64)

    acc.mse_t_sum += mse_t
    acc.mae_t_sum += mae_t
    for i in range(T):
        if np.isfinite(psnr_t[i]):
            acc.psnr_t_sum[i] += psnr_t[i]
            acc.psnr_t_count[i] += 1

    if do_ssim and sk_ssim is not None:
        # compute SSIM on first element in batch for speed; extend if you want full batch
        pred0 = pred[0].numpy()  # (T,C,H,W)
        true0 = true[0].numpy()
        ssim_t = np.zeros((T,), dtype=np.float64)
        for t in range(T):
            if C == 1:
                ssim_t[t] = sk_ssim(true0[t, 0], pred0[t, 0], data_range=data_range)
            else:
                s = 0.0
                for c in range(C):
                    s += sk_ssim(true0[t, c], pred0[t, c], data_range=data_range)
                ssim_t[t] = s / C

        acc.ssim_sum += float(ssim_t.mean())
        acc.ssim_count += 1

        if acc.ssim_t_sum is None:
            acc.ssim_t_sum = np.zeros((T,), dtype=np.float64)
            acc.ssim_t_count = np.zeros((T,), dtype=np.int64)

        acc.ssim_t_sum += ssim_t
        acc.ssim_t_count += 1


def finalize_metrics(acc: MetricAcc) -> Dict:
    out = {
        "num_batches": acc.count,
        "mse": acc.mse_sum / max(acc.count, 1),
        "mae": acc.mae_sum / max(acc.count, 1),
        "psnr": acc.psnr_sum / max(acc.psnr_count, 1),
    }
    if acc.mse_t_sum is not None:
        T = acc.mse_t_sum.shape[0]
        out["mse_t"] = (acc.mse_t_sum / max(acc.count, 1)).tolist()
        out["mae_t"] = (acc.mae_t_sum / max(acc.count, 1)).tolist()
        psnr_t = []
        for i in range(T):
            denom = int(acc.psnr_t_count[i]) if acc.psnr_t_count is not None else 0
            psnr_t.append(float(acc.psnr_t_sum[i] / denom) if denom > 0 else float("inf"))
        out["psnr_t"] = psnr_t

    if acc.ssim_count > 0:
        out["ssim"] = acc.ssim_sum / acc.ssim_count
    if acc.ssim_t_sum is not None and acc.ssim_t_count is not None:
        out["ssim_t"] = (acc.ssim_t_sum / np.maximum(acc.ssim_t_count, 1)).tolist()

    return out


def save_metrics(out_dir: str, name: str, metrics: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # json
    with open(osp.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # csv (framewise if exists)
    if "mse_t" in metrics:
        T = len(metrics["mse_t"])
        has_ssim = "ssim_t" in metrics
        with open(osp.join(out_dir, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            header = ["t", "mse", "mae", "psnr"] + (["ssim"] if has_ssim else [])
            w.writerow(header)
            for t in range(T):
                row = [t, metrics["mse_t"][t], metrics["mae_t"][t], metrics["psnr_t"][t]]
                if has_ssim:
                    row.append(metrics["ssim_t"][t])
                w.writerow(row)


# -------------------------
# main
# -------------------------
def main():
    parser = create_parser()

    # add our args
    parser.add_argument("--ckpt_full", type=str, required=True, help="path to full/long-trained weights (.pth or .ckpt)")
    parser.add_argument("--ckpt_less", type=str, required=True, help="path to less-trained weights (.pth or .ckpt)")
    parser.add_argument("--out_dir", type=str, default="work_dirs/mix_parts", help="root output directory")
    parser.add_argument("--save_vis_batches", type=int, default=1, help="how many batches to save visuals from")
    parser.add_argument("--gif_fps", type=int, default=6)
    parser.add_argument("--ssim", action="store_true", default=False)

    args = parser.parse_args()

    # load config file like OpenSTL scripts do
    cfg_path = osp.join("./configs", args.dataname, f"{args.method}.py") if args.config_file is None else args.config_file
    loaded_cfg = load_config(cfg_path)
    config = update_config(args.__dict__, loaded_cfg,
                           exclude_keys=["method", "val_batch_size", "drop_path", "warmup_epoch"])
    defaults = default_parser()
    for k in defaults.keys():
        if config.get(k, None) is None:
            config[k] = defaults[k]

    # attach dataset defaults
    config.update(dataset_parameters[args.dataname])

    # force reasonable loaders for eval
    config["batch_size"] = config.get("batch_size", 16)
    config["val_batch_size"] = config.get("val_batch_size", 16)
    config["num_workers"] = config.get("num_workers", 4)

    # build test loader
    _, _, test_loader = load_data(**config)

    device = torch.device("cpu") if str(args.device).lower() == "cpu" else torch.device(f"cuda:{args.gpus[0]}")
    print("[INFO] device:", device)

    # model params (fallback to your posted values if missing)
    in_shape = config.get("in_shape", [config["pre_seq_length"], 1, 64, 64])

    model_kwargs = dict(
        in_shape=in_shape,
        hid_S=config.get("hid_S", 64),
        hid_T=config.get("hid_T", 512),
        N_S=config.get("N_S", 4),
        N_T=config.get("N_T", 8),
        model_type=config.get("model_type", "gSTA"),
        mlp_ratio=config.get("mlp_ratio", 8.0),
        drop=config.get("drop", 0.0),
        drop_path=config.get("drop_path", 0.0),
        spatio_kernel_enc=config.get("spatio_kernel_enc", 3),
        spatio_kernel_dec=config.get("spatio_kernel_dec", 3),
    )

    # load both checkpoints into state_dicts compatible with SimVP_Model
    sd_full = load_any_checkpoint(args.ckpt_full)
    sd_less = load_any_checkpoint(args.ckpt_less)

    # sanity: try to load each into a full model to confirm architecture match
    tmp_full = SimVP_Model(**model_kwargs)
    miss, unexp = tmp_full.load_state_dict(sd_full, strict=False)
    print("[CHECK full] missing:", len(miss), " unexpected:", len(unexp))

    tmp_less = SimVP_Model(**model_kwargs)
    miss, unexp = tmp_less.load_state_dict(sd_less, strict=False)
    print("[CHECK less] missing:", len(miss), " unexpected:", len(unexp))

    # split into parts
    # IMPORTANT: submodules expect their *own* key namespace:
    #   SimVP_Model keys: enc.enc.0..., hid.enc.0..., dec.dec.0..., dec.readout...
    #   Encoder module expects: enc.0...
    #   MidMetaNet expects: enc.0...
    #   Decoder expects: dec.0... and readout...
    parts = {
        "full": {
            "enc": sub_state(sd_full, "enc."),   # becomes enc.0... for Encoder
            "hid": sub_state(sd_full, "hid."),   # becomes enc.0... for MidMetaNet
            "dec": sub_state(sd_full, "dec."),   # becomes dec.0.../readout...
        },
        "less": {
            "enc": sub_state(sd_less, "enc."),
            "hid": sub_state(sd_less, "hid."),
            "dec": sub_state(sd_less, "dec."),
        }
    }

    combos = []
    for enc_src in ["full", "less"]:
        for hid_src in ["full", "less"]:
            for dec_src in ["full", "less"]:
                name = f"Enc-{enc_src}__Hid-{hid_src}__Dec-{dec_src}"
                combos.append((name, enc_src, hid_src, dec_src))

    os.makedirs(args.out_dir, exist_ok=True)

    # run all combos
    for combo_name, enc_src, hid_src, dec_src in combos:
        print("\n" + "=" * 80)
        print("[COMBO]", combo_name)

        model = SimVP_Model(**model_kwargs).to(device)
        model.eval()

        # load parts
        miss1, unexp1 = model.enc.load_state_dict(parts[enc_src]["enc"], strict=False)
        miss2, unexp2 = model.hid.load_state_dict(parts[hid_src]["hid"], strict=False)
        miss3, unexp3 = model.dec.load_state_dict(parts[dec_src]["dec"], strict=False)

        print(f"[LOAD enc={enc_src}] missing={len(miss1)} unexpected={len(unexp1)}")
        print(f"[LOAD hid={hid_src}] missing={len(miss2)} unexpected={len(unexp2)}")
        print(f"[LOAD dec={dec_src}] missing={len(miss3)} unexpected={len(unexp3)}")

        # output dirs
        out_root = osp.join(args.out_dir, combo_name)
        arr_dir = osp.join(out_root, "arrays")
        vis_dir = osp.join(out_root, "visuals")
        met_dir = osp.join(out_root, "metrics")
        os.makedirs(arr_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(met_dir, exist_ok=True)

        # metric accumulator
        acc = MetricAcc(
            count=0, mse_sum=0.0, mae_sum=0.0, psnr_sum=0.0, psnr_count=0
        )

        # eval loop
        vis_saved = 0
        max_batches = args.limit_test_batches if args.limit_test_batches is not None else 10
        if max_batches == -1:
            max_batches = None

        with torch.no_grad():
            for bi, batch in enumerate(test_loader):
                if max_batches is not None and bi >= max_batches:
                    break

                x, y = batch  # OpenSTL MovingMNIST returns (input, output)
                x = x.to(device).float()  # (B, pre, C, H, W)
                y = y.to(device).float()  # (B, aft, C, H, W)

                pred = model(x)  # (B, pre, C, H, W) in this SimVP implementation

                # if aft != pre, align by cropping to min
                T = min(pred.shape[1], y.shape[1])
                pred = pred[:, :T]
                yy = y[:, :T]

                update_metrics(acc, pred, yy, data_range=1.0, do_ssim=args.ssim)

                # save a couple of visual examples
                if vis_saved < args.save_vis_batches:
                    # store arrays (first sample only)
                    x0 = x[0].detach().cpu().numpy()      # (pre,C,H,W)
                    y0 = yy[0].detach().cpu().numpy()     # (T,C,H,W)
                    p0 = pred[0].detach().cpu().numpy()   # (T,C,H,W)

                    # clamp for visuals only
                    x0v = np.clip(x0, 0.0, 1.0)
                    y0v = np.clip(y0, 0.0, 1.0)
                    p0v = np.clip(p0, 0.0, 1.0)

                    np.save(osp.join(arr_dir, f"inputs_b{bi}.npy"), x0.astype(np.float32))
                    np.save(osp.join(arr_dir, f"trues_b{bi}.npy"),  y0.astype(np.float32))
                    np.save(osp.join(arr_dir, f"preds_b{bi}.npy"),  p0.astype(np.float32))

                    save_strip(x0v, osp.join(vis_dir, f"input_strip_b{bi}.png"))
                    save_strip(y0v, osp.join(vis_dir, f"true_strip_b{bi}.png"))
                    save_strip(p0v, osp.join(vis_dir, f"pred_strip_b{bi}.png"))
                    save_gif_gt_vs_pred(y0v, p0v, osp.join(vis_dir, f"gt_vs_pred_b{bi}.gif"), fps=args.gif_fps)

                    vis_saved += 1

        metrics = finalize_metrics(acc)
        save_metrics(met_dir, "test_metrics", metrics)

        print("[DONE]", combo_name)
        print("  mse :", metrics.get("mse"))
        print("  mae :", metrics.get("mae"))
        print("  psnr:", metrics.get("psnr"))
        if "ssim" in metrics:
            print("  ssim:", metrics.get("ssim"))

    print("\n[OK] all combos finished. Output root:", args.out_dir)


if __name__ == "__main__":
    main()
