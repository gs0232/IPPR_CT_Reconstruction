"""
train_3dgs.py - per-slice Gaussian-splatting reconstruction for the IPPR sparse-view CT project.

For every (series, slice, n_views):
  1. keep n_views angles of the full sinogram (EXACTLY Sophia's protocol),
  2. initialise Gaussians from the sparse-view FBP (same input as the baseline),
  3. optimise the Gaussians so their projections match the sparse sinogram (L1 + small TV),
  4. render the 888x888 image -> save as float .npy in RAW FBP units (no 0-1 rescale),
  5. evaluate: (a) image vs. full-view FBP ground truth with Baselines/metrics.py,
               (b) view synthesis: rendered vs. true projections at HELD-OUT angles.
The full-view ground truth is only used for evaluation, never for fitting.

=====================================================================================
TODO CHECKLIST - follow in this order  (nothing here has been run yet!)
Tags in the code:  TODO(run) = something you must do   TODO(tune) = hyper-parameter
                   TODO(improve) = optional upgrade for later
=====================================================================================
[ ] 1. SETUP   Put this folder (3DGS/) in the repo root next to Baselines/ (metrics.py is
               imported from there). pip install -r 3DGS/requirements.txt (the root
               requirements.txt lacks scipy/pandas/matplotlib). Use a GPU runtime
               (Colab T4 is enough). CPU works but is far too slow.
               Colab: mount Drive in a NOTEBOOK cell first (drive.mount), then run
               `!python 3DGS/train_3dgs.py ...` (this script does not mount Drive).
[ ] 2. PATHS   Check the path block in `Cfg` (data_root, sino_zip, gt_dir, out_dir) or pass
               --data-root/--gt-dir/--out-dir. Unzipping SinoCT_files.zip took ~6 min
               for Sophia, so keep --sino-dir on the local Colab disk.
[ ] 3. GEOMETRY  `python 3DGS/train_3dgs.py --sanity`   (seconds)
               Must show relative error of a few % for the +1 angle sign. If -1 is better,
               set cfg.angle_sign = -1. The 2nd check (real data) should give a best-fit
               scale close to 1.0; if not, image units != sinogram units and every number
               afterwards is wrong -> fix BEFORE running anything long.
[ ] 4. SMOKE   `python 3DGS/train_3dgs.py --smoke`      (1 slice, 49 views, 300 iterations)
               Look at the printed s/iter, then open <out_dir>/_smoke/previews/*.png.
               Expectation: 3DGS PSNR >= sparse FBP at the same views. Use the s/iter to
               estimate the full runtime:  fits x iters x s/iter.
               If CUDA OOM: lower views_per_step or gauss_chunk.
[ ] 5. TUNE    Hyper-parameters are tuned on the VALIDATION series only (README), not on test:
               `--series series_1023 --max-slices 3 --views 49 98`   (series_1023 = val series
               in Jule's split - confirm with the team). Knobs, roughly by importance:
               iters, n_init, lr_xy/lr_rho, tv_weight {0, .005, .01, .05}, densify_*, init.
               Write the chosen values into Cfg defaults and log them in Log/log_gianne.md.
[ ] 6. FULL    `python 3DGS/train_3dgs.py`  (defaults: test series, slice_step 4, views
               246/98/49/25). Resume-safe: rows already in the CSV are skipped, so a Colab
               disconnect only costs the current fit. TODO(run): ask Sophia which series
               she evaluates (baseline CSV covers all 94 series; running all would be
               94 x 11 slices x 4 views ~ 4000 fits = far too many for the deadline).
[ ] 7. HAND-IN Outputs go to <out_dir>/series_XXXX_sliceYY_vNNN.npy (888x888 float32, raw
               units). Sophia needs them in Drive `outputs/<method>/` by 20.10 - confirm the
               folder name ("3dgs"?). Send her <out_dir>/3dgs_results.csv too.
[ ] 8. REPORT  Fill Log/log_gianne.md; add runtime numbers (runtime_s column) and the
               held-out-view metrics (proj_psnr / proj_ssim) for the view-synthesis part.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.transform import iradon, radon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # gs_ct.py
sys.path.insert(0, str(HERE.parent / "Baselines"))  # metrics.py   
try:
    import metrics  # shared metric functions (PSNR, SSIM, RMSE, head mask)
except ImportError as e:  # pragma: no cover
    raise SystemExit("metrics.py not found - put 3DGS/ next to Baselines/ or edit sys.path above") from e
from gs_ct import GaussianCT, density_reference, init_from_image, init_random

# ----------------------------------------------------------------------------------
# Fixed protocol constants - copied from Baselines/02_fbp_baseline.ipynb.
# DO NOT CHANGE: all methods must see the same angles / geometry as the FBP baseline.
# ----------------------------------------------------------------------------------
N_ANGLES, N_DET = 984, 888
ANGLE_RANGE = 360
OUTPUT_SIZE = 888
FILTER = "ramp"
VIEW_COUNTS = [246, 98, 49, 25]
SLICE_STEP = 4
THETA_FULL_DEG = np.linspace(0, ANGLE_RANGE, N_ANGLES, endpoint=False)


def view_indices(n_views):
    """Same angle selection as the baseline (spread over the first 180 deg only,
    because theta and theta+180 carry the same information)."""
    n_half = N_ANGLES if ANGLE_RANGE == 180 else N_ANGLES // 2
    return np.round(np.linspace(0, n_half, n_views, endpoint=False)).astype(int)


@dataclass
class Cfg:
    # ---- paths (Colab defaults from the baseline notebook) ------------------------
    data_root: str = "/content/drive/MyDrive/MGST_5_IPPR_CT_Reconstruction"  # TODO(run)
    sino_zip: str = "/content/drive/MyDrive/MGST_5_IPPR_CT_Reconstruction/SinoCT_files.zip"       # default: <data_root>/SinoCT_files.zip
    sino_dir: str = "/content/sinograms"
    gt_dir: str = "/content/drive/MyDrive/MGST_5_IPPR_CT_Reconstruction/baseline_results/gt_fullview_npy"         # default: <data_root>/baseline_results/gt_fullview_npy
    out_dir: str = "/content/drive/MyDrive/MGST_5_IPPR_CT_Reconstruction/outputs/3dgs"        # default: <data_root>/outputs/3dgs   TODO(run): confirm name with Sophia

    # ---- what to run ----------------------------------------------------------------
    # TODO(run): U-Net test series from Jule's log; confirm with Sophia. None = all series.
    series: list = field(default_factory=lambda: ["series_1014", "series_1028"])
    views: list = field(default_factory=lambda: list(VIEW_COUNTS))
    slice_step: int = SLICE_STEP
    max_slices: int = 0      # 0 = all slices of the series (every slice_step-th)
    seed: int = 42

    # ---- model / init -----------------------------------------------------------------
    init: str = "fbp"        # "fbp" | "random"   TODO(improve): ablate in the report
    n_init: int = 15000      # TODO(tune)
    max_gaussians: int = 40000   # TODO(tune) memory/time grow linearly with this
    init_gamma: float = 0.5
    init_uniform_frac: float = 0.1
    s_min: float = 0.5       # px, clamp on Gaussian scale
    s_max: float = 60.0
    angle_sign: float = 1.0  # TODO(run): set to -1 only if --sanity says so

    # ---- optimisation ------------------------------------------------------------------
    iters: int = 3000        # TODO(tune) R2-Gaussian uses 30k in 3D; 2D converges much faster
    views_per_step: int = 8  # TODO(tune) lower on OOM
    gauss_chunk: int = 16384 # TODO(tune) lower on OOM
    use_checkpoint: bool = True
    lr_xy: float = 0.1       # TODO(tune) pixels per step (Adam)
    lr_scale: float = 0.01
    lr_rot: float = 0.01
    lr_rho: float = 0.02
    lr_final_ratio: float = 0.05   # lr decays exponentially to this fraction
    tv_weight: float = 0.01  # TODO(tune) 0 disables TV
    tv_patch: int = 32
    tv_chunk: int = 4096
    l2_weight: float = 0.0   # TODO(tune) optional extra L2 term on the sinogram

    # ---- adaptive density control -----------------------------------------------------
    densify_from: float = 0.10    # fractions of `iters`
    densify_until: float = 0.60
    densify_every: int = 100
    densify_top_frac: float = 0.05
    split_px: float = 4.0         # TODO(tune)
    prune_rho_rel: float = 0.003  # prune if density < this * tissue-density reference

    # ---- evaluation / io ------------------------------------------------------------------
    n_heldout: int = 64      # number of held-out angles for the view-synthesis metrics
    save_preview: bool = True
    log_every: int = 250
    smoke: bool = False

    def finalize(self):
        root = Path(self.data_root)
        self.sino_zip = self.sino_zip or str(root / "SinoCT_files.zip")
        self.gt_dir = self.gt_dir or str(root / "baseline_results" / "gt_fullview_npy")
        self.out_dir = self.out_dir or str(root / "outputs" / "3dgs")
        return self


# ----------------------------------------------------------------------------------
# Data helpers (copied from the baseline notebook so layouts stay identical)
# ----------------------------------------------------------------------------------
def as_slices(arr):
    """-> (n_slices, N_DET, N_ANGLES). SinoCT files are stored as (984, 888, n_slices)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    a_ax = [i for i, s in enumerate(arr.shape) if s == N_ANGLES]
    d_ax = [i for i, s in enumerate(arr.shape) if s == N_DET]
    if not a_ax or not d_ax:
        raise ValueError(f"unexpected sinogram shape {arr.shape}")
    s_ax = ({0, 1, 2} - {a_ax[0], d_ax[0]}).pop()
    return np.transpose(arr, (s_ax, d_ax[0], a_ax[0]))


def series_id(path, root):
    m = re.search(r"(\d{3,})", str(Path(path).relative_to(root)))
    return f"series_{m.group(1)}" if m else Path(path).stem


def find_sinograms(cfg):
    """Return the sinogram .npy files for cfg.series, extracting ONLY those series from the
    zip if they are not on disk yet (saves time and space; works on Windows/Linux/Colab)."""
    sino_dir = Path(cfg.sino_dir)
    wanted = set(cfg.series) if cfg.series else None
    have = {series_id(f, sino_dir) for f in sino_dir.rglob("*.npy")} if sino_dir.exists() else set()
    missing = (not have) if wanted is None else bool(wanted - have)
    if missing:
        zp = Path(cfg.sino_zip)
        if not zp.exists():
            raise SystemExit(
                f"Sinogram zip not found: {zp}\n"
                "TODO(run): pass --sino-zip <path to SinoCT_files.zip> (or --data-root), or point "
                "--sino-dir at a folder that already contains the extracted series.")
        print(f"extracting {sorted(wanted) if wanted else 'ALL series'} from {zp.name} ...")
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                m = re.search(r"(\d{3,})", name)
                sid = f"series_{m.group(1)}" if m else None
                if name.endswith(".npy") and (wanted is None or sid in wanted):
                    z.extract(name, sino_dir)
    files = sorted(sino_dir.rglob("*.npy"))
    if wanted:
        files = [f for f in files if series_id(f, sino_dir) in wanted]
    if not files:
        raise SystemExit("no sinogram files found - check --sino-dir / --sino-zip / --series")
    return files, sino_dir


def load_gt(cfg, sid, s, sino_slice):
    """Full-view FBP ground truth, cached by Sophia as <sid>_slice<SS>.npy."""
    path = Path(cfg.gt_dir) / f"{sid}_slice{s:02d}.npy"
    if path.exists():
        return np.load(path)
    print(f"  [warn] {path.name} missing -> computing full-view FBP (~14 s). "
          "TODO(run): make sure gt_dir points at Sophia's gt_fullview_npy folder.")
    gt = iradon(sino_slice, theta=THETA_FULL_DEG, filter_name=FILTER,
                output_size=OUTPUT_SIZE, circle=True).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, gt)
    return gt


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------
# Training pieces
# ----------------------------------------------------------------------------------
def make_optimizer(model, cfg):
    groups = [
        {"params": [model.xy], "lr": cfg.lr_xy},
        {"params": [model.log_scale], "lr": cfg.lr_scale},
        {"params": [model.rot], "lr": cfg.lr_rot},
        {"params": [model.log_rho], "lr": cfg.lr_rho},
    ]
    for g in groups:
        g["base_lr"] = g["lr"]
    return torch.optim.Adam(groups, eps=1e-15)


@torch.no_grad()
def calibrate_density(model, theta, det, target, cfg):
    """Scale all densities so the initial projections have the same total mass as the
    measured sparse sinogram (cheap fix for the init-density approximation)."""
    pred_sum = 0.0
    for i in range(0, len(theta), 16):
        pred_sum += float(model.project(theta[i:i + 16], det, cfg.gauss_chunk, False).sum())
    alpha = float(target.sum()) / max(pred_sum, 1e-12)
    model.log_rho.data += float(np.log(max(alpha, 1e-6)))
    return alpha


def tv_patch_loss(model, patch, rho_ref, chunk):
    """Total variation of a random patch of the rendered image (differentiable).
    Normalised by rho_ref so the weight does not depend on the sinogram scale."""
    c = model.c
    r = 0.8 * c * np.sqrt(np.random.rand())
    a = 2 * np.pi * np.random.rand()
    cx, cy = int(r * np.cos(a)), int(r * np.sin(a))
    dev = model.xy.device
    ar = torch.arange(patch, device=dev, dtype=torch.float32) - patch // 2
    img = model.render_region(ar + cy, ar + cx, chunk=chunk) / rho_ref
    return (img[1:] - img[:-1]).abs().mean() + (img[:, 1:] - img[:, :-1]).abs().mean()


def fit_one(sino, n_views, cfg, device, preview_path=None, gt=None):
    """Fit one slice. sino: raw (N_DET, N_ANGLES). Returns dict with reconstruction etc."""
    set_seed(cfg.seed)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sync()
    t0 = time.perf_counter()

    idx = view_indices(n_views)
    theta_deg = THETA_FULL_DEG[idx]
    sparse = sino[:, idx]                              # (N_DET, V)
    scale = float(sparse.max()) or 1.0                 # normalise sinogram to ~[0,1] for optimisation
    fbp = iradon(sparse, theta=theta_deg, filter_name=FILTER,
                 output_size=OUTPUT_SIZE, circle=True).astype(np.float32)
    fbp_n = fbp / scale
    rho_ref = density_reference(fbp_n, OUTPUT_SIZE)

    if cfg.init == "fbp":
        model = init_from_image(fbp_n, cfg.n_init, OUTPUT_SIZE, cfg.init_uniform_frac,
                                cfg.init_gamma, cfg.seed, cfg.s_min, cfg.s_max)
    else:
        model = init_random(cfg.n_init, OUTPUT_SIZE, cfg.seed, s_min=cfg.s_min, s_max=cfg.s_max)
    model = model.to(device)

    det = torch.arange(N_DET, device=device, dtype=torch.float32) - N_DET // 2
    theta = torch.tensor(np.deg2rad(theta_deg) * cfg.angle_sign, dtype=torch.float32, device=device)
    target = torch.tensor((sparse / scale).T, dtype=torch.float32, device=device)  # (V, D)
    alpha = calibrate_density(model, theta, det, target, cfg)

    opt = make_optimizer(model, cfg)
    V = min(cfg.views_per_step, n_views)
    d_from, d_until = int(cfg.densify_from * cfg.iters), int(cfg.densify_until * cfg.iters)
    gacc, gden = torch.zeros(model.n, device=device), torch.zeros(model.n, device=device)
    loss_val, t_loop = float("nan"), time.perf_counter()

    for it in range(1, cfg.iters + 1):
        decay = cfg.lr_final_ratio ** (it / cfg.iters)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * decay

        sel = torch.randperm(n_views, device=device)[:V]
        pred = model.project(theta[sel], det, cfg.gauss_chunk, cfg.use_checkpoint)
        err = pred - target[sel]
        loss = err.abs().mean()
        if cfg.l2_weight > 0:
            loss = loss + cfg.l2_weight * (err ** 2).mean()
        if cfg.tv_weight > 0:
            loss = loss + cfg.tv_weight * tv_patch_loss(model, cfg.tv_patch, rho_ref, cfg.tv_chunk)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        in_window = d_from <= it <= d_until
        if in_window and model.xy.grad is not None:
            gacc += model.xy.grad.norm(dim=-1)
            gden += 1
        opt.step()
        with torch.no_grad():
            model.xy.data.clamp_(-model.c, model.c)

        if in_window and it % cfg.densify_every == 0:
            info = model.densify_and_prune(gacc, gden, cfg.densify_top_frac, cfg.split_px,
                                           cfg.prune_rho_rel * rho_ref, cfg.max_gaussians)
            opt = make_optimizer(model, cfg)   # TODO(improve): keep Adam state instead of resetting
            gacc, gden = torch.zeros(model.n, device=device), torch.zeros(model.n, device=device)
            print(f"    it {it}: densify {info}")

        if it % cfg.log_every == 0 or it == cfg.iters:
            loss_val = float(loss)
            print(f"    it {it:5d}/{cfg.iters} | loss {loss_val:.5f} | N={model.n} | "
                  f"{(time.perf_counter() - t_loop) / it:.3f} s/it")

    rec_n = model.render_image(chunk=64)
    rec = rec_n.cpu().numpy().astype(np.float32) * scale     # back to RAW FBP units
    sync()
    runtime = time.perf_counter() - t0                       # fit + render (comparable to FBP runtime)

    out = dict(rec=rec, runtime=runtime, n_gauss=model.n, loss=loss_val, alpha=alpha, fbp=fbp)

    # ---- view synthesis: held-out angles (unique half-circle indices not used for fitting) ----
    n_half = N_ANGLES // 2 if ANGLE_RANGE == 360 else N_ANGLES
    held = np.setdiff1d(np.arange(n_half), idx)
    if len(held) > 0:
        held = held[np.unique(np.round(np.linspace(0, len(held) - 1, cfg.n_heldout)).astype(int))]
        th_h = torch.tensor(np.deg2rad(THETA_FULL_DEG[held]) * cfg.angle_sign,
                            dtype=torch.float32, device=device)
        with torch.no_grad():
            cols = [model.project(th_h[i:i + 16], det, cfg.gauss_chunk, False)
                    for i in range(0, len(th_h), 16)]
        pred_sino = torch.cat(cols).cpu().numpy().T * scale   # (D, n_held) raw units
        gt_sino = sino[:, held]
        g_n, p_n = metrics.normalise_pair(gt_sino, pred_sino)  # no head mask in sinogram domain
        out.update(proj_psnr=metrics.psnr(p_n, g_n), proj_ssim=metrics.ssim(p_n, g_n),
                   proj_rmse=metrics.rmse(p_n, g_n))

    if preview_path is not None and gt is not None:
        save_preview(preview_path, gt, fbp, rec)

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def save_preview(path, gt, fbp, rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, f_n = metrics.normalise_pair(gt, fbp)
    g_n, r_n = metrics.normalise_pair(gt, rec)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, im, t in zip(ax[:3], [f_n, r_n, g_n], ["sparse FBP", "3DGS", "ground truth (full FBP)"]):
        a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(t); a.axis("off")
    ax[3].imshow(np.abs(r_n - g_n), cmap="magma", vmin=0, vmax=0.3); ax[3].set_title("|3DGS - GT|")
    ax[3].axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------------
# Sanity check of the geometry (run this first: `--sanity`)
# ----------------------------------------------------------------------------------
def sanity_check(cfg, device):
    S, c = N_DET, N_DET // 2
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    model = GaussianCT(t([[0, 0], [60, -40], [-100, 80]]),
                       t([[10, 5], [4, 12], [15, 15]]).log(), t([0.3, 1.0, 0.0]),
                       t([1.0, 2.0, 0.5]).log(), size=S).to(device)
    img = model.render_image(chunk=3).cpu().numpy()
    theta_deg = np.array([0, 17, 45, 90, 133, 171], dtype=float)
    ref = radon(img, theta=theta_deg, circle=True)             # skimage forward projector
    det = torch.arange(S, dtype=torch.float32, device=device) - c
    print("[1] Gaussian projector vs skimage.radon of the rendered image")
    errs = {}
    for sign in (1.0, -1.0):
        th = torch.tensor(np.deg2rad(theta_deg) * sign, dtype=torch.float32, device=device)
        p = model.project(th, det, chunk=None).detach().cpu().numpy().T
        errs[sign] = np.linalg.norm(p - ref) / np.linalg.norm(ref)
        print(f"    angle_sign {sign:+.0f}: relative error {errs[sign]:.3%}")
    best = min(errs, key=errs.get)
    print(f"    -> use angle_sign = {best:+.0f}" + ("" if errs[best] < 0.05 else
          "   [WARNING: error still > 5% - the projector/renderer geometry is off, do not run!]"))

    print("[2] real data: radon(ground-truth image) vs measured sinogram (needs data paths)")
    try:
        cfg2 = Cfg(**{**cfg.__dict__, "series": cfg.series[:1] if cfg.series else None})
        files, sdir = find_sinograms(cfg2)
        sino = as_slices(np.load(files[0]))[0]
        sid = series_id(files[0], sdir)
        gt = load_gt(cfg, sid, 0, sino)
        idx = view_indices(16)
        q = radon(gt, theta=THETA_FULL_DEG[idx], circle=True)
        m = sino[:, idx]
        a = float((q * m).sum() / (q * q).sum())
        rel = np.linalg.norm(a * q - m) / np.linalg.norm(m)
        print(f"    best-fit scale {a:.3f} (should be ~1.0), residual after fit {rel:.1%}")
    except Exception as e:  # noqa: BLE001
        print(f"    skipped ({type(e).__name__}: {e})")


# ----------------------------------------------------------------------------------
# Main loop (mirrors the baseline notebook: series -> slices -> view counts)
# ----------------------------------------------------------------------------------
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] no GPU found - this will be extremely slow. TODO(run): use a GPU runtime.")
    out_dir = Path(cfg.out_dir)
    csv_name = "3dgs_results.csv"
    if cfg.smoke:
        out_dir, csv_name = out_dir / "_smoke", "3dgs_smoke.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / csv_name

    files, sdir = find_sinograms(cfg)
    if cfg.smoke:
        files = files[:1]
    done = set()
    if csv.exists():
        d = pd.read_csv(csv)
        done = set(zip(d["series"], d["slice"], d["views"]))

    for f in files:
        sid = series_id(f, sdir)
        sinos = as_slices(np.load(f))
        slices = list(range(0, sinos.shape[0], cfg.slice_step))
        if cfg.smoke:
            slices = slices[:1]
        elif cfg.max_slices:
            slices = slices[:cfg.max_slices]
        for s in slices:
            gt = None
            for v in cfg.views:
                if (sid, s, v) in done:
                    print(f"{sid} slice {s:02d} v{v:03d}: already done, skipped"); continue
                gt = load_gt(cfg, sid, s, sinos[s]) if gt is None else gt
                print(f"{sid} slice {s:02d} | {v} views")
                prev = out_dir / "previews" / f"{sid}_slice{s:02d}_v{v:03d}.png" if cfg.save_preview else None
                r = fit_one(sinos[s], v, cfg, device, prev, gt)
                np.save(out_dir / f"{sid}_slice{s:02d}_v{v:03d}.npy", r["rec"])
                m = metrics.evaluate(r["rec"], gt)
                m_fbp = metrics.evaluate(r["fbp"], gt)
                row = {"series": sid, "slice": s, "method": "3DGS", "views": v,
                       "runtime_s": r["runtime"], **m,
                       "proj_psnr": r.get("proj_psnr"), "proj_ssim": r.get("proj_ssim"),
                       "proj_rmse": r.get("proj_rmse"), "n_gaussians": r["n_gauss"],
                       "final_loss": r["loss"], "iters": cfg.iters, "seed": cfg.seed,
                       "fbp_psnr_same_views": m_fbp["psnr"], "fbp_ssim_same_views": m_fbp["ssim"]}
                pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
                print(f"  -> PSNR {m['psnr']:.2f} (FBP {m_fbp['psnr']:.2f}) | SSIM {m['ssim']:.3f} "
                      f"(FBP {m_fbp['ssim']:.3f}) | head-PSNR {m['psnr_head']:.2f} | "
                      f"held-out proj PSNR {r.get('proj_psnr', float('nan')):.2f} | {r['runtime']:.0f} s")

    if csv.exists():
        res = pd.read_csv(csv)
        cols = ["psnr", "ssim", "rmse", "psnr_head", "ssim_head", "rmse_head",
                "proj_psnr", "proj_ssim", "runtime_s"]
        summary = res.groupby("views")[cols].agg(["mean", "std"]).sort_index(ascending=False).round(3)
        summary.to_csv(out_dir / csv_name.replace(".csv", "_summary.csv"))
        print(summary)


def parse_args():
    p = argparse.ArgumentParser(description="Per-slice Gaussian-splatting CT reconstruction")
    p.add_argument("--sanity", action="store_true", help="geometry check against skimage, then exit")
    p.add_argument("--smoke", action="store_true", help="1 slice, 49 views, 300 iters, separate output")
    p.add_argument("--series", nargs="*", help="series ids, e.g. series_1023 (omit = Cfg default)")
    p.add_argument("--all-series", action="store_true", help="ignore the series filter")
    p.add_argument("--views", nargs="*", type=int)
    for name in ["data_root", "sino_zip", "sino_dir", "gt_dir", "out_dir", "init"]:
        p.add_argument("--" + name.replace("_", "-"), default=None)
    for name in ["iters", "n_init", "max_slices", "slice_step", "views_per_step", "gauss_chunk", "seed"]:
        p.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    for name in ["tv_weight", "lr_xy", "lr_rho", "angle_sign"]:
        p.add_argument("--" + name.replace("_", "-"), type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = Cfg()
    for f_ in fields(Cfg):
        v = getattr(a, f_.name, None)
        if v is not None and f_.name not in ("series", "views"):
            setattr(cfg, f_.name, v)
    if a.series:
        cfg.series = a.series
    if a.all_series:
        cfg.series = None
    if a.views:
        cfg.views = a.views
    if a.smoke:
        cfg.smoke, cfg.views, cfg.iters, cfg.n_init, cfg.log_every = True, [49], 300, 5000, 50
        cfg.densify_every = 50
    cfg.finalize()
    if a.sanity:
        sanity_check(cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        main(cfg)