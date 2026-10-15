"""
metrics.py – shared evaluation metrics for the IPPR sparse-view CT project.

Every method (FBP baseline, U-Net, NeRF, 3DGS) should be evaluated with
`evaluate(pred, gt)` from this file, so that all numbers in the report are
computed in exactly the same way.

Conventions
-----------
- `gt`   : ground truth = full-view FBP reconstruction (2D float array)
- `pred` : reconstruction of the method being evaluated (same shape as gt)
- Both images are scaled with the min/max of the *ground truth* (see
  `normalise_pair`), so that intensity errors are not hidden by stretching
  each image separately. Metrics then use a fixed data range of 1.0.
- Metrics are reported for the whole image and inside a head mask
  ("_head" suffix), because the large dark background inflates PSNR/SSIM.

Author: Sophia Gwiggner (Baselines / Evaluation)
"""

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity

DATA_RANGE = 1.0


def normalise_pair(gt, pred):
    """Scale gt and pred with the SAME constants (min/max of gt) to roughly [0, 1].

    Parameters
    ----------
    gt, pred : np.ndarray
        2D images of identical shape.

    Returns
    -------
    gt_n, pred_n : np.ndarray (float32)
        gt_n lies exactly in [0, 1]; pred_n may lie slightly outside, which is
        intended (over-/undershoots are real reconstruction errors).
    """
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    lo, hi = gt.min(), gt.max()
    scale = hi - lo if hi > lo else 1.0
    return (gt - lo) / scale, (pred - lo) / scale


def head_mask(gt_n):
    """Binary mask of the head region, computed automatically from the ground truth.

    Otsu's method picks the threshold between background and tissue by itself,
    so no manual threshold has to be chosen. Holes inside the skull are filled
    and only the largest connected region is kept.
    """
    mask = gt_n > threshold_otsu(gt_n)
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    if n > 1:  # keep the largest connected component (the head)
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = labels == (np.argmax(sizes) + 1)
    return mask


def psnr(pred, gt, mask=None):
    """Peak signal-to-noise ratio in dB (higher = better)."""
    err = (pred - gt) ** 2
    mse = err[mask].mean() if mask is not None else err.mean()
    return float("inf") if mse == 0 else float(10 * np.log10(DATA_RANGE ** 2 / mse))


def rmse(pred, gt, mask=None):
    """Root mean squared error (lower = better)."""
    err = (pred - gt) ** 2
    return float(np.sqrt(err[mask].mean() if mask is not None else err.mean()))


def ssim(pred, gt, mask=None):
    """Structural similarity index (1 = identical). With a mask, the local SSIM
    map is averaged over the masked pixels only."""
    value, ssim_map = structural_similarity(gt, pred, data_range=DATA_RANGE, full=True)
    return float(ssim_map[mask].mean()) if mask is not None else float(value)


def evaluate(pred, gt, already_normalised=False):
    """Compute all metrics for one 2D slice.

    Parameters
    ----------
    pred : np.ndarray   reconstruction of the method
    gt   : np.ndarray   full-view FBP ground truth
    already_normalised : bool
        Set True if both images were already scaled with `normalise_pair`.

    Returns
    -------
    dict with psnr, ssim, rmse (whole image) and psnr_head, ssim_head,
    rmse_head (inside the head mask).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    if not already_normalised:
        gt, pred = normalise_pair(gt, pred)
    mask = head_mask(gt)
    return {
        "psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "rmse": rmse(pred, gt),
        "psnr_head": psnr(pred, gt, mask), "ssim_head": ssim(pred, gt, mask),
        "rmse_head": rmse(pred, gt, mask),
    }


def _self_test():
    """Quick sanity checks – run `python metrics.py`."""
    rng = np.random.default_rng(0)
    gt = np.zeros((128, 128), np.float32)
    gt[32:96, 32:96] = 1.0
    m = evaluate(gt, gt)
    assert m["rmse"] == 0 and np.isinf(m["psnr"]) and abs(m["ssim"] - 1) < 1e-6
    noisy = gt + rng.normal(0, 0.05, gt.shape).astype(np.float32)
    m2 = evaluate(noisy, gt)
    assert 20 < m2["psnr"] < 30 and m2["ssim"] < 1 and m2["rmse"] > 0
    print("metrics.py self-test passed:", {k: round(v, 3) for k, v in m2.items()})


if __name__ == "__main__":
    _self_test()
