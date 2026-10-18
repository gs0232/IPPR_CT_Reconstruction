"""
gaussian_ct.py - Gaussian-splatting model for sparse-view CT (one axial slice, parallel beam)
==============================================================================================

WHY 2D GAUSSIANS FOR A "3DGS" TASK?
-----------------------------------
Our data are single axial slices with parallel-beam sinograms (984 angles x 888 detector
pixels, reconstructed with `skimage.transform.iradon` in Baselines/02_fbp_baseline.ipynb).
A 3D Gaussian cut by one slice plane is a 2D Gaussian, and the line integral of a Gaussian
along a ray is again a Gaussian that we can write down in closed form. This is the idea of
R2-Gaussian (Zha et al., 2024) and X-Gaussian, specialised to one slice, so:

  * no CUDA rasteriser is needed - plain PyTorch runs on a Colab GPU,
  * the forward model is an EXACT X-ray projector, not an alpha-blending approximation
    (3DGS (Kerbl et al., 2023) renders colours with opacity; CT needs line integrals).

THE MATH (all lengths in pixels, origin = image centre)
-------------------------------------------------------
Density of Gaussian i:   G_i(x) = rho_i * exp(-1/2 (x-mu_i)^T Sigma_i^-1 (x-mu_i))
                         Sigma_i = R(phi_i) diag(sigma1^2, sigma2^2) R(phi_i)^T
Detector coordinate for projection angle theta (same convention as skimage's iradon):
                         u = x*cos(theta) - y*sin(theta)      (x = column, y = row offset)
Closed-form line integral of G_i over all rays that hit detector bin u:
                         P_i(u) = rho_i * sqrt(2*pi) * sigma1*sigma2 / s_i
                                  * exp(-(u - m_i)^2 / (2 s_i^2))
                         m_i   = mu_i . n,        n = (cos theta, -sin theta)
                         s_i^2 = n^T Sigma_i n = sigma1^2 cos^2(phi+theta) + sigma2^2 sin^2(phi+theta)
Sinogram at angle theta:  p(u, theta) = sum_i P_i(u)        (sum over all Gaussians)
Reconstructed image:      f(x, y)     = sum_i G_i(x, y)     (same units as the FBP image)

CONTENTS
--------
  GaussianCT.from_fbp()          initialise Gaussians from the sparse-view FBP image
  GaussianCT.project()           differentiable sinogram rendering at given angles
  GaussianCT.render_image()      tile-wise image rendering (no grad)
  GaussianCT.tv_loss()           total variation on a random image patch (regulariser)
  GaussianCT.densify_and_prune() clone / split / prune (adaptive density control)
  _self_test()                   `python gaussian_ct.py` -> checks geometry against skimage.radon

NOTE: written without being run (training would take hours). Run the self-test first.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SQRT_2PI = math.sqrt(2.0 * math.pi)

# TODO(geometry): `python gaussian_ct.py` compares our projector with skimage's radon.
#   If it tells you that the sign-flipped angle matches better, set this to -1.0.
THETA_SIGN = 1.0

PARAM_NAMES = ("xy", "log_scale", "rot", "raw_rho")   # names of the optimised tensors


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus (numerically stable), y > 0."""
    y = y.clamp_min(1e-6)
    return y + torch.log(-torch.expm1(-y))


# ----------------------------------------------------------------------------------------
# Low-level kernels (module level, so that torch.utils.checkpoint can wrap them)
# ----------------------------------------------------------------------------------------

def _project_chunk(xy, sigma, rot, rho, theta, u):
    """Projection of a chunk of Gaussians.

    xy (n,2) | sigma (n,2) | rot (n,) | rho (n,) | theta (B,) radians | u (D,) detector coords
    returns (B, D): summed line integrals of the chunk for each angle / detector bin.
    """
    # centre of the projected 1D Gaussian: m = x*cos(theta) - y*sin(theta)          (n, B)
    m = xy[:, 0:1] * torch.cos(theta)[None, :] - xy[:, 1:2] * torch.sin(theta)[None, :]
    # n^T Sigma n  with  n . e1 = cos(phi+theta),  n . e2 = -sin(phi+theta)         (n, B)
    ang = rot[:, None] + theta[None, :]
    c2 = torch.cos(ang) ** 2
    s2 = 1.0 - c2
    var = sigma[:, 0:1] ** 2 * c2 + sigma[:, 1:2] ** 2 * s2
    # integral of the Gaussian along the ray -> amplitude of the projected 1D Gaussian
    amp = rho[:, None] * SQRT_2PI * (sigma[:, 0:1] * sigma[:, 1:2]) / torch.sqrt(var)
    d = u[None, None, :] - m[:, :, None]                                   # (n, B, D)
    w = amp[:, :, None] * torch.exp(-0.5 * d * d / var[:, :, None])        # (n, B, D)
    return w.sum(dim=0)                                                    # (B, D)


def _eval_density(xs, ys, xy, sigma, rot, rho):
    """Sum of Gaussian densities on a pixel block.

    xs (W,), ys (H,): pixel coordinates (offsets from the image centre)
    returns (H, W) with rows = y, columns = x.
    """
    dx = xs[None, None, :] - xy[:, 0, None, None]          # (n, 1, W)
    dy = ys[None, :, None] - xy[:, 1, None, None]          # (n, H, 1)
    c = torch.cos(rot)[:, None, None]
    s = torch.sin(rot)[:, None, None]
    u1 = dx * c + dy * s                                   # coordinate along e1
    u2 = -dx * s + dy * c                                  # coordinate along e2
    q = (u1 / sigma[:, 0, None, None]) ** 2 + (u2 / sigma[:, 1, None, None]) ** 2
    return (rho[:, None, None] * torch.exp(-0.5 * q)).sum(dim=0)


# ----------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------

class GaussianCT(nn.Module):
    """A set of N anisotropic 2D Gaussians that represents one CT slice.

    Parameters (all in pixel units, origin at the image centre):
        xy         (N,2)  centres, x = column offset, y = row offset
        log_scale  (N,2)  log of the two standard deviations (sigma1, sigma2)
        rot        (N,)   rotation of the principal axes (radians)
        raw_rho    (N,)   density before softplus (keeps rho > 0)
    """

    def __init__(self, xy, sigma, rot, rho, img_size=888, n_det=888,
                 sigma_min=1.0, sigma_max=80.0, use_checkpoint=True):
        super().__init__()
        self.img_size = img_size
        self.n_det = n_det
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_checkpoint = use_checkpoint   # trade ~30% speed for much less GPU memory

        self.xy = nn.Parameter(xy.clone().float())
        self.log_scale = nn.Parameter(torch.log(sigma.clone().float()))
        self.rot = nn.Parameter(rot.clone().float())
        self.raw_rho = nn.Parameter(inv_softplus(rho.clone().float()))

        # detector pixel j sits at u = j - n_det//2 (same as skimage.iradon)
        self.register_buffer("u", torch.arange(n_det, dtype=torch.float32, device=xy.device) - n_det // 2)
        self.optimizer = None
        self._reset_stats()

    # ---- basic accessors -----------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.xy.shape[0]

    def sigma(self) -> torch.Tensor:
        """Standard deviations in pixels, clamped to [sigma_min, sigma_max]."""
        return torch.exp(self.log_scale).clamp(self.sigma_min, self.sigma_max)

    def rho(self) -> torch.Tensor:
        return F.softplus(self.raw_rho)

    # ---- initialisation ---------------------------------------------------------------------
    @classmethod
    def from_fbp(cls, fbp_norm, n_init, device, rng, sigma_min=1.0, sigma_max=80.0,
                 thr=0.05, c=0.5):
        """Sample Gaussians where the sparse-view FBP image has signal (R2-Gaussian style).

        fbp_norm : 2D float array, FBP image divided by its 99.9th percentile (so ~[0, 1])
        rng      : numpy Generator (seeded by the caller -> reproducible)
        thr      : pixels with fbp_norm <= thr count as background (no Gaussians there)
        c        : initial sigma = c * sqrt(support_area / n_init)
        """
        H = fbp_norm.shape[0]
        R = H // 2
        rows, cols = np.mgrid[:H, :H]
        xx, yy = cols - R, rows - R
        inside = xx ** 2 + yy ** 2 <= R ** 2
        f = np.where(inside, np.clip(fbp_norm, 0.0, None), 0.0).astype(np.float64)
        support = f > thr
        if support.sum() < 10:                         # degenerate image -> uniform fallback
            support = inside
            f = inside.astype(np.float64)
        p = (f * support).ravel()
        p /= p.sum()

        # replace=True: duplicates are harmless because of the sub-pixel jitter below
        idx = rng.choice(H * H, size=n_init, replace=True, p=p)
        r, c_ = np.unravel_index(idx, (H, H))
        x = c_ - R + rng.uniform(-0.5, 0.5, n_init)
        y = r - R + rng.uniform(-0.5, 0.5, n_init)

        area = float(support.sum())
        s0 = max(1.2 * sigma_min, c * math.sqrt(area / n_init))
        # neighbouring Gaussians overlap, so divide by the expected overlap 2*pi*c^2
        rho0 = np.maximum(f.ravel()[idx] / (2 * math.pi * c ** 2), 1e-3)

        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=device)
        return cls(
            xy=t(np.stack([x, y], axis=1)),
            sigma=t(np.full((n_init, 2), s0)),
            rot=t(rng.uniform(0.0, math.pi, n_init)),
            rho=t(rho0),
            img_size=H, n_det=H, sigma_min=sigma_min, sigma_max=sigma_max,
        )

    # ---- forward model: sinogram ---------------------------------------------------------------
    def project(self, theta, chunk=4096, theta_sign=None):
        """Differentiable sinogram rendering.

        theta : (B,) projection angles in RADIANS (the same angles as the FBP code uses)
        returns (B, n_det)
        """
        sign = THETA_SIGN if theta_sign is None else theta_sign
        theta = theta * sign
        sigma, rho = self.sigma(), self.rho()
        out = None
        for i in range(0, self.n, chunk):
            args = (self.xy[i:i + chunk], sigma[i:i + chunk], self.rot[i:i + chunk],
                    rho[i:i + chunk], theta, self.u)
            if torch.is_grad_enabled() and self.use_checkpoint:
                part = checkpoint(_project_chunk, *args, use_reentrant=False)
            else:
                part = _project_chunk(*args)
            out = part if out is None else out + part
        return out

    # TODO(speed): this projector evaluates every Gaussian at every detector bin
    #   (N x B x 888 exp() calls per step). A Gaussian only matters within +-4*s of its centre,
    #   so a windowed version (gather ~2*4*s bins, scatter with index_add_) is 10-30x cheaper.
    #   Only do this if the timing from `--smoke` is too slow for the full run.

    # ---- image rendering ---------------------------------------------------------------------------
    def _render_block(self, x0, y0, h, w, n_sigma, max_per_chunk=2048):
        """Density on an (h, w) pixel block whose first pixel is at offset (x0, y0).
        Only Gaussians that reach into the block (within n_sigma) are evaluated."""
        dev = self.xy.device
        xs = x0 + torch.arange(w, dtype=torch.float32, device=dev)
        ys = y0 + torch.arange(h, dtype=torch.float32, device=dev)
        sigma = self.sigma()
        reach = (n_sigma * sigma.max(dim=1).values).detach()
        cx, cy = x0 + (w - 1) / 2.0, y0 + (h - 1) / 2.0
        xyd = self.xy.detach()
        near = ((xyd[:, 0] - cx).abs() <= w / 2.0 + reach) & ((xyd[:, 1] - cy).abs() <= h / 2.0 + reach)
        ids = near.nonzero(as_tuple=True)[0]
        out = torch.zeros(h, w, device=dev)
        if ids.numel() == 0:
            return out
        rho = self.rho()
        for i in range(0, ids.numel(), max_per_chunk):
            j = ids[i:i + max_per_chunk]
            out = out + _eval_density(xs, ys, self.xy[j], sigma[j], self.rot[j], rho[j])
        return out

    @torch.no_grad()
    def render_image(self, tile=64, n_sigma=4.0):
        """Full (img_size x img_size) reconstruction, zero outside the reconstruction circle
        (like skimage's iradon(circle=True), so it is comparable to the ground truth)."""
        H = self.img_size
        R = H // 2
        img = torch.zeros(H, H, device=self.xy.device)
        for row0 in range(0, H, tile):
            for col0 in range(0, H, tile):
                h, w = min(tile, H - row0), min(tile, H - col0)
                img[row0:row0 + h, col0:col0 + w] = self._render_block(col0 - R, row0 - R, h, w, n_sigma)
        ax = torch.arange(H, dtype=torch.float32, device=img.device) - R
        img[(ax[None, :] ** 2 + ax[:, None] ** 2) > R ** 2] = 0.0
        return img

    def tv_loss(self, patch=48, n_sigma=3.0):
        """Total variation of the rendered image on ONE random patch (cheap regulariser).

        The patch is centred on a randomly chosen Gaussian, so it always lies on the head.
        Sparse-view data leave streak artefacts; TV pushes the Gaussians towards piecewise
        smooth images (R2-Gaussian uses the same idea in 3D)."""
        j = int(torch.randint(0, self.n, (1,)).item())
        cx = int(round(self.xy[j, 0].item()))
        cy = int(round(self.xy[j, 1].item()))
        blk = self._render_block(cx - patch // 2, cy - patch // 2, patch, patch, n_sigma)
        return (blk[:, 1:] - blk[:, :-1]).abs().mean() + (blk[1:, :] - blk[:-1, :]).abs().mean()

    # ---- optimiser + adaptive density control ---------------------------------------------------------
    def make_optimizer(self, lr_xy, lr_scale, lr_rot, lr_rho):
        """One Adam param group per tensor (named, so densification can resize them)."""
        groups = [
            {"params": [self.xy], "lr": lr_xy, "name": "xy"},
            {"params": [self.log_scale], "lr": lr_scale, "name": "log_scale"},
            {"params": [self.rot], "lr": lr_rot, "name": "rot"},
            {"params": [self.raw_rho], "lr": lr_rho, "name": "raw_rho"},
        ]
        self.optimizer = torch.optim.Adam(groups, eps=1e-8)
        return self.optimizer

    def _reset_stats(self):
        dev = self.xy.device
        self.grad_accum = torch.zeros(self.n, device=dev)   # sum of |d loss / d xy|
        self.denom = torch.zeros(self.n, device=dev)        # number of accumulated steps

    @torch.no_grad()
    def accumulate_grad_stats(self):
        """Call after loss.backward(), before optimizer.step()."""
        g = self.xy.grad
        if g is None:
            return
        self.grad_accum += g.norm(dim=1)
        self.denom += 1.0

    @torch.no_grad()
    def _rebuild(self, keep, new_rows):
        """Keep the rows where `keep` is True and append `new_rows[name]` (dict of tensors).
        Adam's moment estimates are kept for surviving rows and set to 0 for new rows
        (same trick as the original 3DGS code), so training continues smoothly."""
        assert self.optimizer is not None, "call make_optimizer() first"
        n_new = 0
        for group in self.optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            rows = new_rows.get(name)
            data = old.detach()[keep]
            if rows is not None:
                data = torch.cat([data, rows], dim=0)
                n_new = rows.shape[0]
            new = nn.Parameter(data.contiguous())
            state = self.optimizer.state.pop(old, None)
            if state is not None:
                for k in ("exp_avg", "exp_avg_sq"):
                    s = state[k][keep]
                    if rows is not None:
                        s = torch.cat([s, torch.zeros_like(rows)], dim=0)
                    state[k] = s
                self.optimizer.state[new] = state
            group["params"][0] = new
            setattr(self, name, new)
        ga, dn = self.grad_accum[keep], self.denom[keep]
        if n_new:
            ga = torch.cat([ga, torch.zeros(n_new, device=ga.device)])
            dn = torch.cat([dn, torch.zeros(n_new, device=dn.device)])
        self.grad_accum, self.denom = ga, dn

    @torch.no_grad()
    def prune(self, rho_min, margin=8.0, min_keep=100):
        """Remove nearly transparent Gaussians and Gaussians far outside the image circle."""
        R = self.img_size // 2
        keep = (self.rho() >= rho_min) & (self.xy.norm(dim=1) <= R + margin)
        n_keep = int(keep.sum())
        if n_keep < min_keep or n_keep == self.n:
            return 0
        removed = self.n - n_keep
        self._rebuild(keep, {})
        return removed

    @torch.no_grad()
    def densify_and_prune(self, frac, split_sigma, n_max, rho_min, scale_div=1.6):
        """Adaptive density control (3DGS idea, simplified).

        * The `frac` fraction of Gaussians with the largest average position gradient is
          "under-reconstructed". A relative (top-k) rule is used instead of an absolute
          threshold because the gradient scale depends on our loss normalisation.
        * small ones (max sigma <= split_sigma) are CLONED (density is halved, copy is jittered),
        * big ones are SPLIT into two children with sigma / scale_div, sampled inside the parent.
        * afterwards nearly transparent Gaussians are pruned.
        """
        dev = self.xy.device
        stats = {"clone": 0, "split": 0, "prune": 0}
        avg = self.grad_accum / self.denom.clamp_min(1.0)
        k = min(int(frac * self.n), max(0, n_max - self.n))     # respect the Gaussian budget
        if k > 0:
            thr = torch.topk(avg, k).values[-1]
            sel = (avg >= thr) & (avg > 0)
            sigma = self.sigma()
            big = sigma.max(dim=1).values > split_sigma
            split_mask, clone_mask = sel & big, sel & ~big
            rows = {name: [] for name in PARAM_NAMES}

            ci = clone_mask.nonzero(as_tuple=True)[0]
            if ci.numel() > 0:
                half = inv_softplus(self.rho()[ci] * 0.5)
                self.raw_rho.data[ci] = half                     # parent keeps half the density
                jitter = torch.randn(ci.numel(), 2, device=dev) * sigma[ci].min(dim=1, keepdim=True).values * 0.5
                rows["xy"].append(self.xy[ci] + jitter)
                rows["log_scale"].append(self.log_scale[ci].clone())
                rows["rot"].append(self.rot[ci].clone())
                rows["raw_rho"].append(half.clone())
                stats["clone"] = int(ci.numel())

            pi = split_mask.nonzero(as_tuple=True)[0]
            if pi.numel() > 0:
                sig_p, rot_p = sigma[pi], self.rot[pi]
                cs, sn = torch.cos(rot_p), torch.sin(rot_p)
                for _ in range(2):                               # two children per parent
                    local = torch.randn(pi.numel(), 2, device=dev) * sig_p      # in the principal-axis frame
                    dx = local[:, 0] * cs - local[:, 1] * sn
                    dy = local[:, 0] * sn + local[:, 1] * cs
                    rows["xy"].append(self.xy[pi] + torch.stack([dx, dy], dim=1))
                    rows["log_scale"].append(torch.log(sig_p / scale_div))
                    rows["rot"].append(rot_p.clone())
                    rows["raw_rho"].append(self.raw_rho[pi].clone())
                stats["split"] = int(pi.numel())

            if stats["clone"] or stats["split"]:
                new_rows = {name: torch.cat(v, dim=0) for name, v in rows.items()}
                self._rebuild(~split_mask, new_rows)             # parents of splits are removed

        stats["prune"] = self.prune(rho_min)
        self._reset_stats()
        return stats

    def save(self, path):
        """Store the Gaussian parameters (for later analysis / figures)."""
        torch.save({k: getattr(self, k).detach().cpu() for k in PARAM_NAMES}, path)


# ----------------------------------------------------------------------------------------
# Self-test: is our projector consistent with skimage's radon (and hence with iradon)?
# ----------------------------------------------------------------------------------------

def _self_test():
    """Run `python gaussian_ct.py` BEFORE any long job (takes a few seconds, CPU is fine).

    A few random Gaussians are rendered to an image, skimage's radon() projects that image
    numerically, and our analytic projector must reproduce the result. If the angle sign or
    the image axes were wrong, the reconstruction would be mirrored/rotated w.r.t. the
    ground truth even though the sinogram loss looks fine - this test catches that."""
    from skimage.transform import radon

    H = 256
    rng = np.random.default_rng(0)
    n = 12
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    model = GaussianCT(
        xy=to_t(rng.uniform(-45, 45, (n, 2))), sigma=to_t(rng.uniform(3, 10, (n, 2))),
        rot=to_t(rng.uniform(0, math.pi, n)), rho=to_t(rng.uniform(0.5, 1.5, n)),
        img_size=H, n_det=H, sigma_min=1.0, use_checkpoint=False)

    img = model.render_image().numpy()
    theta_deg = np.array([0.0, 20.0, 45.0, 90.0, 137.0, 170.0])
    theta = torch.deg2rad(to_t(theta_deg))

    errors = {}
    for img_name, im in (("image", img), ("image.T", img.T.copy())):
        ref = radon(im, theta=theta_deg, circle=True)                  # (H, n_angles)
        for sign in (+1.0, -1.0):
            with torch.no_grad():
                pred = model.project(theta, theta_sign=sign).numpy().T    # (H, n_angles)
            errors[(img_name, sign)] = float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

    for (img_name, sign), e in errors.items():
        print(f"  relative error vs skimage.radon   {img_name:8s}  theta_sign={sign:+.0f}:  {e:.4f}")

    best = min(errors, key=errors.get)
    if best == ("image", 1.0) and errors[best] < 0.08:
        print("geometry self-test PASSED  (keep THETA_SIGN = +1.0)")
    elif best == ("image", -1.0) and errors[best] < 0.08:
        print("geometry self-test: the sign is flipped -> set THETA_SIGN = -1.0 at the top of this file, re-run.")
    else:
        print("geometry self-test FAILED: best match is", best, "with error", round(errors[best], 4),
              "\n  -> the renderer and projector disagree about the image axes (x = column, y = row).")


if __name__ == "__main__":
    _self_test()
