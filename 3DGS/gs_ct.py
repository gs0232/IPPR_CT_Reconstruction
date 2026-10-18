"""
gs_ct.py - 2D Gaussian-splatting model for sparse-view parallel-beam CT.

One model = one CT slice (per-scan optimisation, no training set). The slice is
represented by N anisotropic 2D Gaussians. Each Gaussian i has
    xy_i        centre (x, y) in PIXEL units, origin = image centre (pixel 444)
    s_i = (s1, s2)   std-devs along its two principal axes (pixels)
    rot_i       rotation angle phi of the principal axes (radians)
    rho_i       peak density (attenuation per pixel, same units as the FBP image)

Parameterisation (like 3DGS): log-scales, log-density, free rotation angle.

----------------------------------------------------------------------------
Why this is "3DGS for CT" (X-ray / R2-Gaussian style) and not RGB splatting
----------------------------------------------------------------------------
There is no colour, no alpha compositing and no depth sorting. A CT measurement
is a LINE INTEGRAL of attenuation, so Gaussians simply ADD UP. For parallel-beam
CT the line integral of a 2D Gaussian along a ray has a closed form (it is again
a 1D Gaussian on the detector), so the differentiable "rasteriser" is exact:

    detector coordinate  t = x*cos(th) - y*sin(th)            (see geometry below)
    sigma_t^2 = s1^2 cos^2(th+phi) + s2^2 sin^2(th+phi)       (= u^T Sigma u)
    p_i(t)    = rho * 2*pi*s1*s2 * N(t ; m_i, sigma_t^2)       with m_i = x_i cos th - y_i sin th
    sinogram(t, th) = sum_i p_i(t)

The detector-bin width is approximated by adding 1/12 pixel^2 to sigma_t^2
(box filter ~ Gaussian), which also keeps very thin Gaussians numerically safe.

----------------------------------------------------------------------------
Geometry - MUST match the FBP baseline (Baselines/02_fbp_baseline.ipynb)
----------------------------------------------------------------------------
Taken from skimage.transform.iradon(..., circle=True) (version 0.26 source):
  * image pixel (row i, col j)  <->  y = i - 444, x = j - 444
  * detector bin k              <->  t = k - 444
  * t = x*cos(th) - y*sin(th)   with th in radians
  * reconstruction is zero outside the circle x^2 + y^2 > 444^2
  * image units: sinogram value = sum of image values along the ray with 1 px
    spacing (so Gaussians fitted to the raw sinogram live in the same units as
    the FBP ground truth -> we never have to rescale the result).
`train_3dgs.py --sanity` checks all of this numerically against skimage.radon.
If it reports that ANGLE_SIGN = -1 fits better, set cfg.angle_sign = -1.

TODO(improve): the dense (views x gaussians x detector-bins) evaluation below is
the simple, readable baseline. A tile-based CUDA/Triton kernel that only touches
bins within +-4*sigma_t of each Gaussian would be 10-100x faster/leaner.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

TWO_PI = 2.0 * math.pi
PIX_VAR = 1.0 / 12.0  # variance of a 1-pixel-wide box -> detector bin integration


# ----------------------------------------------------------------------------
# Differentiable parallel-beam projector (the "splatting" step)
# ----------------------------------------------------------------------------
def _project_chunk(xy, log_scale, rot, log_rho, theta, det, s_min, s_max):
    """Project a set of Gaussians for several views.

    xy (n,2), log_scale (n,2), rot (n,), log_rho (n,)
    theta (V,) radians, det (D,) detector coordinates in pixels
    returns (V, D) projections.
    """
    s = torch.exp(log_scale).clamp(s_min, s_max)           # (n,2)
    rho = torch.exp(log_rho)                                # (n,)
    s1, s2 = s[:, 0], s[:, 1]

    cos_t = torch.cos(theta)[:, None]                       # (V,1)
    sin_t = torch.sin(theta)[:, None]
    m = xy[:, 0][None, :] * cos_t - xy[:, 1][None, :] * sin_t   # (V,n) centre on detector

    phi = theta[:, None] + rot[None, :]                     # (V,n)
    var = (s1 ** 2)[None, :] * torch.cos(phi) ** 2 \
        + (s2 ** 2)[None, :] * torch.sin(phi) ** 2 + PIX_VAR    # (V,n)
    amp = (rho * TWO_PI * s1 * s2)[None, :] / torch.sqrt(TWO_PI * var)  # (V,n)

    diff = det[None, None, :] - m[:, :, None]               # (V,n,D)
    prof = torch.exp(-0.5 * diff * diff / var[:, :, None])  # (V,n,D)
    return torch.bmm(amp.unsqueeze(1), prof).squeeze(1)     # (V,D) = sum_n amp * prof


class GaussianCT(nn.Module):
    def __init__(self, xy, log_scale, rot, log_rho, size=888, s_min=0.5, s_max=60.0):
        super().__init__()
        self.xy = nn.Parameter(xy.contiguous().float())               # (N,2)
        self.log_scale = nn.Parameter(log_scale.contiguous().float()) # (N,2)
        self.rot = nn.Parameter(rot.contiguous().float())             # (N,)
        self.log_rho = nn.Parameter(log_rho.contiguous().float())     # (N,)
        self.size = size
        self.c = size // 2
        self.s_min = float(s_min)
        self.s_max = float(s_max)

    # -- convenience --------------------------------------------------------
    @property
    def n(self):
        return self.xy.shape[0]

    def scales(self):
        return torch.exp(self.log_scale).clamp(self.s_min, self.s_max)

    def rho(self):
        return torch.exp(self.log_rho)

    # -- forward model: Gaussians -> sinogram columns -------------------------
    def project(self, theta, det, chunk=16384, use_checkpoint=True):
        """theta (V,) radians (angle_sign already applied), det (D,) -> (V,D).

        chunk: Gaussians per chunk. With use_checkpoint=True every chunk is
        recomputed in the backward pass, so peak memory is bounded by
        V * chunk * D floats instead of V * N * D.
        TODO(tune): lower `chunk` (or views_per_step) if you hit CUDA OOM.
        """
        args = (self.xy, self.log_scale, self.rot, self.log_rho)
        if chunk is None or chunk >= self.n:
            return _project_chunk(*args, theta, det, self.s_min, self.s_max)
        out = 0.0
        for i in range(0, self.n, chunk):
            sl = slice(i, i + chunk)
            a = tuple(p[sl] for p in args)
            if use_checkpoint and torch.is_grad_enabled():
                out = out + checkpoint(_project_chunk, *a, theta, det,
                                       self.s_min, self.s_max, use_reentrant=False)
            else:
                out = out + _project_chunk(*a, theta, det, self.s_min, self.s_max)
        return out

    # -- image-domain rendering (evaluation + TV regulariser) -----------------
    def render_region(self, ys, xs, chunk=64):
        """Evaluate the density on the grid ys (H,) x xs (W,) (pixel offsets from
        the centre, float tensors). Differentiable. Returns (H, W)."""
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        s, rho = self.scales(), self.rho()
        cp, sp = torch.cos(self.rot), torch.sin(self.rot)
        out = torch.zeros_like(gx)
        for i in range(0, self.n, chunk):
            sl = slice(i, i + chunk)
            dx = gx[None, :] - self.xy[sl, 0:1]                      # (c,P)
            dy = gy[None, :] - self.xy[sl, 1:2]
            a = (dx * cp[sl, None] + dy * sp[sl, None]) / s[sl, 0:1]   # along e1
            b = (-dx * sp[sl, None] + dy * cp[sl, None]) / s[sl, 1:2]  # along e2
            out = out + (rho[sl, None] * torch.exp(-0.5 * (a * a + b * b))).sum(0)
        inside = (gx ** 2 + gy ** 2) <= self.c ** 2   # same circle as iradon(circle=True)
        return (out * inside).view(len(ys), len(xs))

    @torch.no_grad()
    def render_image(self, chunk=64):
        """Full size x size reconstruction in the model's (normalised) units."""
        dev = self.xy.device
        co = torch.arange(self.size, device=dev, dtype=torch.float32) - self.c
        return self.render_region(co, co, chunk=chunk)

    # -- adaptive density control (clone / split / prune) ---------------------
    def _set(self, xy, log_scale, rot, log_rho):
        self.xy = nn.Parameter(xy.contiguous())
        self.log_scale = nn.Parameter(log_scale.contiguous())
        self.rot = nn.Parameter(rot.contiguous())
        self.log_rho = nn.Parameter(log_rho.contiguous())

    @torch.no_grad()
    def densify_and_prune(self, grad_accum, denom, top_frac, split_px, rho_prune, max_n):
        """Standard 3DGS-style control, simplified for 2D CT.

        * candidates = the `top_frac` fraction of Gaussians with the largest mean
          positional-gradient norm (RELATIVE threshold, so it does not depend on
          the loss scale). TODO(tune): switch to an absolute threshold if wanted.
        * candidate with max-scale <= split_px -> CLONE (density halved on both copies)
        * candidate with max-scale  > split_px -> SPLIT into 2 children sampled
          from the parent, scale/1.6, density x1.28 (total mass is preserved)
        * afterwards: prune Gaussians with density < rho_prune.
        Net growth per call is at most `top_frac*N`, capped by `max_n`.
        """
        n, dev = self.n, self.xy.device
        avg = grad_accum / denom.clamp(min=1.0)
        s = self.scales()
        budget = int(min(top_frac * n, max(0, max_n - n)))
        cand = torch.zeros(n, dtype=torch.bool, device=dev)
        if budget > 0:
            cand[torch.topk(avg, budget).indices] = True
        big = s.max(dim=1).values > split_px
        clone_m, split_m = cand & ~big, cand & big

        xy, ls = self.xy.data, self.log_scale.data
        rot, lr_ = self.rot.data, self.log_rho.data.clone()
        lr_[clone_m] -= math.log(2.0)          # clone: halve density of both copies

        def offset(m, factor):                 # sample a point from each Gaussian
            z = torch.randn(int(m.sum()), 2, device=dev) * s[m] * factor
            c, sn = torch.cos(rot[m]), torch.sin(rot[m])
            return torch.stack([z[:, 0] * c - z[:, 1] * sn, z[:, 0] * sn + z[:, 1] * c], 1)

        parts_xy = [xy[~split_m], xy[clone_m] + offset(clone_m, 0.5)]
        parts_ls = [ls[~split_m], ls[clone_m]]
        parts_rot = [rot[~split_m], rot[clone_m]]
        parts_lr = [lr_[~split_m], lr_[clone_m]]
        for _ in range(2):                     # two children per split Gaussian
            parts_xy.append(xy[split_m] + offset(split_m, 1.0))
            parts_ls.append(ls[split_m] - math.log(1.6))
            parts_rot.append(rot[split_m])
            parts_lr.append(lr_[split_m] + math.log(1.6 ** 2 / 2.0))

        new_xy, new_ls = torch.cat(parts_xy), torch.cat(parts_ls)
        new_rot, new_lr = torch.cat(parts_rot), torch.cat(parts_lr)
        keep = torch.exp(new_lr) > rho_prune
        if keep.sum() < 100:                   # safety net
            keep[:] = True
        self._set(new_xy[keep], new_ls[keep], new_rot[keep], new_lr[keep])
        return dict(cloned=int(clone_m.sum()), split=int(split_m.sum()),
                    pruned=int((~keep).sum()), n=self.n)


# ----------------------------------------------------------------------------
# Initialisation
# ----------------------------------------------------------------------------
def density_reference(img, size=888, pct=99.5):
    """A robust 'tissue density' scale of an image (used to make the TV term and
    the pruning threshold independent of the sinogram scale)."""
    c = size // 2
    yy, xx = np.mgrid[:size, :size] - c
    inside = (xx ** 2 + yy ** 2) <= c ** 2
    v = img[inside & (img > 0)]
    return float(np.percentile(v, pct)) if v.size else 1.0


def init_from_image(img, n, size=888, uniform_frac=0.1, gamma=0.5, seed=0, s_min=0.5, s_max=60.0):
    """Sample Gaussian centres from a (sparse-view FBP) image - R2-Gaussian style.

    img : 2D float array in the units the model will be trained in.
    Points are drawn with probability ~ max(img,0)^gamma inside the head
    (gamma<1 flattens the bone/soft-tissue contrast so soft tissue gets points too).
    `uniform_frac` of the points are scattered uniformly in the circle so that
    streak artefacts in air can still be 'explained' (and pruned if useless).
    TODO(tune): n, gamma, uniform_frac, 5% threshold below.
    """
    rng = np.random.default_rng(seed)
    S, c = size, size // 2
    yy, xx = np.mgrid[:S, :S] - c
    inside = (xx ** 2 + yy ** 2) <= c ** 2
    pos = np.clip(img, 0, None) * inside
    pos = np.where(pos > 0.05 * pos.max(), pos, 0.0)        # drop air / streak noise
    area = max(int((pos > 0).sum()), 1)

    n_f = int(round(n * (1 - uniform_frac)))
    n_u = n - n_f
    w = pos.ravel() ** gamma
    w /= w.sum()
    flat = rng.choice(S * S, size=n_f, replace=True, p=w)
    py, px = np.unravel_index(flat, (S, S))
    x_f = px - c + rng.uniform(-0.5, 0.5, n_f)
    y_f = py - c + rng.uniform(-0.5, 0.5, n_f)
    r = 0.95 * c * np.sqrt(rng.uniform(size=n_u))
    a = TWO_PI * rng.uniform(size=n_u)
    x = np.concatenate([x_f, r * np.cos(a)])
    y = np.concatenate([y_f, r * np.sin(a)])

    ix = np.clip(np.round(x + c).astype(int), 0, S - 1)
    iy = np.clip(np.round(y + c).astype(int), 0, S - 1)
    val = np.clip(img[iy, ix], 0.01 * img.max(), None)      # density floor = 1% of peak
    a2 = area / n                                            # image area per Gaussian
    s0 = float(np.clip(0.4 * np.sqrt(a2), 1.0, 6.0))         # initial isotropic scale
    rho0 = val * a2 / (TWO_PI * s0 ** 2)                     # ~ val when s0 = 0.4*sqrt(a2)

    t = lambda v: torch.tensor(v, dtype=torch.float32)
    return GaussianCT(
        xy=t(np.stack([x, y], 1)),
        log_scale=torch.full((n, 2), math.log(s0)),
        rot=t(rng.uniform(0, math.pi, n)),
        log_rho=t(np.log(rho0)),
        size=size, s_min=s_min, s_max=s_max,
    )


def init_random(n, size=888, seed=0, s0=3.0, rho0=1e-3, s_min=0.5, s_max=60.0):
    """Ablation: uniform random initialisation inside the reconstruction circle.
    Density is calibrated afterwards by train_3dgs.calibrate_density()."""
    rng = np.random.default_rng(seed)
    c = size // 2
    r = 0.95 * c * np.sqrt(rng.uniform(size=n))
    a = TWO_PI * rng.uniform(size=n)
    xy = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    return GaussianCT(
        xy=t(xy),
        log_scale=torch.full((n, 2), math.log(s0)),
        rot=t(rng.uniform(0, math.pi, n)),
        log_rho=torch.full((n,), math.log(rho0)),
        size=size, s_min=s_min, s_max=s_max,
    )
