# Implementation Log Gianne

## Task
3D Gaussian Splatting for sparse-view CT (3DGS – Gianne)

Question: Can a set of optimised Gaussians (here: their 2D slice version) turn a sparse-view CT slice into a good full reconstruction?

Group protocol (as agreed)
- Input: the sparse-view reconstructions as PNG slices, `sparse_output/series_XXXX/slice_YY.png` (FBP from 98 of the 984 angles).
- Output: the reconstructed slice as PNG, named `series_XXXX_sliceYY_v098.png`, only for the test series and only for slices that have a ground truth (00, 04, 08, …). Sophia compares them with her ground truth.
- Models are tuned on the validation patients `series_1030` … `series_1039`. 3DGS has no training set: every slice is optimised on its own, so the validation patients are used only to tune hyper-parameters.



## What I did/learned
### Approach
1. Load the sparse PNG, remove its constant background grey level (min–max normalisation put “air” at ≈ 16/255, not 0).
2. Turn it into the projections it was made from: parallel-beam `radon` at the 98 known angles (= pseudo-sinogram, in PNG intensity units).
3. Represent the slice by $N$ anisotropic Gaussians (initialised from the sparse image) and optimise position, scale, rotation and density so that their analytic projections match the pseudo-sinogram (L1 + small TV term, adaptive clone / split / prune).
4. Render the 888×888 image, restore the background level and save it as an 8-bit PNG in the same intensity convention as the input PNG.
5. Evaluate with the shared `metrics.py` against the ground-truth PNG (used only for evaluation) – and compare with the sparse input itself, which is the baseline to beat.

### Setup
The shared Drive folder only shows up in Colab if it was added to My Drive as a shortcut. Need Editor access to `outputs/` to write results there.

### Data helpers
The PNGs are min–max normalised FBP images, so air is a constant grey, not 0. Subtract that level before projecting, and add it back before saving, so the output PNG has the same intensity convention as the input PNG (and, presumably, as the ground-truth PNG).
Because the sparse PNG is itself an FBP of the measured projections at the 98 known angles, `radon` of it gives (approximately) those projections back: that is the data the Gaussians are fitted to. Nothing from the ground truth is used for fitting.

### The Gaussian Model
Why this is "3DGS for CT" and not RGB splatting - there is no colour, no alpha compositing and no depth sorting. A CT measurement is a line integral of attenuation, so Gaussians simply add up. For parallel-beam CT the line integral of a 2D Gaussian along a ray has a closed form (again a 1D Gaussian on the detector), so the differentiable "rasteriser" is exact and plain PyTorch is enough.

Geometry is taken from `skimage.transform.iradon(circle=True)` / `radon(circle=True)`, i.e. the frame of the FBP baseline: pixel (row *i*, col *j*) ↔ $y=i-444,\ x=j-444$; detector bin $k$ ↔ $t=k-444$; zero outside the circle of radius 444. §5 checks the projector numerically against `skimage.radon`.

The dense (views × Gaussians × bins) evaluation is the simple, readable baseline. A tile-based CUDA/Triton kernel touching only bins within ±4σ would be 10–100× faster.

### Initialisation
R2-Gaussian style: sample the Gaussian centres from the **sparse input image** (background removed). `init="random"` is available as an ablation for the report.

### Training and main loop
- The pseudo-sinogram is divided by its maximum before optimisation (values ~[0, 1]); the rendered image is multiplied back, the background level is added again and the result is clipped to [0, 1].
- Loss = L1 on the projections of a random subset of `views_per_step` angles (+ optional L2) + `tv_weight` × TV of a random rendered patch.
- Densification (clone / split / prune) between 10 % and 60 % of the iterations; learning rates decay exponentially.
- `view_synthesis=True`: after fitting, projections at angles that were not among the 98 are rendered and compared with the projections of the ground-truth image (`proj_psnr / proj_ssim / proj_rmse`; a global gain is matched first because the sparse and the full PNG are normalised separately).
- `runtime_s` = PNG→sinogram + init + fitting + render; evaluation is not counted.
- Results are scored from the saved 8-bit PNG, exactly what Sophia will load. `input_psnr / input_ssim` = the sparse input PNG itself vs. the ground truth → the number 3DGS has to beat.
- A CSV row is appended after every fit and finished fits are skipped, so a Colab disconnect only costs the current fit – just run the cell again. Smoke and tuning runs (`tag`) write to `<out_dir>/_<tag>/` and never touch the official outputs.

### Geometry check 
1. The analytic projector vs. `skimage.radon` of the rendered image of 3 test Gaussians -> relative error should be well below 5 %, and it tells which `angle_sign` is right.
2. PNG input: prints the background level (expect ≈ 0.06), the fraction of clipped pixels (expect ≈ 0 %) and the pseudo-sinogram shape (888, 98). If many pixels are clipped, the PNG lost bone intensity and nothing downstream can recover it.
3. Result of check 1 (first run): `angle_sign +1` → 0.077 % error, `-1` → 108.5 % → keep `angle_sign = +1`.

### Smoke test
Writes to `<out_dir>/_smoke/` only. Check:
- s/it in the log → estimate the full runtime = *number of slices × iterations × s/it* (each series has about 10 slices on the evaluation grid).
- the preview PNG in `<out_dir>/_smoke/previews/`; 3DGS PSNR/SSIM should be at least as good as `input_psnr / input_ssim` (the sparse PNG itself).
- CUDA out of memory → lower `views_per_step` and/or `gauss_chunk`.


### Validation Tuning
Group decision: validation = `series_1030` … `series_1039`. Hyper-parameters are tuned **only** there, never on the test series.
Every run gets its own `tag`, so outputs go to `<out_dir>/_<tag>/`. Change one thing at a time, compare the summary tables, then write the winning values into the `Cfg` defaults (§1) and into `Log/log_gianne.md`.
Knobs by importance: `iters`, `n_init`, `lr_xy`, `lr_rho`, `tv_weight` {0, 0.005, 0.01, 0.05}, `densify_*`, `split_px`, `init` ("image" vs "random").

### Summary Notes
- Protocol: sparse-view PNG in → reconstructed PNG out (`series_XXXX_sliceYY_v098.png`, 8-bit greyscale, same intensity convention as the input PNG), test series only, slices 00/04/08/… that have a ground truth.
- Per-scan optimisation: one Gaussian set per slice, no training data. Validation patients (series_1030–1039) were used only to tune hyper-parameters; the ground truth is used only for evaluation.
- The PNGs are min–max-normalised FBP images: background grey level (≈ 16/255) is removed before projecting and restored before saving. The pseudo-sinogram is `radon` of the sparse PNG at the 98 known angles (= `angle_indices.npy`), so it only contains the information that is in the PNG (8-bit, FBP-filtered) – a limitation compared with fitting the raw measured sinograms.
- Initialised from the sparse input image; `init="random"` ablation possible.
- View synthesis is scored against projections of the ground-truth image (not raw scanner data), after matching a global gain.
- `runtime_s` includes PNG→sinogram + optimisation + render; compare with FBP (~seconds) and U-Net (single forward pass after training) in the report.
- Limitations to mention: 2D slice-wise model (no 3D consistency across slices); dense projector is slow; 8-bit input; hyper-parameters tuned on a few validation slices only.