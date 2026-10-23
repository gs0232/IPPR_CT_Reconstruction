# Implementation Log Sophia

## Task
Baseline and metric evaluation

## What I did/learned
### 7.10 - Github
- Set up Github

### 13.10 - Overview
- Overview of what's to do in order to compute the baselines:
  1. Check data for size, values, ... (1_data_check.ipynb)
  2. Compute metrics
  3. Normalization and baseline numbers
  4. Plots
- Problem occurred: generated sparse images do not match the provided original images --> Waiting for answer: I suggested to Souram that we should change the output of the FBP code so that we get the sparse-view AND full-view images from the sinograms instead of only the sparse-view.

### 15.10 - Image Preprocessing + Baseline files
- Checking if the images are now comparable --> Worked!
- Checking the size, values, ... of the data
  - Images are barely different (Baselines/figures/comparison_full_sparse_histogram.png) --> the sparse images need stronger sparsity to be a meaningful baseline
  - Images are 888x888 and 8-bit RGBA PNGs (uint8)
  - Limitation found: PNGs are saved as 8-bit images.
- Create baseline and metric python file using Claude
  - Otsu method is used to select masks of head
  - Computes PSNR, SSIM and RMSE (whole image and head only)
  - Makes a summary table (mean ± SD), the plot of PSNR/SSIM vs. number of views, and an example figure
- Problem occurred: In parallel-beam CT, the projections at $\theta$ and $\theta + 180°$ contain the same information. Over 360°, 246 views therefore contain only 123 different views. --> The views should be spread evenly over 180°

The baseline and metric file will be run in my next session. First, only 2 patients will be tested (MAX_SERIES=2) and if everything works out, MAX_SERIES=None to run the code on all patients.

### 17.10 - Baseline
- Running the baseline file on two images/sinograms
  - Creating the images for 25%, 10%, 5%, and 2.5% of sparse
  - Only every 4th slice was used (SLICE_STEP = 4) to save time.
  - Souram’s “50%” (every 2nd angle over 360°) contains the same information as our 246 views (25%)
  - Sanity check: our full-view FBP matches Souram’s full-view PNG (SSIM $= 0.999$), so ANGLE_RANGE = 360 is correct.
  - Using the Otsu method to extract the head masks (Baselines/figures/sparse_images_head_mask.png)
- Current limitation: imgaes had to be sparsed more; the SinoCT sinograms were computed from the finished images, so our artefacts come only from missing angles (no real scanner noise). Results are therefore slightly optimistic.; the U-Net, NeRF and 3DGS must use my ground truth and the same angles, otherwise their results aren’t comparable to the baseline (`baseline_results/gt_fullview_npy/series_XXXX_sliceYY.npy`).
 
### 18.10 - Baseline
- Run main loop: FBP baseline over all series
  - The full-view ground truth is cached as `.npy` on Drive (`baseline_results/gt_fullview_npy/series_XXXX_sliceYY.npy`), so it is only computed once
  (U-Net / NeRF / 3DGS can use the same files as their ground truth).
  - Results are appended to `fbp_baseline.csv`
- Extract SSIM, PSNR, RMSE for head and whole image (mean +- std)
  - Results in `fbp_baseline_summary.csv`
  - Plot in `psnr_ssim_baseline.png`

**IMPORTANT NOTE TO JULE, SAMANTHA AND GIANNE LOIS:**
- The ground truths are in the Drive folder here: [https://drive.google.com/drive/folders/1VzUAMYD5p1QxiqmAUdYOtaDazxYKPoT9?usp=sharing](https://drive.google.com/drive/folders/1sKGq4ms9Y9b9G2BW_m6IgPDBlgAqYTtq?usp=sharing)
- I uploaded the full view .npy files in `baseline_results/gt_fullview_npy`
- Souram’s “50%” (every 2nd angle over 360°) contains the same information as my 246 views (25%)
- Make your sparse input from the sinogram with exactly these angles, so all methods
  get the same input as the FBP baseline:
```python
  idx   = np.round(np.linspace(0, 492, n_views, endpoint=False)).astype(int)  # n_views = 246, 98, 49 or 25
  theta = np.linspace(0, 360, 984, endpoint=False)[idx]   # angles in degrees
  sparse_sino = sino[:, idx]                              # sino shape: (888 detectors, 984 angles)
```
- Save your reconstructions as float `.npy`, 888×888, named `series_XXXX_sliceYY_vNNN.npy`
  (e.g. `series_1009_slice08_v098.npy`). Do NOT rescale them to 0–1 yourself.
- Put your reconstructions in the Drive folder `outputs/<method>/` (e.g. `outputs/unet/`) by 20.10.
  I (Sophia) run the final evaluation for all methods with the same code.
  Optional while developing: check your own results with `from metrics import evaluate` → `evaluate(pred, gt)`.

### 21.10 - Data availability
- Agreed on: the model-makers will give me their output as a reconstructed PNGs, I will compare them to my gt
  - Output naming: series_XXXX_sliceYY_vNNN.png (example: series_1009_slice08_v098.png)
- Agreed on: reducing number of patients for training and validation
  - 21 training patients (1009-1029)
  - 10 validation patients (1030-1039)
 
### 23.10 - Evaluation
- Evaluation of NeRF and 3DGS by running the 03_evaluation.ipynb notebook
- Figures are in the Google Drive under evaluation_results (summary_metrics_nerf_3dgs.png, evaluation_bars.png, comparison_example.png)
  https://drive.google.com/drive/folders/1oAEy2eY6jiNFTqkNCpEWnyM5ddjrrEGJ?usp=drive_link 

# Results:
- PSNR is practically the same for all three methods (about $22.3$–$22.5$ dB). The differences are far smaller than the SD.
- SSIM: FBP $0.795$ > NeRF $0.679$ > 3DGS $0.367$.
- Equal PSNR combined with lower SSIM usually means the methods get the average grey values right but lose fine structure. NeRF blurs; 3DGS shows patchy or noisy texture. FBP has streaks but keeps edges and detail.

For the report: at 98 views (10%), NeRF and 3DGS remove the streak artefacts in the background, but they don’t reconstruct the head itself better than FBP. Our success criterion (“higher PSNR and SSIM than FBP”) is met for the whole image but not inside the head.

### FBP:
- Fine streaks in the whole background. That’s why its whole-image SSIM is lower.
- Inside the head the image looks closest to the ground truth: sharp skull, smooth brain.
- The error is concentrated in a thin ring along the skull edge, where the contrast is highest.

### NeRF:
- Very blurred. The skull is no longer a sharp ring but a soft edge, and the brain has a blotchy texture.
- The structures below the skull (neck, soft tissue) are missing almost entirely.
- The background is clean, which is why its whole-image SSIM is very high.
- Head-only, it’s clearly worse than FBP: typical underfitting / missing fine detail. The cause may be too few training iterations, too low a resolution or too weak positional encoding. Ask Samantha.

### 3DGS:
- The skull is reproduced well, but the brain is full of fine grainy texture. You can see the individual Gaussians.
- The background has scattered dots, so-called “floaters”.
- The ground-truth brain is very smooth, so this grain lowers the head SSIM a lot ($0.367$). It varies from slice to slice depending on how well the Gaussians converged, which explains the high SD.

