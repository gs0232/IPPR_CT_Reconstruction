# Implementation Log Sophia

## Task
Baseline and metric evaluation

## What I did/learned
### 7.10
- Set up Github

### 13.10
- Overview of what's to do in order to compute the baselines:
  1. Check data for size, values, ... (1_data_check.ipynb)
  2. Compute metrics
  3. Normalization and baseline numbers
  4. Plots
- Problem occurred: generated sparse images do not match the provided original images --> Waiting for answer: I suggested to Souram that we should change the output of the FBP code so that we get the sparse-view AND full-view images from the sinograms instead of only the sparse-view.

### 15.10
- Checking if the images are now comparable --> Worked!
- Checking the size, values, ... of the data
  - Images are barely different (Baslines/figures/comparison_full_sparse_histogram.png) --> sparse images even more
  - Imgages are 888x888 and 8-bit RGBA PNGs (unit8)
  - Limitation found: PNGs are saved as 8-bit images.
- Create baseline and metric python file using Claude
  - Otsu method is used to select masks of head
  - Computes PSNR, SSIM and RMSE (whole image and head only)
  - Makes a summary table (mean ± SD), the plot of PSNR/SSIM vs. number of views, and an example figure
- Problem occurred: In parallel-beam CT, the projections at $\theta$ and $\theta + 180°$ contain the same information. Over 360°, 246 views therefore contain only 123 different views. --> The views should be spread evenly over 180°

The baseline and metric file will be run in my next session. First, only 2 patients will be tested (MAX_SERIES=2) and if everything works out, MAX_SERIES=None to run the code on all patients.

### 17.10
- Running the baseline file on two images/sinograms
  - Creating the images for 25%, 10%, 5%, and 2.5% of sparse
  - Using the Otsu method to extract the head masks (Baselines/figures/sparse_images_head_mask.png)
