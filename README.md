# IPPR_CT_Reconstruction

## Introduction
When undergoing a Computer Tomography (CT), patients are directly exposed to radiation.
Since DNA gene expression is said to be affected by such procedures (Schmid et al., 2025), the
aim of novel healthcare technologies is to minimize radiation exposure, especially sensitive
patient groups like children. Sparse-view CT scans are a promising method to reduce radiation
exposure by acquiring fewer projections per scan (Li et al., 2025). However, less projections
mean that the Signal-to-Noise Ratio (SNR) reduces as well, which leads to increased noise and
possible artefacts (Chung et al., 2022). In order to preserve diagnostic accuracy, CT imaging
slices are reconstructed using software-based technologies like Neural Radiance Fields (NeRFs)
and 3D Gaussian Splatting (3DGS). In this way radiation exposure can be minimized while
diagnostic accuracy remains stable.

The aim of this project is to showcase the techniques and effectiveness of sparse-view CT
reconstruction using image processing and pattern recognition methods. To do so, datasets of CT
images are evaluated, tested, and sparsed. A model is then trained to reconstruct the missing
projections and compare them to the ground truth based on loss functions. Metrics like the
Structural Similarity Index Measure (SSIM) then provide information about the reconstruction
accuracy on which the evaluation of the used techniques will be based. Furthermore,
requirements and specifications of the data and the model will be evaluated.

## Dataset
SinoCT - https://aimi.stanford.edu/datasets/sinoct
- Stanford University Hospital
- Over 9,000 head CT scans
- The full dataset is 1.3 TB
- Reconstructed images: 512x512 pixels
- Sinograms: 984x888 pixels

## Methology
Using SinoCT, the dataset will be split by patient, not by slice, into training (70%), validation
(15%) and test (15%) sets, using a fixed random seed. Neighbouring slices from the same patient
look very similar, so a slice-level split would let the model see near-copies of test images during
training and give overly optimistic results. The test set is only used once, for final results.
The team will utilise multi-domain pairing, as each sample has two domains:

- Sinogram domain: the projection data, made up of many angles.
- Image domain: the reconstructed CT slice.
- 
To simulate sparse-view scans, we will keep only every n-th angle from the full sinogram, giving
[e.g. 60, 30 and 15 views] out of [e.g. 180 or 360]. The full-view reconstruction is the ground
truth. Each training pair is therefore a sparse-view FBP image and a full-view reference image.
We will also check that all images have the same resolution, orientation and intensity range
(normalised to 0–1 with the same scaling for input and target), so that predictions and ground
truth line up pixel by pixel before any metric is calculated.

Every method is run on the same test patients and the same number of views. The algorithm is
trained on the training set, its hyperparameters (learning rate, epochs) are tuned on the validation
set, and final scores are computed on the test set.

### Neural Radiance Fields (NeRFs)
The performance of the NeRF based reconstruction will be evaluated at each sparsity level in
order to determine how accurately the NeRF model can recover information when less measured
projections are available. Additionally, the reconstructed CT output will be compared against the
full view reference to show if projection synthesis can result in improved reconstruction quality.

### 3D Gaussian Splatting (3DGS)
The 3DGS model will be evaluated using sparse view scans, similar to the NeRF model. The
available projections will be used to optimise the representations based on 3D Gaussian
primitives. The resultant synthesised views will then be compared with their corresponding
ground truth projections, based on the chosen evaluatino metrics, including PSNR and SSIM.

### Evaluation Metrics
In order to evaluate and compare model performance, the following metrics will be assessed for
each method:
1. Peak Signal to Noise Ratio (PSNR): PSNR measures the pixel level error to determine
how closely a processed image resembles the original. A higher ratio value indicates a
closer match to the original, or higher quality output.
2. Structural Similarity Index Measure (SSIM): SSIM measures the similarity of the
processed image by comparing local pixel patterns. To assess quality, this metric
measures luminance, contrast and structure. A value of 0 indicates no similarity, while a
value of 1 indicates perfect similarity.
3. Root Mean Squared Error (RMSE): RMSE measures the average difference between the
model's predicted values and actual values.
4. Runtime: Reconstruction time per slice and U-Net training time, to compare practicality.

### Results and Synthesis
For view synthesis, we compute the same metrics between predicted and true projections at held-
out angles. This involves reporting mean and standard deviation across test slices for each
method and each view count. We will then plot PSNR/SSIM against the number of views to
show how each method copes as data becomes sparser. From then on, we compare side-by-side
images and different maps to identify remaining artefacts such as streaks, blurring, or lost detail.
If abnormal cases are labelled, report results separately for normal and abnormal scans, and
visually check that abnormalities are not smoothed away.
Given our scope, we consider the project successful if:
- the algorithm achieves higher PSNR and SSIM than FBP at every tested view count, and
ideally also outperforms SART;
- the algorithm at a reduced view count (e.g. 30 views) gives quality close to FBP at a
higher view count (e.g. 60 views), which indicates possible dose reduction;
- reconstructions show no obvious loss of abnormal features;
- the pipeline is reproducible with fixed seeds and documented settings that can be trained
on Colab.
