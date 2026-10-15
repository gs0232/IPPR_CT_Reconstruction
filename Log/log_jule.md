# Implementation Log Jule

## Task 
Unet impelmentation and training

## What I did/learned
### 14.10
setting up doubleConv:
One conv alone has a small receptive field; stacking two approximates a larger effective one cheaply

residual learning instead of predicting clean image directly:
model learns to predict the difference between steaky and clean image, wich is then subtracted. converges faster and is more stabel

Original images and sparsed images do not overlap pixelwise
- unet is supposed to get rid of stripes but now also has to account for the shift in image
- so reconstruction could be correct but is shifted and therefore punished
- comparison with nerf and 3dgs becomes less accurate

difference in shade does not affect performance that much as long as it stays constant across images, can be solved by normalising

l1loss is supposed to preserve sharper edges than l2/mse loss but has to be compared to a combined loss later

### 15.10
first complete trainrun on old dataset completed:

Train patients: ['patient_4', 'patient_8', 'patient_2', 'patient_5', 'patient_0', 'patient_1']
Val patients:   ['patient_7']
Test patients:  ['patient_3', 'patient_6']
[warning] patient_2: 32 originals, 26 sparse, 26 matched
[warning] patient_0: 32 originals, 26 sparse, 26 matched
Images -> train: 228, val: 36, test: 80
Epoch 001 | train L1 0.17542 | val L1 0.15234  <- saved
Epoch 002 | train L1 0.08598 | val L1 0.07597  <- saved
Epoch 003 | train L1 0.07238 | val L1 0.07081  <- saved
Epoch 004 | train L1 0.06503 | val L1 0.06476  <- saved
Epoch 005 | train L1 0.06199 | val L1 0.06507
Epoch 006 | train L1 0.05766 | val L1 0.07726
Epoch 007 | train L1 0.05754 | val L1 0.05978  <- saved
Epoch 008 | train L1 0.05308 | val L1 0.05631  <- saved
Epoch 009 | train L1 0.05203 | val L1 0.05657
Epoch 010 | train L1 0.05103 | val L1 0.05481  <- saved
Epoch 011 | train L1 0.05017 | val L1 0.05055  <- saved
Epoch 012 | train L1 0.04828 | val L1 0.05231
Epoch 013 | train L1 0.04897 | val L1 0.05109
Epoch 014 | train L1 0.04824 | val L1 0.05258
Epoch 015 | train L1 0.04556 | val L1 0.05165
Epoch 016 | train L1 0.04706 | val L1 0.05063
Epoch 017 | train L1 0.04588 | val L1 0.05083
Epoch 018 | train L1 0.04521 | val L1 0.04782  <- saved
Epoch 019 | train L1 0.04480 | val L1 0.04789
Epoch 020 | train L1 0.04374 | val L1 0.04988
Epoch 021 | train L1 0.04326 | val L1 0.04604  <- saved
Epoch 022 | train L1 0.04151 | val L1 0.04709
Epoch 023 | train L1 0.04176 | val L1 0.04679
Epoch 024 | train L1 0.04291 | val L1 0.04789
Epoch 025 | train L1 0.04145 | val L1 0.04944
Epoch 026 | train L1 0.04090 | val L1 0.04746
Epoch 027 | train L1 0.04137 | val L1 0.04723
Epoch 028 | train L1 0.04077 | val L1 0.04819
Epoch 029 | train L1 0.04065 | val L1 0.04668
Epoch 030 | train L1 0.03990 | val L1 0.04488  <- saved
Epoch 031 | train L1 0.03919 | val L1 0.04687
Epoch 032 | train L1 0.03921 | val L1 0.04451  <- saved
Epoch 033 | train L1 0.03823 | val L1 0.04261  <- saved
Epoch 034 | train L1 0.03950 | val L1 0.04867
Epoch 035 | train L1 0.03895 | val L1 0.04468
Epoch 036 | train L1 0.03833 | val L1 0.04351
Epoch 037 | train L1 0.03713 | val L1 0.04778
Epoch 038 | train L1 0.03730 | val L1 0.04798
Epoch 039 | train L1 0.03824 | val L1 0.04334
Epoch 040 | train L1 0.03771 | val L1 0.04558
Epoch 041 | train L1 0.03610 | val L1 0.04426
Epoch 042 | train L1 0.03722 | val L1 0.04456
Epoch 043 | train L1 0.03549 | val L1 0.04453
Epoch 044 | train L1 0.03573 | val L1 0.04556
Epoch 045 | train L1 0.03597 | val L1 0.04610
Epoch 046 | train L1 0.03598 | val L1 0.04573
Epoch 047 | train L1 0.03528 | val L1 0.04565
Epoch 048 | train L1 0.03511 | val L1 0.04364
Epoch 049 | train L1 0.03554 | val L1 0.04809
Epoch 050 | train L1 0.03496 | val L1 0.04581

=== Test set (held-out patients) ===
 input:  PSNR 8.78 dB   SSIM 0.2397
  unet:  PSNR 23.63 dB   SSIM 0.7712

shows mentioned issues in comparability with baseline

none-sparse- and sparse images from the same reconstruction algorithm were included
20 patients with ca. 30 images per patient should be enough training material
