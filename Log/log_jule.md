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

angefangener zweiter trainingsdruchlauf:
Found 21 series
Train patients: ['series_1013', 'series_1018', 'series_1022', 'series_1024', 'series_1027', 'series_1015', 'series_1021', 'series_1026', 'series_1019', 'series_1010', 'series_1020', 'series_1011', 'series_1025', 'series_1016', 'series_1017', 'series_1009', 'series_1012', 'series_1029']
Val patients:   ['series_1023']
Test patients:  ['series_1028', 'series_1014']
Images -> train: 712, val: 32, test: 72
Epoch 001 | train L1 0.03520 | val L1 0.03242  <- saved
Epoch 002 | train L1 0.01450 | val L1 0.01041  <- saved
Epoch 003 | train L1 0.01170 | val L1 0.01175
Epoch 004 | train L1 0.01014 | val L1 0.01646
Epoch 005 | train L1 0.00949 | val L1 0.00836  <- saved
Epoch 006 | train L1 0.00809 | val L1 0.01073
Epoch 007 | train L1 0.00684 | val L1 0.00695  <- saved
Epoch 008 | train L1 0.00654 | val L1 0.00661  <- saved
Epoch 009 | train L1 0.00597 | val L1 0.00720
Epoch 010 | train L1 0.00497 | val L1 0.00669
Epoch 011 | train L1 0.00505 | val L1 0.01962
Epoch 012 | train L1 0.00471 | val L1 0.00493  <- saved
Epoch 013 | train L1 0.00432 | val L1 0.00492  <- saved
Epoch 014 | train L1 0.00445 | val L1 0.00459  <- saved
Epoch 015 | train L1 0.00422 | val L1 0.00705
Epoch 016 | train L1 0.00422 | val L1 0.00430  <- saved
Epoch 017 | train L1 0.00389 | val L1 0.00594
Epoch 018 | train L1 0.00375 | val L1 0.00510
Epoch 019 | train L1 0.00380 | val L1 0.00629
Epoch 020 | train L1 0.00345 | val L1 0.00297  <- saved
Epoch 021 | train L1 0.00350 | val L1 0.00303
Epoch 022 | train L1 0.00331 | val L1 0.00302
Epoch 023 | train L1 0.00338 | val L1 0.00280  <- saved
Epoch 024 | train L1 0.00326 | val L1 0.00329


### 16.10
major issues with completing model training
training takes multiple hours
notebook for collab created
running in collab results in issues when trying to mount drive
no solution found yet

first complete trainrun on 20 patients:

Found 21 series
Train patients: ['series_1013', 'series_1018', 'series_1022', 'series_1024', 'series_1027', 'series_1015', 'series_1021', 'series_1026', 'series_1019', 'series_1010', 'series_1020', 'series_1011', 'series_1025', 'series_1016', 'series_1017', 'series_1009', 'series_1012', 'series_1029']
Val patients:   ['series_1023']
Test patients:  ['series_1028', 'series_1014']
Images -> train: 712, val: 32, test: 72
Epoch 001 | train L1 0.03520 | val L1 0.03242  <- saved
Epoch 002 | train L1 0.01450 | val L1 0.01041  <- saved
Epoch 003 | train L1 0.01170 | val L1 0.01175
Epoch 004 | train L1 0.01014 | val L1 0.01646
Epoch 005 | train L1 0.00949 | val L1 0.00836  <- saved
Epoch 006 | train L1 0.00809 | val L1 0.01073
Epoch 007 | train L1 0.00684 | val L1 0.00695  <- saved
Epoch 008 | train L1 0.00654 | val L1 0.00661  <- saved
Epoch 009 | train L1 0.00597 | val L1 0.00720
Epoch 010 | train L1 0.00497 | val L1 0.00669
Epoch 011 | train L1 0.00505 | val L1 0.01962
Epoch 012 | train L1 0.00471 | val L1 0.00493  <- saved
Epoch 013 | train L1 0.00432 | val L1 0.00492  <- saved
Epoch 014 | train L1 0.00445 | val L1 0.00459  <- saved
Epoch 015 | train L1 0.00422 | val L1 0.00705
Epoch 016 | train L1 0.00422 | val L1 0.00430  <- saved
Epoch 017 | train L1 0.00389 | val L1 0.00594
Epoch 018 | train L1 0.00375 | val L1 0.00510
Epoch 019 | train L1 0.00380 | val L1 0.00629
Epoch 020 | train L1 0.00345 | val L1 0.00297  <- saved
Epoch 021 | train L1 0.00350 | val L1 0.00303
Epoch 022 | train L1 0.00331 | val L1 0.00302
Epoch 023 | train L1 0.00338 | val L1 0.00280  <- saved
Epoch 024 | train L1 0.00326 | val L1 0.00329
Epoch 025 | train L1 0.00334 | val L1 0.00316
Epoch 026 | train L1 0.00324 | val L1 0.00322
Epoch 027 | train L1 0.00362 | val L1 0.00311
Epoch 028 | train L1 0.00333 | val L1 0.00235  <- saved
Epoch 029 | train L1 0.00300 | val L1 0.00374
Epoch 030 | train L1 0.00314 | val L1 0.00281
Epoch 031 | train L1 0.00337 | val L1 0.00246
Epoch 032 | train L1 0.00293 | val L1 0.00190  <- saved
Epoch 033 | train L1 0.00294 | val L1 0.00214
Epoch 034 | train L1 0.00284 | val L1 0.00344
Epoch 035 | train L1 0.00311 | val L1 0.00302
Epoch 036 | train L1 0.00305 | val L1 0.00264
Epoch 037 | train L1 0.00267 | val L1 0.00326
Epoch 038 | train L1 0.00272 | val L1 0.00414
Epoch 039 | train L1 0.00275 | val L1 0.00237
Epoch 040 | train L1 0.00257 | val L1 0.00170  <- saved
Epoch 041 | train L1 0.00253 | val L1 0.00227
Epoch 042 | train L1 0.00268 | val L1 0.00280
Epoch 043 | train L1 0.00239 | val L1 0.00163  <- saved
Epoch 044 | train L1 0.00266 | val L1 0.00228
Epoch 045 | train L1 0.00230 | val L1 0.00204
Epoch 046 | train L1 0.00230 | val L1 0.00174
Epoch 047 | train L1 0.00243 | val L1 0.00219
Epoch 048 | train L1 0.00260 | val L1 0.00182
Epoch 049 | train L1 0.00231 | val L1 0.00231
Epoch 050 | train L1 0.00252 | val L1 0.00203

=== Test set (held-out patients) ===
 input:  PSNR 54.33 dB   SSIM 0.9977
  unet:  PSNR 52.61 dB   SSIM 0.9981

  => 
  - Unet pipeline works
  - no indicator of whether unet works for reconstruction or not

  possible issue: images are not sparsed enough

  ### 18.10
  comparison skript for images yiels following results:

  Series found: 21
  series_1009/slice_22.png: mode=RGBA, size=(888, 888)
  series_1009/slice_22.png: mode=RGBA, size=(888, 888)
series_1009: mean diff = 0.72, max diff = 8.0 (gray levels of 255)
  series_1010/slice_23.png: mode=RGBA, size=(888, 888)
  series_1010/slice_23.png: mode=RGBA, size=(888, 888)
series_1010: mean diff = 0.58, max diff = 7.0 (gray levels of 255)
  series_1011/slice_16.png: mode=RGBA, size=(888, 888)
  series_1011/slice_16.png: mode=RGBA, size=(888, 888)
series_1011: mean diff = 0.62, max diff = 8.0 (gray levels of 255)
  series_1012/slice_22.png: mode=RGBA, size=(888, 888)
  series_1012/slice_22.png: mode=RGBA, size=(888, 888)
series_1012: mean diff = 0.58, max diff = 8.0 (gray levels of 255)
  series_1013/slice_20.png: mode=RGBA, size=(888, 888)
  series_1013/slice_20.png: mode=RGBA, size=(888, 888)
series_1013: mean diff = 0.69, max diff = 9.0 (gray levels of 255)
  series_1014/slice_16.png: mode=RGBA, size=(888, 888)
  series_1014/slice_16.png: mode=RGBA, size=(888, 888)
series_1014: mean diff = 0.47, max diff = 6.0 (gray levels of 255)
  series_1015/slice_20.png: mode=RGBA, size=(888, 888)
  series_1015/slice_20.png: mode=RGBA, size=(888, 888)
series_1015: mean diff = 0.56, max diff = 6.0 (gray levels of 255)
  series_1016/slice_16.png: mode=RGBA, size=(888, 888)
  series_1016/slice_16.png: mode=RGBA, size=(888, 888)
series_1016: mean diff = 0.74, max diff = 8.0 (gray levels of 255)
  series_1017/slice_20.png: mode=RGBA, size=(888, 888)
  series_1017/slice_20.png: mode=RGBA, size=(888, 888)
series_1017: mean diff = 0.88, max diff = 9.0 (gray levels of 255)
  series_1018/slice_18.png: mode=RGBA, size=(888, 888)
  series_1018/slice_18.png: mode=RGBA, size=(888, 888)
series_1018: mean diff = 0.92, max diff = 7.0 (gray levels of 255)
  series_1019/slice_19.png: mode=RGBA, size=(888, 888)
  series_1019/slice_19.png: mode=RGBA, size=(888, 888)
series_1019: mean diff = 0.56, max diff = 8.0 (gray levels of 255)
  series_1020/slice_20.png: mode=RGBA, size=(888, 888)
  series_1020/slice_20.png: mode=RGBA, size=(888, 888)
series_1020: mean diff = 0.86, max diff = 8.0 (gray levels of 255)
  series_1021/slice_20.png: mode=RGBA, size=(888, 888)
  series_1021/slice_20.png: mode=RGBA, size=(888, 888)
series_1021: mean diff = 0.60, max diff = 7.0 (gray levels of 255)
  series_1022/slice_20.png: mode=RGBA, size=(888, 888)
  series_1022/slice_20.png: mode=RGBA, size=(888, 888)
series_1022: mean diff = 0.57, max diff = 7.0 (gray levels of 255)
  series_1023/slice_16.png: mode=RGBA, size=(888, 888)
  series_1023/slice_16.png: mode=RGBA, size=(888, 888)
series_1023: mean diff = 0.45, max diff = 7.0 (gray levels of 255)
  series_1024/slice_20.png: mode=RGBA, size=(888, 888)
  series_1024/slice_20.png: mode=RGBA, size=(888, 888)
series_1024: mean diff = 0.81, max diff = 8.0 (gray levels of 255)
  series_1025/slice_20.png: mode=RGBA, size=(888, 888)
  series_1025/slice_20.png: mode=RGBA, size=(888, 888)
series_1025: mean diff = 0.80, max diff = 7.0 (gray levels of 255)
  series_1026/slice_22.png: mode=RGBA, size=(888, 888)
  series_1026/slice_22.png: mode=RGBA, size=(888, 888)
series_1026: mean diff = 0.66, max diff = 8.0 (gray levels of 255)
  series_1027/slice_20.png: mode=RGBA, size=(888, 888)
  series_1027/slice_20.png: mode=RGBA, size=(888, 888)
series_1027: mean diff = 0.50, max diff = 8.0 (gray levels of 255)
  series_1028/slice_20.png: mode=RGBA, size=(888, 888)
  series_1028/slice_20.png: mode=RGBA, size=(888, 888)
series_1028: mean diff = 0.86, max diff = 7.0 (gray levels of 255)
  series_1029/slice_18.png: mode=RGBA, size=(888, 888)
  series_1029/slice_18.png: mode=RGBA, size=(888, 888)
series_1029: mean diff = 1.15, max diff = 11.0 (gray levels of 255)

plus check_sparse_vs_full.png

suggest that images are too similar
especially in the middle where skull is is almost no difference

final train run + image reconstruction for evaluation and comparison:

Found 30 series
Train patients (20): ['series_1009', 'series_1010', 'series_1011', 'series_1012', 'series_1013', 'series_1014', 'series_1015', 'series_1016', 'series_1017', 'series_1018', 'series_1019', 'series_1020', 'series_1021', 'series_1022', 'series_1023', 'series_1024', 'series_1025', 'series_1026', 'series_1027', 'series_1028']
Val patients   (0): []
Recon patients (10): ['series_1029', 'series_1030', 'series_1031', 'series_1032', 'series_1033', 'series_1034', 'series_1035', 'series_1036', 'series_1037', 'series_1038']
Images -> train: 780, val: 0
Epoch 001 | train L1 0.03603
Epoch 002 | train L1 0.01687
Epoch 003 | train L1 0.01294
Epoch 004 | train L1 0.01006
Epoch 005 | train L1 0.00825
Epoch 006 | train L1 0.00699
Epoch 007 | train L1 0.00616
Epoch 008 | train L1 0.00541
Epoch 009 | train L1 0.00521
Epoch 010 | train L1 0.00509
Epoch 011 | train L1 0.00470
Epoch 012 | train L1 0.00457
Epoch 013 | train L1 0.00410
Epoch 014 | train L1 0.00417
Epoch 015 | train L1 0.00404
Epoch 016 | train L1 0.00372
Epoch 017 | train L1 0.00321
Epoch 018 | train L1 0.00367
Epoch 019 | train L1 0.00375
Epoch 020 | train L1 0.00390
Epoch 021 | train L1 0.00346
Epoch 022 | train L1 0.00342
Epoch 023 | train L1 0.00319
Epoch 024 | train L1 0.00328
Epoch 025 | train L1 0.00320
Epoch 026 | train L1 0.00306
Epoch 027 | train L1 0.00311
Epoch 028 | train L1 0.00286
Epoch 029 | train L1 0.00278
Epoch 030 | train L1 0.00295
Epoch 031 | train L1 0.00289
Epoch 032 | train L1 0.00267
Epoch 033 | train L1 0.00267
Epoch 034 | train L1 0.00270
Epoch 035 | train L1 0.00261
Epoch 036 | train L1 0.00287
Epoch 037 | train L1 0.00260
Epoch 038 | train L1 0.00259
Epoch 039 | train L1 0.00268
Epoch 040 | train L1 0.00295
Epoch 041 | train L1 0.00240
Epoch 042 | train L1 0.00239
Epoch 043 | train L1 0.00245
Epoch 044 | train L1 0.00265
Epoch 045 | train L1 0.00242
Epoch 046 | train L1 0.00238
Epoch 047 | train L1 0.00217
Epoch 048 | train L1 0.00239
Epoch 049 | train L1 0.00236
Epoch 050 | train L1 0.00224