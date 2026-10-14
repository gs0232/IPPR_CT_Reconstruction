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