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
