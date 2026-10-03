# Dataset card: Demo dataset

*Task:* sparse_view_ct  
*Checked:* 2026-10-03  

**Verdict:** ❌ NOT SUITABLE AS IS — 1 blocker(s). Fix them or reject the dataset.

## Provenance

| Field | Value |
|---|---|
| source_url | — |
| publisher | — |
| paper_to_cite | — |
| version_and_date | — |
| licence | CC BY-NC-ND 4.0 |
| access | open |
| ethics_approval | — |
| de_identification | — |
| acquisition | — |
| population | — |
| label_source | not needed (reconstruction) |
| raw_data_available | — |
| official_split | — |
| known_issues | — |

## Key numbers

| Fact | Value |
|---|---|
| files | 50 |
| size_on_disk_GB | 0.0 |
| samples (images/volumes) | 49 |
| 2D slices total | 49 |
| in-plane sizes | (256, 256), (256, 200) |
| spacing known | 0/49 |
| intensity scale | 8-bit display values (0–255) |
| patients | 21 |
| official split | {'train': 37, 'test': 12} |
| classes | 2 |
| imbalance ratio (max/min) | 5.1 |

## Findings

### Origin & licence

- 📝 **TODO** 'source_url' unknown — Where can it be downloaded from? Keep the link for the report.
- 📝 **TODO** 'publisher' unknown — Who released it? A known hospital/challenge is more trustworthy than an anonymous Kaggle upload.
- 📝 **TODO** 'paper_to_cite' unknown — Look for a data-descriptor paper (Scientific Data, Medical Physics, challenge paper).
- 📝 **TODO** 'version_and_date' unknown — Datasets get updated; note the version you used so results are reproducible.
- 📝 **TODO** 'ethics_approval' unknown — For human data the paper should state ethics approval / consent.
- 📝 **TODO** 'de_identification' unknown — How were names, dates, faces (head CT/MR!) removed?
- 📝 **TODO** 'acquisition' unknown — Scanner vendors, sites, years, protocol. One scanner = poor generalisation.
- 📝 **TODO** 'population' unknown — Who is included? Bias in age/sex/disease affects what your model learns.
- 📝 **TODO** 'raw_data_available' unknown — For reconstruction: are raw projections/sinograms given, or only images?
- 📝 **TODO** 'official_split' unknown — An official split makes your results comparable with published work.
- 📝 **TODO** 'known_issues' unknown — Search '<dataset name> issues / errata / github issues'.
- ℹ️ **INFO** Non-commercial (NC): fine for uni projects/theses, NOT for company or start-up work.
- ⚠️ **WARN** No-derivatives (ND): you may use it, but don't publish modified copies (e.g. your sparsified/simulated data). Share your code instead.
- ✅ **PASS** Open access.

### Files & size

- ℹ️ **INFO** 50 files, 0.00 GB on disk.
- ✅ **PASS** Documentation found: README.txt — read it!

### Readability

- ✅ **PASS** All 49 image files readable.

### Size & resolution

- ℹ️ **INFO** 49 samples (images or volumes) = 49 2D slices in total.
- ⚠️ **WARN** 2 different in-plane sizes (most common: (256, 256), (256, 200)) — you'll need resizing/cropping/padding.
- ⚠️ **WARN** Smallest image side is 200 px (< 256). Fine details may be lost.
- ℹ️ **INFO** 1 of 49 images are not square.
- ⚠️ **WARN** Pixel size in mm is unknown (PNG/JPG/NPY carry none). No physical scanner geometry, no mm-based metrics (e.g. HD95 in mm). Check the paper for it.

### Pixel values

- ℹ️ **INFO** Checked 30 random samples. Stored data type(s): float32, uint8.
- ⚠️ **WARN** Mixed intensity scales across samples: {'8-bit display values (0–255)': 26, 'Hounsfield units (HU)': 4} — normalise per source.
- ⚠️ **WARN** 8-bit images: intensities were windowed/compressed for display, original HU / raw values are lost.

### Patients & splits

- ℹ️ **INFO** 21 patients; samples per patient: min 2, median 2, max 4.
- ✅ **PASS** 21 patients: enough for a patient-level train/val/test split.
- ✅ **PASS** Split given by the folders: {'train': 37, 'test': 12}.
- ❌ **FAIL** LEAKAGE: 1 patient(s) appear in more than one split (e.g. P001).

### Labels & class balance

- ⚠️ **WARN** Imbalanced: 2 classes, largest/smallest = 5.1× ('normal' 41 vs 'lesion' 8). Use class weights/balanced sampling and per-class metrics (not plain accuracy).
- ⚠️ **WARN** 1 class(es) with < 30 samples: {'lesion': 8}.

### Balance & diversity

- ✅ **PASS** split: 2 groups, largest 76%.
- 📝 **TODO** No acquisition/demographic metadata in the files (typical for PNG/NPY). Get scanner, site, sex, age distribution from the paper or a metadata table.

### Duplicates

- ⚠️ **WARN** 1 exact duplicate files (e.g. ['train/normal/P002_slice00.png', 'train/normal/P002_slice00_copy.png']).
- ⚠️ **WARN** 1 near-identical sample(s) among 49 checked (e.g. ['train/normal/P002_slice00.png', 'train/normal/P002_slice00_copy.png']). Look at them.

### Privacy

- ℹ️ **INFO** No DICOM headers to check. Still look at the images for burned-in text, and at head scans for recognisable faces.

### Task fit

- 📝 **TODO** Modality not stored in the files — confirm it is CT.
- ⚠️ **WARN** 8-bit display images: projections simulated from these are not physically realistic. OK as proof of concept — must be stated as limitation.
- ⚠️ **WARN** Pixel size unknown → geometry in pixel units only.
- ℹ️ **INFO** Only reconstructed images → you'll simulate sparse projections yourself (standard practice). Avoid the 'inverse crime': simulate on a finer grid / different projector than you reconstruct with, and add noise.
- ℹ️ **INFO** Labels: none needed — the full-view image is the target. Note that the 'ground truth' is itself a reconstruction (noise, kernel, artefacts).
