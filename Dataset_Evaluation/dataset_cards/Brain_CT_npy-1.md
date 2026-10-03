# Dataset card: Brain CT npy-1

*Task:* sparse_view_ct  
*Checked:* 2026-10-03  

**Verdict:** ❌ NOT SUITABLE AS IS — 1 blocker(s). Fix them or reject the dataset.

## Provenance

| Field | Value |
|---|---|
| source_url | https://www.kaggle.com/datasets/josepc/brain-ct-npy-1 |
| publisher | Jose Pérez Cano |
| paper_to_cite | — |
| version_and_date | 2021 |
| licence | Community Data License Agreement - Sharing - Version 1.0 |
| access | https://www.kaggle.com/datasets/josepc/brain-ct |
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
| files | 9997 |
| size_on_disk_GB | 62.9 |
| samples (images/volumes) | 9997 |
| 2D slices total | 5118464 |
| in-plane sizes | (512, 3) |
| spacing known | 0/9997 |
| intensity scale | normalised (0–1) |

## Findings

### Origin & licence

- 📝 **TODO** 'paper_to_cite' unknown — Look for a data-descriptor paper (Scientific Data, Medical Physics, challenge paper).
- 📝 **TODO** 'ethics_approval' unknown — For human data the paper should state ethics approval / consent.
- 📝 **TODO** 'de_identification' unknown — How were names, dates, faces (head CT/MR!) removed?
- 📝 **TODO** 'acquisition' unknown — Scanner vendors, sites, years, protocol. One scanner = poor generalisation.
- 📝 **TODO** 'population' unknown — Who is included? Bias in age/sex/disease affects what your model learns.
- 📝 **TODO** 'raw_data_available' unknown — For reconstruction: are raw projections/sinograms given, or only images?
- 📝 **TODO** 'official_split' unknown — An official split makes your results comparable with published work.
- 📝 **TODO** 'known_issues' unknown — Search '<dataset name> issues / errata / github issues'.
- ⚠️ **WARN** Custom / restricted terms: read the agreement (sharing, publication, storage rules).

### Files & size

- ℹ️ **INFO** 9997 files, 62.90 GB on disk.
- 📝 **TODO** No README/LICENSE in the download: get licence & description from the source website.

### Readability

- ✅ **PASS** All 9997 image files readable.

### Size & resolution

- ℹ️ **INFO** 9997 samples (images or volumes) = 5118464 2D slices in total.
- ✅ **PASS** All images have the same in-plane size (512, 3).
- ⚠️ **WARN** Smallest image side is 3 px (< 256). Fine details may be lost.
- ℹ️ **INFO** 9997 of 9997 images are not square.
- ℹ️ **INFO** Volumes: 512–512 slices (median 512).
- ⚠️ **WARN** Pixel size in mm is unknown (PNG/JPG/NPY carry none). No physical scanner geometry, no mm-based metrics (e.g. HD95 in mm). Check the paper for it.

### Pixel values

- ℹ️ **INFO** Checked 30 random samples. Stored data type(s): float64.
- ✅ **PASS** Consistent intensity scale: normalised (0–1) (range 0 … 1).

### Patients & splits

- ⚠️ **WARN** No patient ID found (no DICOM tag, no PATIENT_ID_REGEX). You cannot prove a patient-level split → leakage risk. Find the ID in file names or a table.
- ℹ️ **INFO** No train/val/test folders found: you'll make your own split (per patient, fixed seed).

### Labels & class balance

- ⚠️ **WARN** 9997 of 9997 samples have no label.
- ❌ **FAIL** No labels found.

### Balance & diversity

- 📝 **TODO** No acquisition/demographic metadata in the files (typical for PNG/NPY). Get scanner, site, sex, age distribution from the paper or a metadata table.

### Duplicates

- ⚠️ **WARN** 91 exact duplicate files (e.g. ['ID_00e2957ab.npy', 'ID_021b4a7ce.npy']).
- ✅ **PASS** No near-duplicates among 300 sampled images.

### Privacy

- ℹ️ **INFO** No DICOM headers to check. Still look at the images for burned-in text, and at head scans for recognisable faces.

### Task fit

- 📝 **TODO** Modality not stored in the files — confirm it is CT.
- ⚠️ **WARN** Intensities are normalised (0–1): find the conversion to HU/μ in the docs, otherwise simulated projections are only relative.
- ⚠️ **WARN** Pixel size unknown → geometry in pixel units only.
- ℹ️ **INFO** Only reconstructed images → you'll simulate sparse projections yourself (standard practice). Avoid the 'inverse crime': simulate on a finer grid / different projector than you reconstruct with, and add noise.
- ℹ️ **INFO** Labels: none needed — the full-view image is the target. Note that the 'ground truth' is itself a reconstruction (noise, kernel, artefacts).
