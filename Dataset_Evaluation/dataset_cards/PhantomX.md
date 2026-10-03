# Dataset card: PhantomX

*Task:* sparse_view_ct  
*Checked:* 2026-10-03  

**Verdict:** ❌ NOT SUITABLE AS IS — 1 blocker(s). Fix them or reject the dataset.

## Provenance

| Field | Value |
|---|---|
| source_url | https://www.kaggle.com/datasets/phantomxdata/head-ct-collection-phantomx?resource=download |
| publisher | PhantomX GmbH |
| paper_to_cite | PhantomX Abdomen Lesion CT Collection |
| version_and_date | 2024 |
| licence | CC BY-SA 4.0 |
| access | https://phantomx.de/product-tag/data/ |
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
| files | 3640 |
| size_on_disk_GB | 1.92 |
| samples (images/volumes) | 9 |
| 2D slices total | 3639 |
| in-plane sizes | (512, 512) |
| spacing known | 9/9 |
| in-plane spacing mm | 0.468–0.507 |
| slice spacing mm | 0.50–0.50 |
| intensity scale | Hounsfield units (HU) |
| patients | 3 |
| classes | 9 |
| imbalance ratio (max/min) | 1.0 |

## Findings

### Origin & licence

- 📝 **TODO** 'ethics_approval' unknown — For human data the paper should state ethics approval / consent.
- 📝 **TODO** 'de_identification' unknown — How were names, dates, faces (head CT/MR!) removed?
- 📝 **TODO** 'acquisition' unknown — Scanner vendors, sites, years, protocol. One scanner = poor generalisation.
- 📝 **TODO** 'population' unknown — Who is included? Bias in age/sex/disease affects what your model learns.
- 📝 **TODO** 'raw_data_available' unknown — For reconstruction: are raw projections/sinograms given, or only images?
- 📝 **TODO** 'official_split' unknown — An official split makes your results comparable with published work.
- 📝 **TODO** 'known_issues' unknown — Search '<dataset name> issues / errata / github issues'.
- ℹ️ **INFO** Share-alike (SA): anything you publish from it must use the same licence.
- ✅ **PASS** CC BY: free use with citation.

### Files & size

- ℹ️ **INFO** 3640 files, 1.92 GB on disk.
- 📝 **TODO** No README/LICENSE in the download: get licence & description from the source website.
- ℹ️ **INFO** Metadata/label tables found: dataset_info.csv

### Readability

- ✅ **PASS** All 3639 image files readable.

### Size & resolution

- ℹ️ **INFO** 9 samples (images or volumes) = 3639 2D slices in total.
- ✅ **PASS** All images have the same in-plane size (512, 512).
- ℹ️ **INFO** Volumes: 325–565 slices (median 325).
- ℹ️ **INFO** In-plane pixel size 0.468–0.507 mm (known for 9/9).
- ℹ️ **INFO** Slice spacing 0.50–0.50 mm.

### Pixel values

- ℹ️ **INFO** Checked 9 random samples. Stored data type(s): int16.
- ✅ **PASS** Consistent intensity scale: Hounsfield units (HU) (range -2048 … 1903).
- ℹ️ **INFO** Values below −1024 HU present (outside-scan-field padding like −2048/−3024). Clip to −1024 before computing anything physical.

### Patients & splits

- ℹ️ **INFO** 3 patients; samples per patient: min 3, median 3, max 3.
- ❌ **FAIL** Only 3 patients: no meaningful train/val/test split possible.
- ℹ️ **INFO** No train/val/test folders found: you'll make your own split (per patient, fixed seed).

### Labels & class balance

- ✅ **PASS** Reasonably balanced: 9 classes, largest/smallest = 1.0× ('AIDR3D0.5mm_FC26_300_174818.022_202' 1 vs 'AIDR3D0.5mm_FC26_300_174818.022_202' 1).
- ⚠️ **WARN** 9 class(es) with < 30 samples: {'AIDR3D0.5mm_FC26_300_174818.022_202': 1, 'FBP0.5mm_FC26_300_174818.022_205': 1, 'AiCE0.5mm_BRAIN-CTA_300_174818.022_206': 1, 'AiCE_BRAIN_CTA_300_0.5_5': 1, 'AIDR3D_FC08_300_0.5_13': 1, 'FBP_FC08_300_0.5_15': 1, '94_120_300_FC26_AIDR3D_154123.594': 1, '95_120_300_FC26_FBP_154123.594': 1, '98_120_300_BRAIN-CTA_AiCE_154123.594': 1}.

### Balance & diversity

- ⚠️ **WARN** Manufacturer: all patients = 'Canon Medical Systems' — no diversity, results may not generalise beyond it.
- ⚠️ **WARN** ManufacturerModelName: all patients = 'Aquilion ONE' — no diversity, results may not generalise beyond it.
- ✅ **PASS** InstitutionName: 2 groups, largest 67%.
- ⚠️ **WARN** KVP: all patients = '120.0' — no diversity, results may not generalise beyond it.
- ✅ **PASS** ConvolutionKernel: 2 groups, largest 67%.
- ⚠️ **WARN** BodyPartExamined: all patients = 'NECK' — no diversity, results may not generalise beyond it.
- ⚠️ **WARN** PatientSex: all patients = 'M' — no diversity, results may not generalise beyond it.
- ✅ **PASS** age_group: 2 groups, largest 67%.
- ⚠️ **WARN** SliceThickness: all patients = '0.5' — no diversity, results may not generalise beyond it.

### Duplicates

- ✅ **PASS** No exact duplicate files.
- ⚠️ **WARN** 5 near-identical sample(s) among 9 checked (e.g. ['phantomx_head_dataset/50-03/300/95_120_300_FC26_FBP_154123.594 [series …6.891789]', 'phantomx_head_dataset/50-03/300/98_120_300_BRAIN-CTA_AiCE_154123.594 [series …3.335578]']). Look at them.

### Privacy

- ⚠️ **WARN** Possibly identifying DICOM tags filled in: {'PatientName': 9, 'PatientBirthDate': 9, 'PatientAddress': 6, 'ReferringPhysicianName': 9, 'InstitutionAddress': 9}. Check whether real or pseudonyms; keep data private either way.
- ℹ️ **INFO** Study dates present (often shifted for anonymisation — don't rely on them).

### Task fit

- ✅ **PASS** All series are CT.
- ✅ **PASS** Intensities are physical (Hounsfield units (HU)) → forward projection is meaningful.
- ✅ **PASS** Pixel size known → you can define a realistic scanner geometry (mm).
- ✅ **PASS** Square slices — fits standard parallel/fan-beam simulation.
- ℹ️ **INFO** Only reconstructed images → you'll simulate sparse projections yourself (standard practice). Avoid the 'inverse crime': simulate on a finer grid / different projector than you reconstruct with, and add noise.
- ℹ️ **INFO** Labels: none needed — the full-view image is the target. Note that the 'ground truth' is itself a reconstruction (noise, kernel, artefacts).
- ℹ️ **INFO** Reconstruction kernels: {'FC26': 4, 'BRAIN_CTA': 3, 'FC08': 2} (sharp kernels = noisier targets).
