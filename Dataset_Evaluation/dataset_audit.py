"""
dataset_audit.py
================
A small, transparent toolkit to check whether an (imaging) dataset is suitable
for a project BEFORE you train anything on it.

Every check:
  1. prints WHY it matters (one line),
  2. prints WHAT it found (tables / plots),
  3. adds a traffic-light finding to a Report:
        PASS  ✅  fine
        WARN  ⚠️  usable, but write it down as a limitation / handle it
        FAIL  ❌  blocker for this task
        INFO  ℹ️  fact worth knowing, no judgement
        TODO  📝  something only you can find out (paper, website, licence)

Supported formats (each reader is optional, missing libraries are skipped):
  DICOM (.dcm or no extension)        -> pydicom
  NIfTI / NRRD / MHA (.nii, .nii.gz, ...) -> SimpleITK (or nibabel)
  PNG / JPG / TIFF / BMP              -> Pillow
  NumPy (.npy, .npz)                  -> numpy
  HDF5 (.h5, .hdf5)                   -> h5py

Nothing here modifies your data. Everything is read-only.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # plots are optional
    plt = None

# ---- optional readers -------------------------------------------------------
try:
    import pydicom
except ImportError:
    pydicom = None
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import h5py
except ImportError:
    h5py = None


def available_readers():
    """Print which file readers are installed."""
    libs = {"pydicom (DICOM)": pydicom, "SimpleITK (NIfTI/NRRD/MHA)": sitk,
            "nibabel (NIfTI fallback)": nib, "Pillow (PNG/JPG/TIFF)": Image,
            "h5py (HDF5)": h5py, "matplotlib (plots)": plt}
    for name, mod in libs.items():
        print(("  ✓ " if mod is not None else "  ✗ ") + name)


# =============================================================================
# 1. REPORT  (collects all findings, prints summary, saves a dataset card)
# =============================================================================
ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️", "TODO": "📝"}


class Report:
    def __init__(self, dataset_name: str, task: str):
        self.dataset_name = dataset_name
        self.task = task
        self.findings: list[dict] = []
        self.facts: dict = {}          # key numbers, used for comparing datasets
        self.provenance: dict = {}

    def add(self, section: str, status: str, message: str):
        status = status.upper()
        assert status in ICON, f"unknown status {status}"
        # replace an older finding with identical section+message (re-running a cell)
        self.findings = [f for f in self.findings
                         if not (f["section"] == section and f["message"] == message)]
        self.findings.append({"section": section, "status": status, "message": message})
        print(f"  {ICON[status]} {status:4s} | {message}")

    def clear_section(self, section: str):
        """Called at the start of every check so re-running a cell doesn't duplicate."""
        self.findings = [f for f in self.findings if f["section"] != section]

    def fact(self, key: str, value):
        self.facts[key] = value

    # ---- output -------------------------------------------------------------
    def table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.findings, columns=["section", "status", "message"])
        df.insert(0, "", df["status"].map(ICON))
        return df

    def verdict(self) -> str:
        c = Counter(f["status"] for f in self.findings)
        if c["FAIL"]:
            return f"❌ NOT SUITABLE AS IS — {c['FAIL']} blocker(s). Fix them or reject the dataset."
        if c["WARN"]:
            v = f"⚠️ USABLE WITH CAVEATS — {c['WARN']} warning(s). Handle them or list them as limitations."
        else:
            v = "✅ SUITABLE — no blockers or warnings found by the automatic checks."
        if c["TODO"]:
            v += f"  📝 {c['TODO']} open question(s) still to answer."
        return v

    def summary(self):
        c = Counter(f["status"] for f in self.findings)
        print("=" * 78)
        print(f"DATASET: {self.dataset_name}    TASK: {self.task}")
        print("  ".join(f"{ICON[s]} {s}: {c[s]}" for s in ICON))
        print(self.verdict())
        print("=" * 78)
        order = {"FAIL": 0, "WARN": 1, "TODO": 2, "INFO": 3, "PASS": 4}
        df = self.table()
        return df.sort_values("status", key=lambda s: s.map(order)).reset_index(drop=True)

    def save(self, folder="dataset_cards"):
        """Writes <name>.md (human readable dataset card) and <name>.json (for comparing)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", self.dataset_name).strip("_")
        data = {"dataset_name": self.dataset_name, "task": self.task, "date": str(date.today()),
                "verdict": self.verdict(), "facts": _jsonable(self.facts),
                "provenance": self.provenance, "findings": self.findings}
        (folder / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))

        lines = [f"# Dataset card: {self.dataset_name}", "",
                 f"*Task:* {self.task}  ", f"*Checked:* {date.today()}  ", "",
                 f"**Verdict:** {self.verdict()}", ""]
        if self.provenance:
            lines += ["## Provenance", "", "| Field | Value |", "|---|---|"]
            lines += [f"| {k} | {str(v).replace('|', '/') or '—'} |" for k, v in self.provenance.items()]
            lines.append("")
        if self.facts:
            lines += ["## Key numbers", "", "| Fact | Value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in _jsonable(self.facts).items()]
            lines.append("")
        lines += ["## Findings", ""]
        for section in dict.fromkeys(f["section"] for f in self.findings):
            lines += [f"### {section}", ""]
            lines += [f"- {ICON[f['status']]} **{f['status']}** {f['message']}"
                      for f in self.findings if f["section"] == section]
            lines.append("")
        (folder / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved {folder / (stem + '.md')} and {folder / (stem + '.json')}")


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (tuple, set)):
            v = list(v)
        out[k] = v
    return out


def compare_reports(folder="dataset_cards") -> pd.DataFrame:
    """Side-by-side table of all saved dataset cards (one column per dataset)."""
    cols = {}
    for p in sorted(Path(folder).glob("*.json")):
        d = json.loads(p.read_text())
        c = Counter(f["status"] for f in d["findings"])
        col = {"task": d["task"], "checked": d["date"],
               "verdict": d["verdict"].split("—")[0].strip(),
               "❌ FAIL": c["FAIL"], "⚠️ WARN": c["WARN"], "📝 TODO": c["TODO"]}
        col.update({k: str(v) for k, v in d["facts"].items()})
        col["licence"] = d.get("provenance", {}).get("licence", "")
        cols[d["dataset_name"]] = col
    return pd.DataFrame(cols)


def _header(title, why):
    print(f"\n### {title}")
    print(f"Why it matters: {why}\n")


# =============================================================================
# 2. PROVENANCE  (filled in by hand — no code can know who made a dataset)
# =============================================================================
PROVENANCE_TEMPLATE = {
    "source_url": "",          # where you found it (Zenodo, TCIA, PhysioNet, Kaggle, grand-challenge...)
    "publisher": "",           # institution / group / challenge that released it
    "paper_to_cite": "",       # the data descriptor paper
    "version_and_date": "",    # which version, when released, is it still maintained?
    "licence": "",             # e.g. CC BY 4.0, CC BY-NC-ND 4.0, TCIA Restricted, custom DUA
    "access": "",              # open / registration / credentialed (training needed) / DUA signed
    "ethics_approval": "",     # IRB / ethics committee mentioned? patient consent?
    "de_identification": "",   # how were patients anonymised?
    "acquisition": "",         # scanners, sites, countries, years, protocol, dose
    "population": "",          # who is in it? healthy/patients, age range, inclusion criteria
    "label_source": "",        # who labelled (experts? how many? agreement?), or 'no labels'
    "raw_data_available": "",  # e.g. projections/sinograms, k-space, or only reconstructed images
    "official_split": "",      # given train/val/test split? patient-level?
    "known_issues": "",        # errata, forum complaints, retracted cases
}

_PROV_HINTS = {
    "source_url": "Where can it be downloaded from? Keep the link for the report.",
    "publisher": "Who released it? A known hospital/challenge is more trustworthy than an anonymous Kaggle upload.",
    "paper_to_cite": "Look for a data-descriptor paper (Scientific Data, Medical Physics, challenge paper).",
    "version_and_date": "Datasets get updated; note the version you used so results are reproducible.",
    "licence": "No licence = no permission. Check NC (non-commercial) and ND (no derivatives).",
    "access": "Credentialed access (e.g. PhysioNet CITI course, TCIA restricted) can take days to weeks.",
    "ethics_approval": "For human data the paper should state ethics approval / consent.",
    "de_identification": "How were names, dates, faces (head CT/MR!) removed?",
    "acquisition": "Scanner vendors, sites, years, protocol. One scanner = poor generalisation.",
    "population": "Who is included? Bias in age/sex/disease affects what your model learns.",
    "label_source": "Who drew the labels and how? Labels are only as good as their source.",
    "raw_data_available": "For reconstruction: are raw projections/sinograms given, or only images?",
    "official_split": "An official split makes your results comparable with published work.",
    "known_issues": "Search '<dataset name> issues / errata / github issues'.",
}


def check_provenance(card: dict, report: Report):
    sec = "Origin & licence"
    report.clear_section(sec)
    _header("Origin, licence, ethics", "you must be allowed to use the data, be able to cite it, "
            "and know where it comes from to judge bias.")
    report.provenance = dict(card)
    for key, val in card.items():
        if not str(val).strip():
            report.add(sec, "TODO", f"'{key}' unknown — {_PROV_HINTS.get(key, '')}")

    lic = str(card.get("licence", "")).upper().replace("-", " ")
    if lic.strip():
        if any(w in lic for w in ["NONE", "UNKNOWN", "NO LICENSE", "NO LICENCE"]):
            report.add(sec, "FAIL", "No licence: legally you have no permission to use it. Ask the authors or skip it.")
        if "CC0" in lic or "PUBLIC DOMAIN" in lic:
            report.add(sec, "PASS", "CC0 / public domain: free to use, citing is still good practice.")
        if re.search(r"\bNC\b", lic):
            report.add(sec, "INFO", "Non-commercial (NC): fine for uni projects/theses, NOT for company or start-up work.")
        if re.search(r"\bND\b", lic):
            report.add(sec, "WARN", "No-derivatives (ND): you may use it, but don't publish modified copies "
                                   "(e.g. your sparsified/simulated data). Share your code instead.")
        if re.search(r"\bSA\b", lic):
            report.add(sec, "INFO", "Share-alike (SA): anything you publish from it must use the same licence.")
        if re.search(r"\bBY\b", lic) and not re.search(r"\b(NC|ND)\b", lic):
            report.add(sec, "PASS", "Attribution licence (BY): free use, including derivatives, as long as you cite it.")
        if "CDLA" in lic or "COMMUNITY DATA LICENSE" in lic:
            report.add(sec, "INFO", "CDLA-Sharing: free use; if you publish the data (modified or not) it must stay under CDLA-Sharing. "
                                    "Check that the uploader had the right to re-license the original data.")
        if any(w in lic for w in ["DUA", "AGREEMENT", "RESTRICTED", "CUSTOM"]) and "CDLA" not in lic \
                and "COMMUNITY DATA LICENSE" not in lic:
            report.add(sec, "WARN", "Custom / restricted terms: read the agreement (sharing, publication, storage rules).")
    acc = str(card.get("access", "")).lower()
    if any(w in acc for w in ["credential", "dua", "restricted", "application", "approval"]):
        report.add(sec, "WARN", "Access needs approval/training — plan days to weeks before you can download.")
    elif "open" in acc:
        report.add(sec, "PASS", "Open access.")


# =============================================================================
# 3. INVENTORY  (what files are there?)
# =============================================================================
KINDS = {
    "dicom": {".dcm", ".dicom", ".ima"},
    "volume": {".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd"},
    "image": {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"},
    "array": {".npy", ".npz"},
    "hdf5": {".h5", ".hdf5", ".hdf"},
    "table": {".csv", ".tsv", ".xlsx", ".xls", ".json", ".xml"},
    "document": {".txt", ".md", ".pdf", ".html", ".rst", ".docx"},
    "archive": {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"},
}
IMAGE_KINDS = {"dicom", "volume", "image", "array", "hdf5"}
SPLIT_WORDS = {"train": "train", "training": "train", "tr": "train",
               "val": "val", "valid": "val", "validation": "val", "dev": "val",
               "test": "test", "testing": "test", "ts": "test"}


def _full_ext(p: Path) -> str:
    s = p.name.lower()
    for double in (".nii.gz", ".tar.gz"):
        if s.endswith(double):
            return double
    return p.suffix.lower()


def _is_dicom_without_ext(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def inventory(root, report: Report, max_files: int | None = None) -> pd.DataFrame:
    sec = "Files & size"
    report.clear_section(sec)
    _header("Inventory", "know what you are dealing with: formats, size on disk, documentation.")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith(".") or "__MACOSX" in p.parts:
            continue
        ext = _full_ext(p)
        kind = next((k for k, exts in KINDS.items() if ext in exts), "other")
        if kind == "other" and ext == "" and _is_dicom_without_ext(p):
            kind = "dicom"
        rel = p.relative_to(root)
        rows.append({"path": str(p), "rel": str(rel), "ext": ext or "(none)", "kind": kind,
                     "size_mb": p.stat().st_size / 1e6,
                     "top_folder": rel.parts[0] if len(rel.parts) > 1 else "."})
        if max_files and len(rows) >= max_files:
            report.add(sec, "INFO", f"Stopped after max_files={max_files}; numbers below are for this subset.")
            break
    inv = pd.DataFrame(rows)
    if inv.empty:
        report.add(sec, "FAIL", "No files found.")
        return inv

    size_gb = inv["size_mb"].sum() / 1e3
    by_kind = inv.groupby("kind").agg(files=("path", "size"), size_gb=("size_mb", lambda s: round(s.sum() / 1e3, 3)))
    print(by_kind.to_string(), "\n")
    print("Top-level folders:", ", ".join(sorted(inv["top_folder"].unique())[:15]))
    report.fact("files", len(inv))
    report.fact("size_on_disk_GB", round(size_gb, 2))
    report.add(sec, "INFO", f"{len(inv)} files, {size_gb:.2f} GB on disk.")

    n_img = inv["kind"].isin(IMAGE_KINDS).sum()
    if n_img == 0:
        report.add(sec, "FAIL", "No readable image files found (check formats or unpack archives).")
    if (inv["kind"] == "archive").any():
        report.add(sec, "WARN", f"{(inv['kind'] == 'archive').sum()} archive(s) still packed — unzip them, then re-run.")
    if (inv["kind"] == "other").any():
        exts = inv.loc[inv["kind"] == "other", "ext"].value_counts().head(5).to_dict()
        report.add(sec, "INFO", f"Unrecognised file types (ignored): {exts}")
    if size_gb > 80:
        report.add(sec, "WARN", f"{size_gb:.0f} GB: too big for free Colab disk / Google Drive. Plan a subset or local storage.")

    docs = inv[inv["rel"].str.contains(r"readme|licen[sc]e|citation|data.?card|terms|description",
                                       case=False, regex=True)]
    if len(docs):
        report.add(sec, "PASS", f"Documentation found: {', '.join(docs['rel'].head(5))} — read it!")
    else:
        report.add(sec, "TODO", "No README/LICENSE in the download: get licence & description from the source website.")
    tables = inv[inv["kind"] == "table"]
    if len(tables):
        report.add(sec, "INFO", f"Metadata/label tables found: {', '.join(tables['rel'].head(5))}")
    return inv


# =============================================================================
# 4. HEADERS  (read metadata without loading all pixels) and group into items
# =============================================================================
DICOM_FIELDS = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "Modality", "Manufacturer",
                "ManufacturerModelName", "InstitutionName", "BodyPartExamined", "SeriesDescription",
                "KVP", "XRayTubeCurrent", "CTDIvol", "SliceThickness", "ConvolutionKernel",
                "Rows", "Columns", "BitsStored", "RescaleIntercept", "RescaleSlope",
                "PatientSex", "PatientAge", "StudyDate", "BurnedInAnnotation", "PhotometricInterpretation"]
PHI_FIELDS = ["PatientName", "PatientBirthDate", "PatientAddress", "OtherPatientIDs",
              "ReferringPhysicianName", "InstitutionAddress", "PatientTelephoneNumbers"]
_PIL_DTYPE = {"1": "bool", "L": "uint8", "P": "uint8", "RGB": "uint8", "RGBA": "uint8",
              "I;16": "uint16", "I;16B": "uint16", "I": "int32", "F": "float32"}


def _dcm_value(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return "\\".join(str(x) for x in v)
    try:
        return float(v) if type(v).__name__ in ("DSfloat", "IS", "DSdecimal") else str(v)
    except Exception:
        return str(v)


def _read_one(row) -> dict:
    """Header of one file -> dict. Never loads full pixel data."""
    path, kind = row["path"], row["kind"]
    out = {"shape": None, "spacing": None, "dtype": None, "error": None}
    try:
        if kind == "dicom":
            if pydicom is None:
                raise ImportError("pip install pydicom")
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            for f in DICOM_FIELDS:
                out[f] = _dcm_value(getattr(ds, f, None))
            for f in PHI_FIELDS:
                out["phi_" + f] = _dcm_value(getattr(ds, f, None))
            ps = getattr(ds, "PixelSpacing", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            out["z"] = float(ipp[2]) if ipp is not None and len(ipp) == 3 else None
            rows, cols = getattr(ds, "Rows", None), getattr(ds, "Columns", None)
            out["shape"] = (int(rows), int(cols)) if rows and cols else None
            out["spacing"] = (float(ps[0]), float(ps[1])) if ps is not None else None
            bits = getattr(ds, "BitsStored", None)
            out["dtype"] = f"{bits}-bit" if bits else None
            out["transfer_syntax"] = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
        elif kind == "volume":
            if sitk is not None:
                r = sitk.ImageFileReader()
                r.SetFileName(path)
                r.ReadImageInformation()
                out["shape"] = tuple(int(s) for s in r.GetSize()[::-1])        # -> (z, y, x)
                out["spacing"] = tuple(round(float(s), 4) for s in r.GetSpacing()[::-1])
                out["dtype"] = sitk.GetPixelIDValueAsString(r.GetPixelIDValue())
            elif nib is not None:
                img = nib.load(path)
                shp, zooms = img.shape, img.header.get_zooms()
                out["shape"] = tuple(int(s) for s in shp[::-1])
                out["spacing"] = tuple(round(float(z), 4) for z in zooms[::-1])
                out["dtype"] = str(img.get_data_dtype())
            else:
                raise ImportError("pip install SimpleITK")
        elif kind == "image":
            if Image is None:
                raise ImportError("pip install pillow")
            with Image.open(path) as im:
                w, h = im.size
                ch = len(im.getbands())
                n = getattr(im, "n_frames", 1)
                out["shape"] = ((n,) if n > 1 else ()) + (h, w) + ((ch,) if ch > 1 else ())
                out["dtype"] = _PIL_DTYPE.get(im.mode, im.mode)
                out["mode"] = im.mode
        elif kind == "array":
            if path.endswith(".npz"):
                with np.load(path) as z:
                    arrays = {k: z[k] for k in z.files}
                key = max(arrays, key=lambda k: arrays[k].size)
                out["shape"], out["dtype"], out["array_key"] = arrays[key].shape, str(arrays[key].dtype), key
                out["npz_keys"] = ",".join(arrays)
            else:
                a = np.load(path, mmap_mode="r")
                out["shape"], out["dtype"] = tuple(a.shape), str(a.dtype)
        elif kind == "hdf5":
            if h5py is None:
                raise ImportError("pip install h5py")
            found = []
            with h5py.File(path, "r") as f:
                f.visititems(lambda n, o: found.append((n, o.shape, str(o.dtype)))
                             if isinstance(o, h5py.Dataset) else None)
            if found:
                name, shp, dt = max(found, key=lambda t: int(np.prod(t[1])) if t[1] else 0)
                out["shape"], out["dtype"], out["array_key"] = tuple(shp), dt, name
                out["h5_datasets"] = "; ".join(f"{n}{s}" for n, s, _ in found[:10])
    except Exception as e:  # unreadable file is itself a finding
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def read_headers(inv: pd.DataFrame, report: Report, patient_id_regex: str | None = None,
                 max_headers: int = 20000, seed: int = 0) -> pd.DataFrame:
    """One row per image file with its metadata."""
    sec = "Readability"
    report.clear_section(sec)
    _header("Reading headers", "unreadable or inconsistent files break pipelines later; "
            "headers tell you scanner, spacing, patient, etc.")
    img = inv[inv["kind"].isin(IMAGE_KINDS)].copy()
    if len(img) > max_headers:
        report.add(sec, "INFO", f"{len(img)} image files; reading a random {max_headers} for speed.")
        img = img.sample(max_headers, random_state=seed)
    meta = pd.concat([img.reset_index(drop=True),
                      pd.DataFrame([_read_one(r) for _, r in img.iterrows()])], axis=1)

    # patient ID: DICOM tag first, else regex on the relative path
    pid = meta["PatientID"] if "PatientID" in meta else pd.Series([None] * len(meta))
    if patient_id_regex:
        from_path = meta["rel"].str.extract(patient_id_regex, expand=False)
        if isinstance(from_path, pd.DataFrame):
            from_path = from_path.iloc[:, 0]
        pid = pid.where(pid.notna() & (pid.astype(str) != ""), from_path)
    meta["patient_id"] = pid
    # split from folder names (train/val/test)
    meta["split"] = meta["rel"].apply(
        lambda r: next((SPLIT_WORDS[p.lower()] for p in Path(r).parts[:-1] if p.lower() in SPLIT_WORDS), None))

    err = meta["error"].notna()
    if err.any():
        print(meta.loc[err, ["rel", "error"]].head(10).to_string(), "\n")
        missing_lib = meta.loc[err, "error"].str.contains("ImportError|pip install").all()
        report.add(sec, "WARN" if missing_lib else "FAIL",
                   f"{err.sum()} of {len(meta)} files could not be read"
                   + (" (missing library — install it and re-run)" if missing_lib else
                      " — corrupt or unusual format, see table above."))
    else:
        report.add(sec, "PASS", f"All {len(meta)} image files readable.")
    return meta


def _has_channels(kind, shape) -> bool:
    """(H, W, 3/4) = one 2D image with colour/window channels, not a 3-slice-wide volume."""
    s = tuple(shape)
    if kind == "image":
        return len(s) >= 3 and s[-1] in (3, 4)
    if kind in ("array", "hdf5"):
        return len(s) == 3 and s[-1] in (3, 4) and min(s[0], s[1]) > 16
    return False


def build_items(meta: pd.DataFrame) -> pd.DataFrame:
    """
    One row per *sample*: a 2D image file, a 3D volume file, or a whole DICOM series
    (DICOM stores one slice per file, so slices are grouped by SeriesInstanceUID).
    """
    ok = meta[meta["error"].isna()].copy()
    parts = []
    nond = ok[ok["kind"] != "dicom"].copy()
    if len(nond):
        nond["item_id"] = nond["rel"]
        nond["paths"] = nond["path"].apply(lambda p: [p])
        nond["n_files"] = 1
        parts.append(nond)
    dcm = ok[ok["kind"] == "dicom"].copy()
    if len(dcm):
        key = dcm["SeriesInstanceUID"].fillna(dcm["rel"].apply(lambda r: str(Path(r).parent)))
        rows = []
        for sid, g in dcm.groupby(key):
            g = g.sort_values("z") if g["z"].notna().all() else g
            first = g.iloc[0].to_dict()
            zs = np.sort(g["z"].dropna().unique())
            dz = float(np.median(np.diff(zs))) if len(zs) > 1 else None
            st = pd.to_numeric(g["SliceThickness"], errors="coerce").median() if "SliceThickness" in g else None
            shp = first["shape"]
            sp = first["spacing"]
            if len(g) > 1 and shp:
                shp = (len(g),) + tuple(shp)
                if sp:
                    sp = (round(dz if dz else (st if pd.notna(st) else np.nan), 4),) + tuple(sp)
            first.update({"item_id": str(Path(first["rel"]).parent) + f" [series …{str(sid)[-8:]}]",
                          "paths": list(g["path"]), "n_files": len(g), "shape": shp, "spacing": sp,
                          "z_spacing": dz})
            for f in PHI_FIELDS:   # any slice with PHI counts
                col = "phi_" + f
                if col in g:
                    vals = g[col].dropna()
                    first[col] = vals.iloc[0] if len(vals) else None
            rows.append(first)
        parts.append(pd.DataFrame(rows))
    items = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if items.empty:
        return items

    def inplane(r):
        s = r["shape"]
        if not s:
            return None
        s = tuple(s)
        if _has_channels(r["kind"], s):
            s = s[:-1]
        return s[-2:] if len(s) >= 2 else None

    def nslices(r):
        s = r["shape"]
        if not s:
            return None
        s = tuple(s)
        if _has_channels(r["kind"], s):
            s = s[:-1]
        return s[0] if len(s) == 3 else 1

    items["inplane"] = items.apply(inplane, axis=1)
    items["n_slices"] = items.apply(nslices, axis=1)
    items["inplane_spacing"] = items["spacing"].apply(lambda s: tuple(s[-2:]) if s else None)
    return items


# =============================================================================
# 5. SIZE & RESOLUTION
# =============================================================================
def check_size_resolution(items: pd.DataFrame, report: Report, min_inplane: int = 256):
    sec = "Size & resolution"
    report.clear_section(sec)
    _header("Size & resolution", "models need a fixed input size; physical pixel size (mm) "
            "defines scanner geometry and mm-based metrics; too few samples = no proper validation.")
    n_items = len(items)
    n_2d = int(items["n_slices"].fillna(1).sum())
    report.fact("samples (images/volumes)", n_items)
    report.fact("2D slices total", n_2d)
    report.add(sec, "INFO", f"{n_items} samples (images or volumes) = {n_2d} 2D slices in total.")

    print("Most common shapes:")
    print(items["shape"].astype(str).value_counts().head(8).to_string(), "\n")
    sizes = items["inplane"].dropna()
    uniq = sizes.astype(str).value_counts()
    report.fact("in-plane sizes", ", ".join(uniq.index[:3]))
    if len(uniq) == 1:
        report.add(sec, "PASS", f"All images have the same in-plane size {uniq.index[0]}.")
    elif len(uniq) > 1:
        report.add(sec, "WARN", f"{len(uniq)} different in-plane sizes (most common: "
                               f"{', '.join(uniq.index[:3])}) — you'll need resizing/cropping/padding.")
    if len(sizes):
        smallest = min(min(s) for s in sizes)
        nonsquare = sum(s[0] != s[1] for s in sizes)
        if smallest < min_inplane:
            report.add(sec, "WARN", f"Smallest image side is {smallest} px (< {min_inplane}). Fine details may be lost.")
        if nonsquare:
            report.add(sec, "INFO", f"{nonsquare} of {len(sizes)} images are not square.")

    nch = items.apply(lambda r: bool(r["shape"]) and _has_channels(r["kind"], r["shape"]), axis=1)
    if nch.any():
        report.add(sec, "INFO", f"{nch.sum()} samples have 3–4 channels (colour, or several CT windows stacked). "
                                "Several windows = intensities were already windowed, raw HU are gone.")

    sl = items["n_slices"].dropna()
    if (sl > 1).any():
        report.add(sec, "INFO", f"Volumes: {int(sl.min())}–{int(sl.max())} slices (median {int(sl.median())}).")

    sp = items["inplane_spacing"].dropna()
    report.fact("spacing known", f"{len(sp)}/{n_items}")
    if len(sp) == 0:
        report.add(sec, "WARN", "Pixel size in mm is unknown (PNG/JPG/NPY carry none). No physical scanner "
                               "geometry, no mm-based metrics (e.g. HD95 in mm). Check the paper for it.")
    else:
        px = np.array([s[0] for s in sp])
        report.fact("in-plane spacing mm", f"{px.min():.3f}–{px.max():.3f}")
        report.add(sec, "INFO" if len(sp) == n_items else "WARN",
                   f"In-plane pixel size {px.min():.3f}–{px.max():.3f} mm (known for {len(sp)}/{n_items}).")
        if px.max() / max(px.min(), 1e-9) > 1.5:
            report.add(sec, "WARN", "Pixel sizes vary a lot between samples — resample to a common spacing.")
        zs = items["spacing"].dropna().apply(lambda s: s[0] if len(s) == 3 else np.nan).dropna()
        if len(zs):
            report.fact("slice spacing mm", f"{zs.min():.2f}–{zs.max():.2f}")
            report.add(sec, "INFO", f"Slice spacing {zs.min():.2f}–{zs.max():.2f} mm.")
            if zs.median() > 3 * np.median(px):
                report.add(sec, "INFO", "Strongly anisotropic voxels (slices much thicker than pixels): "
                                        "2D per-slice processing is usually the natural choice.")

    if plt is not None and len(sizes):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        uniq.head(10).plot.barh(ax=ax[0], color="#4C78A8")
        ax[0].set_title("In-plane size (count)")
        ax[0].invert_yaxis()
        if len(sp):
            ax[1].hist([s[0] for s in sp], bins=20, color="#4C78A8")
            ax[1].set_title("Pixel spacing (mm)")
        else:
            ax[1].text(0.5, 0.5, "no spacing information", ha="center", va="center")
            ax[1].axis("off")
        plt.tight_layout()
        plt.show()


# =============================================================================
# 6. PIXEL VALUES
# =============================================================================
def load_slices(item, n_slices: int = 3) -> tuple[list[np.ndarray], str]:
    """Return a few representative 2D slices (float) + the stored dtype. DICOM -> HU."""
    kind, paths = item["kind"], item["paths"]
    if kind == "dicom":
        idx = sorted(set(np.linspace(0, len(paths) - 1, n_slices + 2).astype(int)[1:-1])) or [0]
        out, dt = [], ""
        for i in idx:
            ds = pydicom.dcmread(paths[i], force=True)
            a = ds.pixel_array.astype(np.float32)
            dt = str(ds.pixel_array.dtype)
            a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
            out.append(a)
        return out, dt
    if kind == "volume":
        if sitk is not None:
            vol = sitk.GetArrayFromImage(sitk.ReadImage(paths[0]))
        else:
            vol = np.asarray(nib.load(paths[0]).dataobj).T
        arr = vol
    elif kind == "image":
        with Image.open(paths[0]) as im:
            arr = np.array(im)
    elif kind == "array":
        if paths[0].endswith(".npz"):
            with np.load(paths[0]) as z:
                arr = z[item.get("array_key") or z.files[0]]
        else:
            arr = np.load(paths[0], mmap_mode="r")
    elif kind == "hdf5":
        f = h5py.File(paths[0], "r")
        arr = f[item["array_key"]]
    else:
        raise ValueError(kind)
    dt = str(arr.dtype)
    # colour image -> luminance; array with channels (e.g. 3 CT windows) -> first channel
    if _has_channels(kind, arr.shape):
        arr = arr[..., :3].mean(-1) if kind == "image" else np.asarray(arr[..., 0])
    if arr.ndim == 2:
        slices = [np.asarray(arr, dtype=np.float32)]
    else:
        while arr.ndim > 3:
            arr = arr[0]
        n = arr.shape[0]
        idx = sorted(set(np.linspace(0, n - 1, n_slices + 2).astype(int)[1:-1])) or [0]
        slices = [np.asarray(arr[i], dtype=np.float32) for i in idx]
    if kind == "hdf5":
        f.close()
    return slices, dt


def _guess_scale(mn, mx, dtype):
    if dtype in ("uint8", "bool") or (mn >= 0 and mx <= 255 and float(mx).is_integer() and mx > 1):
        return "8-bit display values (0–255)"
    if mn >= -0.01 and mx <= 1.01:
        return "normalised (0–1)"
    if mn < -500 and mx > 300:
        return "Hounsfield units (HU)"
    if mn >= 0 and mx < 0.1:
        return "attenuation coefficients (μ, 1/mm)"
    if mn >= 0 and mx <= 65535:
        return "raw integer values (e.g. 12/16-bit, unknown scale)"
    return "unknown scale"


def check_pixels(items: pd.DataFrame, report: Report, n: int = 30, seed: int = 0) -> pd.DataFrame:
    sec = "Pixel values"
    report.clear_section(sec)
    _header("Pixel values", "the intensity scale (HU? 8-bit?) decides what you can do physically; "
            "NaNs, blank images and mixed scales silently break training.")
    sample = items.sample(min(n, len(items)), random_state=seed)
    rows = []
    for _, it in sample.iterrows():
        try:
            sl, dt = load_slices(it)
            a = np.stack([s.ravel() for s in sl]) if len({s.size for s in sl}) == 1 else np.concatenate([s.ravel() for s in sl])
            finite = a[np.isfinite(a)]
            mn, mx = (float(finite.min()), float(finite.max())) if finite.size else (np.nan, np.nan)
            rows.append({"item_id": it["item_id"], "dtype": dt, "min": mn, "max": mx,
                         "mean": float(finite.mean()) if finite.size else np.nan,
                         "p1": float(np.percentile(finite, 1)) if finite.size else np.nan,
                         "p99": float(np.percentile(finite, 99)) if finite.size else np.nan,
                         "nan_or_inf_%": 100 * (1 - finite.size / a.size),
                         "below_-1024_%": 100 * float((finite < -1024).mean()) if finite.size else 0,
                         "constant": bool(finite.size and mn == mx),
                         "scale_guess": _guess_scale(mn, mx, dt), "error": None})
        except Exception as e:
            rows.append({"item_id": it["item_id"], "error": f"{type(e).__name__}: {e}"[:150]})
    st = pd.DataFrame(rows)
    show = [c for c in ["item_id", "dtype", "min", "max", "mean", "p1", "p99", "scale_guess"] if c in st]
    print(st[show].head(10).round(3).to_string(), "\n")

    bad = st["error"].notna()
    if bad.any():
        hint = " (compressed DICOM? pip install pylibjpeg pylibjpeg-libjpeg)" if st.loc[bad, "error"].str.contains(
            "decompress|handler|JPEG", case=False).any() else ""
        report.add(sec, "FAIL", f"{bad.sum()} of {len(st)} sampled samples failed to load pixels{hint}.")
    ok = st[~bad]
    if ok.empty:
        return st
    report.add(sec, "INFO", f"Checked {len(ok)} random samples. Stored data type(s): {', '.join(ok['dtype'].unique())}.")
    scales = ok["scale_guess"].value_counts()
    report.fact("intensity scale", scales.index[0])
    if len(scales) == 1:
        report.add(sec, "PASS" if "unknown" not in scales.index[0] else "WARN",
                   f"Consistent intensity scale: {scales.index[0]} (range {ok['min'].min():.4g} … {ok['max'].max():.4g}).")
    else:
        report.add(sec, "WARN", f"Mixed intensity scales across samples: {scales.to_dict()} — normalise per source.")
    if any(s.startswith("8-bit") for s in scales.index):
        report.add(sec, "WARN", "8-bit images: intensities were windowed/compressed for display, "
                               "original HU / raw values are lost.")
    if (ok["nan_or_inf_%"] > 0).any():
        report.add(sec, "WARN", f"{(ok['nan_or_inf_%'] > 0).sum()} samples contain NaN/Inf values.")
    if ok["constant"].any():
        report.add(sec, "WARN", f"{ok['constant'].sum()} samples are completely blank/constant.")
    if (ok["below_-1024_%"] > 0.5).any():
        report.add(sec, "INFO", "Values below −1024 HU present (outside-scan-field padding like −2048/−3024). "
                               "Clip to −1024 before computing anything physical.")
    return st


def show_samples(items: pd.DataFrame, n: int = 12, window: tuple | None = None, seed: int = 1):
    """
    Look at the data! window=(level, width), e.g. (40, 400) soft tissue, (400, 1800) bone, (-600, 1500) lung.
    Without window: 1st–99th percentile per image.
    """
    _header("Visual check", "numbers don't show artefacts, burned-in text, wrong orientation, "
            "cropped field of view or weird anatomy. Always look.")
    if plt is None:
        print("matplotlib not installed")
        return
    sample = items.sample(min(n, len(items)), random_state=seed)
    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    all_vals = []
    for ax, (_, it) in zip(axes, sample.iterrows()):
        try:
            sl, _ = load_slices(it, n_slices=1)
            a = sl[len(sl) // 2]
            all_vals.append(a[::4, ::4].ravel())
            if window:
                lo, hi = window[0] - window[1] / 2, window[0] + window[1] / 2
            else:
                lo, hi = np.percentile(a, [1, 99])
            ax.imshow(a, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"{Path(str(it['item_id']).split(' [')[0]).name[-30:]}\n{tuple(it['shape'])}", fontsize=7)
        except Exception as e:
            ax.text(0.5, 0.5, f"load error\n{type(e).__name__}", ha="center", va="center", fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    if all_vals:
        v = np.concatenate(all_vals)
        plt.figure(figsize=(8, 2.5))
        plt.hist(v[np.isfinite(v)], bins=150, color="#4C78A8", log=True)
        plt.title("Intensity histogram of the shown samples (log count)")
        plt.tight_layout()
        plt.show()


# =============================================================================
# 7. PATIENTS & SPLITS  (data leakage)
# =============================================================================
def check_patients(items: pd.DataFrame, report: Report, split_col: str = "split"):
    sec = "Patients & splits"
    report.clear_section(sec)
    _header("Patients & splits", "slices of the same patient in train AND test make results look far "
            "better than they are (data leakage). Splits must be per patient.")
    pid = items["patient_id"].replace("", np.nan)
    known = pid.notna()
    if not known.any():
        report.add(sec, "WARN", "No patient ID found (no DICOM tag, no PATIENT_ID_REGEX). You cannot prove a "
                               "patient-level split → leakage risk. Find the ID in file names or a table.")
    else:
        n_pat = pid.nunique()
        per = items[known].groupby(pid[known]).size()
        report.fact("patients", n_pat)
        report.add(sec, "INFO", f"{n_pat} patients; samples per patient: min {per.min()}, "
                               f"median {per.median():.0f}, max {per.max()}.")
        if (~known).any():
            report.add(sec, "WARN", f"{(~known).sum()} samples have no patient ID.")
        if per.max() > 0.2 * len(items) and n_pat > 3:
            report.add(sec, "WARN", f"Patient '{per.idxmax()}' contributes {per.max() / len(items):.0%} of all samples.")
        if n_pat < 5:
            report.add(sec, "FAIL", f"Only {n_pat} patients: no meaningful train/val/test split possible.")
        elif n_pat < 20:
            report.add(sec, "WARN", f"Only {n_pat} patients: test set will be tiny; consider cross-validation.")
        else:
            report.add(sec, "PASS", f"{n_pat} patients: enough for a patient-level train/val/test split.")

    if split_col in items and items[split_col].notna().any():
        counts = items[split_col].value_counts()
        print("Samples per split:", counts.to_dict())
        report.fact("official split", str(counts.to_dict()))
        report.add(sec, "PASS", f"Split given by the folders: {counts.to_dict()}.")
        if known.any():
            sp = items[known].groupby(pid[known])[split_col].nunique()
            leak = sp[sp > 1]
            if len(leak):
                report.add(sec, "FAIL", f"LEAKAGE: {len(leak)} patient(s) appear in more than one split "
                                       f"(e.g. {', '.join(map(str, leak.index[:5]))}).")
            else:
                report.add(sec, "PASS", "No patient appears in more than one split.")
    else:
        report.add(sec, "INFO", "No train/val/test folders found: you'll make your own split (per patient, fixed seed).")


# =============================================================================
# 8. LABELS
# =============================================================================
def labels_from_folders(items: pd.DataFrame, level: int = -2) -> pd.Series:
    """Class = name of the folder at `level` in the path (-2 = parent folder). Skips split words."""
    def lab(rel):
        parts = [p for p in Path(rel).parts if p.lower() not in SPLIT_WORDS]
        return parts[level] if len(parts) >= -level else None
    return pd.Series(items["rel"].apply(lab).values, index=items.index, name="label")


def labels_from_table(items: pd.DataFrame, table_path, id_col: str, label_col: str,
                      match: str = "stem") -> pd.Series:
    """
    Read labels from a CSV/XLSX. match='stem': table id == file name without extension;
    match='patient': table id == patient_id.
    """
    t = pd.read_excel(table_path) if str(table_path).endswith(("xlsx", "xls")) else pd.read_csv(table_path, sep=None, engine="python")
    lut = dict(zip(t[id_col].astype(str), t[label_col]))
    if match == "patient":
        keys = items["patient_id"].astype(str)
    else:
        keys = items["rel"].apply(lambda r: Path(r).name.split(".")[0])
    return pd.Series(keys.map(lut).values, index=items.index, name="label")


def check_labels(labels: pd.Series, report: Report, max_ratio_warn: float = 3.0, min_per_class: int = 30):
    """Class labels (classification) or a continuous target (regression)."""
    sec = "Labels & class balance"
    report.clear_section(sec)
    _header("Labels & class balance", "a model trained on 95% 'healthy' can score 95% accuracy by "
            "always saying healthy. Small classes = unreliable metrics for them.")
    missing = labels.isna().sum()
    if missing:
        report.add(sec, "WARN", f"{missing} of {len(labels)} samples have no label.")
    lab = labels.dropna()
    if lab.empty:
        report.add(sec, "FAIL", "No labels found.")
        return
    numeric = pd.to_numeric(lab, errors="coerce")
    if numeric.notna().all() and lab.nunique() > 20:
        print(numeric.describe().round(3).to_string())
        report.add(sec, "INFO", f"Continuous target (regression): range {numeric.min():.3g}–{numeric.max():.3g}.")
        return
    counts = lab.value_counts()
    print(pd.DataFrame({"count": counts, "share": (counts / counts.sum()).round(3)}).to_string(), "\n")
    report.fact("classes", len(counts))
    ratio = counts.max() / counts.min()
    report.fact("imbalance ratio (max/min)", round(float(ratio), 1))
    msg = f"{len(counts)} classes, largest/smallest = {ratio:.1f}× ('{counts.idxmax()}' {counts.max()} vs '{counts.idxmin()}' {counts.min()})."
    if len(counts) == 1:
        report.add(sec, "FAIL", "Only one class present.")
    elif ratio <= max_ratio_warn:
        report.add(sec, "PASS", "Reasonably balanced: " + msg)
    else:
        report.add(sec, "WARN", "Imbalanced: " + msg + " Use class weights/balanced sampling and "
                               "per-class metrics (not plain accuracy).")
    small = counts[counts < min_per_class]
    if len(small):
        report.add(sec, "WARN", f"{len(small)} class(es) with < {min_per_class} samples: {small.to_dict()}.")
    if plt is not None:
        counts.head(20).plot.bar(figsize=(7, 2.5), color="#4C78A8", title="Samples per class")
        plt.tight_layout()
        plt.show()


def split_images_masks(items: pd.DataFrame, mask_pattern: str = r"(?:mask|seg|label|gt|annot)"):
    """Separate image files from mask files by a regex on the path. Returns (images, masks)."""
    is_mask = items["rel"].str.contains(mask_pattern, case=False, regex=True)
    return items[~is_mask].copy(), items[is_mask].copy()


def check_masks(images: pd.DataFrame, masks: pd.DataFrame, report: Report,
                mask_pattern: str = r"(?:mask|seg|label|gt|annot)", n: int = 20, seed: int = 0):
    """Segmentation: are there masks, do they match the images, how big are the structures?"""
    sec = "Labels & class balance"
    report.clear_section(sec)
    _header("Segmentation masks", "each image needs a matching mask of the same size; tiny "
            "structures (<1% of pixels) are hard and need special losses/metrics.")
    if masks.empty:
        report.add(sec, "FAIL", f"No mask files found with pattern {mask_pattern!r}.")
        return
    norm = lambda r: re.sub(mask_pattern, "", Path(r).name.split(".")[0], flags=re.I).strip("_- ")
    img_keys = dict(zip(images["rel"].apply(norm), images.index))
    matched = masks["rel"].apply(norm).map(img_keys)
    report.add(sec, "INFO", f"{len(images)} images, {len(masks)} masks, {matched.notna().sum()} masks matched to an image by name.")
    if matched.notna().sum() < len(images):
        report.add(sec, "WARN", f"{len(images) - matched.notna().sum()} images have no matching mask (or names differ).")
    shape_mismatch = sum(tuple(images.loc[i, "shape"] or ()) != tuple(masks.loc[m, "shape"] or ())
                         for m, i in matched.dropna().items())
    if shape_mismatch:
        report.add(sec, "FAIL", f"{shape_mismatch} image/mask pairs differ in shape.")
    elif matched.notna().any():
        report.add(sec, "PASS", "Matched image/mask pairs have identical shapes.")

    voxels, empty = Counter(), 0
    total = 0
    for _, m in masks.sample(min(n, len(masks)), random_state=seed).iterrows():
        try:
            sl, _ = load_slices(m, n_slices=9)
            a = np.concatenate([s.ravel() for s in sl]).astype(int)
            total += a.size
            vals, cnt = np.unique(a, return_counts=True)
            voxels.update(dict(zip(vals.tolist(), cnt.tolist())))
            if (vals != 0).sum() == 0:
                empty += 1
        except Exception:
            pass
    if total:
        tab = pd.DataFrame({"label": list(voxels), "pixel_share_%": [100 * c / total for c in voxels.values()]})
        print(tab.sort_values("label").round(4).to_string(index=False), "\n")
        fg = tab[tab["label"] != 0]
        if len(tab) > 40:
            report.add(sec, "WARN", f"{len(tab)} distinct values in masks — are these really label maps (not images)?")
        for _, r in fg.iterrows():
            if r["pixel_share_%"] < 1:
                report.add(sec, "INFO", f"Label {int(r['label'])} covers only {r['pixel_share_%']:.3f}% of pixels "
                                       "(small structure → Dice/Tversky loss, distance metrics).")
        if empty:
            report.add(sec, "INFO", f"{empty} of the sampled masks are empty (no foreground in sampled slices).")


# =============================================================================
# 9. BALANCE OF METADATA  (who/what is in the dataset?)
# =============================================================================
DEFAULT_BALANCE_COLS = ["Manufacturer", "ManufacturerModelName", "InstitutionName", "KVP",
                        "ConvolutionKernel", "BodyPartExamined", "PatientSex", "age_group",
                        "SliceThickness", "split"]


def _age_group(a):
    m = re.match(r"(\d+)([YMWD]?)", str(a or ""))
    if not m:
        return None
    v = int(m.group(1)) if m.group(2) in ("", "Y") else 0
    return f"{v // 10 * 10}s"


def check_balance(items: pd.DataFrame, report: Report, columns: list | None = None,
                  extra: pd.DataFrame | None = None, dominance: float = 0.8):
    """
    Distribution of acquisition & demographic metadata, counted per PATIENT when possible.
    extra: optional table with a 'patient_id' column (e.g. clinical CSV) whose columns are added.
    """
    sec = "Balance & diversity"
    report.clear_section(sec)
    _header("Balance & diversity", "if 95% comes from one scanner, sex or age group, your model "
            "learns that and may fail elsewhere. Also a key limitation to state.")
    df = items.copy()
    if "PatientAge" in df:
        df["age_group"] = df["PatientAge"].apply(_age_group)
    if extra is not None and "patient_id" in extra:
        df = df.merge(extra, on="patient_id", how="left", suffixes=("", "_extra"))
        columns = (columns or DEFAULT_BALANCE_COLS) + [c for c in extra.columns if c != "patient_id"]
    columns = [c for c in (columns or DEFAULT_BALANCE_COLS) if c in df]
    if df["patient_id"].notna().any():
        per = df.drop_duplicates("patient_id")
        unit = "patients"
    else:
        per, unit = df, "samples"
    shown = 0
    for c in columns:
        s = per[c].replace("", np.nan)
        if s.isna().all():
            continue
        shown += c != "split"
        vc = s.value_counts(dropna=True)
        miss = s.isna().mean()
        print(f"{c}  ({unit}, {miss:.0%} missing):  " +
              ", ".join(f"{k}: {v}" for k, v in vc.head(8).items()) + (" …" if len(vc) > 8 else ""))
        top_share = vc.iloc[0] / vc.sum()
        if len(vc) == 1:
            report.add(sec, "WARN", f"{c}: all {unit} = '{vc.index[0]}' — no diversity, results may not generalise beyond it.")
        elif top_share > dominance:
            report.add(sec, "WARN", f"{c}: '{vc.index[0]}' dominates ({top_share:.0%}).")
        else:
            report.add(sec, "PASS", f"{c}: {len(vc)} groups, largest {top_share:.0%}.")
        if miss > 0.5:
            report.add(sec, "INFO", f"{c}: {miss:.0%} missing.")
    if shown == 0:
        report.add(sec, "TODO", "No acquisition/demographic metadata in the files (typical for PNG/NPY). "
                               "Get scanner, site, sex, age distribution from the paper or a metadata table.")


# =============================================================================
# 10. DUPLICATES
# =============================================================================
def _ahash(a: np.ndarray, k: int = 16) -> str:
    a = np.nan_to_num(a.astype(np.float32))
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    if h == 0 or w == 0:
        return ""
    small = a[:h, :w].reshape(k, h // k, k, w // k).mean((1, 3))
    return np.packbits(small > small.mean()).tobytes().hex()


def check_duplicates(items: pd.DataFrame, report: Report, n_visual: int = 300,
                     max_file_mb: float = 500, near_tol: float = 0.01, seed: int = 0):
    sec = "Duplicates"
    report.clear_section(sec)
    _header("Duplicates", "the same image twice (especially across train/test) inflates results; "
            "many duplicates also mean the dataset is smaller than it looks.")
    # exact: file content hash
    hashes = {}
    for _, it in items.iterrows():
        p = it["paths"][0]
        if Path(p).stat().st_size / 1e6 > max_file_mb:
            continue
        h = hashlib.md5(Path(p).read_bytes()).hexdigest()
        hashes.setdefault(h, []).append(it["item_id"])
    dup = [v for v in hashes.values() if len(v) > 1]
    if dup:
        report.add(sec, "WARN", f"{sum(len(v) - 1 for v in dup)} exact duplicate files "
                               f"(e.g. {dup[0][:2]}).")
    else:
        report.add(sec, "PASS", "No exact duplicate files.")
    # near: bucket by a tiny average-hash, then confirm with a normalised 32x32 thumbnail
    sample = items.sample(min(n_visual, len(items)), random_state=seed)
    buckets = {}
    for _, it in sample.iterrows():
        try:
            sl, _ = load_slices(it, n_slices=1)
            a = sl[0]
            buckets.setdefault(_ahash(a), []).append((it, _thumb(a)))
        except Exception:
            pass
    groups = []
    for k, members in buckets.items():
        if not k or len(members) < 2:
            continue
        used = set()
        for i, (it_i, t_i) in enumerate(members):
            if i in used:
                continue
            grp = [it_i]
            for j in range(i + 1, len(members)):
                if j not in used and np.abs(t_i - members[j][1]).mean() < near_tol:
                    grp.append(members[j][0])
                    used.add(j)
            if len(grp) > 1:
                groups.append(grp)
    if groups:
        n_pairs = sum(len(g) - 1 for g in groups)
        cross = [g for g in groups if len({r.get("split") for r in g} - {None}) > 1]
        report.add(sec, "WARN", f"{n_pairs} near-identical sample(s) among {len(sample)} checked "
                               f"(e.g. {[r['item_id'] for r in groups[0][:2]]}). Look at them. Same scan reconstructed "
                               "with different algorithms/kernels also looks near-identical: keep such groups in ONE split.")
        if cross:
            report.add(sec, "FAIL", f"{len(cross)} near-duplicate group(s) span different splits → leakage.")
    else:
        report.add(sec, "PASS", f"No near-duplicates among {len(sample)} sampled images.")


def _thumb(a: np.ndarray, k: int = 32) -> np.ndarray:
    """32x32 thumbnail, z-scored, so 8-bit and HU copies of the same image still match."""
    a = np.nan_to_num(a.astype(np.float32))
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    if h == 0 or w == 0:
        return np.zeros((k, k), np.float32)
    t = a[:h, :w].reshape(k, h // k, k, w // k).mean((1, 3))
    return (t - t.mean()) / (t.std() + 1e-6)


# =============================================================================
# 11. PRIVACY
# =============================================================================
_ANON = re.compile(r"^(|none|anon.*|deident.*|removed|x+|0+|\^*|unknown|n/?a|patient.?\d*)$", re.I)


def check_privacy(items: pd.DataFrame, report: Report):
    sec = "Privacy"
    report.clear_section(sec)
    _header("Privacy (DICOM tags)", "medical data must be de-identified; if names/birth dates are "
            "inside, don't upload it anywhere public (GitHub, public Colab) and check the terms.")
    dcm = items[items["kind"] == "dicom"]
    if dcm.empty:
        report.add(sec, "INFO", "No DICOM headers to check. Still look at the images for burned-in text, "
                               "and at head scans for recognisable faces.")
        return
    found = {}
    for f in PHI_FIELDS:
        col = "phi_" + f
        if col in dcm:
            vals = dcm[col].dropna().astype(str)
            real = vals[~vals.str.strip().str.match(_ANON)]
            if len(real):
                found[f] = len(real)
    if found:
        report.add(sec, "WARN", f"Possibly identifying DICOM tags filled in: {found}. Check whether real or "
                               "pseudonyms (phantom datasets use made-up names); keep data private either way.")
    else:
        report.add(sec, "PASS", "Name / birth date / address tags empty or anonymised.")
    if "BurnedInAnnotation" in dcm and (dcm["BurnedInAnnotation"].astype(str).str.upper() == "YES").any():
        report.add(sec, "WARN", "Some images flag burned-in annotations (text in the pixels).")
    if "StudyDate" in dcm and dcm["StudyDate"].notna().any():
        report.add(sec, "INFO", "Study dates present (often shifted for anonymisation — don't rely on them).")


# =============================================================================
# 12. TASK FIT  (does it fit what YOU want to do?)
# =============================================================================
def check_task_fit(task: str, items: pd.DataFrame, report: Report, pixel_stats: pd.DataFrame | None = None,
                   provenance: dict | None = None):
    """
    task: 'sparse_view_ct' | 'reconstruction' | 'segmentation' | 'classification' | 'other'
    """
    sec = "Task fit"
    report.clear_section(sec)
    _header(f"Task fit: {task}", "a 'good' dataset can still be the wrong one for your task.")
    prov = provenance or {}
    scale = (pixel_stats["scale_guess"].dropna().value_counts().index[0]
             if pixel_stats is not None and "scale_guess" in pixel_stats and pixel_stats["scale_guess"].notna().any()
             else "")
    if task in ("sparse_view_ct", "reconstruction"):
        if "Modality" in items and items["Modality"].notna().any():
            mods = items["Modality"].dropna().unique()
            if set(mods) == {"CT"}:
                report.add(sec, "PASS", "All series are CT.")
            else:
                report.add(sec, "WARN", f"Modalities found: {list(mods)} — keep only CT.")
        else:
            report.add(sec, "TODO", "Modality not stored in the files — confirm it is CT.")
        if "Hounsfield" in scale or "attenuation" in scale:
            report.add(sec, "PASS", f"Intensities are physical ({scale}) → forward projection is meaningful.")
        elif "normalised" in scale or "raw" in scale:
            report.add(sec, "WARN", f"Intensities are {scale}: find the conversion to HU/μ in the docs, otherwise "
                                    "simulated projections are only relative.")
        elif "8-bit" in scale:
            report.add(sec, "WARN", "8-bit display images: projections simulated from these are not physically "
                                    "realistic. OK as proof of concept — must be stated as limitation.")
        if items["inplane_spacing"].notna().any():
            report.add(sec, "PASS", "Pixel size known → you can define a realistic scanner geometry (mm).")
        else:
            report.add(sec, "WARN", "Pixel size unknown → geometry in pixel units only.")
        sizes = items["inplane"].dropna()
        if len(sizes) and all(s[0] == s[1] for s in sizes):
            report.add(sec, "PASS", "Square slices — fits standard parallel/fan-beam simulation.")
        raw = str(prov.get("raw_data_available", "")).lower()
        if any(w in raw for w in ["yes", "sinogram", "projection", "raw"]) and not raw.startswith("no"):
            report.add(sec, "PASS", "Raw projection data available → you can test on real measured data.")
        else:
            report.add(sec, "INFO", "Only reconstructed images → you'll simulate sparse projections yourself "
                                    "(standard practice). Avoid the 'inverse crime': simulate on a finer grid / "
                                    "different projector than you reconstruct with, and add noise.")
        report.add(sec, "INFO", "Labels: none needed — the full-view image is the target. Note that the "
                                "'ground truth' is itself a reconstruction (noise, kernel, artefacts).")
        if "ConvolutionKernel" in items and items["ConvolutionKernel"].notna().any():
            report.add(sec, "INFO", f"Reconstruction kernels: {items['ConvolutionKernel'].value_counts().head(4).to_dict()} "
                                    "(sharp kernels = noisier targets).")
    elif task == "segmentation":
        report.add(sec, "INFO", "Needs: masks for every image, matching shape, known label meaning (label map "
                                "legend), annotator info. See 'Labels' section.")
        if not str(prov.get("label_source", "")).strip():
            report.add(sec, "TODO", "Who drew the masks, and with what protocol? (label quality = ceiling of your model)")
    elif task == "classification":
        report.add(sec, "INFO", "Needs: one reliable label per sample, enough samples per class, patient-level split. "
                                "See 'Labels' section.")
        if not str(prov.get("label_source", "")).strip():
            report.add(sec, "TODO", "How were labels obtained (radiologist read, biopsy, report text mining)?")
    else:
        report.add(sec, "TODO", "Write down which properties your task needs and check them by hand.")


# =============================================================================
# 13. DEMO DATASET  (so you can try the notebook without real data)
# =============================================================================
def make_demo_dataset(root="demo_dataset", seed=0):
    """
    Synthetic 'CT-like' dataset with PLANTED PROBLEMS, to see what the checks report:
    mixed formats (8-bit PNG + float HU .npy), one patient in train AND test, one duplicate,
    one blank image, imbalanced classes, a non-square image.
    """
    if Image is None:
        raise ImportError("pip install pillow")
    rng = np.random.default_rng(seed)
    root = Path(root)
    yy, xx = np.mgrid[-1:1:256j, -1:1:256j]

    def phantom(r_body=0.85, lesion=False):
        img = np.full((256, 256), -1000.0)
        ox, oy, ecc = rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(0.65, 0.95)
        body = ((xx - ox) / r_body) ** 2 + ((yy - oy) / (r_body * ecc)) ** 2 < 1
        img[body] = 40 + rng.normal(0, 15, body.sum())
        for _ in range(rng.integers(2, 5)):                 # 'bones' at random places
            bx, by, br = rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(0.08, 0.2)
            img[((xx - bx) ** 2 + (yy - by) ** 2 < br ** 2) & body] = rng.uniform(400, 1000)
        if lesion:
            cx, cy = rng.uniform(-0.3, 0.3, 2)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 < 0.01] = 120
        return img

    def to8(img):  # window L40 W400 -> 8-bit
        return np.clip((img - (40 - 200)) / 400 * 255, 0, 255).astype(np.uint8)

    plan = ([("train", f"P{i:03d}", "normal") for i in range(1, 15)] +
            [("train", f"P{i:03d}", "lesion") for i in range(15, 18)] +
            [("test", f"P{i:03d}", "normal") for i in range(18, 22)] +
            [("test", "P020", "lesion"), ("test", "P001", "normal")])   # P001 leaks into test
    for split, pid, cls in plan:
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for s in range(2):
            img = phantom(rng.uniform(0.7, 0.9), lesion=(cls == "lesion"))
            name = f"{pid}_slice{s:02d}"
            if pid in ("P005", "P006"):
                np.save(d / f"{name}.npy", img.astype(np.float32))     # HU floats
            else:
                Image.fromarray(to8(img)).save(d / f"{name}.png")
    # planted extras
    import shutil
    shutil.copy(root / "train/normal/P002_slice00.png", root / "train/normal/P002_slice00_copy.png")
    Image.fromarray(np.zeros((256, 256), np.uint8)).save(root / "train/normal/P010_slice05.png")
    Image.fromarray(to8(phantom())[:, :200]).save(root / "train/normal/P011_slice07.png")
    (root / "README.txt").write_text("Synthetic demo dataset made by dataset_audit.make_demo_dataset()\n")
    print(f"Demo dataset written to {root.resolve()}")
    return str(root)
