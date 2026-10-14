"""
Training script for the sparse-view CT U-Net (image-domain, FBPConvNet-style).

Images are paired per patient by the NUMBER in the filename
(image_0 <-> slice_00, image_1 <-> slice_01, ...).

The split is done at PATIENT level (no slices of the same patient appear in
more than one split), which avoids data leakage.
"""

import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

from unet_ct import UNet


# Configuration
ORIGINAL_DIR = Path("Original_images")
SPARSE_DIR = Path("Sparse_images")

IMG_SIZE = 256          # must be divisible by 16 (4 pooling stages in the U-Net)
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
SEED = 42

N_TEST_PATIENTS = 2     # 8 patients -> 5 train / 1 val / 2 test
N_VAL_PATIENTS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Pairing images
def file_index(path):
    """Extract the (last) number in a filename: 'slice_07.png' -> 7."""
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1])


def collect_pairs(patient_name):
    """Return a list of (sparse_path, original_path) for one patient,
    matched by the number in the filename (NOT by alphabetical order, since
    'image_10' would otherwise sort before 'image_2')."""
    orig_files = {file_index(p): p for p in (ORIGINAL_DIR / patient_name).glob("*.jpg")}
    sparse_files = {file_index(p): p for p in (SPARSE_DIR / patient_name).glob("*.png")}

    common = sorted(set(orig_files) & set(sparse_files))
    if len(common) != len(orig_files) or len(common) != len(sparse_files):
        print(f"[warning] {patient_name}: {len(orig_files)} originals, "
              f"{len(sparse_files)} sparse, {len(common)} matched")
    return [(sparse_files[i], orig_files[i]) for i in common]



# Dataset

def load_gray(path):
    """Load as grayscale, resize, scale to [0, 1], shape (1, H, W)."""
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class CTPairDataset(Dataset):
    """Holds all (sparse, original) pairs of a list of patients in memory
    (small dataset, so this is fine and makes training fast)."""

    def __init__(self, patient_names, augment=False):
        self.augment = augment
        self.sparse, self.original = [], []
        for name in patient_names:
            for sparse_path, orig_path in collect_pairs(name):
                self.sparse.append(load_gray(sparse_path))
                self.original.append(load_gray(orig_path))

    def __len__(self):
        return len(self.sparse)

    def __getitem__(self, idx):
        x, y = self.sparse[idx], self.original[idx]
        if self.augment:
            # The SAME flip must be applied to input and target
            if random.random() < 0.5:
                x, y = torch.flip(x, dims=[2]), torch.flip(y, dims=[2])
            if random.random() < 0.5:
                x, y = torch.flip(x, dims=[1]), torch.flip(y, dims=[1])
        return x, y


# Patient-level split
def split_patients():
    patients = sorted(p.name for p in ORIGINAL_DIR.iterdir() if p.is_dir())
    random.Random(SEED).shuffle(patients)

    test = patients[:N_TEST_PATIENTS]
    val = patients[N_TEST_PATIENTS:N_TEST_PATIENTS + N_VAL_PATIENTS]
    train = patients[N_TEST_PATIENTS + N_VAL_PATIENTS:]
    print(f"Train patients: {train}\nVal patients:   {val}\nTest patients:  {test}")
    return train, val, test


# Training / evaluation
def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over the loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_metrics(model, dataset):
    """PSNR / SSIM of (a) the raw sparse FBP input and (b) the U-Net output,
    both compared to the ground truth. (a) is your FBP baseline."""
    model.eval()
    rows = {"input": {"psnr": [], "ssim": []}, "unet": {"psnr": [], "ssim": []}}
    for x, y in dataset:
        pred = model(x.unsqueeze(0).to(DEVICE)).squeeze().clamp(0, 1).cpu().numpy()
        x_np, y_np = x.squeeze().numpy(), y.squeeze().numpy()
        for name, img in (("input", x_np), ("unet", pred)):
            rows[name]["psnr"].append(peak_signal_noise_ratio(y_np, img, data_range=1.0))
            rows[name]["ssim"].append(structural_similarity(y_np, img, data_range=1.0))
    for name, r in rows.items():
        print(f"{name:>6}:  PSNR {np.mean(r['psnr']):.2f} dB   SSIM {np.mean(r['ssim']):.4f}")


def main():
    set_seed(SEED)
    train_p, val_p, test_p = split_patients()

    train_ds = CTPairDataset(train_p, augment=True)
    val_ds = CTPairDataset(val_p)
    test_ds = CTPairDataset(test_p)
    print(f"Images -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = UNet(in_channels=1, out_channels=1, base_channels=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, loss_fn, optimizer)
        va = run_epoch(model, val_loader, loss_fn)
        flag = ""
        if va < best_val:          # keep the model that does best on validation
            best_val = va
            torch.save(model.state_dict(), "best_unet.pt")
            flag = "  <- saved"
        print(f"Epoch {epoch:03d} | train L1 {tr:.5f} | val L1 {va:.5f}{flag}")

    # Final evaluation on the untouched test patients, using the best checkpoint
    model.load_state_dict(torch.load("best_unet.pt", map_location=DEVICE))
    print("\n=== Test set (held-out patients) ===")
    evaluate_metrics(model, test_ds)


if __name__ == "__main__":
    main()
