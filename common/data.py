import hashlib
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

SEED = int("".join(c for c in "m26cse018" if c.isdigit()))
OUT = Path("results")
OUT.mkdir(exist_ok=True)
CFG = json.loads(Path("config.json").read_text())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def read_data(test=False):
    archive = Path("data/cifar-10-python.tar.gz")
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        temp = archive.with_suffix(".partial")
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", temp)
        temp.replace(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != "c58f30108f718f92721af3b95e74349a":
        raise ValueError("Incomplete CIFAR10 archive. Remove the archive and retry.")
    keys = ["test_batch"] if test else [f"data_batch_{i}" for i in range(1, 6)]
    with tarfile.open(archive) as tar:
        parts = [
            pickle.load(tar.extractfile("cifar-10-batches-py/" + key), encoding="bytes")
            for key in keys
        ]
    x = torch.from_numpy(np.concatenate([p[b"data"] for p in parts]).reshape(-1, 3, 32, 32))
    y = torch.tensor(np.concatenate([p[b"labels"] for p in parts]))
    return x, y


def indices(y):
    """Create the roll-seeded stratified split and draw both subsets from its training side."""
    tr, va = train_test_split(np.arange(len(y)), test_size=5000, stratify=y, random_state=SEED)
    sub, _ = train_test_split(tr, train_size=CFG["svm_samples"], stratify=y[tr], random_state=SEED)
    mem, _ = train_test_split(
        tr, train_size=CFG["memorisation_samples"], stratify=y[tr], random_state=SEED
    )
    np.savez(OUT / "split.npz", train=tr, validation=va, svm=sub, memory=mem)
    return tr, va, sub, mem


def permutation(condition):
    rng = np.random.default_rng(SEED)
    # Treat each RGB triplet as one location. The seeded mapping is identical for every image.
    if condition == "pixel":
        return torch.tensor(rng.permutation(1024))
    # Tile-row, within-row, tile-column, within-column -> 64 intact 4x4 tiles.
    tiles = np.arange(1024).reshape(8, 4, 8, 4).transpose(0, 2, 1, 3).reshape(64, 4, 4)
    return torch.tensor(
        tiles[rng.permutation(64)].reshape(8, 8, 4, 4).transpose(0, 2, 1, 3).ravel()
    )


def batches(data, ids, condition="clean", shuffle=False, epoch=0):
    x, y = data
    rng = np.random.default_rng(SEED + epoch)
    order = rng.permutation(ids) if shuffle else ids
    perm = permutation(condition) if condition in ("pixel", "tile") else None
    for start in range(0, len(order), CFG["batch_size"]):
        group = order[start : start + CFG["batch_size"]]
        clean = x[group].to(DEVICE).float().div_(255)
        inp = clean
        if perm is not None:
            inp = clean.flatten(2)[:, :, perm].reshape_as(clean)
        # A different training epoch changes the noise. During evaluation epoch=0
        # and the same example order give identical noise for all three views.
        if condition == "noise":
            noise = torch.from_numpy(rng.standard_normal(tuple(clean.shape), dtype=np.float32)).to(
                DEVICE
            )
            inp = (clean + CFG["noise_sd"] * noise).clamp(0, 1)
        yield inp, y[group].to(DEVICE), clean, group
