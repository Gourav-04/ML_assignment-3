"""CSV writing and frozen feature extraction shared across questions."""

import csv
import numpy as np
import torch
from .data import OUT, batches


def write_table(name, rows):
    with (OUT / f"{name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0])
        w.writeheader()
        w.writerows(rows)


@torch.inference_mode()
def encode(net, data, ids):
    return np.concatenate([net.encoder(x).cpu().numpy() for x, _, _, _ in batches(data, ids)])
