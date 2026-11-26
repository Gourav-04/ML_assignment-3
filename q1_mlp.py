"""Q1. Why do neighbouring pixels help an MLP, and when does it learn random labels?

Read part_a and part_b in that order. The first compares models; the second
compares successive epochs of one model. A final test score is meaningful
for (a), while (b) specifically asks about behaviour throughout training.

Reference run (seed 26018): contiguous groups scored 52.10% versus 48.40%
for random groups. Compare these two rows to isolate the effect of locality.
The clean validation threshold was crossed at epoch 1 and the assigned-label
threshold at epoch 33. This describes a substantial level of memorisation;
it cannot date the first memorised example.
"""

import csv
import numpy as np
import torch
from common.data import CFG, OUT, SEED, indices, read_data
from common.training import fit, evaluate, load
from common.evidence import write_table


def part_a(data, train, validation):
    """Hold the patch MLP fixed and change which pixels are grouped together."""
    # Dense baseline: flatten the image and feed all coordinates into one layer.
    fit("mlp_clean", "clean", data, train, validation)
    # Shared embedding: 64 groups of 4x4 RGB pixels, each mapped from 48 to 32 features.
    # Both patch models have exactly the same number of weights and receive the same updates.
    for condition in ("clean", "pixel"):
        fit(f"patch_{condition}", condition, data, train, validation)


def part_b(data, memory_ids, validation):
    """Give 25% of a stratified 4,000-image subset fixed, deliberately wrong labels."""
    rng = np.random.default_rng(SEED)
    count = int(len(memory_ids) * CFG["flip_fraction"])
    flipped = rng.choice(memory_ids, count, replace=False)
    assigned = data[1].clone()
    # Adding an offset from 1 to 9 modulo 10 always changes the CIFAR10 class.
    # Keep the original labels in data so we can score both answers on each flipped image.
    offsets = torch.tensor(rng.integers(1, 10, count))
    assigned[flipped] = (assigned[flipped] + offsets) % 10
    np.savez(OUT / "flips.npz", indices=flipped, assigned=assigned[flipped].numpy())
    fit("memory", "clean", data, memory_ids, validation, labels=assigned, flipped=flipped)

    # Validation measures transfer. Predicting the arbitrary assigned label measures
    # memorisation. Compare their first crossings of the prespecified 20% threshold.
    with (OUT / "memory.csv").open() as handle:
        trajectory = list(csv.DictReader(handle))
    for column in ("val_acc", "flipped_assigned_acc"):
        crossing = next((r["epoch"] for r in trajectory if float(r[column]) > 0.20), None)
        print(f"Q1(b) {column}: first epoch above 20% = {crossing}")


def prepare(data, split):
    train, validation, _, memory_ids = split
    part_a(data, train, validation)
    part_b(data, memory_ids, validation)


def test_models(test):
    """Use the selected checkpoints to measure the controlled comparison on official test images."""
    ids = np.arange(len(test[1]))
    rows = []
    for kind, condition in (("mlp", "clean"), ("patch", "clean"), ("patch", "pixel")):
        metrics = evaluate(load(f"{kind}_{condition}"), test, ids, condition)
        rows.append(dict(model=kind, condition=condition, **metrics))
        print(f"Q1(a) {kind} {condition}: {metrics['acc']:.2%}")
    write_table("q1_test", rows)
    return rows


if __name__ == "__main__":
    from figures import Q1_CONDITIONS, classifier_curves, memory_curve

    training = read_data()
    prepare(training, indices(training[1]))
    test_models(read_data(test=True))
    classifier_curves(Q1_CONDITIONS, "q1_curves.pdf")
    memory_curve()
