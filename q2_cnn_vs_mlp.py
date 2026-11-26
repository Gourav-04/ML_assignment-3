"""Q2. Does the CNN advantage depend on local image structure?

The dense MLP from Q1 has 101,098 weights. The CNN has 100,074. Both use the
same seven epochs, minibatches and Adam settings. The three parts change the
input geometry. Shape, RGB values and sample identities are preserved.

Reference test results (clean / pixel shuffle / tile shuffle):
    MLP  41.57% / 43.21% / 40.59%
    CNN  52.66% / 38.82% / 45.28%
Keeping tiles intact recovered 646 correct CNN decisions relative to pixel
shuffling. Confidence on CNN errors rose from 31.89% to 37.43%. Because
the mistaken images change between conditions, interpret accuracy and
confidence together.
"""

import numpy as np
from common.data import indices, read_data
from common.training import fit, evaluate, load
from common.evidence import write_table


def train_pair(condition, data, train, validation):
    # A new condition has its own checkpoint. In particular the shuffled models
    # learn directly from shuffled training images before they see shuffled test images.
    for kind in ("mlp", "cnn"):
        fit(f"{kind}_{condition}", condition, data, train, validation)


def part_a(data, train, validation):
    """Measure the clean architecture gap. Reuse Q1's dense baseline if available."""
    train_pair("clean", data, train, validation)


def part_b(data, train, validation):
    """Break pixel adjacency using one fixed permutation shared across every split.

    A dense first layer can learn weights for the reordered columns. A convolutional
    filter now repeatedly encounters groups of unrelated pixels. Predict a larger CNN drop.
    """
    train_pair("pixel", data, train, validation)


def part_c(data, train, validation):
    """Move 4x4 tiles intact. Local neighbours within a tile survive the shuffle.

    Recovery over (b) supports local structure as an explanation. The remaining
    gap to (a) can reflect changes in larger shapes and edges at tile boundaries.
    """
    train_pair("tile", data, train, validation)


def prepare(data, split):
    train, validation, _, _ = split
    part_a(data, train, validation)
    part_b(data, train, validation)
    part_c(data, train, validation)


def test_models(test):
    ids = np.arange(len(test[1]))
    rows = []
    for kind in ("mlp", "cnn"):
        for condition in ("clean", "pixel", "tile"):
            # The condition selects the same mapping as in training. wrong_conf
            # averages the largest softmax probability only over incorrect predictions.
            metrics = evaluate(load(f"{kind}_{condition}"), test, ids, condition)
            rows.append(dict(model=kind, condition=condition, **metrics))
            print(
                f"Q2 {kind} {condition}: accuracy {metrics['acc']:.2%}, "
                f"errors {metrics['errors']}, confidence on errors {metrics['wrong_conf']:.2%}"
            )
    write_table("q2_test", rows)
    return rows


if __name__ == "__main__":
    from figures import Q2_CONDITIONS, classifier_curves

    training = read_data()
    prepare(training, indices(training[1]))
    test_models(read_data(test=True))
    classifier_curves(Q2_CONDITIONS, "q2_curves.pdf")
