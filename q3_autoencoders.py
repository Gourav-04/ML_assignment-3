"""Q3. Three measurements of an encoder: pixel fidelity, recovery under noise,
and class information readable by a linear classifier. Each uses a different target.

Reference run: width 128 minimized validation MSE (0.01132). Denoising
fixed 2,379 CNN errors but spoiled 2,352 correct predictions. The resulting
27-image gain gives little evidence of recovered class information. Probe
accuracy rose from 29.22% to 39.81% across the three widths. The rankings
agree in this sweep; better pixel fidelity alone need not cause better classification.
"""

import json
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common.data import CFG, OUT, batches, indices, read_data
from common.training import fit, evaluate, load
from common.evidence import encode, write_table


def part_a(data, train, validation):
    """Train three clean autoencoders. Choose the width with lowest validation MSE."""
    scores = {}
    for width in CFG["bottlenecks"]:
        name = f"ae_{width}"
        fit(name, "clean", data, train, validation)
        scores[width] = evaluate(load(name), data, validation, "clean", ae=True)["loss"]
    selected = min(scores, key=scores.get)
    (OUT / "selected.json").write_text(
        json.dumps(dict(bottleneck=selected, val_mse=scores), indent=2)
    )
    print(f"Q3(a) validation MSE by width: {scores}; selected {selected}")
    return selected, scores


def part_b(data, train, validation, width):
    """Train a new denoiser at the selected width and obtain the clean Q2 classifier."""
    # During training batches() adds clipped Gaussian noise with SD 0.2. The
    # autoencoder loss compares decoded pixels with the clean image; it ignores class labels.
    fit(f"dae_{width}", "noise", data, train, validation)
    fit("cnn_clean", "clean", data, train, validation)


def part_c(data, train):
    """Fit one probe per frozen encoder, using class labels only for the probe."""
    probes = {}
    for width in CFG["bottlenecks"]:
        features = encode(load(f"ae_{width}"), data, train)
        # One row per image and one column per latent coordinate. The scaler
        # sees training features only. The extracted arrays have no gradient graph.
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1500))
        probes[width] = probe.fit(features, data[1].numpy()[train])
    return probes


def prepare(data, split):
    train, validation, _, _ = split
    width, scores = part_a(data, train, validation)
    part_b(data, train, validation, width)
    return width, scores, part_c(data, train)


@torch.inference_mode()
def test_models(test, prepared):
    width, val_mse, probes = prepared
    ids = np.arange(len(test[1]))
    cnn = load("cnn_clean")
    denoiser = load(f"dae_{width}")
    denoising = []
    for view, decoder in [
        ("corrupted", None),
        ("clean AE", load(f"ae_{width}")),
        ("denoising AE", denoiser),
    ]:
        metrics = evaluate(cnn, test, ids, "noise", decoder=decoder)
        denoising.append(dict(input=view, **metrics))
        print(f"Q3(b) {view}: fixed CNN accuracy {metrics['acc']:.2%}")
    write_table("denoising", denoising)

    # Counting individual changed decisions distinguishes a small consistent gain
    # from thousands of corrections almost cancelled by new mistakes.
    gains = losses = 0
    for noisy, labels, _, _ in batches(test, ids, "noise"):
        before = cnn(noisy).argmax(1) == labels
        after = cnn(denoiser(noisy)).argmax(1) == labels
        gains += int((after & ~before).sum())
        losses += int((before & ~after).sum())
    discordant = gains + losses
    p = binomtest(gains, discordant, 0.5).pvalue if discordant else 1.0
    (OUT / "paired_denoising.json").write_text(
        json.dumps(dict(gains=gains, losses=losses, exact_mcnemar_p=p), indent=2)
    )

    measures = []
    for size, probe in probes.items():
        net = load(f"ae_{size}")
        mse = evaluate(net, test, ids, "clean", ae=True)["loss"]
        accuracy = probe.score(encode(net, test, ids), test[1].numpy())
        measures.append(dict(size=size, val_mse=val_mse[size], test_mse=mse, probe_test=accuracy))
        print(f"Q3(c) width {size}: test MSE {mse:.5f}, frozen probe accuracy {accuracy:.2%}")
    write_table("representation", measures)


if __name__ == "__main__":
    training = read_data()
    state = prepare(training, indices(training[1]))
    test_models(read_data(test=True), state)
