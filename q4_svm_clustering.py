"""Q4. Supervised margins and unsupervised clusters in the same frozen feature space.
Read part_a (SVM selection), part_b (boundary inspection) and part_c (cluster selection).
prepare() fits everything on training/validation; test_models() scores the locked choices.

Reference run: RBF with C=1 reached 42.79% test accuracy. Cat/dog was
the most confused pair (353 direct swaps). Eight clusters gave ARI 0.049
and NMI 0.082. Cat and deer cluster distributions overlap by 90.4%, yet
the SVM directly confuses them on just 153 of their 2,000 test images.
The code below derives these statistics from the learned representations.
"""

import itertools
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common.data import CFG, OUT, SEED, indices, read_data
from common.training import load
from common.evidence import encode, write_table
from q3_autoencoders import part_a as select_autoencoder

CLASSES = "airplane automobile bird cat deer dog frog horse ship truck".split()


def part_a(training_features, train_labels, validation_features, val_labels):
    """Fit 15 SVMs on one fixed subset. Select the best validation accuracy.
    Strict improvement keeps the first configuration in the event of a tie.
    sv_fraction counts each training image once even if it supports several pairs.
    """
    sweep = []
    best = None
    for kernel, c in itertools.product(CFG["kernels"], CFG["C"]):
        svm = SVC(
            C=c,
            kernel=kernel,
            gamma="scale",
            degree=3,
            decision_function_shape="ovo",
            cache_size=400,
        )
        svm.fit(training_features, train_labels)
        val = svm.score(validation_features, val_labels)
        print(f"{kernel}, C={c:g}, validation={val:.4f}", flush=True)
        if best is None or val > best[0]:
            best = val, svm
        sweep.append(
            dict(
                kernel=kernel,
                C=c,
                train=svm.score(training_features, train_labels),
                val=val,
                sv_fraction=len(svm.support_) / len(training_features),
            )
        )
    write_table("svm_sweep", sweep)
    return best[1]


def part_b(svm, training_features, test_features, subset_ids, data, test, predictions):
    """Select the three most confused pairs and inspect each binary boundary.

    A point may support one class pair but not another. The dual coefficient
    filter keeps only support vectors participating in this pair. A decision
    value near zero measures proximity to this boundary in score space.
    """
    train_y = data[1].numpy()[subset_ids]
    y = test[1].numpy()
    confusion = confusion_matrix(y, predictions, labels=np.arange(10))
    np.savetxt(OUT / "svm_confusion.csv", confusion, fmt="%d", delimiter=",")
    pairs = list(itertools.combinations(range(10), 2))
    difficult = sorted(pairs, key=lambda p: -(confusion[p] + confusion[p[::-1]]))[:3]
    # In libsvm, pair (i,j) uses row j-1 for class i SVs and row i for class j SVs.
    support_y = train_y[svm.support_]
    decisions = svm.decision_function(test_features)
    train_scores = svm.decision_function(training_features[svm.support_])
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(9, 7))
    for row, (i, j) in enumerate(difficult):
        pair_column = pairs.index((i, j))
        for offset, label in enumerate((i, j)):
            dual_row = j - 1 if label == i else i
            eligible = np.flatnonzero((support_y == label) & (svm.dual_coef_[dual_row] != 0))
            nearest_support = eligible[np.argmin(abs(train_scores[eligible, pair_column]))]
            candidate_test_ids = np.flatnonzero(y == label)
            nearest_test = candidate_test_ids[
                np.argmin(abs(decisions[candidate_test_ids, pair_column]))
            ]
            for side, image_id, image, score in (
                (
                    "SV",
                    int(subset_ids[svm.support_[nearest_support]]),
                    data[0][subset_ids[svm.support_[nearest_support]]],
                    train_scores[nearest_support, pair_column],
                ),
                (
                    "test",
                    int(nearest_test),
                    test[0][nearest_test],
                    decisions[nearest_test, pair_column],
                ),
            ):
                ax = axes[row, offset + (2 if side == "test" else 0)]
                ax.imshow(image.permute(1, 2, 0))
                ax.set_title(f"{side}: {CLASSES[label]}\nf={score:.2g}", fontsize=9)
                ax.axis("off")
                rows.append(
                    dict(
                        class_i=i,
                        class_j=j,
                        errors_ij=int(confusion[i, j]),
                        errors_ji=int(confusion[j, i]),
                        source=side,
                        image_id=image_id,
                        true=label,
                        score=float(score),
                        predicted=int(predictions[nearest_test]) if side == "test" else "",
                    )
                )
    fig.tight_layout()
    fig.savefig(OUT / "pairs.pdf")
    plt.close(fig)
    write_table("pairs", rows)


def part_c(training_features, validation_features):
    """Fit each k-means on training features. Validation silhouette needs no class labels."""
    clustering = []
    best_cluster = None
    for k in CFG["cluster_counts"]:
        kmeans = KMeans(k, n_init=10, random_state=SEED).fit(training_features)
        assigned_clusters = kmeans.predict(validation_features)
        silhouette = silhouette_score(
            validation_features,
            assigned_clusters,
            sample_size=min(2000, len(validation_features)),
            random_state=SEED,
        )
        clustering.append(dict(k=k, silhouette=silhouette))
        if best_cluster is None or silhouette > best_cluster[0]:
            best_cluster = silhouette, kmeans
    write_table("cluster_selection", clustering)
    return best_cluster[1]


def prepare(data, split):
    train, validation, subset, _ = split
    width, _ = select_autoencoder(data, train, validation)
    encoder = load(f"ae_{width}")
    features = encode(encoder, data, subset)
    scaler = StandardScaler().fit(features)
    training_features = scaler.transform(features)
    validation_features = scaler.transform(encode(encoder, data, validation))
    labels = data[1].numpy()
    svm = part_a(training_features, labels[subset], validation_features, labels[validation])
    clusters = part_c(training_features, validation_features)
    return encoder, scaler, svm, clusters, training_features, subset


def test_models(data, test, prepared):
    encoder, scaler, svm, clusters, training_features, subset = prepared
    ids = np.arange(len(test[1]))
    test_features = scaler.transform(encode(encoder, test, ids))
    # Memberships and class predictions are now fixed; labels enter only to score them.
    groups = clusters.predict(test_features)
    predictions = svm.predict(test_features)
    labels = test[1].numpy()
    part_b(svm, training_features, test_features, subset, data, test, predictions)
    table = np.zeros((clusters.n_clusters, 10), dtype=int)
    np.add.at(table, (groups, labels), 1)
    np.savetxt(OUT / "contingency.csv", table, fmt="%d", delimiter=",")
    summary = dict(
        kernel=svm.kernel,
        C=svm.C,
        test_acc=float(np.mean(predictions == labels)),
        k=clusters.n_clusters,
        ARI=adjusted_rand_score(labels, groups),
        NMI=normalized_mutual_info_score(labels, groups),
    )
    (OUT / "bonus.json").write_text(json.dumps(summary, indent=2))
    # Compare distributions over clusters for two ground-truth classes. The overlap
    # sums the shared probability mass; confusion counts only direct SVM swaps.
    confusion = confusion_matrix(labels, predictions, labels=np.arange(10))
    comparisons = []
    for i, j in itertools.combinations(range(10), 2):
        overlap = np.minimum(table[:, i] / table[:, i].sum(), table[:, j] / table[:, j].sum()).sum()
        swap_rate = (confusion[i, j] + confusion[j, i]) / (confusion[i].sum() + confusion[j].sum())
        comparisons.append(
            dict(first=CLASSES[i], second=CLASSES[j], overlap=overlap, svm_swap_rate=swap_rate)
        )
    write_table("q4_class_pairs", comparisons)
    print("Q4 selected models and test scores:", summary)


if __name__ == "__main__":
    from figures import svm_curves

    training = read_data()
    state = prepare(training, indices(training[1]))
    test_models(training, read_data(test=True), state)
    svm_curves()
