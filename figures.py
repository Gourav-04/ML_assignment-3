import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")


def rows(name):
    with (R / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


def series(ax, data, x, y, label, **kwargs):
    ax.plot([float(r[x]) for r in data], [float(r[y]) for r in data], label=label, **kwargs)


def classifier_curves(conditions, output):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.4 * len(conditions), 3), sharey=True)
    for ax, name in zip(axes, conditions):
        for condition, color in zip(["clean", "pixel", "tile"], ["#126466", "#a23f26", "#53428a"]):
            if condition not in conditions[name]:
                continue
            d = rows(f"{name}_{condition}")
            for key, line in [("train_acc", "--"), ("val_acc", "-")]:
                series(
                    ax, d, "epoch", key, condition + " " + key.split("_")[0], color=color, ls=line
                )
        ax.set(title=name.upper(), xlabel="Epoch", ylim=(0, 1))
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(R / output)
    plt.close(fig)


def memory_curve():
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for key, label in [
        ("unchanged_acc", "Unchanged labels"),
        ("flipped_original_acc", "Flipped: original label"),
        ("flipped_assigned_acc", "Flipped: assigned label"),
        ("val_acc", "Clean validation"),
    ]:
        series(ax, rows("memory"), "epoch", key, label)
    ax.set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.03))
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(R / "memory.pdf")
    plt.close(fig)


def svm_curves():
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.4), sharey=True)
    for ax, kernel in zip(axes, ["linear", "rbf", "poly"]):
        d = [r for r in rows("svm_sweep") if r["kernel"] == kernel]
        series(ax, d, "C", "val", "Validation accuracy", marker="o")
        series(ax, d, "C", "sv_fraction", "Support-vector fraction", marker="s")
        ax.set(xscale="log", xlabel="C", title=kernel, ylim=(0, 1.04))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(R / "svm.pdf")
    plt.close(fig)


Q1_CONDITIONS = {"patch": ("clean", "pixel"), "mlp": ("clean",)}
Q2_CONDITIONS = {"mlp": ("clean", "pixel", "tile"), "cnn": ("clean", "pixel", "tile")}


def run():
    classifier_curves(Q1_CONDITIONS, "q1_curves.pdf")
    classifier_curves(Q2_CONDITIONS, "q2_curves.pdf")
    classifier_curves({**Q1_CONDITIONS, **Q2_CONDITIONS}, "curves.pdf")
    memory_curve()
    svm_curves()


if __name__ == "__main__":
    run()
