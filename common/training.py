"""Training never loads the official test batch."""

import csv
import json
import numpy as np
import torch
from torch import nn
from .data import CFG, OUT, DEVICE, SEED, batches, read_data, indices
from .models import classifier, PatchMLP, Autoencoder


def model(name):
    if name.startswith(("ae", "dae")):
        return Autoencoder(int(name.split("_")[1]))
    if name.startswith("patch"):
        return PatchMLP()
    return classifier(name.split("_")[0])


def load(name):
    net = model(name).to(DEVICE)
    net.load_state_dict(torch.load(OUT / f"{name}.pt", map_location=DEVICE, weights_only=True))
    return net.eval()


def evaluate(net, data, ids, condition, ae=False, decoder=None, labels=None):
    net.eval()
    totals = np.zeros(4)
    with torch.inference_mode():
        for x, y, clean, group in batches(data, ids, condition):
            if labels is not None:
                y = labels[group].to(DEVICE)
            out = net(decoder(x) if decoder is not None else x)
            if ae:
                totals[0] += nn.functional.mse_loss(out, clean).item() * len(y)
            else:
                totals[0] += nn.functional.cross_entropy(out, y, reduction="sum").item()
                conf, pred = out.softmax(1).max(1)
                wrong = pred != y
                totals[1:] += [int((~wrong).sum()), float(conf[wrong].sum()), int(wrong.sum())]
    return dict(
        loss=totals[0] / len(ids),
        acc=totals[1] / len(ids),
        wrong_conf=totals[2] / max(1, totals[3]),
        errors=int(totals[3]),
    )


def fit(name, condition, data, train, val, labels=None, flipped=None):
    """Fit on training examples. Save the earliest best validation checkpoint and the full epoch history."""
    log = OUT / f"{name}.csv"
    if log.exists() and (OUT / f"{name}.pt").exists():
        return
    torch.manual_seed(SEED)
    net = model(name).to(DEVICE)
    ae = isinstance(net, Autoencoder)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["learning_rate"])
    epochs = CFG[
        "ae_epochs" if ae else "memorisation_epochs" if name == "memory" else "classifier_epochs"
    ]
    best = float("inf")
    rows = []
    for epoch in range(1, epochs + 1):
        net.train()
        for x, y, clean, group in batches(data, train, condition, True, epoch):
            out = net(x)
            if labels is not None:
                y = labels[group].to(DEVICE)
            loss = nn.functional.mse_loss(out, clean) if ae else nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        va = evaluate(net, data, val, condition, ae)
        tr = evaluate(net, data, train, condition, ae, labels=labels)
        row = dict(
            epoch=epoch,
            train_loss=tr["loss"],
            val_loss=va["loss"],
            parameters=sum(p.numel() for p in net.parameters()),
        )
        if not ae:
            row.update(train_acc=tr["acc"], val_acc=va["acc"])
        # Compare both possible labels on exactly the same images at the end of each epoch.
        if flipped is not None:
            row["flipped_assigned_acc"] = evaluate(net, data, flipped, condition, labels=labels)[
                "acc"
            ]
            row["flipped_original_acc"] = evaluate(net, data, flipped, condition)["acc"]
            row["unchanged_acc"] = evaluate(net, data, np.setdiff1d(train, flipped), condition)[
                "acc"
            ]
        # Minimizing negative accuracy is equivalent to maximizing accuracy.
        score = va["loss"] if ae else -va["acc"]
        if score < best:
            best = score
            torch.save(net.state_dict(), OUT / f"{name}.pt")
        rows.append(row)
        print(name, epoch, round(va["loss"], 5), flush=True)
    with log.with_suffix(".tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    log.with_suffix(".tmp").replace(log)
