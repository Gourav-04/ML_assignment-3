"""Run every question, freezing all choices before opening the official test set."""

import q1_mlp as q1
import q2_cnn_vs_mlp as q2
import q3_autoencoders as q3
import q4_svm_clustering as q4
from common.data import read_data, indices
from common.evidence import write_table
from figures import run as draw_figures


def run():
    data = read_data()
    split = indices(data[1])
    q1.prepare(data, split)
    q2.prepare(data, split)
    probes = q3.prepare(data, split)
    margin_and_clusters = q4.prepare(data, split)

    test = read_data(test=True)
    q1_rows = q1.test_models(test)
    q2_rows = q2.test_models(test)
    # Q1 and Q2 share the same clean MLP. Include it only once in the report table.
    write_table("classifiers_test", q2_rows + [row for row in q1_rows if row["model"] == "patch"])
    q3.test_models(test, probes)
    q4.test_models(data, test, margin_and_clusters)
    draw_figures()


if __name__ == "__main__":
    run()
