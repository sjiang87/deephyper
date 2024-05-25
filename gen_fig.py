import os
import argparse

import proplot as pplt
from deephyper.gnn_uq.figure import (
    plot_search_reward,
    plot_parity,
    plot_err_unc,
    plot_conf_calib,
    plot_unc_decomp,
    plot_search_reward_simple,
    plot_uq_metrics,
    plot_conf_curve,
    plot_tsne,
    plot_pc9_all,
    plot_loss_curve,
)


def get_dir(default_path, provided_path):
    if provided_path:
        if not os.path.exists(provided_path):
            os.makedirs(provided_path)
        return provided_path
    else:
        if not os.path.exists(default_path):
            os.makedirs(default_path)
        return default_path


LABEL = ["Lipo", "ESOL", "FreeSolv", "QM7", "QM9"]

COLOR = []

cycle = pplt.Cycle("ggplot")
cycle2 = pplt.Cycle("Dark2")
for c in cycle:
    COLOR.append(c["color"])
for c in cycle2:
    COLOR.append(c["color"])


def main():
    parser = argparse.ArgumentParser(description="Generate all figure.")

    parser.add_argument("--ROOT_DIR", type=str, help="Root directory")
    parser.add_argument("--PLOT_DIR", type=str, help="Plot directory")
    parser.add_argument("--RESULT_DIR", type=str, help="Result directory")
    parser.add_argument("--DATA_DIR", type=str, help="Data directory")

    args = parser.parse_args()

    ROOT_DIR = get_dir("./autognnuq/", args.ROOT_DIR)
    PLOT_DIR = get_dir("./autognnuq/fig/", args.PLOT_DIR)
    RESULT_DIR = get_dir("./autognnuq/result/", args.RESULT_DIR)
    DATA_DIR = get_dir("./autognnuq/data/", args.DATA_DIR)

    print(f"# Your ROOT DIR: {ROOT_DIR}")
    print(f"# Your PLOT DIR: {PLOT_DIR}")
    print(f"# Your RESULT_DIR: {RESULT_DIR}")
    print(f"# Your DATA_DIR: {DATA_DIR}\n")

    # plot_search_reward(
    #     ROOT_DIR=ROOT_DIR, LABEL=LABEL, COLOR=COLOR, PLOT_DIR=PLOT_DIR, format="png"
    # )
    # print("# Search reward plots done...")

    # plot_search_reward_simple(
    #     ROOT_DIR=ROOT_DIR, LABEL=LABEL, COLOR=COLOR, PLOT_DIR=PLOT_DIR, format="png"
    # )
    # print("# Search reward plots with simple representation done...")

    # plot_parity(RESULT_DIR=RESULT_DIR, PLOT_DIR=PLOT_DIR, format="pdf")
    # print("# Parity plots done...")

    # plot_err_unc(RESULT_DIR=RESULT_DIR, PLOT_DIR=PLOT_DIR, COLOR=COLOR, format="pdf")
    # print("# Error vs uncertainty plots done...")

    # plot_conf_calib(RESULT_DIR=RESULT_DIR, PLOT_DIR=PLOT_DIR, COLOR=COLOR, format="png")
    # print("# Confidence-based calibration curve plots done...")

    # plot_unc_decomp(RESULT_DIR=RESULT_DIR, PLOT_DIR=PLOT_DIR, COLOR=COLOR, format="png")
    # print("# Uncertainty decomposition plots done...")

    # plot_uq_metrics(
    #     DATA_DIR=DATA_DIR,
    #     RESULT_DIR=RESULT_DIR,
    #     PLOT_DIR=PLOT_DIR,
    #     COLOR=COLOR,
    #     format="png",
    # )
    # print("# UQ metrics plots done...")

    # plot_conf_curve(RESULT_DIR=RESULT_DIR, PLOT_DIR=PLOT_DIR, COLOR=COLOR, format="png")
    # print("# Confidence curve plots done...")

    # plot_tsne(
    #     ROOT_DIR=ROOT_DIR,
    #     RESULT_DIR=RESULT_DIR,
    #     DATA_DIR=DATA_DIR,
    #     COLOR=COLOR,
    #     PLOT_DIR=PLOT_DIR,
    #     format="png"
    # )

    # print("# t-SNE plots done...")

    # plot_pc9_all(
    #     ROOT_DIR=ROOT_DIR,
    #     RESULT_DIR=RESULT_DIR,
    #     DATA_DIR=DATA_DIR,
    #     LABEL=LABEL,
    #     COLOR=COLOR,
    #     PLOT_DIR=PLOT_DIR,
    #     format="png",
    # )

    # print("# OOD PC9 plots done...")

    plot_loss_curve(
        ROOT_DIR=ROOT_DIR,
        RESULT_DIR=RESULT_DIR,
        COLOR=COLOR,
        PLOT_DIR=PLOT_DIR,
        format="png",
    )
    plot_loss_curve(
        ROOT_DIR=ROOT_DIR,
        RESULT_DIR=RESULT_DIR,
        COLOR=COLOR,
        PLOT_DIR=PLOT_DIR,
        format="png",
        if_simple=True,
    )

    print("# Loss curves done...")


if __name__ == "__main__":
    main()
