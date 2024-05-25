import os
import argparse

from deephyper.gnn_uq.analysis import (
    calculate_conf_calib,
    result_to_pickle,
    random_result_to_pickle,
    simple_result_to_pickle,
    simple_random_result_to_pickle,
    mc_dropout_result_to_pickle,
    result_to_csv,
    comp_time
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


def main():
    parser = argparse.ArgumentParser(description="Generate all results.")

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

    # result_to_pickle(DATA_DIR=DATA_DIR, ROOT_DIR=ROOT_DIR, RESULT_DIR=RESULT_DIR)
    # print("# Result saved to pickle files ...")

    # random_result_to_pickle(DATA_DIR=DATA_DIR, ROOT_DIR=ROOT_DIR, RESULT_DIR=RESULT_DIR)
    # print("# Random result saved to pickle files ...")

    # simple_result_to_pickle(DATA_DIR=DATA_DIR, ROOT_DIR=ROOT_DIR, RESULT_DIR=RESULT_DIR)
    # print("# Simple representation result saved to pickle files ...")

    # simple_random_result_to_pickle(
    #     DATA_DIR=DATA_DIR, ROOT_DIR=ROOT_DIR, RESULT_DIR=RESULT_DIR
    # )
    # print("# Simple representation random result saved to pickle files ...")

    # mc_dropout_result_to_pickle(
    #     DATA_DIR=DATA_DIR, ROOT_DIR=ROOT_DIR, RESULT_DIR=RESULT_DIR
    # )
    # print("# MC dropout result saved to pickle files ...")

    # result_to_csv(RESULT_DIR=RESULT_DIR)
    # print("# Metrics saved to csv files ...")

    # calculate_conf_calib(RESULT_DIR=RESULT_DIR)
    # print("# Uncertainty calibration done ...")

    comp_time(ROOT_DIR=ROOT_DIR, RESULT_DIR=RESULT_DIR)
    print("# Time analysis done ...")


if __name__ == "__main__":
    main()
