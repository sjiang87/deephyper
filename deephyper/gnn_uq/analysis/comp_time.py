import os
import pickle
import glob
import argparse
import numpy as np
import pandas as pd


def get_dir(default_path, provided_path):
    if provided_path:
        if not os.path.exists(provided_path):
            os.makedirs(provided_path)
        return provided_path
    else:
        if not os.path.exists(default_path):
            os.makedirs(default_path)
        return default_path


def comp_time(ROOT_DIR, RESULT_DIR):

    csv_out1 = []
    csv_out2 = []

    for dataset in ["lipo", "delaney", "freesolv", "qm7", "qm9"]:
        t_search = []
        if dataset == "qm9":
            split = "811"
        else:
            split = "523"
        for i in range(8):
            df = pd.read_csv(
                os.path.join(
                    ROOT_DIR, f"NEW_RE_{dataset}_random_{i}_split_{split}/results.csv"
                )
            )

            dt = df["m:timestamp_gather"].values[-1] / 60 / 60

            t_search.append(dt)

        t_search = np.array(t_search)

        # print(
        #     f"{dataset:<8} time (hour): {t_search.mean():0.2f} ({t_search.std():0.2f})"
        # )

        csv_out1.append([dataset, t_search.mean(), t_search.std()])

    for dataset in ["lipo", "delaney", "freesolv", "qm7", "qm9"]:
        t_post = []
        
        if dataset == "qm9":
            split = "811"
        else:
            split = "523"
        for i in range(8):

            files = glob.glob(
                os.path.join(
                    ROOT_DIR,
                    f"NEW_POST_RESULT/post_result_{dataset}_random_{i}_split_{split}/test_*.pickle",
                )
            )

            ts = []
            for file in files:
                ts.append(os.path.getctime(file))

            t_post.append((np.max(ts) - np.min(ts)) / (len(ts) - 1))

        t_post = np.array(t_post) / 60

        # print(f"{dataset:<8} time (min): {t_post.mean():0.2f} ({t_post.std():0.2f})")

        csv_out2.append([dataset, t_post.mean(), t_post.std()])

    df1 = pd.DataFrame(
        csv_out1,
        columns=["dataset", "time_mean_hour", "time_std_hour"],
    )

    out_csv_file1 = os.path.join(RESULT_DIR, "nas_search_time.csv")
    df1.to_csv(out_csv_file1, index=False)

    df2 = pd.DataFrame(
        csv_out2,
        columns=["dataset", "time_mean_minute", "time_std_minute"],
    )

    out_csv_file2 = os.path.join(RESULT_DIR, "nas_post_training_time.csv")
    df2.to_csv(out_csv_file2, index=False)
