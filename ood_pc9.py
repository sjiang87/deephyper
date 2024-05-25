import os
import ast
import pickle
import argparse
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.optimizers.legacy import Adam
from deephyper.gnn_uq.load_data import load_data
from deephyper.gnn_uq.data_utils import split_data, get_data
from deephyper.gnn_uq.gnn_model import RegressionUQSpace, nll
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_dir(default_path, provided_path):
    if provided_path:
        if not os.path.exists(provided_path):
            os.makedirs(provided_path)
        return provided_path
    else:
        if not os.path.exists(default_path):
            os.makedirs(default_path)
        return default_path


def ood_pc9(args, arch, name):
    ROOT_DIR = get_dir("./autognnuq/", args.ROOT_DIR)
    POST_DIR = get_dir("./autognnuq/", args.POST_DIR)
    DATA_DIR = get_dir("./autognnuq/data/", args.DATA_DIR)

    SPLIT_TYPE = args.SPLIT_TYPE
    seed = int(args.seed)
    dataset = "qm9"

    POST_MODEL_DIR = os.path.join(
        POST_DIR,
        f"NEW_POST_MODEL/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
    )
    POST_RESULT_DIR = os.path.join(
        POST_DIR,
        f"NEW_POST_RESULT_PC9/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
    )

    POST_MODEL_H5 = os.path.join(
        POST_MODEL_DIR,
        f"best_{name}.h5",
    )
    POST_RESULT_PICKLE = os.path.join(
        POST_RESULT_DIR,
        f"test_{name}.pickle",
    )

    if not os.path.exists(POST_RESULT_PICKLE):
        tf.keras.backend.clear_session()

        if SPLIT_TYPE == "811":
            sizes = (0.8, 0.1, 0.1)
        elif SPLIT_TYPE == "523":
            sizes = (0.5, 0.2, 0.3)

        tasks = [
            "mu",
            "alpha",
            "homo",
            "lumo",
            "gap",
            "r2",
            "zpve",
            "cv",
            "u0",
            "u298",
            "h298",
            "g298",
        ]

        data = get_data(
            os.path.join(DATA_DIR, f"{dataset}.csv"), tasks, max_data_size=None
        )

        x_train_qm9, y_train_qm9, x_valid_qm9, y_valid_qm9, x_test_qm9, y_test_qm9 = (
            split_data(
                data, split_type="random", sizes=sizes, show_mol=False, seed=seed
            )
        )

        ss = StandardScaler()
        y_train_qm9 = ss.fit_transform(y_train_qm9)
        y_valid_qm9 = ss.transform(y_valid_qm9)
        y_test_qm9 = ss.transform(y_test_qm9)
        mean = ss.mean_
        std = ss.var_**0.5

        ss2 = MinMaxScaler()
        d = x_train_qm9[0]
        d_ = d.reshape(-1, d.shape[-1])
        d_ = ss2.fit_transform(d_)
        d_ = d_.reshape(d.shape)
        x_train_qm9[0] = d_

        d = x_valid_qm9[0]
        d_ = d.reshape(-1, d.shape[-1])
        d_ = ss2.transform(d_)
        d_ = d_.reshape(d.shape)
        x_valid_qm9[0] = d_

        d = x_test_qm9[0]
        d_ = d.reshape(-1, d.shape[-1])
        d_ = ss2.transform(d_)
        d_ = d_.reshape(d.shape)
        x_test_qm9[0] = d_

        ss3 = MinMaxScaler()
        d = x_train_qm9[2]
        d_ = d.reshape(-1, d.shape[-1])
        d_ = ss3.fit_transform(d_)
        d_ = d_.reshape(d.shape)
        x_train_qm9[2] = d_

        d = x_valid_qm9[2]
        d_ = d.reshape(-1, d.shape[-1])
        d_ = ss3.transform(d_)
        d_ = d_.reshape(d.shape)
        x_valid_qm9[2] = d_

        d = x_test_qm9[2]
        d_ = d.reshape(-1, d.shape[-1])
        d_ = ss3.transform(d_)
        d_ = d_.reshape(d.shape)
        x_test_qm9[2] = d_

        ###

        tasks = ["energy", "homo", "lumo", "multi"]

        data = get_data(os.path.join(DATA_DIR, "pc9.csv"), tasks, max_data_size=None)

        x_pc9, y_pc9, _, _, _, _ = split_data(
            data, split_type="random", sizes=(1.0, 0.0, 0.0), show_mol=False, seed=0
        )

        d = x_pc9[0]
        d_ = d.reshape(-1, d.shape[-1])
        d_ = ss2.transform(d_)
        d_ = d_.reshape(d.shape)
        x_pc9[0] = d_

        d = x_pc9[2]
        d_ = d.reshape(-1, d.shape[-1])
        d_ = ss3.transform(d_)
        d_ = d_.reshape(d.shape)
        x_pc9[2] = d_

        arch = ast.literal_eval(arch)

        input_shape = [item.shape[1:] for item in x_train_qm9]
        output_shape = y_train_qm9.shape[1:]
        shapes = dict(input_shape=input_shape, output_shape=output_shape)

        space = RegressionUQSpace(**shapes).build()

        model = space.sample(choice=arch)

        model.compile(loss=nll, optimizer=Adam(learning_rate=0.001))

        model.load_weights(POST_MODEL_H5)
        y_test = y_pc9.squeeze()
        y_preds = []
        y_uncs = []

        batch = int(np.ceil(len(x_pc9[0]) / 2048))

        for i in tqdm(range(batch)):
            x_test_ = [x_pc9[j][i * 2048 : (i + 1) * 2048] for j in range(len(x_pc9))]
            y_dist_ = model(x_test_)
            y_preds.append(y_dist_.loc.numpy())
            y_uncs.append(y_dist_.scale.numpy())

        y_pred = np.concatenate(y_preds, axis=0).squeeze()
        y_unc = np.concatenate(y_uncs, axis=0).squeeze()

        with open(
            POST_RESULT_PICKLE,
            "wb",
        ) as handle:
            pickle.dump(y_test, handle)
            pickle.dump(y_pred, handle)
            pickle.dump(y_unc, handle)
            pickle.dump(mean, handle)
            pickle.dump(std, handle)


def main():
    parser = argparse.ArgumentParser(description="Neural architecture search.")

    parser.add_argument(
        "--ROOT_DIR", type=str, help="Root directory", default="./autognnuq/"
    )
    parser.add_argument(
        "--POST_DIR", type=str, help="Post training directory", default="./autognnuq/"
    )
    parser.add_argument(
        "--DATA_DIR", type=str, help="Data directory", default="./autognnuq/data/"
    )
    parser.add_argument(
        "--SPLIT_TYPE", type=str, help="Split ratio 811 or 523", default="811"
    )
    parser.add_argument("--seed", type=int, help="Random seed data split", default=0)

    args = parser.parse_args()

    ROOT_DIR = get_dir("./autognnuq/", args.ROOT_DIR)
    POST_DIR = get_dir("./autognnuq/", args.POST_DIR)
    DATA_DIR = get_dir("./autognnuq/data/", args.DATA_DIR)

    SPLIT_TYPE = args.SPLIT_TYPE
    seed = int(args.seed)
    dataset = "qm9"

    print(f"# ROOT DIR     : {ROOT_DIR}")
    print(f"# DATA DIR     : {DATA_DIR}")
    print(f"# POST DIR     : {POST_DIR}")

    print(f"# dataset      : {dataset}")
    print(f"# split ratio  : {SPLIT_TYPE}")
    print(f"# random seed  : {seed}")

    MODEL_DIR = os.path.join(
        ROOT_DIR, f"NEW_RE_{dataset}_random_{seed}_split_{SPLIT_TYPE}/save/model/"
    )
    arch_path = MODEL_DIR.split("save")[0] + "results.csv"
    df = pd.read_csv(arch_path)
    loss_min = []
    arch_min = []
    id_min = []
    for i in range(len(df)):
        loss_min_ = np.argsort(df["objective"])[::-1].values[i]
        arch_min_ = df["p:arch_seq"][loss_min_]
        id_min_ = df["job_id"][loss_min_]

        if not any(np.array_equal(arch_min_, x) for x in arch_min):
            loss_min.append(loss_min_)
            arch_min.append(arch_min_)
            id_min.append(id_min_)

    for i in range(10):
        print(f"Model {i + 1} started... previous loss {df['objective'][loss_min_]}")
        ood_pc9(
            args,
            arch=arch_min[i],
            name=id_min[i],
        )


if __name__ == "__main__":
    main()
