import os
import ast
import pickle
import argparse
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers.legacy import Adam
from deephyper.gnn_uq.load_data import load_data
from deephyper.gnn_uq.gnn_model import (
    RegressionUQSpace,
    nll,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def get_dir(default_path, provided_path):
    if provided_path:
        if not os.path.exists(provided_path):
            os.makedirs(provided_path)
        return provided_path
    else:
        if not os.path.exists(default_path):
            os.makedirs(default_path)
        return default_path


def post_train(args, arch, name):
    ROOT_DIR = get_dir("./autognnuq/", args.ROOT_DIR)
    POST_DIR = get_dir("./autognnuq/", args.POST_DIR)
    DATA_DIR = get_dir("./autognnuq/data/", args.DATA_DIR)

    SPLIT_TYPE = args.SPLIT_TYPE
    seed = int(args.seed)
    dataset = args.dataset
    bs = int(args.batch_size)
    lr = float(args.learning_rate)
    epoch = int(args.epoch)
    mode = args.mode

    if mode == "normal":
        POST_MODEL_DIR = os.path.join(
            POST_DIR,
            f"NEW_POST_MODEL/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )
        POST_RESULT_DIR = os.path.join(
            POST_DIR,
            f"NEW_POST_RESULT/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )
    elif mode == "simple":
        POST_MODEL_DIR = os.path.join(
            POST_DIR,
            f"NEW_SIMPLE_MODEL/simplepost_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )
        POST_RESULT_DIR = os.path.join(
            POST_DIR,
            f"NEW_SIMPLE_RESULT/simplepost_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )
    elif mode == "random":
        POST_MODEL_DIR = os.path.join(
            POST_DIR,
            f"NEW_POST_MODEL_RANDOM/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )
        POST_RESULT_DIR = os.path.join(
            POST_DIR,
            f"NEW_POST_RANDOM_RESULT/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )

    POST_MODEL_H5 = os.path.join(
        POST_MODEL_DIR,
        f"best_{name}.h5",
    )
    POST_RESULT_PICKLE = os.path.join(
        POST_RESULT_DIR,
        f"test_{name}.pickle",
    )
    POST_RESULT_VAL_PICKLE = os.path.join(
        POST_RESULT_DIR,
        f"val_{name}.pickle",
    )

    if not os.path.exists(POST_MODEL_DIR):
        os.makedirs(POST_MODEL_DIR)
    if not os.path.exists(POST_RESULT_DIR):
        os.makedirs(POST_RESULT_DIR)

    if not os.path.exists(POST_RESULT_PICKLE):
        tf.keras.backend.clear_session()

        if SPLIT_TYPE == "811":
            sizes = (0.8, 0.1, 0.1)
        elif SPLIT_TYPE == "523":
            sizes = (0.5, 0.2, 0.3)

        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (mean, std) = (
            load_data(
                DATA_DIR=DATA_DIR,
                dataset=dataset,
                test=1,
                split_type="random",
                seed=seed,
                sizes=sizes,
            )
        )

        # turn str of architecture choice to a list
        arch = ast.literal_eval(arch)

        input_shape = [item.shape[1:] for item in x_train]
        output_shape = y_train.shape[1:]
        shapes = dict(input_shape=input_shape, output_shape=output_shape)

        space = RegressionUQSpace(**shapes).build()

        model = space.sample(choice=arch)
        print(model.summary())

        model.compile(loss=nll, optimizer=Adam(learning_rate=lr))

        cp = ModelCheckpoint(
            POST_MODEL_H5,
            monitor="val_loss",
            verbose=2,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )

        es = EarlyStopping(monitor="val_loss", mode="min", patience=200)
        history = model.fit(
            x_train,
            y_train,
            batch_size=bs,
            epochs=epoch,
            callbacks=[cp, es],
            validation_data=(x_valid, y_valid),
            verbose=2,
        ).history
        # make prediction
        model.load_weights(POST_MODEL_H5)
        y_test = y_test.squeeze()
        y_preds = []
        y_uncs = []

        batch = int(np.ceil(len(x_test[0]) / 128))

        for i in range(batch):
            x_test_ = [x_test[j][i * 128 : (i + 1) * 128] for j in range(len(x_test))]
            y_dist_ = model(x_test_)
            y_preds.append(y_dist_.loc.numpy())
            y_uncs.append(y_dist_.scale.numpy())

        y_pred = np.concatenate(y_preds, axis=0).squeeze()
        y_unc = np.concatenate(y_uncs, axis=0).squeeze()

        with open(POST_RESULT_PICKLE, "wb") as handle:
            pickle.dump(y_test, handle)
            pickle.dump(y_pred, handle)
            pickle.dump(y_unc, handle)
            pickle.dump(history, handle)

        y_valid = y_valid.squeeze()

        y_val_preds = []
        y_val_uncs = []

        batch = int(np.ceil(len(x_valid[0]) / 128))

        for i in range(batch):
            x_valid_ = [
                x_valid[j][i * 128 : (i + 1) * 128] for j in range(len(x_valid))
            ]
            y_dist_ = model(x_valid_)
            y_val_preds.append(y_dist_.loc.numpy())
            y_val_uncs.append(y_dist_.scale.numpy())

        y_val_pred = np.concatenate(y_val_preds, axis=0).squeeze()
        y_val_unc = np.concatenate(y_val_uncs, axis=0).squeeze()

        with open(POST_RESULT_VAL_PICKLE, "wb") as handle:
            pickle.dump(y_valid, handle)
            pickle.dump(y_val_pred, handle)
            pickle.dump(y_val_unc, handle)


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
        "--SPLIT_TYPE", type=str, help="Split ratio 811 or 523", default="523"
    )
    parser.add_argument("--seed", type=int, help="Random seed data split", default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        help="lipo, delaney, qm7, freesolv, qm9",
        default="delaney",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate", default=1e-3
    )
    parser.add_argument(
        "--epoch", type=int, help="Number of search epochs", default=1000
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Training mode [normal, simple, ranodm]",
        default="normal",
    )

    args = parser.parse_args()

    ROOT_DIR = get_dir("./autognnuq/", args.ROOT_DIR)
    POST_DIR = get_dir("./autognnuq/", args.POST_DIR)
    DATA_DIR = get_dir("./autognnuq/data/", args.DATA_DIR)

    SPLIT_TYPE = args.SPLIT_TYPE
    seed = int(args.seed)
    dataset = args.dataset
    bs = int(args.batch_size)
    lr = float(args.learning_rate)
    epoch = int(args.epoch)
    mode = args.mode

    print(f"# ROOT DIR     : {ROOT_DIR}")
    print(f"# DATA DIR     : {DATA_DIR}")
    print(f"# POST DIR     : {POST_DIR}")

    print(f"# dataset      : {dataset}")
    print(f"# split ratio  : {SPLIT_TYPE}")
    print(f"# random seed  : {seed}")
    print(f"# batch size   : {bs}")
    print(f"# learning rate: {lr}")
    print(f"# epoch        : {epoch}")
    print(f"# mode         : {mode}")

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
        post_train(args, arch_min[i], id_min[i])


if __name__ == "__main__":
    main()
