import os
import ast
import pickle
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


if os.path.exists("/scratch/gpfs/sj0161/autognnuq/"):
    ROOT_DIR = "/scratch/gpfs/sj0161/autognnuq/"
else:
    ROOT_DIR = "../autognnuq/"

if os.path.exists("/scratch/gpfs/sj0161/autognnuq/fig/"):
    PLOT_DIR = "/scratch/gpfs/sj0161/autognnuq/fig/"
else:
    PLOT_DIR = "../autognnuq/fig/"

if os.path.exists("/scratch/gpfs/sj0161/autognnuq/result/"):
    RESULT_DIR = "/scratch/gpfs/sj0161/autognnuq/result/"
else:
    RESULT_DIR = "../autognnuq/result/"

if os.path.exists("/scratch/gpfs/sj0161/autognnuq/data/"):
    DATA_DIR = "/scratch/gpfs/sj0161/autognnuq/data/"
else:
    DATA_DIR = "../autognnuq/data/"


def ood_pc9(ROOT_DIR, arch, name, dataset, seed, SPLIT_TYPE):
    if not os.path.exists(
        os.path.join(
            ROOT_DIR,
            f"NEW_POST_RESULT_PC9/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )
    ):
        os.makedirs(
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_RESULT_PC9/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
            )
        )

    if not os.path.exists(
        os.path.join(
            ROOT_DIR,
            f"NEW_POST_RESULT_PC9/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/test_{name}.pickle",
        )
    ):
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

        data = get_data(os.path.join(DATA_DIR, f"{dataset}.csv"), tasks, max_data_size=None)

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

        model.load_weights(
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_MODEL/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/best_{name}.h5",
            )
        )
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
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_RESULT_PC9/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/test_{name}.pickle",
            ),
            "wb",
        ) as handle:
            pickle.dump(y_test, handle)
            pickle.dump(y_pred, handle)
            pickle.dump(y_unc, handle)
            pickle.dump(mean, handle)
            pickle.dump(std, handle)


def main(
    ROOT_DIR="/scratch/gpfs/sj0161/autognnuq/",
    dataset="qm9",
    topk=10,
    seed=0,
    SPLIT_TYPE="811",
):
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

    for i in range(topk):
        print(f"Model {i + 1} started... previous loss {df['objective'][loss_min_]}")
        ood_pc9(
            ROOT_DIR=ROOT_DIR,
            arch=arch_min[i],
            name=id_min[i],
            dataset=dataset,
            seed=seed,
            SPLIT_TYPE=SPLIT_TYPE,
        )


def get_combinations_for_index(idx, total_gpus=2):
    seeds = [3,4,5,6,7]
    datasets = ["qm9"]
    split_types = ["811"]

    combinations2 = list(itertools.product(seeds, datasets, split_types))

    combinations = combinations2  # combinations + combinations2

    combos_per_index = len(combinations) // total_gpus

    start = idx * combos_per_index
    end = start + combos_per_index if idx < total_gpus - 1 else len(combinations)

    return combinations[start:end]


if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_idx = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1

    for combo in get_combinations_for_index(idx, max_idx):
        seed, dataset, split_type = combo
        main(dataset=dataset, topk=10, seed=seed, SPLIT_TYPE=split_type)
