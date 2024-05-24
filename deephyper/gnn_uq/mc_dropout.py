import os
import ast
import pickle
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers.legacy import Adam
from deephyper.gnn_uq.load_data import load_data
from deephyper.gnn_uq.gnn_model import RegressionUQSpace_MCDropout, nll
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


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


def mc_dropout(ROOT_DIR, arch, name, dataset, seed, idx, SPLIT_TYPE):
    if not os.path.exists(
        os.path.join(
            ROOT_DIR,
            f"NEW_POST_MODEL_MC/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )
    ):
        os.makedirs(
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_MODEL_MC/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
            )
        )
    if not os.path.exists(
        os.path.join(
            ROOT_DIR,
            f"NEW_POST_RESULT_MC/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
        )
    ):
        os.makedirs(
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_RESULT_MC/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
            )
        )

    if not os.path.exists(
        os.path.join(
            ROOT_DIR,
            f"NEW_POST_RESULT_MC/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/test_{name}_{idx}.pickle",
        )
    ):
        tf.keras.backend.clear_session()

        if SPLIT_TYPE == "811":
            sizes = (0.8, 0.1, 0.1)
        elif SPLIT_TYPE == "523":
            sizes = (0.5, 0.2, 0.3)

        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (mean, std) = (
            load_data(DATA_DIR=DATA_DIR, dataset=dataset, test=1, split_type="random", seed=seed, sizes=sizes)
        )

        # turn str of architecture choice to a list
        arch = ast.literal_eval(arch)

        # define model input and output shapes
        input_shape = [item.shape[1:] for item in x_train]
        output_shape = y_train.shape[1:]
        shapes = dict(input_shape=input_shape, output_shape=output_shape)

        # define regression uncertainty quantification space
        space = RegressionUQSpace_MCDropout(**shapes).build()

        # load the model with a certain architecture
        model = space.sample(choice=arch)
        print(model.summary())

        if dataset == "lipo":
            batch_size = 256
            epochs = 1000
            patience = 200
            lr = 1e-3
        else:
            batch_size = 512
            epochs = 1000
            patience = 200
            lr = 1e-3  # compile and train the model

        model.compile(loss=nll, optimizer=Adam(learning_rate=lr))

        if dataset == "lipo":
            batch_size = 256
            epochs = 1000
            patience = 200
            lr = 1e-3
        else:
            batch_size = 512
            epochs = 1000
            patience = 200
            lr = 1e-3

        # compile and train the model
        model.compile(loss=nll, optimizer=Adam(learning_rate=lr))
        
        if not os.path.exists(
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_RESULT_MC/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/test_{name}_0.pickle",
            )
        ):
            cp = ModelCheckpoint(
                os.path.join(
                    ROOT_DIR,
                    f"NEW_POST_MODEL_MC/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/best_{name}.h5",
                ),
                monitor="val_loss",
                verbose=2,
                save_best_only=True,
                save_weights_only=True,
                mode="min",
            )

            es = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
            
            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[cp, es],
                validation_data=(x_valid, y_valid),
                verbose=2,
            ).history
        
        # make prediction
        model.load_weights(
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_MODEL_MC/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/best_{name}.h5",
            )
        )
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

        with open(
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_RESULT_MC/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/test_{name}_{idx}.pickle",
            ),
            "wb",
        ) as handle:
            pickle.dump(y_test, handle)
            pickle.dump(y_pred, handle)
            pickle.dump(y_unc, handle)
            # pickle.dump(history, handle)

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

        with open(
            os.path.join(
                ROOT_DIR,
                f"NEW_POST_RESULT_MC/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/val_{name}_{idx}.pickle",
            ),
            "wb",
        ) as handle:
            pickle.dump(y_valid, handle)
            pickle.dump(y_val_pred, handle)
            pickle.dump(y_val_unc, handle)


# def mc_dropout_random(ROOT_DIR, arch, name, dataset, seed, idx, SPLIT_TYPE):
#     if not os.path.exists(
#         os.path.join(
#             ROOT_DIR,
#             f"NEW_POST_RESULT_MC_RANDOM/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
#         )
#     ):
#         os.makedirs(
#             os.path.join(
#                 ROOT_DIR,
#                 f"NEW_POST_RESULT_MC_RANDOM/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/",
#             )
#         )

#     if not os.path.exists(
#         os.path.join(
#             ROOT_DIR,
#             f"NEW_POST_RESULT_MC_RANDOM/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/test_{name}_{idx}.pickle",
#         )
#     ):
#         tf.keras.backend.clear_session()

#         if SPLIT_TYPE == "811":
#             sizes = (0.8, 0.1, 0.1)
#         elif SPLIT_TYPE == "523":
#             sizes = (0.5, 0.2, 0.3)

#         (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (mean, std) = (
#             load_data(
#                 DATA_DIR=DATA_DIR,
#                 dataset=dataset,
#                 test=1,
#                 split_type="random",
#                 seed=seed,
#                 sizes=sizes,
#             )
#         )

#         # turn str of architecture choice to a list
#         arch = ast.literal_eval(arch)

#         # define model input and output shapes
#         input_shape = [item.shape[1:] for item in x_train]
#         output_shape = y_train.shape[1:]
#         shapes = dict(input_shape=input_shape, output_shape=output_shape)

#         # define regression uncertainty quantification space
#         space = RegressionUQSpace_MCDropout(**shapes).build()

#         # load the model with a certain architecture
#         model = space.sample(choice=arch)
#         print(model.summary())

#         if dataset == "lipo":
#             batch_size = 256
#             epochs = 1000
#             patience = 200
#             lr = 1e-3
#         else:
#             batch_size = 512
#             epochs = 1000
#             patience = 200
#             lr = 1e-3  # compile and train the model

#         model.compile(loss=nll, optimizer=Adam(learning_rate=lr))

#         # make prediction
#         model.load_weights(
#             os.path.join(
#                 ROOT_DIR,
#                 f"NEW_POST_MODEL_RANDOM/post_model_{dataset}_random_{seed}_split_{SPLIT_TYPE}/best_{name}.h5",
#             )
#         )
#         y_test = y_test.squeeze()
#         y_preds = []
#         y_uncs = []

#         batch = int(np.ceil(len(x_test[0]) / 128))

#         for i in range(batch):
#             x_test_ = [x_test[j][i * 128 : (i + 1) * 128] for j in range(len(x_test))]
#             y_dist_ = model(x_test_)
#             y_preds.append(y_dist_.loc.numpy())
#             y_uncs.append(y_dist_.scale.numpy())

#         y_pred = np.concatenate(y_preds, axis=0).squeeze()
#         y_unc = np.concatenate(y_uncs, axis=0).squeeze()

#         with open(
#             os.path.join(
#                 ROOT_DIR,
#                 f"NEW_POST_RESULT_MC_RANDOM/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/test_{name}_{idx}.pickle",
#             ),
#             "wb",
#         ) as handle:
#             pickle.dump(y_test, handle)
#             pickle.dump(y_pred, handle)
#             pickle.dump(y_unc, handle)

#         y_valid = y_valid.squeeze()

#         y_val_preds = []
#         y_val_uncs = []

#         batch = int(np.ceil(len(x_valid[0]) / 128))

#         for i in range(batch):
#             x_valid_ = [
#                 x_valid[j][i * 128 : (i + 1) * 128] for j in range(len(x_valid))
#             ]
#             y_dist_ = model(x_valid_)
#             y_val_preds.append(y_dist_.loc.numpy())
#             y_val_uncs.append(y_dist_.scale.numpy())

#         y_val_pred = np.concatenate(y_val_preds, axis=0).squeeze()
#         y_val_unc = np.concatenate(y_val_uncs, axis=0).squeeze()

#         with open(
#             os.path.join(
#                 ROOT_DIR,
#                 f"NEW_POST_RESULT_MC_RANDOM/post_result_{dataset}_random_{seed}_split_{SPLIT_TYPE}/val_{name}_{idx}.pickle",
#             ),
#             "wb",
#         ) as handle:
#             pickle.dump(y_valid, handle)
#             pickle.dump(y_val_pred, handle)
#             pickle.dump(y_val_unc, handle)


def main(
    ROOT_DIR="/scratch/gpfs/sj0161/autognnuq/",
    dataset="delaney",
    topk=1,
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

    for i in range(1):
        print(f"Model {i + 1} started... previous loss {df['objective'][loss_min_]}")
        for idx in range(10):
            mc_dropout(
                ROOT_DIR=ROOT_DIR,
                arch=arch_min[i],
                name=id_min[i],
                dataset=dataset,
                seed=seed,
                idx=idx,
                SPLIT_TYPE=SPLIT_TYPE,
            )

    # id_min_ = df["job_id"][0]
    # arch_min_ = df["p:arch_seq"][0]
    # for idx in range(10):
    #     mc_dropout_random(
    #         ROOT_DIR=ROOT_DIR,
    #         arch=arch_min_,
    #         name=id_min_,
    #         dataset=dataset,
    #         seed=seed,
    #         idx=idx,
    #         SPLIT_TYPE=SPLIT_TYPE
    #     )


def get_combinations_for_index(idx, total_gpus=2):
    # seeds = range(8)
    # datasets = ["qm7", "delaney", "freesolv", "lipo"]
    # split_types = ["523"]

    # combinations = list(itertools.product(seeds, datasets, split_types))

    seeds = range(8)
    datasets = ["qm9"]
    split_types = ["811"]

    combinations2 = list(itertools.product(seeds, datasets, split_types))

    combinations = combinations2 # combinations + combinations2

    combos_per_index = len(combinations) // total_gpus

    start = idx * combos_per_index
    end = start + combos_per_index if idx < total_gpus - 1 else len(combinations)

    return combinations[start:end]


if __name__ == "__main__":
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    max_idx = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1

    for combo in get_combinations_for_index(idx, max_idx):
        seed, dataset, split_type = combo
        main(dataset=dataset, topk=1, seed=seed, SPLIT_TYPE=split_type)
