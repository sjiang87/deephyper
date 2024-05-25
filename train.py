import os
import argparse
import itertools
import tensorflow as tf
import ray

from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback
from deephyper.nas.run import run_base_trainer
from deephyper.problem import NaProblem
from deephyper.search.nas import RegularizedEvolution
from deephyper.gnn_uq.gnn_model import RegressionUQSpace, nll
from deephyper.gnn_uq.load_data import load_data, load_data_simple


def get_dir(default_path, provided_path):
    if provided_path:
        if not os.path.exists(provided_path):
            os.makedirs(provided_path)
        return provided_path
    else:
        if not os.path.exists(default_path):
            os.makedirs(default_path)
        return default_path


def get_evaluator(run_function):
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [LoggerCallback()],
    }

    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function, method="ray", method_kwargs=method_kwargs
    )
    print(
        f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}",
    )

    return evaluator


def main():
    parser = argparse.ArgumentParser(description="Neural architecture search.")

    parser.add_argument(
        "--ROOT_DIR", type=str, help="Root directory", default="./autognnuq/"
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
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--epoch", type=int, help="Number of search epochs", default=30)
    parser.add_argument(
        "--simple", type=int, help="Simple representation 1 or not 0", default=1
    )
    parser.add_argument(
        "--max_eval",
        type=int,
        help="Maximum number of architecture evaluations",
        default=1000,
    )

    args = parser.parse_args()
    ROOT_DIR = get_dir("./autognnuq/", args.ROOT_DIR)
    DATA_DIR = get_dir("./autognnuq/data/", args.DATA_DIR)

    SPLIT_TYPE = args.SPLIT_TYPE
    seed = int(args.seed)
    dataset = args.dataset
    bs = int(args.batch_size)
    lr = float(args.learning_rate)
    epoch = int(args.epoch)
    simple = int(args.simple)
    max_eval = int(args.max_eval)

    print(f"# ROOT DIR     : {ROOT_DIR}")
    print(f"# DATA DIR     : {DATA_DIR}")

    print(f"# dataset      : {dataset}")
    print(f"# split ratio  : {SPLIT_TYPE}")
    print(f"# random seed  : {seed}")
    print(f"# batch size   : {bs}")
    print(f"# learning rate: {lr}")
    print(f"# epoch        : {epoch}")
    print(f"# simple repre : {simple==1}")
    print(f"# max eval     : {max_eval}")

    if SPLIT_TYPE == "811":
        splits = (0.8, 0.1, 0.1)
    elif SPLIT_TYPE == "523":
        splits = (0.5, 0.2, 0.3)

    problem = NaProblem()

    if simple == 1:
        problem.load_data(
            load_data_simple,
            DATA_DIR=DATA_DIR,
            dataset=dataset,
            sizes=splits,
            split_type="random",
            seed=seed,
        )
    else:
        problem.load_data(
            load_data,
            DATA_DIR=DATA_DIR,
            dataset=dataset,
            sizes=splits,
            split_type="random",
            seed=seed,
        )

    problem.search_space(RegressionUQSpace)
    problem.hyperparameters(
        batch_size=bs,
        learning_rate=lr,
        optimizer="adam",
        num_epochs=epoch,
        callbacks=dict(
            EarlyStopping=dict(monitor="val_loss", mode="min", verbose=0, patience=30),
            ModelCheckpoint=dict(
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=0,
                filepath="model.h5",
                save_weights_only=True,
            ),
        ),
    )

    problem.loss(nll)
    problem.metrics(["mae"])
    problem.objective("-val_loss")

    if simple == 1:
        regevo_search = RegularizedEvolution(
            problem,
            get_evaluator(run_base_trainer),
            log_dir=os.path.join(
                ROOT_DIR, f"SIMPLE_RE_{dataset}_random_{seed}_split_{SPLIT_TYPE}"
            ),
        )
    else:
        regevo_search = RegularizedEvolution(
            problem,
            get_evaluator(run_base_trainer),
            log_dir=os.path.join(
                ROOT_DIR, f"NEW_RE_{dataset}_random_{seed}_split_{SPLIT_TYPE}"
            ),
        )
    regevo_search.search(max_evals=max_eval)


if __name__ == "__main__":
    available_gpus = tf.config.list_physical_devices("GPU")
    n_gpus = len(available_gpus)
    is_gpu_available = n_gpus > 0

    if is_gpu_available:
        print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
    else:
        print("No GPU available")

    if not (ray.is_initialized()):
        if is_gpu_available:
            ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)
        else:
            ray.init(num_cpus=4, log_to_driver=False)

        main()
