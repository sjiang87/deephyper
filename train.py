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
from deephyper.gnn_uq.load_data import load_data


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
    """
    Creates and returns an Evaluator object for running the provided `run_function`.

    Args:
        run_function (callable): The function to be executed by the Evaluator.

    Returns:
        Evaluator: An Evaluator object configured based on the availability of GPU resources.

    """
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


def main(seed, dataset, SPLIT_TYPE):
    """
    Main function for executing the Neural Architecture Search process.

    Args:
        seed (int): Seed for random number generation.
        dataset (str): Name of the dataset.
        SPLIT_TYPE (str): Type of data split to be used.

    """
    parser = argparse.ArgumentParser(description="Generate all figure.")

    parser.add_argument("--ROOT_DIR", type=str, help="Root directory")
    parser.add_argument("--SPLIT_TYPE", type=str, help="Split ratio 811 or 523")
    parser.add_argument("--seed", type=int, help="Random seed data split")
    parser.add_argument("--dataset", type=str, help="Dataset [lipo, delaney, qm7, freesolv, qm9]")

    args = parser.parse_args()
    ROOT_DIR = get_dir("./autognnuq/", args.ROOT_DIR)
    SPLIT_TYPE = args.SPLIT_TYPE
    seed = int(args.seed)
    dataset = args.dataset

    print(f"# Your ROOT DIR   : {ROOT_DIR}")
    print(f"# Your dataset    : {dataset}")
    print(f"# Your split ratio: {SPLIT_TYPE}")
    print(f"# Your random seed: {seed}")

    if SPLIT_TYPE == "811":
        splits = (0.8, 0.1, 0.1)
    elif SPLIT_TYPE == "523":
        splits = (0.5, 0.2, 0.3)

    if dataset == "lipo":
        bs = 128
    else:
        bs = 512
    problem = NaProblem()
    problem.load_data(
        load_data, dataset=dataset, sizes=splits, split_type="random", seed=seed
    )
    problem.search_space(RegressionUQSpace)
    problem.hyperparameters(
        batch_size=bs,
        learning_rate=1e-3,
        optimizer="adam",
        num_epochs=30,
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

    regevo_search = RegularizedEvolution(
        problem,
        get_evaluator(run_base_trainer),
        log_dir=os.path.join(
            ROOT_DIR, f"NEW_RE_{dataset}_random_{seed}_split_{SPLIT_TYPE}"
        ),
    )
    regevo_search.search(max_evals=1000)


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
