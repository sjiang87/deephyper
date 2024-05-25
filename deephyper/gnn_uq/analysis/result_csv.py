import os
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from deephyper.gnn_uq.analysis import combine_result


def load_data_(RESULT_DIR):
    file = os.path.join(RESULT_DIR, "val_test_result.pickle")
    file_random = os.path.join(RESULT_DIR, "val_test_result_random.pickle")
    file_simple = os.path.join(RESULT_DIR, "val_test_result_simple.pickle")
    file_mc = os.path.join(RESULT_DIR, "val_test_result_mc_dropout.pickle")
    file_qm9 = os.path.join(RESULT_DIR, "val_test_result_qm9.pickle")
    file_qm9_random = os.path.join(RESULT_DIR, "val_test_result_qm9_random.pickle")
    file_qm9_simple = os.path.join(RESULT_DIR, "val_test_result_qm9_simple.pickle")
    file_qm9_mc = os.path.join(RESULT_DIR, "val_test_result_qm9_mc_dropout.pickle")
 
    with open(file, "rb") as handle:
        result = pickle.load(handle)

    with open(file_random, "rb") as handle:
        result_random = pickle.load(handle)

    with open(file_simple, "rb") as handle:
        result_simple = pickle.load(handle)

    with open(file_mc, "rb") as handle:
        result_mc = pickle.load(handle)


    with open(file_qm9, "rb") as handle:
        result_qm9 = pickle.load(handle)

    with open(file_qm9_random, "rb") as handle:
        result_qm9_random = pickle.load(handle)

    with open(file_qm9_simple, "rb") as handle:
        result_qm9_simple = pickle.load(handle)

    with open(file_qm9_mc, "rb") as handle:
        result_qm9_mc = pickle.load(handle)


    return (
        result,
        result_qm9,
        result_random,
        result_qm9_random,
        result_simple,
        result_qm9_simple,
        result_mc,
        result_qm9_mc,
    )
    
    
def result_to_csv_(RESULT, RESULT_QM9):
    
    output = []

    tasks = ["u0_atom", "lipo", "freesolv", "logSolubility"]
    units = ["log D", "log mol/L", "kcal/mol", "kcal/mol"]

    for i, dataset in enumerate(["lipo", "delaney", "freesolv", "qm7"]):
        metric_total = np.zeros(8)
        for seed in range(8):
            y_test_temp = np.copy(RESULT[(dataset, "523", seed)][0])
            y_pred_temp = np.copy(RESULT[(dataset, "523", seed)][1])

            if dataset == "qm7":
                metric_temp = skm.mean_absolute_error(y_test_temp, y_pred_temp)
                # metric = "MAE"
            else:
                metric_temp = skm.mean_squared_error(y_test_temp, y_pred_temp) ** 0.5
                # metric = "RMSE"

            metric_total[seed] = metric_temp

        nll, cnll, ma, sp = combine_result(RESULT, dataset=dataset, SPLIT_TYPE="523")

        output.append(
            [
                dataset,
                tasks[i],
                units[i],
                metric_total.mean(),
                metric_total.std(),
                nll.mean(),
                nll.std(),
                cnll.mean(),
                cnll.std(),
                ma.mean(),
                ma.std(),
                sp.mean(),
                sp.std(),
            ]
        )


    dataset = "qm9"

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

    units = [
        "D",
        "a_0^3",
        "eV",
        "eV",
        "eV",
        "a_0^2",
        "eV",
        "cal/mol K",
        "kcal/mol",
        "kcal/mol",
        "kcal/mol",
        "kcal/mol",
    ]

    # unit bohr = a_0, 1 hartree = 27.2114 eV

    for i, task in enumerate(tasks):
        metric_total = np.zeros(8)

        for seed in range(8):
            y_test_temp = np.copy(RESULT_QM9[(dataset, "811", seed)][0][:, i])
            y_pred_temp = np.copy(RESULT_QM9[(dataset, "811", seed)][1][:, i])

            if task in ["homo", "lumo", "gap", "zpve"]:
                y_test_temp *= 27.2114
                y_pred_temp *= 27.2114
            metric_temp = skm.mean_absolute_error(y_test_temp, y_pred_temp)
            metric_total[seed] = metric_temp

        nll, cnll, ma, sp = combine_result(
            RESULT_QM9, dataset=dataset, SPLIT_TYPE="811", idx=i
        )

        output.append(
            [
                dataset,
                tasks[i],
                units[i],
                metric_total.mean(),
                metric_total.std(),
                nll.mean(),
                nll.std(),
                cnll.mean(),
                cnll.std(),
                ma.mean(),
                ma.std(),
                sp.mean(),
                sp.std(),
            ]
        )

    df = pd.DataFrame(
        output,
        columns=[
            "dataset",
            "task",
            "unit",
            "err_mean",
            "err_std",
            "nll_mean",
            "nll_std",
            "cnll_mean",
            "cnll_std",
            "ma_mean",
            "ma_std",
            "sp_mean",
            "sp_std",
        ],
    )
    return df

def result_to_csv(RESULT_DIR):
    (
        result,
        result_qm9,
        result_random,
        result_qm9_random,
        result_simple,
        result_qm9_simple,
        result_mc,
        result_qm9_mc,
    ) = load_data_(RESULT_DIR)
    
    
    df = result_to_csv_(result, result_qm9)
    out_file = os.path.join(RESULT_DIR, "metrics.csv")
    df.to_csv(out_file, index=False)

    df = result_to_csv_(result_random, result_qm9_random)
    out_file = os.path.join(RESULT_DIR, "metrics_random.csv")
    df.to_csv(out_file, index=False)

    df = result_to_csv_(result_simple, result_qm9_simple)
    out_file = os.path.join(RESULT_DIR, "metrics_simple.csv")
    df.to_csv(out_file, index=False)

    df = result_to_csv_(result_mc, result_qm9_mc)
    out_file = os.path.join(RESULT_DIR, "metrics_mc.csv")
    df.to_csv(out_file, index=False)
