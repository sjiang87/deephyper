import os
import pickle
from deephyper.gnn_uq.analysis import (
    get_result,
    get_result_random,
    get_result_random_simple,
    get_result_simple,
    get_result_mc_dropout,
    get_result_mc_dropout_random,
)


def result_to_pickle(ROOT_DIR, RESULT_DIR, DATA_DIR):
    # qm9
    datasets = ["qm9"]

    split_types = ["811"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_qm9.pickle")

    if os.path.exists(result_file):
        print("# QM9 result file exists ...")

    else:
        print("# QM9 result file does not exist ...")
        out_result = get_result(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)

    # other
    datasets = ["lipo", "delaney", "freesolv", "qm7"]
    split_types = ["523", "811"]
    range_seeds = range(8)
    
    result_file = os.path.join(RESULT_DIR, "val_test_result.pickle")

    if os.path.exists(result_file):
        print("# Lipo, ESOL, FreeSOlv and QM7 result file exists ...")

    else:
        print("# Lipo, ESOL, FreeSOlv and QM7 result file does not exist ...")
        out_result = get_result(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)
            
            
def random_result_to_pickle(ROOT_DIR, RESULT_DIR, DATA_DIR):
    # qm9
    datasets = ["qm9"]

    split_types = ["811"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_qm9_random.pickle")

    if os.path.exists(result_file):
        print("# QM9 result file exists ...")

    else:
        print("#QM9 result file does not exist ...")
        out_result = get_result_random(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)

    # other
    datasets = ["lipo", "delaney", "freesolv", "qm7"]
    split_types = ["523"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_random.pickle")

    if os.path.exists(result_file):
        print("# Lipo, ESOL, FreeSOlv and QM7 result file exists ...")

    else:
        print("# Lipo, ESOL, FreeSOlv and QM7 result file does not exist ...")
        out_result = get_result_random(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)


def simple_random_result_to_pickle(ROOT_DIR, RESULT_DIR, DATA_DIR):
    # qm9
    datasets = ["qm9"]

    split_types = ["811"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_qm9_random_simple.pickle")

    if os.path.exists(result_file):
        print("# QM9 result file exists ...")

    else:
        print("#QM9 result file does not exist ...")
        out_result = get_result_random_simple(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)

    # other
    datasets = ["lipo", "delaney", "freesolv", "qm7"]
    split_types = ["523"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_random_simple.pickle")

    if os.path.exists(result_file):
        print("# Lipo, ESOL, FreeSOlv and QM7 result file exists ...")

    else:
        print("# Lipo, ESOL, FreeSOlv and QM7 result file does not exist ...")
        out_result = get_result_random_simple(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)
            
            
def simple_result_to_pickle(ROOT_DIR, RESULT_DIR, DATA_DIR):
    # qm9
    datasets = ["qm9"]

    split_types = ["811"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_qm9_simple.pickle")

    if os.path.exists(result_file):
        print("# QM9 result file exists ...")

    else:
        print("#QM9 result file does not exist ...")
        out_result = get_result_simple(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)

    # other
    datasets = ["lipo", "delaney", "freesolv", "qm7"]
    split_types = ["523"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_simple.pickle")

    if os.path.exists(result_file):
        print("# Lipo, ESOL, FreeSOlv and QM7 result file exists ...")

    else:
        print("# Lipo, ESOL, FreeSOlv and QM7 result file does not exist ...")
        out_result = get_result_simple(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)
            
def mc_dropout_result_to_pickle(ROOT_DIR, RESULT_DIR, DATA_DIR):
    # qm9
    datasets = ["qm9"]

    split_types = ["811"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_qm9_mc_dropout.pickle")

    if os.path.exists(result_file):
        print("# QM9 result file exists ...")

    else:
        print("#QM9 result file does not exist ...")
        out_result = get_result_mc_dropout(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)

    # other
    datasets = ["lipo", "delaney", "freesolv", "qm7"]
    split_types = ["523"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_mc_dropout.pickle")

    if os.path.exists(result_file):
        print("# Lipo, ESOL, FreeSOlv and QM7 result file exists ...")

    else:
        print("# Lipo, ESOL, FreeSOlv and QM7 result file does not exist ...")
        out_result = get_result_mc_dropout(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)
            
            
def mc_dropout_random_result_to_pickle(ROOT_DIR, RESULT_DIR, DATA_DIR):
    # qm9
    datasets = ["qm9"]

    split_types = ["811"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_qm9_mc_dropout_random.pickle")

    if os.path.exists(result_file):
        print("# QM9 result file exists ...")

    else:
        print("#QM9 result file does not exist ...")
        out_result = get_result_mc_dropout_random(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)

    # other
    datasets = ["lipo", "delaney", "freesolv", "qm7"]
    split_types = ["523"]
    range_seeds = range(8)

    result_file = os.path.join(RESULT_DIR, "val_test_result_mc_dropout_random.pickle")

    if os.path.exists(result_file):
        print("# Lipo, ESOL, FreeSOlv and QM7 result file exists ...")

    else:
        print("# Lipo, ESOL, FreeSOlv and QM7 result file does not exist ...")
        out_result = get_result_mc_dropout_random(
            DATA_DIR=DATA_DIR,
            ROOT_DIR=ROOT_DIR,
            datasets=datasets,
            split_types=split_types,
            range_seeds=range_seeds,
        )

        with open(result_file, "wb") as handle:
            pickle.dump(out_result, handle)
