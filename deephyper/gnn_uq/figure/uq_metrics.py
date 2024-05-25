import os
import proplot as pplt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deephyper.gnn_uq.figure import sat

pplt.rc["errorbar.capsize"] = 0


def benchmark_result(DATA_DIR, dataset, estimator):
    benchmark = pd.read_csv(os.path.join(DATA_DIR, "benchmark_result.csv"))
    benchmark = benchmark[
        (benchmark["Data Set"] == dataset)
        & (benchmark["Estimator"] == estimator)
        & (benchmark["Split"] == "Random Split")
    ]
    return benchmark


def load_data_(RESULT_DIR, DATA_DIR):
    df = pd.read_csv(os.path.join(RESULT_DIR, "metrics.csv"))
    df_simple = pd.read_csv(os.path.join(RESULT_DIR, "metrics_simple.csv"))
    df_random = pd.read_csv(os.path.join(RESULT_DIR, "metrics_random.csv"))
    df_mc = pd.read_csv(os.path.join(RESULT_DIR, "metrics_mc.csv"))

    agu_nll = df["nll_mean"].values
    agu_cnll = df["cnll_mean"].values
    agu_ma = df["ma_mean"].values
    agu_sp = df["sp_mean"].values

    random_nll = df_random["nll_mean"].values
    random_cnll = df_random["cnll_mean"].values
    random_ma = df_random["ma_mean"].values
    random_sp = df_random["sp_mean"].values

    mc_nll = df_mc["nll_mean"].values
    mc_cnll = df_mc["cnll_mean"].values
    mc_ma = df_mc["ma_mean"].values
    mc_sp = df_mc["sp_mean"].values

    simple_nll = df_simple["nll_mean"].values
    simple_cnll = df_simple["cnll_mean"].values
    simple_ma = df_simple["ma_mean"].values
    simple_sp = df_simple["sp_mean"].values

    agu_nll_std = df["nll_std"].values
    agu_cnll_std = df["cnll_std"].values
    agu_ma_std = df["ma_std"].values
    agu_sp_std = df["sp_std"].values

    random_nll_std = df_random["nll_std"].values
    random_cnll_std = df_random["cnll_std"].values
    random_ma_std = df_random["ma_std"].values
    random_sp_std = df_random["sp_std"].values

    mc_nll_std = df_mc["nll_std"].values
    mc_cnll_std = df_mc["cnll_std"].values
    mc_ma_std = df_mc["ma_std"].values
    mc_sp_std = df_mc["sp_std"].values

    simple_nll_std = df_simple["nll_std"].values
    simple_cnll_std = df_simple["cnll_std"].values
    simple_ma_std = df_simple["ma_std"].values
    simple_sp_std = df_simple["sp_std"].values

    ref_nll = np.zeros((4, 8))
    ref_cnll = np.zeros((4, 8))
    ref_ma = np.zeros((4, 8))
    ref_sp = np.zeros((4, 8))

    datasets = ["lipo", "freesolv", "Delaney", "QM7"]

    metrics = {
        "Average NLL": ref_nll,
        "Average Calibrated NLL": ref_cnll,
        "Miscalibration Area": ref_ma,
        "Spearman's Coefficient": ref_sp,
    }

    for index, dataset in enumerate(datasets):
        results = benchmark_result(
            DATA_DIR=DATA_DIR, dataset=dataset, estimator="MPNN Ensemble"
        )
        for metric, array in metrics.items():
            array[index] = results[metric]

    ref_nll = metrics["Average NLL"]
    ref_cnll = metrics["Average Calibrated NLL"]
    ref_ma = metrics["Miscalibration Area"]
    ref_sp = metrics["Spearman's Coefficient"]

    agu_d = [agu_nll, agu_cnll, agu_ma, agu_sp]
    rad_d = [random_nll, random_cnll, random_ma, random_sp]
    mc_d = [mc_nll, mc_cnll, mc_ma, mc_sp]
    simple_d = [simple_nll, simple_cnll, simple_ma, simple_sp]

    ref_d = [
        ref_nll.mean(axis=1),
        ref_cnll.mean(axis=1),
        ref_ma.mean(axis=1),
        ref_sp.mean(axis=1),
    ]

    agu_d_std = [agu_nll_std, agu_cnll_std, agu_ma_std, agu_sp_std]
    rad_d_std = [random_nll_std, random_cnll_std, random_ma_std, random_sp_std]
    mc_d_std = [mc_nll_std, mc_cnll_std, mc_ma_std, mc_sp_std]
    simple_d_std = [simple_nll_std, simple_cnll_std, simple_ma_std, simple_sp_std]

    ref_d_std = [
        ref_nll.std(axis=1),
        ref_cnll.std(axis=1),
        ref_ma.std(axis=1),
        ref_sp.std(axis=1),
    ]

    return (
        agu_d,
        agu_d_std,
        rad_d,
        rad_d_std,
        mc_d,
        mc_d_std,
        ref_d,
        ref_d_std,
        simple_d,
        simple_d_std,
    )


def plot_uq_metrics(DATA_DIR, RESULT_DIR, PLOT_DIR, COLOR, format="pdf"):
    LABEL = ["Lipo", "ESOL", "FreeSolv", "QM7"]

    YLABELS = ["NLL", "cNLL", "Miscalibration Area", "Spearman's Coefficient"]

    (
        agu_d,
        agu_d_std,
        rad_d,
        rad_d_std,
        mc_d,
        mc_d_std,
        ref_d,
        ref_d_std,
        simple_d,
        simple_d_std,
    ) = load_data_(RESULT_DIR=RESULT_DIR, DATA_DIR=DATA_DIR)

    fig, ax = pplt.subplots(nrows=1, ncols=4, sharey=False, refwidth=2.3, refheight=2)

    xx = np.arange(4) * 6

    for i in range(4):
        d1 = agu_d[i][:4]
        d2 = rad_d[i][:4]
        d3 = ref_d[i][:4]
        d4 = mc_d[i][:4]
        d5 = simple_d[i][:4]

        d1_std = agu_d_std[i][:4]
        d2_std = rad_d_std[i][:4]
        d3_std = ref_d_std[i][:4]
        d4_std = mc_d_std[i][:4]
        d5_std = simple_d_std[i][:4]

        if i == 0:
            ax[i].bar(
                xx - 2,
                d1,
                yerr=d1_std,
                width=0.125,
                label="AutoGNNUQ",
                c=COLOR[1],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx - 1,
                d5,
                yerr=d5_std,
                width=0.125,
                label="AutoGNNUQ-Simple",
                c=COLOR[0],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx - 0,
                d4,
                yerr=d4_std,
                width=0.125,
                label="MC Dropout",
                c=COLOR[2],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx + 1,
                d2,
                yerr=d2_std,
                width=0.125,
                label="Random Ensemble",
                c="#4CB384",
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx + 2,
                d3,
                yerr=d3_std,
                width=0.125,
                label="Benchmark",
                c=sat(COLOR[3], 0.3),
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
        else:
            ax[i].bar(
                xx - 2,
                d1,
                yerr=d1_std,
                width=0.125,
                c=COLOR[1],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx - 1,
                d5,
                yerr=d5_std,
                width=0.125,
                c=COLOR[0],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx - 0,
                d4,
                yerr=d4_std,
                width=0.125,
                c=COLOR[2],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx + 1,
                d2,
                yerr=d2_std,
                width=0.125,
                c="#4CB384",
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx + 2,
                d3,
                yerr=d3_std,
                width=0.125,
                c=sat(COLOR[3], 0.2),
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
        ax[i].set_xticks(xx)
        ax[i].set_xticklabels(LABEL)
        ax[i].xaxis.set_minor_locator(plt.NullLocator())
        ax[i].format(
            ylabel=YLABELS[i],
            xticklabelsize=11,
            yticklabelsize=12,
            xlabelsize=13,
            ylabelsize=13,
        )

    fig.legend(loc="b", prop={"size": 12}, ncol=5)

    out_file = os.path.join(PLOT_DIR, f"uq_metric.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)

    #### QM9
    fig, ax = pplt.subplots(nrows=2, ncols=2, sharey=False, refwidth=5.28, refheight=2)

    xx = np.arange(12) * 5

    LABELS_QM9 = [
        r"$\mu$",
        r"$\alpha$",
        r"$\epsilon_{\mathrm{HOMO}}$",
        r"$\epsilon_{\mathrm{LUMO}}$",
        r"$\Delta \epsilon$",
        r"$\langle R^2 \rangle$",
        r"ZPVE",
        r"$c_v$",
        r"$U_0$",
        r"$U$",
        r"$H$",
        r"$G$",
    ]

    for i in range(4):
        d1 = agu_d[i][4:]
        d2 = rad_d[i][4:]
        d4 = mc_d[i][4:]
        d5 = simple_d[i][4:]

        d1_std = agu_d_std[i][4:]
        d2_std = rad_d_std[i][4:]
        d4_std = mc_d_std[i][4:]
        d5_std = simple_d[i][4:]

        if i == 0:
            ax[i].bar(
                xx - 1.5,
                d1,
                yerr=d1_std,
                width=0.15,
                label="AutoGNNUQ",
                c=COLOR[1],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx - 0.5,
                d5,
                yerr=d5_std,
                width=0.15,
                label="AutoGNNUQ-Simple",
                c=COLOR[0],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx + 0.5,
                d4,
                yerr=d4_std,
                width=0.15,
                label="MC Dropout",
                c=COLOR[2],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx + 1.5,
                d2,
                yerr=d2_std,
                width=0.15,
                label="Random Ensemble",
                c="#4CB384",
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
        else:
            ax[i].bar(
                xx - 1.5,
                d1,
                yerr=d1_std,
                width=0.15,
                c=COLOR[1],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx - 0.5,
                d5,
                yerr=d5_std,
                width=0.15,
                c=COLOR[0],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx + 0.5,
                d4,
                yerr=d4_std,
                width=0.15,
                c=COLOR[2],
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )
            ax[i].bar(
                xx + 1.5,
                d2,
                yerr=d2_std,
                width=0.15,
                c="#4CB384",
                error_kw=dict(elinewidth=1, ecolor=COLOR[3]),
            )

        if i == 0 or i == 1:
            yticks = [10, 5, 0, -5]
            ylim = [None, None]
        elif i == 2:
            yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
            ylim = [0, 1]
        elif i == 3:
            yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
            ylim = [-0.1, 1]

        ax[i].set_xticks(xx)
        ax[i].set_xticklabels(LABELS_QM9)
        ax[i].xaxis.set_minor_locator(plt.NullLocator())
        ax[i].format(
            ylabel=YLABELS[i],
            xticklabelsize=12,
            ylim=ylim,
            yticklabelsize=12,
            xlabelsize=13,
            ylabelsize=13,
            yticks=yticks,
            xlim=[-3, 58],
        )

        for label_idx, label in enumerate(ax[i].get_xticklabels()):
            if label_idx in [2, 3, 6]:
                label.set_rotation(45)

    fig.legend(loc="b", prop={"size": 12}, ncol=4)

    out_file = os.path.join(PLOT_DIR, f"uq_metric_qm9.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)
