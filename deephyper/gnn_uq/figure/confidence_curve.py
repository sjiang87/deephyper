import os
import pickle
import proplot as pplt
import numpy as np
from deephyper.gnn_uq.figure import sat

def load_data_(RESULT_DIR):
    with open(os.path.join(RESULT_DIR, "val_test_result.pickle"), "rb") as handle:
        result = pickle.load(handle)
    
    with open(os.path.join(RESULT_DIR, "val_test_result_qm9.pickle"), "rb") as handle:
        result_qm9 = pickle.load(handle)
        
    return result, result_qm9


def conf_curve_(err, unc, mode="rmse"):
    idx = np.argsort(-unc)

    err = err[idx]
    unc = unc[idx]

    cum_err = []

    percent = np.arange(0, 101, 1)

    for p in percent:
        n = int(p * len(err) / 101)
        if mode == "rmse":
            err_mean = np.mean(err[n:] ** 2) ** 0.5
        elif mode == "mae":
            err_mean = np.mean(err[n:])
        cum_err.append(err_mean)

    return percent, np.array(cum_err)


def plot_conf_curve(RESULT_DIR, PLOT_DIR, COLOR, format="pdf"):
    LABELS = [
        "Lipo" + "\n" + "[log D]",
        "ESOL" + "\n" + "[log mol/L]",
        "FreeSolv" + "\n" + "[kcal/mol]",
        "QM7" + "\n" + "[kcal/mol]",
        r"QM9 $\mu$" + "\n" + "[D]",
        r"QM9 $\alpha$" + "\n" + r"[$a_0^3$]",
        r"QM9 $\epsilon_{\mathrm{HOMO}}$" + "\n" + "[eV]",
        r"QM9 $\epsilon_{\mathrm{LUMO}}$" + "\n" + "[eV]",
        r"QM9 $\Delta \epsilon$" + "\n" + "[eV]",
        r"QM9 $\langle R^2 \rangle$" + "\n" + r"[$a_0^2$]",
        r"QM9 ZPVE" + "\n" + "[eV]",
        r"QM9 $c_v$" + "\n" + "[cal/mol$\cdot$K]",
        r"QM9 $U_0$" + "\n" + "[kcal/mol]",
        r"QM9 $U$" + "\n" + "[kcal/mol]",
        r"QM9 $H$" + "\n" + "[kcal/mol]",
        r"QM9 $G$" + "\n" + "[kcal/mol]",
    ]
    
    result, result_qm9 = load_data_(RESULT_DIR=RESULT_DIR)

    fig, ax = pplt.subplots(refheight=2, refwidth=2.5, ncols=4, nrows=4, sharey=False)

    for i in range(16):
        err_ale = np.zeros((8, 101))
        err_epi = np.zeros((8, 101))
        err_tot = np.zeros((8, 101))
        oracle = np.zeros((8, 101))

        if i == 0:
            dataset = "lipo"
        elif i == 1:
            dataset = "delaney"
        elif i == 2:
            dataset = "freesolv"
        elif i == 3:
            dataset = "qm7"
        else:
            dataset = "qm9"

        if i >= 3:
            mode = "mae"
            ylabel = "MAE"
        else:
            mode = "rmse"
            ylabel = "RMSE"

        for seed in range(8):
            if dataset == "qm9":
                split = "811"

                y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = result_qm9[
                    dataset, split, seed
                ]
                y_true = np.copy(y_true)[..., i - 4]
                y_pred = np.copy(y_pred)[..., i - 4]
                y_alea = np.copy(y_alea)[..., i - 4]
                y_epis = np.copy(y_epis)[..., i - 4]

                if i in [6, 7, 8, 10]:
                    y_true = y_true * 27.2114
                    y_pred = y_pred * 27.2114
                    y_alea = y_alea * (27.2114**2)
                    y_epis = y_epis * (27.2114**2)

            else:
                split = "523"

                y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = result[
                    dataset, split, seed
                ]

            err = np.abs(y_true - y_pred)

            p, oracle[seed] = conf_curve_(err, err, mode=mode)
            _, err_ale[seed] = conf_curve_(err, y_alea**0.5, mode=mode)
            _, err_epi[seed] = conf_curve_(err, y_epis**0.5, mode=mode)
            _, err_tot[seed] = conf_curve_(err, (y_alea + y_epis) ** 0.5, mode=mode)

        ax[i].plot(p, err_tot.mean(axis=0), label="Total", c=COLOR[0])
        ax[i].fill_between(
            p,
            err_tot.mean(axis=0) - err_tot.std(axis=0),
            err_tot.mean(axis=0) + err_tot.std(axis=0),
            c=sat(COLOR[0], 0.85),
        )

        ax[i].plot(p, err_ale.mean(axis=0), label="Aleatoric", c=COLOR[1])
        ax[i].fill_between(
            p,
            err_ale.mean(axis=0) - err_ale.std(axis=0),
            err_ale.mean(axis=0) + err_ale.std(axis=0),
            c=sat(COLOR[1], 0.85),
        )

        ax[i].plot(p, err_epi.mean(axis=0), label="Epistemic", c=COLOR[2])
        ax[i].fill_between(
            p,
            err_epi.mean(axis=0) - err_epi.std(axis=0),
            err_epi.mean(axis=0) + err_epi.std(axis=0),
            c=sat(COLOR[2], 0.85),
        )

        ax[i].plot(p, oracle.mean(axis=0), "k--", label="Oracle")

        ax[i].fill_between(
            p,
            oracle.mean(axis=0) - oracle.std(axis=0),
            oracle.mean(axis=0) + oracle.std(axis=0),
            c=sat("#000000", 0.85),
        )

        ax[i].text(
            0.02,
            0.02,
            LABELS[i],
            ha="left",
            va="bottom",
            transform=ax[i].transAxes,
            fontsize=10,
            color="k",
            weight="bold",
        )

        ax[i].format(
            xlabel=r"Confidence Percentile",
            ylabel=ylabel,
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=10,
            yticklabelsize=10,
            xlim=[0, 100],
        )

        if i == 3:
            ax[i].legend(ncol=1, loc="upper right", prop={"size": 11})

    out_file = os.path.join(PLOT_DIR, f"conf_curve.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)
