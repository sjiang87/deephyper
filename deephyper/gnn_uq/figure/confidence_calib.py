import os
import pickle
import proplot as pplt
import numpy as np
import uncertainty_toolbox as uct
from deephyper.gnn_uq.figure import sat
from deephyper.gnn_uq.analysis import conf_level, miscal_area


def plot_conf_calib(RESULT_DIR, PLOT_DIR, COLOR, format="pdf"):
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
    result_file = os.path.join(RESULT_DIR, "val_test_result.pickle")
    qm9_result_file = os.path.join(RESULT_DIR, "val_test_result_qm9.pickle")

    with open(result_file, "rb") as handle:
        result = pickle.load(handle)

    with open(qm9_result_file, "rb") as handle:
        result_qm9 = pickle.load(handle)

    fig, ax = pplt.subplots(refheight=1, refwidth=2, ncols=4, nrows=4, share=True)
    for i in range(16):
        perc_tot = np.zeros((8, 100))
        perc_ale = np.zeros((8, 100))
        perc_epi = np.zeros((8, 100))
        MCA = np.zeros((8,))

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

        for seed in range(8):
            if dataset == "qm9":
                split = "811"

                y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = (
                    result_qm9[dataset, split, seed]
                )
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

            cf1, perc_tot[seed] = conf_level(y_true, y_pred, (y_alea + y_epis) ** 0.5)
            cf2, perc_ale[seed] = conf_level(y_true, y_pred, (y_alea) ** 0.5)
            cf3, perc_epi[seed] = conf_level(y_true, y_pred, (y_epis) ** 0.5)
            MCA[seed] = miscal_area((y_alea + y_epis), y_pred, y_true)[0]

        ax[i].plot(cf1, perc_tot.mean(axis=0), label="Total", c=COLOR[0])
        ax[i].fill_between(
            cf1,
            perc_tot.mean(axis=0) - perc_tot.std(axis=0),
            perc_tot.mean(axis=0) + perc_tot.std(axis=0),
            c=sat(COLOR[0], 0.85),
        )

        ax[i].plot(cf2, perc_ale.mean(axis=0), label="Aleatoric", c=COLOR[1])
        ax[i].fill_between(
            cf2,
            perc_ale.mean(axis=0) - perc_ale.std(axis=0),
            perc_ale.mean(axis=0) + perc_ale.std(axis=0),
            c=sat(COLOR[1], 0.85),
        )

        ax[i].plot(cf3, perc_epi.mean(axis=0), label="Epistemic", c=COLOR[2])
        ax[i].fill_between(
            cf3,
            perc_epi.mean(axis=0) - perc_epi.std(axis=0),
            perc_epi.mean(axis=0) + perc_epi.std(axis=0),
            c=sat(COLOR[2], 0.85),
        )

        ax[i].plot([0, 1], [0, 1], "k--", lw=1)

        ax[i].text(
            0.98,
            0.02,
            LABELS[i],
            ha="right",
            va="bottom",
            transform=ax[i].transAxes,
            fontsize=10,
            color="k",
            weight="bold",
        )

        ax[i].format(
            xlabel=r"Confidence Level",
            ylabel="Empirical Coverage",
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=10,
            yticklabelsize=10,
            xlim=[0, 1],
        )

        if i == 15:
            ax[i].legend(ncol=1, loc="lower right", prop={"size": 11})

    out_file = os.path.join(PLOT_DIR, f"conf_calib_before.{format}")

    # after calibration

    fig.save(out_file, bbox_inches="tight", dpi=600)

    fig, ax = pplt.subplots(refheight=1, refwidth=2, ncols=4, nrows=4, share=True)
    for i in range(16):
        perc_tot = np.zeros((8, 100))
        perc_ale = np.zeros((8, 100))
        perc_epi = np.zeros((8, 100))
        MCA = np.zeros((8,))

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

        for seed in range(8):
            if dataset == "qm9":
                split = "811"

                y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = (
                    result_qm9[dataset, split, seed]
                )
                y_true = np.copy(y_true)[..., i - 4]
                y_pred = np.copy(y_pred)[..., i - 4]
                y_alea = np.copy(y_alea)[..., i - 4]
                y_epis = np.copy(y_epis)[..., i - 4]
                v_true = np.copy(v_true)[..., i - 4]
                v_pred = np.copy(v_pred)[..., i - 4]
                v_alea = np.copy(v_alea)[..., i - 4]
                v_epis = np.copy(v_epis)[..., i - 4]

                if i in [6, 7, 8, 10]:
                    y_true = y_true * 27.2114
                    y_pred = y_pred * 27.2114
                    y_alea = y_alea * (27.2114**2)
                    y_epis = y_epis * (27.2114**2)

                    v_true = v_true * 27.2114
                    v_pred = v_pred * 27.2114
                    v_alea = v_alea * (27.2114**2)
                    v_epis = v_epis * (27.2114**2)

            else:
                split = "523"

                y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = result[
                    dataset, split, seed
                ]

            a = uct.recalibration.optimize_recalibration_ratio(
                y_mean=v_pred,
                y_std=(v_alea + v_epis) ** 0.5,
                y_true=v_true,
                criterion="miscal",
            )

            cf1, perc_tot[seed] = conf_level(
                y_true, y_pred, (a**2 * y_alea + a**2 * y_epis) ** 0.5
            )
            cf2, perc_ale[seed] = conf_level(y_true, y_pred, (a**2 * y_alea) ** 0.5)
            cf3, perc_epi[seed] = conf_level(y_true, y_pred, (a**2 * y_epis) ** 0.5)

            MCA[seed] = miscal_area((a**2 * y_alea + a**2 * y_epis), y_pred, y_true)[0]

        ax[i].plot(cf1, perc_tot.mean(axis=0), label="Total", c=COLOR[0])
        ax[i].fill_between(
            cf1,
            perc_tot.mean(axis=0) - perc_tot.std(axis=0),
            perc_tot.mean(axis=0) + perc_tot.std(axis=0),
            c=sat(COLOR[0], 0.85),
        )

        ax[i].plot(cf2, perc_ale.mean(axis=0), label="Aleatoric", c=COLOR[1])
        ax[i].fill_between(
            cf2,
            perc_ale.mean(axis=0) - perc_ale.std(axis=0),
            perc_ale.mean(axis=0) + perc_ale.std(axis=0),
            c=sat(COLOR[1], 0.85),
        )

        ax[i].plot(cf3, perc_epi.mean(axis=0), label="Epistemic", c=COLOR[2])
        ax[i].fill_between(
            cf3,
            perc_epi.mean(axis=0) - perc_epi.std(axis=0),
            perc_epi.mean(axis=0) + perc_epi.std(axis=0),
            c=sat(COLOR[2], 0.85),
        )

        ax[i].plot([0, 1], [0, 1], "k--", lw=1)

        ax[i].text(
            0.02,
            0.98,
            LABELS[i],
            ha="left",
            va="top",
            transform=ax[i].transAxes,
            fontsize=10,
            color="k",
            weight="bold",
        )

        ax[i].format(
            xlabel=r"Confidence Level",
            ylabel="Empirical Coverage",
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=10,
            yticklabelsize=10,
            xlim=[0, 1],
        )

    out_file = os.path.join(PLOT_DIR, f"conf_calib_after.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)
