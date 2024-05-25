import os
import pickle
import proplot as pplt
import numpy as np
import pandas as pd
import uncertainty_toolbox as uct
from deephyper.gnn_uq.figure import sat
from deephyper.gnn_uq.analysis import conf_level, miscal_area


def load_data_(RESULT_DIR, mode="normal"):

    if mode == "normal":
        result_file_1 = os.path.join(RESULT_DIR, "val_test_result.pickle")
        result_file_2 = os.path.join(RESULT_DIR, "val_test_result_qm9.pickle")

    elif mode == "mc":
        result_file_1 = os.path.join(RESULT_DIR, "val_test_result_mc_dropout.pickle")
        result_file_2 = os.path.join(
            RESULT_DIR, "val_test_result_qm9_mc_dropout.pickle"
        )

    elif mode == "random":
        result_file_1 = os.path.join(RESULT_DIR, "val_test_result_random.pickle")
        result_file_2 = os.path.join(RESULT_DIR, "val_test_result_qm9_random.pickle")

    elif mode == "simple":
        result_file_1 = os.path.join(RESULT_DIR, "val_test_result_simple.pickle")
        result_file_2 = os.path.join(RESULT_DIR, "val_test_result_qm9_simple.pickle")

    with open(result_file_1, "rb") as handle:
        result = pickle.load(handle)

    with open(result_file_2, "rb") as handle:
        result_qm9 = pickle.load(handle)

    return result, result_qm9


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

    for data_mode in ["normal", "simple", "mc", "random"]:
        result, result_qm9 = load_data_(RESULT_DIR=RESULT_DIR, mode=data_mode)

        csv_out = []

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

                    y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = (
                        result[dataset, split, seed]
                    )

                cf1, perc_tot[seed] = conf_level(
                    y_true, y_pred, (y_alea + y_epis) ** 0.5
                )
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

        diff_before = np.abs(perc_tot - cf1)
        ece_before = np.mean(diff_before, axis=1)
        mce_before = np.max(diff_before, axis=1)
        MCA_before = np.copy(MCA)

        # after calibration

        fig.save(out_file, bbox_inches="tight", dpi=600)

        fig, ax = pplt.subplots(refheight=1, refwidth=2, ncols=4, nrows=4, share=True)
        for i in range(16):
            perc_tot = np.zeros((8, 100))
            perc_ale = np.zeros((8, 100))
            perc_epi = np.zeros((8, 100))
            MCA = np.zeros((8,))
            ratios = np.zeros((8,))

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

                    y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = (
                        result[dataset, split, seed]
                    )

                a = uct.recalibration.optimize_recalibration_ratio(
                    y_mean=v_pred,
                    y_std=(v_alea + v_epis) ** 0.5,
                    y_true=v_true,
                    criterion="miscal",
                )

                ratios[seed] = a

                cf1, perc_tot[seed] = conf_level(
                    y_true, y_pred, (a**2 * y_alea + a**2 * y_epis) ** 0.5
                )
                cf2, perc_ale[seed] = conf_level(y_true, y_pred, (a**2 * y_alea) ** 0.5)
                cf3, perc_epi[seed] = conf_level(y_true, y_pred, (a**2 * y_epis) ** 0.5)

                MCA[seed] = miscal_area(
                    (a**2 * y_alea + a**2 * y_epis), y_pred, y_true
                )[0]

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

            diff = np.abs(perc_tot - cf1)
            ece = np.mean(diff, axis=1)
            mce = np.max(diff, axis=1)

            csv_out.append(
                [
                    LABELS[i].split("\n")[0],
                    ece_before.mean(),
                    ece_before.std(),
                    mce_before.mean(),
                    mce_before.std(),
                    MCA_before.mean(),
                    MCA_before.std(),
                    ece.mean(),
                    ece.std(),
                    mce.mean(),
                    mce.std(),
                    MCA.mean(),
                    MCA.std(),
                    ratios.mean(),
                    ratios.std(),
                ]
            )

        df = pd.DataFrame(
            csv_out,
            columns=[
                "dataset",
                "before_ece_mean",
                "before_ece_std",
                "before_mce_mean",
                "before_mce_std",
                "before_mca_mean",
                "before_mca_std",
                "after_ece_mean",
                "after_ece_std",
                "after_mce_mean",
                "after_mce_std",
                "after_mca_mean",
                "after_mca_std",
                "ratio_mean",
                "ratio_std",
            ],
        )

        out_csv_file = os.path.join(RESULT_DIR, f"conf_calib_ratio_mca_{data_mode}.csv")
        df.to_csv(out_csv_file, index=False)

        out_file = os.path.join(PLOT_DIR, f"conf_calib_after_{data_mode}.{format}")

        fig.save(out_file, bbox_inches="tight", dpi=600)
