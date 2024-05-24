import os
import pickle
import proplot as pplt
import numpy as np
from deephyper.gnn_uq.figure import sat


def plot_unc_decomp(RESULT_DIR, PLOT_DIR, COLOR, format="pdf"):
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

    VMAXS = [1, 1.2, 2, 500, 3, 2, 0.6, 0.6, 1, 200, 0.04, 1, 0.3, 0.3, 0.3, 0.3]
    result_file = os.path.join(RESULT_DIR, "val_test_result.pickle")
    qm9_result_file = os.path.join(RESULT_DIR, "val_test_result_qm9.pickle")

    with open(result_file, "rb") as handle:
        result = pickle.load(handle)

    with open(qm9_result_file, "rb") as handle:
        result_qm9 = pickle.load(handle)

    fig, ax = pplt.subplots(
        refheight=2.5 * 3 / 4, refwidth=2.5, ncols=4, nrows=4, share=False
    )

    for i in range(16):
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

        aleatoric_values = []
        epistemic_values = []

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

            aleatoric_values.append(y_alea**0.5)
            epistemic_values.append(y_epis**0.5)

        aleatoric_values = np.array(aleatoric_values)
        epistemic_values = np.array(epistemic_values)

        sorted_aleatoric_y = []

        for c in range(8):
            val = aleatoric_values[c]
            sort_indices = np.argsort(val)
            sorted_aleatoric_mean = val[sort_indices]
            cdf_values = np.linspace(0, 1, len(sorted_aleatoric_mean))
            sorted_aleatoric_y.append(sorted_aleatoric_mean)

        sorted_aleatoric_y = np.array(sorted_aleatoric_y)[:-1]
        sorted_aleatoric_mean = np.mean(sorted_aleatoric_y, axis=0)[:-1]
        sorted_aleatoric_std = np.std(sorted_aleatoric_y, axis=0)[:-1]

        lower_bound = sorted_aleatoric_mean - sorted_aleatoric_std
        upper_bound = sorted_aleatoric_mean + sorted_aleatoric_std

        sorted_epistemic_y = []

        for c in range(8):
            val = epistemic_values[c]
            sort_indices2 = np.argsort(val)
            sorted_epistemic_mean = val[sort_indices2]
            cdf_values2 = np.linspace(0, 1, len(sorted_epistemic_mean))
            sorted_epistemic_y.append(sorted_epistemic_mean)

        sorted_epistemic_y = np.array(sorted_epistemic_y)[:-1]
        sorted_epistemic_mean = np.mean(sorted_epistemic_y, axis=0)[:-1]
        sorted_epistemic_std = np.std(sorted_epistemic_y, axis=0)[:-1]

        lower_bound = sorted_aleatoric_mean - sorted_aleatoric_std
        upper_bound = sorted_aleatoric_mean + sorted_aleatoric_std
        lower_bound2 = sorted_epistemic_mean - sorted_epistemic_std
        upper_bound2 = sorted_epistemic_mean + sorted_epistemic_std

        if sorted_aleatoric_mean[-1] > sorted_epistemic_mean[-1]:
            sorted_epistemic_mean = np.append(
                sorted_epistemic_mean, sorted_aleatoric_mean[-1]
            )
            cdf_values2 = np.append(cdf_values2, cdf_values[-1])
        else:
            sorted_aleatoric_mean = np.append(
                sorted_aleatoric_mean, sorted_epistemic_mean[-1]
            )
            cdf_values = np.append(cdf_values, cdf_values2[-1])
            
        if i == 2:
            sorted_aleatoric_mean = np.append(sorted_aleatoric_mean, [2])
            cdf_values = np.append(cdf_values, 1)
            sorted_epistemic_mean = np.append(sorted_epistemic_mean, [2])
            cdf_values2 = np.append(cdf_values2, 1)

        ax[i].plot(
            sorted_aleatoric_mean,
            cdf_values[:-1],
            drawstyle="steps-post",
            label="Aleatoric",
            c=COLOR[0],
        )
        ax[i].plot(
            sorted_epistemic_mean,
            cdf_values2[:-1],
            drawstyle="steps-post",
            label="Epistemic",
            c=COLOR[1],
        )
        ax[i].fill_betweenx(
            cdf_values[: len(lower_bound)],
            lower_bound,
            upper_bound,
            c=sat(COLOR[0], 0.85),
        )
        ax[i].fill_betweenx(
            cdf_values2[: len(lower_bound2)],
            lower_bound2,
            upper_bound2,
            c=sat(COLOR[1], 0.85),
        )

        ax[i].text(
            0.98,
            0.02,
            LABELS[i],
            ha="right",
            va="bottom",
            transform=ax[i].transAxes,
            fontsize=13,
            color="k",
            weight="bold",
        )

        ax[i].format(
            xlim=[0, VMAXS[i]],
            ylim=[0, 1.05],
            xlabelsize=14,
            ylabelsize=14,
            xticklabelsize=11,
            yticklabelsize=11,
        )

        ax[i].ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))

        if i == 12:
            ax[i].legend(ncol=1, loc="center right", prop={"size": 11})

    fig.text(
        0.015,
        0.52,
        "Cumulative Distribution",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=15,
        transform=fig.transFigure,
    )
    fig.text(
        0.52,
        0.01,
        "Uncertainty",
        va="center",
        ha="center",
        rotation=0,
        fontsize=15,
        transform=fig.transFigure,
    )

    out_file = os.path.join(PLOT_DIR, f"unc_decomp.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)
