import os
import pickle
import proplot as pplt
import numpy as np
from deephyper.gnn_uq.figure import sat


def plot_err_unc(RESULT_DIR, PLOT_DIR, COLOR, format="pdf"):
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

    ratios = []
    for seed in range(8):
        ratio = np.zeros((16, 3))

        fig, ax = pplt.subplots(refheight=2.5*3/4, refwidth=2.5, ncols=4, nrows=4, share=False)

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

            std = (y_alea + y_epis) ** 0.5
            dif = y_true - y_pred

            vmax = std.max()
            xx = np.linspace(0, vmax, 100)

            emax = np.max(np.abs(dif))
            emax = np.max([vmax * 3])

            ax[i].plot(xx, np.zeros_like(xx), "k", lw=0.5)
            ax[i].fill_between(xx, xx, -xx, color=sat(COLOR[3], 0.7))
            ax[i].fill_between(xx, 2 * xx, xx, color=sat(COLOR[1], 0.7))
            ax[i].fill_between(xx, -2 * xx, -xx, color=sat(COLOR[1], 0.7))

            idx1 = np.where(np.abs(dif) - 2 * std >= 0)[0]
            idx2 = np.where(np.abs(dif) - 2 * std < 0)[0]
            idx3 = np.where(np.abs(dif) - 1 * std < 0)[0]

            ax[i].scatter(std[idx1], dif[idx1], color=COLOR[0], s=2)
            ax[i].scatter(std[idx2], dif[idx2], color="k", s=2, label=LABELS[i])

            ax[i].text(
                0.97,
                0.9,
                f"{len(idx1) / len(std) *100:0.0f}%",
                ha="right",
                fontsize=12,
                color="r",
                weight="bold",
                transform=ax[i].transAxes,
            )
            ax[i].text(
                0.97,
                0.75,
                f"{(len(idx2)-len(idx3)) / len(std) *100:0.0f}%",
                ha="right",
                fontsize=12,
                color="blue",
                weight="bold",
                transform=ax[i].transAxes,
            )
            ax[i].text(
                0.97,
                0.55,
                f"{(len(idx3)) / len(std) *100:0.0f}%",
                ha="right",
                fontsize=12,
                color="k",
                weight="bold",
                transform=ax[i].transAxes,
            )

            ratio[i, 0] = len(idx1) / len(std)
            ratio[i, 1] = (len(idx2) - len(idx3)) / len(std)
            ratio[i, 2] = (len(idx3)) / len(std)

            ax[i].text(
                0.02,
                0.98,
                LABELS[i],
                ha="left",
                va="top",
                transform=ax[i].transAxes,
                fontsize=13,
                color="k",
                weight="bold",
            )

            ax[i].format(
                xlim=[0, None],
                ylim=[-emax, emax],
                xlabelsize=14,
                ylabelsize=14,
                xticklabelsize=11,
                yticklabelsize=11,
            )

            ax[i].ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
            


        fig.text(
            0.015,
            0.52,
            "Error",
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=15,
            transform=fig.transFigure
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

        out_file = os.path.join(PLOT_DIR, f"unc_err_{seed}.{format}")

        fig.save(out_file, bbox_inches="tight", dpi=600)
        
        ratios.append(ratio)

    ratios = np.array(ratios)

    # print(ratios)
