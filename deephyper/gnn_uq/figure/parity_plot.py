import os
import pickle
import numpy as np
import proplot as pplt


def parity_seed_(
    result,
    dataset,
    idx,
    ax,
    vmin,
    vmax,
    label,
    unit,
    SPLIT_TYPE="811",
    ticks=[0, 1, 2],
    colorticks=[0, 1, 2],
    seed=0,
    task=None,
):
    y_true, y_pred, _, _, _, _, _, _ = result[(dataset, SPLIT_TYPE, seed)]

    if dataset == "qm9":
        if task in ["homo", "lumo", "gap", "zpve"]:
            y_true_temp = y_true[..., idx] * 27.2114
            y_pred_temp = y_pred[..., idx] * 27.2114
        else:
            y_true_temp = y_true[..., idx]
            y_pred_temp = y_pred[..., idx]
    else:
        y_true_temp = y_true
        y_pred_temp = y_pred

    im = ax.hist2d(
        y_true_temp,
        y_pred_temp,
        bins=np.linspace(vmin, vmax, 50),
        cmap="Dusk",
        density=False,
    )

    ax.plot([vmin, vmax], [vmin, vmax], "--", c="k", lw=0.5)
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])

    cbar = ax.colorbar(im[-1], ax=ax, ticks=colorticks)
    cbar.formatter.set_powerlimits((-3, 2))
    cbar.ax.yaxis.set_offset_position("left")
    cbar.update_ticks()
    ax.text(
        0.05,
        0.95,
        label + "\n" + unit,
        transform=ax.transAxes,
        va="top",
        ha="left",
        size=12,
        weight="bold",
    )

    ax.format(
        xlabel=None,
        ylabel=None,
        xticklabelsize=10,
        yticklabelsize=10,
        xlabelsize=12,
        ylabelsize=12,
        xticks=ticks,
        yticks=ticks,
    )


def plot_parity(RESULT_DIR, PLOT_DIR, format="pdf"):
    result_file = os.path.join(RESULT_DIR, "val_test_result.pickle")
    qm9_result_file = os.path.join(RESULT_DIR, "val_test_result_qm9.pickle")

    with open(result_file, "rb") as handle:
        result = pickle.load(handle)

    with open(qm9_result_file, "rb") as handle:
        result_qm9 = pickle.load(handle)

    for seed in range(8):
        fig, ax = pplt.subplots(nrows=4, ncols=4, share=False, refwidth=2, refheight=2)

        parity_seed_(
            result=result,
            dataset="lipo",
            idx=None,
            ax=ax[0],
            vmin=-2,
            vmax=6,
            label="Lipo",
            unit=r"[log D]",
            ticks=[-2, 0, 2, 4, 6],
            colorticks=[0, 4, 8, 12, 16],
            seed=seed,
        )
        parity_seed_(
            result=result,
            dataset="delaney",
            idx=None,
            ax=ax[1],
            vmin=-9,
            vmax=4,
            label="ESOL",
            unit=r"[log mol/L]",
            ticks=[-8, -4, 0, 4],
            colorticks=[0, 2, 4, 6, 8],
            seed=seed,
        )
        parity_seed_(
            result=result,
            dataset="freesolv",
            idx=None,
            ax=ax[2],
            vmin=-12,
            vmax=5,
            label="FreeSolv",
            unit=r"[kcal/mol]",
            ticks=[-10, -5, 0, 5],
            colorticks=[0, 1, 2, 3, 4, 5],
            seed=seed,
        )
        parity_seed_(
            result=result,
            dataset="qm7",
            idx=None,
            ax=ax[3],
            vmin=-2500,
            vmax=-800,
            label="QM7",
            unit=r"[kcal/mol]",
            ticks=[-2500, -2000, -1500, -1000],
            colorticks=[0, 20, 40, 60, 80, 100],
            seed=seed,
        )

        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=0,
            ax=ax[4],
            vmin=-1,
            vmax=10,
            label=r"QM9 $\mu$",
            unit=r"[D]",
            ticks=[0, 2, 4, 6, 8, 10],
            colorticks=[0, 50, 100, 150, 200, 250],
            seed=seed,
            task="mu",
        )

        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=1,
            ax=ax[5],
            vmin=20,
            vmax=120,
            label=r"QM9 $\alpha$",
            unit=r"[$a_0^3$]",
            ticks=[20, 40, 60, 80, 100, 120],
            colorticks=[0, 300, 600, 900, 1200],
            seed=seed,
            task="alpha",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=2,
            ax=ax[6],
            vmin=-9,
            vmax=-3,
            label=r"QM9 $\epsilon_{\mathrm{HOMO}}$",
            unit=r"[eV]",
            ticks=[-9, -7, -5, -3],
            colorticks=[0, 200, 400, 600],
            seed=seed,
            task="homo",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=3,
            ax=ax[7],
            vmin=-5,
            vmax=3,
            label=r"QM9 $\epsilon_{\mathrm{LUMO}}$",
            unit=r"[eV]",
            ticks=[-5, -3, -1, 1, 3],
            colorticks=[0, 100, 200, 300, 400],
            seed=seed,
            task="lumo",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=4,
            ax=ax[8],
            vmin=1,
            vmax=12,
            label=r"QM9 $\Delta \epsilon$",
            unit=r"[eV]",
            ticks=[2, 4, 6, 8, 10, 12],
            colorticks=[0, 200, 400, 600],
            seed=seed,
            task="gap",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=5,
            ax=ax[9],
            vmin=0,
            vmax=3500,
            label=r"QM9 $\langle R^2 \rangle$",
            unit=r"[$a_0^2$]",
            ticks=[0, 1000, 2000, 3000],
            colorticks=[0, 500, 1000, 1500],
            seed=seed,
            task="r2",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=6,
            ax=ax[10],
            vmin=0,
            vmax=8,
            label=r"QM9 ZPVE",
            unit=r"[eV]",
            ticks=[0, 2, 4, 6, 8],
            colorticks=[0, 300, 600, 900, 1200],
            seed=seed,
            task="zpve",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=7,
            ax=ax[11],
            vmin=10,
            vmax=50,
            label=r"QM9 $c_v$",
            unit=r"[cal/mol$\cdot$K]",
            ticks=[10, 20, 30, 40, 50],
            colorticks=[0, 300, 600, 900, 1200],
            seed=seed,
            task="cv",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=8,
            ax=ax[12],
            vmin=-600,
            vmax=-200,
            label=r"QM9 $U_0$",
            unit=r"[kcal/mol]",
            ticks=[-600, -500, -400, -300, -200],
            colorticks=[0, 400, 800, 1200, 1600],
            seed=seed,
            task="u0",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=9,
            ax=ax[13],
            vmin=-600,
            vmax=-200,
            label=r"QM9 $U$",
            unit=r"[kcal/mol]",
            ticks=[-600, -500, -400, -300, -200],
            colorticks=[0, 400, 800, 1200, 1600],
            seed=seed,
            task="u298",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=10,
            ax=ax[14],
            vmin=-600,
            vmax=-200,
            label=r"QM9 $H$",
            unit=r"[kcal/mol]",
            ticks=[-600, -500, -400, -300, -200],
            colorticks=[0, 400, 800, 1200, 1600],
            seed=seed,
            task="h298",
        )
        parity_seed_(
            result=result_qm9,
            dataset="qm9",
            idx=11,
            ax=ax[15],
            vmin=-600,
            vmax=-200,
            label=r"QM9 $G$",
            unit=r"[kcal/mol]",
            ticks=[-600, -500, -400, -300, -200],
            colorticks=[0, 400, 800, 1200, 1600],
            seed=seed,
            task="g298",
        )

        for i in [3, 9, 11, 12, 13, 14]:
            ax[i].ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

        fig.text(
            0.015,
            0.52,
            "Predicted",
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=15,
        )
        fig.text(0.52, 0.005, "True", va="center", ha="center", rotation=0, fontsize=15)

        out_file = os.path.join(PLOT_DIR, f"parity_{seed}.{format}")

        fig.save(out_file, bbox_inches="tight", dpi=600)
