import os
import glob
import pickle
import proplot as pplt
import numpy as np
from deephyper.gnn_uq.figure import sat

from deephyper.gnn_uq.data_utils import get_data, split_data
from deephyper.gnn_uq.analysis import conf_level
from rdkit import Chem
from scipy.stats import spearmanr
from rdkit.Chem import MACCSkeys

from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde


def load_pca_data_(DATA_DIR, RESULT_DIR):
    pca_file = os.path.join(RESULT_DIR, "pc9_pca.pickle")

    if os.path.exists(pca_file):
        with open(pca_file, "rb") as handle:
            pca_pc9 = pickle.load(handle)
            pca_qm9 = pickle.load(handle)
            x_pc9 = pickle.load(handle)
            x_qm9 = pickle.load(handle)
    else:
        # pc9
        pc9_file = os.path.join(DATA_DIR, "pc9.csv")
        data = get_data(pc9_file, tasks=["homo"], max_data_size=None)
        mol_train, _, mol_valid, _, mol_test, _ = split_data(
            data, split_type="random", sizes=(0.5, 0.2, 0.3), show_mol=True, seed=0
        )
        mol_pc9 = np.concatenate((mol_train, mol_valid, mol_test))
        mols = [Chem.MolFromSmiles(smi) for smi in mol_pc9]
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        x_pc9 = np.array(fps)

        # qm9
        qm9_file = os.path.join(DATA_DIR, "qm9.csv")
        data = get_data(qm9_file, tasks=["homo"], max_data_size=None)
        mol_train, _, mol_valid, _, mol_test, _ = split_data(
            data, split_type="random", sizes=(0.8, 0.1, 0.1), show_mol=True, seed=0
        )
        mol_qm9 = np.concatenate((mol_train, mol_valid, mol_test))
        mols = [Chem.MolFromSmiles(smi) for smi in mol_qm9]
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        x_qm9 = np.array(fps)

        pca = PCA(n_components=x_qm9.shape[-1], whiten=True, random_state=42)
        pca.fit(x_pc9)
        pca_pc9 = pca.transform(x_pc9)
        pca_qm9 = pca.transform(x_qm9)

        with open(pca_file, "wb") as handle:
            pickle.dump(pca_pc9, handle)
            pickle.dump(pca_qm9, handle)
            pickle.dump(x_pc9, handle)
            pickle.dump(x_qm9, handle)

    return pca_pc9.squeeze(), pca_qm9.squeeze(), x_pc9.squeeze(), x_qm9.squeeze()


def load_data_(ROOT_DIR, RESULT_DIR):
    pc9_file = os.path.join(RESULT_DIR, "pc9_pred.pickle")

    if os.path.exists(pc9_file):
        with open(pc9_file, "rb") as handle:
            y_true = pickle.load(handle)
            y_pred = pickle.load(handle)
            y_epis = pickle.load(handle)
            y_alea = pickle.load(handle)
    else:
        files = sorted(
            glob.glob(os.path.join(ROOT_DIR, "NEW_POST_RESULT_PC9/*/test_*.pickle"))
        )

        y_preds = []
        y_uncs = []

        for file in files:
            with open(file, "rb") as handle:
                y_true = pickle.load(handle)
                y_pred = pickle.load(handle)
                y_unc = pickle.load(handle)
                mean = pickle.load(handle)
                std = pickle.load(handle)

            y_pred = y_pred[:, [2, 3]]
            y_unc = y_unc[:, [2, 3]]

            mean = mean[[2, 3]]
            std = std[[2, 3]]

            y_preds.append((y_pred * std + mean))
            y_uncs.append((y_unc * std))

        y_pred = np.mean(y_preds, axis=0) * 27.2114
        y_epis = np.var(y_preds, axis=0) * 27.2114**2
        y_alea = np.mean(np.array(y_uncs) ** 2, axis=0) * 27.2114**2

        with open(pc9_file, "wb") as handle:
            pickle.dump(y_true, handle)
            pickle.dump(y_pred, handle)
            pickle.dump(y_epis, handle)
            pickle.dump(y_alea, handle)

    return y_true, y_pred, y_epis, y_alea


def plot_pc9_pca(ROOT_DIR, RESULT_DIR, DATA_DIR, COLOR, PLOT_DIR, format="pdf"):
    y_true, y_pred, y_epis, y_alea = load_data_(ROOT_DIR, RESULT_DIR)
    idx = np.where(y_true[:, 3] == 1)[0]

    pca_pc9, pca_qm9, _, _ = load_pca_data_(DATA_DIR, RESULT_DIR)
    pca_pc9 = pca_pc9[:, 0]
    pca_qm9 = pca_qm9[:, 0]

    fig, ax = pplt.subplots(refheight=2, refwidth=2, ncols=1, nrows=1, share=False)

    ax.hist(pca_pc9, density=True, bins=np.arange(-3, 3, 0.05), alpha=0.5, c=COLOR[0])
    ax.hist(pca_qm9, density=True, bins=np.arange(-3, 3, 0.05), alpha=0.5, c=COLOR[1])

    kde_pc9 = gaussian_kde(pca_pc9, bw_method=0.05)
    kde_qm9 = gaussian_kde(pca_qm9, bw_method=0.05)

    xx = np.linspace(-3, 3, 1000)
    ax.plot(xx, kde_pc9(xx), color=COLOR[0], lw=2, label="PC9")
    ax.plot(xx, kde_qm9(xx), color=COLOR[1], lw=2, label="QM9")

    ax.legend(ncol=1, loc="upper right", prop={"size": 11})

    ax.format(
        xlabel="Principal Component",
        ylabel="Density",
        xlim=[-3, 3],
        ylim=[0, 0.8],
        xlabelsize=12,
        ylabelsize=12,
        xticklabelsize=10,
        yticklabelsize=10,
    )

    out_file = os.path.join(PLOT_DIR, f"pc9_pca.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)


def plot_pc9_parity(
    ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format="pdf"
):
    y_true, y_pred, y_epis, y_alea = load_data_(ROOT_DIR, RESULT_DIR)

    fig, ax = pplt.subplots(refheight=2, refwidth=2, ncols=2, nrows=1, share=False)

    idx1 = np.where(y_true[:, 3] == 1)[0]
    idx2 = np.where(y_true[:, 3] == 2)[0]
    idx3 = np.where(y_true[:, 3] >= 3)[0]

    xlabels = [r"$\epsilon_\mathrm{HOMO}$ [eV]", r"$\epsilon_\mathrm{LUMO}$ [eV]"]
    lims = [[-12, 0], [-8, 4]]
    ticks = [[-12, -8, -4, 0], [-8, -4, 0, 4]]

    for i in range(2):
        ax[i].scatter(
            y_true[idx1, i + 1] * 0.0433634,
            y_pred[idx1, i],
            s=1,
            c=sat(COLOR[1], 0.1),
            label=r"$m_s$=1",
        )
        ax[i].scatter(
            y_true[idx2, i + 1] * 0.0433634,
            y_pred[idx2, i],
            s=1,
            c=sat(COLOR[0], 0.1),
            label=r"$m_s$=2",
        )
        ax[i].scatter(
            y_true[idx3, i + 1] * 0.0433634,
            y_pred[idx3, i],
            s=1,
            c=sat(COLOR[4], 0.1),
            label=r"$m_s$=3",
        )
        ax[i].plot([-12, 6], [-12, 6], "k--", lw=1)

        if i == 1:
            ax[i].legend(
                ncol=1,
                loc="lower right",
                prop={"size": 11},
                scatterpoints=1,
                markerscale=4,
            )

        ax[i].format(
            xlabel="True " + xlabels[i],
            ylabel="Predicted " + xlabels[i],
            xlim=lims[i],
            ylim=lims[i],
            xticks=ticks[i],
            yticks=ticks[i],
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=10,
            yticklabelsize=10,
        )

    out_file = os.path.join(PLOT_DIR, f"pc9_pca_parity.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)


def plot_pc9_decomp(
    ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format="pdf"
):
    y_true, y_pred, y_epis, y_alea = load_data_(ROOT_DIR, RESULT_DIR)

    fig, ax = pplt.subplots(refheight=2, refwidth=2, ncols=1, nrows=1, share=False)

    linestyles = ["-", "--"]
    xlabels = [r"$\epsilon_\mathrm{HOMO}$", r"$\epsilon_\mathrm{LUMO}$"]

    for i in range(2):
        aleatoric_values = np.array(y_alea)[:, i] ** 0.5
        epistemic_values = np.array(y_epis)[:, i] ** 0.5

        val = aleatoric_values
        sort_indices = np.argsort(val)
        sorted_aleatoric_mean = val[sort_indices]
        cdf_values = np.linspace(0, 1, len(sorted_aleatoric_mean))
        sorted_aleatoric_y = sorted_aleatoric_mean

        sorted_aleatoric_y = np.array(sorted_aleatoric_y)[:-1]

        val = epistemic_values
        sort_indices2 = np.argsort(val)
        sorted_epistemic_mean = val[sort_indices2]
        cdf_values2 = np.linspace(0, 1, len(sorted_epistemic_mean))
        sorted_epistemic_y = sorted_epistemic_mean

        sorted_epistemic_y = np.array(sorted_epistemic_y)[:-1]

        if sorted_aleatoric_y[-1] > sorted_epistemic_y[-1]:
            sorted_epistemic_y = np.append(sorted_epistemic_y, sorted_epistemic_y[-1])
            cdf_values2 = np.append(cdf_values2, cdf_values[-1])
        else:
            sorted_aleatoric_y = np.append(sorted_aleatoric_y, sorted_epistemic_y[-1])
            cdf_values = np.append(cdf_values, cdf_values2[-1])

        ax.plot(
            sorted_aleatoric_y,
            cdf_values[:-1],
            drawstyle="steps-post",
            label="Alea. " + xlabels[i],
            c=COLOR[0],
            linestyle=linestyles[i],
        )
        ax.plot(
            sorted_epistemic_y,
            cdf_values2[:-1],
            drawstyle="steps-post",
            label="Epis. " + xlabels[i],
            c=COLOR[1],
            linestyle=linestyles[i],
        )

        ax.legend(ncol=1, loc="lower right", prop={"size": 10})

        ax.format(
            xlabel="Uncertainty",
            ylabel="Cumulative Distribution",
            xlim=[0, 1],
            ylim=[0, 1.05],
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=10,
            yticklabelsize=10,
        )

        ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))

    out_file = os.path.join(PLOT_DIR, f"pc9_pca_decomp.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)


def plot_pc9_conf(ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format="pdf"):
    y_true, y_pred, y_epis, y_alea = load_data_(ROOT_DIR, RESULT_DIR)

    xlabels = [r"$\epsilon_\mathrm{HOMO}$ [eV]", r"$\epsilon_\mathrm{LUMO}$ [eV]"]
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig, ax = pplt.subplots(refheight=2, refwidth=2, ncols=2, nrows=1, share=True)

    for i in range(2):
        cf1, perc_tot = conf_level(
            y_true[:, i + 1] * 0.0433634, y_pred[:, i], (y_alea + y_epis)[:, i] ** 0.5
        )
        cf2, perc_ale = conf_level(
            y_true[:, i + 1] * 0.0433634, y_pred[:, i], (y_alea)[:, i] ** 0.5
        )
        cf3, perc_epi = conf_level(
            y_true[:, i + 1] * 0.0433634, y_pred[:, i], (y_epis)[:, i] ** 0.5
        )

        ax[i].plot(cf1, perc_tot, label="Total", c=COLOR[0])
        ax[i].plot(cf2, perc_ale, label="Aleatoric", c=COLOR[1])
        ax[i].plot(cf3, perc_epi, label="Epistemic", c=COLOR[2])

        ax[i].plot([0, 1], [0, 1], "k--", lw=1)

        ax[i].text(
            0.98,
            0.02,
            xlabels[i],
            ha="right",
            va="bottom",
            transform=ax[i].transAxes,
            fontsize=12,
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
            xticks=ticks,
            yticks=ticks,
        )

        if i == 0:
            ax[i].legend(ncol=1, loc="upper left", prop={"size": 10})

    out_file = os.path.join(PLOT_DIR, f"pc9_pca_conf_calib.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)


def plot_pc9_unc_err(
    ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format="pdf"
):
    y_true, y_pred, y_epis, y_alea = load_data_(ROOT_DIR, RESULT_DIR)

    xlabels = [r"$\epsilon_\mathrm{HOMO}$ [eV]", r"$\epsilon_\mathrm{LUMO}$ [eV]"]

    fig, ax = pplt.subplots(refheight=2, refwidth=2, ncols=2, nrows=1, share=False)

    for i in range(2):
        std = (y_alea[:, i] + y_epis[:, i]) ** 0.5
        dif = y_true[:, i + 1] * 0.0433634 - y_pred[:, i]

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

        ax[i].scatter(std[idx1], dif[idx1], color=COLOR[0], s=1)
        ax[i].scatter(std[idx2], dif[idx2], color="k", s=1)

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

        ax[i].text(
            0.02,
            0.98,
            xlabels[i],
            ha="left",
            va="top",
            transform=ax[i].transAxes,
            fontsize=12,
            color="k",
            weight="bold",
        )

        ax[i].format(
            xlabel="Uncertainty",
            ylabel="Error",
            xlim=[0, None],
            ylim=[-emax, emax],
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=10,
            yticklabelsize=10,
        )

    out_file = os.path.join(PLOT_DIR, f"pc9_pca_unc_err.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)


def plot_pc9_similarity(
    ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format="pdf"
):
    xlabels = [r"$\epsilon_\mathrm{HOMO}$ [eV]", r"$\epsilon_\mathrm{LUMO}$ [eV]"]

    pca_pc9, pca_qm9, _, _ = load_pca_data_(DATA_DIR, RESULT_DIR)

    y_true, y_pred, y_epis, y_alea = load_data_(ROOT_DIR, RESULT_DIR)

    fig, ax = pplt.subplots(refheight=2, refwidth=2, ncols=2, nrows=1, share=True)

    for j in range(2):
        vmin = min([min(pca_pc9[:, 0]), min(pca_qm9[:, 0])])
        vmax = max([max(pca_pc9[:, 0]), max(pca_qm9[:, 0])])

        den1, bins = np.histogram(
            pca_pc9[:, 0], density=True, bins=np.linspace(vmin, vmax, 25)
        )
        den2, bins = np.histogram(
            pca_qm9[:, 0], density=True, bins=np.linspace(vmin, vmax, 25)
        )
        epi_bin = []
        den_diff = []
        for i in range(len(bins) - 1):
            idx = np.where((pca_pc9[:, 0] >= bins[i]) & (pca_pc9[:, 0] <= bins[i + 1]))[
                0
            ]
            if len(idx) >= 1:
                y_epis_mean = (y_epis[idx, j] ** 0.5).mean()
                y_epis_std = (y_epis[idx, j] ** 0.5).std() / (len(idx) ** 0.5)
                epi_bin.append([y_epis_mean, y_epis_std])

                rel_diff = den1[i] / den2[i]
                den_diff.append(rel_diff)

        epi_bin = np.array(epi_bin)
        den_diff = np.array(den_diff)

        print(spearmanr(den_diff, epi_bin[:, 0]).correlation)

        ax[j].errorbar(
            den_diff,
            epi_bin[:, 0],
            epi_bin[:, 1],
            marker="o",
            c=COLOR[1],
            linestyle="none",
        )
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")

        ax[j].text(
            0.02,
            0.98,
            xlabels[j],
            ha="left",
            va="top",
            transform=ax[j].transAxes,
            fontsize=12,
            color="k",
            weight="bold",
        )

        ax[j].format(
            xlabel=r"$\rho_\mathrm{OOD}$",
            ylabel="Epistemic Uncertainty",
            yticks=[0.2, 0.3, 0.4],
            xlim=[0.1, 100],
            xlabelsize=12,
            ylabelsize=12,
            xticklabelsize=10,
            yticklabelsize=10,
        )

    out_file = os.path.join(PLOT_DIR, f"pc9_pca_simi.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)


def plot_pc9_all(ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format="png"):
    plot_pc9_pca(ROOT_DIR, RESULT_DIR, DATA_DIR, COLOR, PLOT_DIR, format)
    plot_pc9_parity(ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format)
    plot_pc9_decomp(ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format)
    plot_pc9_conf(ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format)
    plot_pc9_unc_err(ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format)
    plot_pc9_similarity(ROOT_DIR, RESULT_DIR, DATA_DIR, LABEL, COLOR, PLOT_DIR, format)
