import os
import glob
import random
import pickle
import proplot as pplt
import numpy as np
from deephyper.gnn_uq.figure import sat

from deephyper.gnn_uq.data_utils import get_data, split_data
from deephyper.gnn_uq.analysis import (
    calculate_silhouette_score,
    extract_functional_groups,
)
from rdkit import Chem
from sklearn.cluster import KMeans
from rdkit.Chem import MACCSkeys
from sklearn.manifold import TSNE

import warnings

warnings.filterwarnings("ignore")


def load_data_(RESULT_DIR):
    with open(os.path.join(RESULT_DIR, "val_test_result.pickle"), "rb") as handle:
        result = pickle.load(handle)

    with open(os.path.join(RESULT_DIR, "val_test_result_qm9.pickle"), "rb") as handle:
        result_qm9 = pickle.load(handle)

    return result, result_qm9


def load_tsne_(RESULT_DIR, DATA_DIR, dataset):
    if "qm7" in dataset:
        tasks = ["u0_atom"]
    elif "lipo" in dataset:
        tasks = ["lipo"]
    elif "freesolv" in dataset:
        tasks = ["freesolv"]
    elif "delaney" in dataset:
        tasks = ["logSolubility"]
    elif "qm9" in dataset:
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
    tsne_dir = os.path.join(RESULT_DIR, f"tsne_{dataset}.pickle")

    if os.path.exists(tsne_dir):
        with open(tsne_dir, "rb") as handle:
            X_tsne, clusters, mol_test, best_k = pickle.load(handle)

    else:
        seed = 0
        data_dir = os.path.join(DATA_DIR, f"{dataset}.csv")
        data = get_data(data_dir, tasks, max_data_size=None)

        if dataset != "qm9":

            mol_train, _, mol_valid, _, mol_test, _ = split_data(
                data,
                split_type="random",
                sizes=(0.5, 0.2, 0.3),
                show_mol=True,
                seed=seed,
            )

        else:
            mol_train, _, mol_valid, _, mol_test, _ = split_data(
                data,
                split_type="random",
                sizes=(0.8, 0.1, 0.1),
                show_mol=True,
                seed=seed,
            )

        k_values = np.arange(3, 30)

        mols = [Chem.MolFromSmiles(smi) for smi in mol_test]

        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]

        X = np.array(fps)

        np.random.seed(0)
        random.seed(0)

        tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)

        X_tsne = tsne.fit_transform(X)

        silhouette_scores = [calculate_silhouette_score(X_tsne, k) for k in k_values]

        best_k = k_values[np.argmax(silhouette_scores)]

        kmeans = KMeans(n_clusters=best_k, random_state=0)
        clusters = kmeans.fit_predict(X_tsne)

        with open(tsne_dir, "wb") as handle:
            pickle.dump([X_tsne, clusters, mol_test, best_k], handle)

    print(f"# Best k = {best_k}")

    return X_tsne, clusters, mol_test, best_k


def plot_tsne_(
    ROOT_DIR, RESULT_DIR, DATA_DIR, COLOR, PLOT_DIR, dataset, mode="epis", format="pdf"
):
    if "qm7" in dataset:
        unit = ["[kcal/mol]"]
    elif "lipo" in dataset:
        unit = ["[log D]"]
    elif "freesolv" in dataset:
        unit = ["[kcal/mol]"]
    elif "delaney" in dataset:
        unit = ["[log mol/L]"]
    elif "qm9" in dataset:
        unit = [
            r"$\mu$" + " [D]",
            r"$\alpha$" + r" [$a_0^3$]",
            r"$\epsilon_{\mathrm{HOMO}}$" + " [eV]",
            r"$\epsilon_{\mathrm{LUMO}}$" + " [eV]",
            r"$\Delta \epsilon$" + " [eV]",
            r"$\langle R^2 \rangle$" + r" [$a_0^2$]",
            r"ZPVE" + " [eV]",
            r"$c_v$" + r" [cal/mol$\cdot$K]",
            r"$U_0$" + " [kcal/mol]",
            r"$U$" + " [kcal/mol]",
            r"$H$" + " [kcal/mol]",
            r"$G$" + " [kcal/mol]",
        ]

    result, result_qm9 = load_data_(RESULT_DIR=RESULT_DIR)
    X_tsne, clusters, mol_test, best_k = load_tsne_(
        RESULT_DIR=RESULT_DIR, DATA_DIR=DATA_DIR, dataset=dataset
    )

    for idx in range(len(unit)):
        stds = []

        for seed in range(1):
            if dataset != "qm9":
                y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = result[
                    (dataset.lower(), "523", seed)
                ]
            else:
                y_true, y_pred, y_epis, y_alea, v_true, v_pred, v_epis, v_alea = (
                    result_qm9[(dataset.lower(), "811", seed)]
                )
                y_true = y_true[..., idx]
                y_pred = y_pred[..., idx]
                y_epis = y_epis[..., idx]
                y_alea = y_alea[..., idx]

                if idx in [2, 3, 4, 6]:
                    y_true = y_true * 27.2114
                    y_pred = y_pred * 27.2114
                    y_alea = y_alea * (27.2114**2)
                    y_epis = y_epis * (27.2114**2)

            if mode == "epis":
                std = y_epis**0.5
            elif mode == "alea":
                std = y_alea**0.5

            stds.append(std)

        stds = np.array(stds)
        std = np.mean(stds, axis=0)

        vmin = std.min()
        vmax = std.max()
        mean_std = std.mean()
        cmap = pplt.Colormap("coolwarm")

        std_sub_mean = std - mean_std

        adjusted_vmin = np.percentile(std_sub_mean, 10)
        adjusted_vmax = np.percentile(std_sub_mean, 90)

        colors = (clusters - clusters.min()) / (clusters.max() - clusters.min())

        fig, ax = pplt.subplots(ncols=2, refwidth=3, refheight=3)

        im = ax[0].scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            alpha=0.9,
            edgecolor="k",
            s=20,
            c=clusters,
            cmap="ggplot",
        )

        im = ax[1].scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            alpha=0.9,
            edgecolor="k",
            s=20,
            c=std_sub_mean,
            cmap="coolwarm",
            vmin=adjusted_vmin,
            vmax=adjusted_vmax,
        )

        cbar = fig.colorbar(im, ax=ax[1])

        if mode == "epis":
            cbar.set_label(r"Epistemic Uncertainty " + unit[idx], size=12)
        elif mode == "alea":
            cbar.set_label(r"Aleatoric Uncertainty " + unit[idx], size=12)

        if dataset == "qm7":
            cbar.set_ticks(np.linspace(adjusted_vmin, adjusted_vmax, num=6))
            cbar.set_ticklabels([5, 10, 15, 20, 25, 30])

            for i in range(2):
                ax[i].format(
                    xlabel="t-SNE 1",
                    ylabel="t-SNE 2",
                    xlim=[-80, 80],
                    ylim=[-60, 60],
                    xticklabelsize=10,
                    yticklabelsize=10,
                    xlabelsize=12,
                    ylabelsize=12,
                )
        else:
            cbar.set_ticks(np.linspace(adjusted_vmin, adjusted_vmax, num=6))
            cbar.set_ticklabels(
                np.round(np.linspace(adjusted_vmin, adjusted_vmax, num=6) + mean_std, 2)
            )

            for i in range(2):
                ax[i].format(
                    xlabel="t-SNE 1",
                    ylabel="t-SNE 2",
                    xticklabelsize=10,
                    yticklabelsize=10,
                    xlabelsize=12,
                    ylabelsize=12,
                )

        out_file = os.path.join(
            PLOT_DIR, f"tsne_{dataset}_{idx}_{seed}_{mode}.{format}"
        )

        fig.save(out_file, bbox_inches="tight", dpi=600)


def plot_tsne(ROOT_DIR, RESULT_DIR, DATA_DIR, COLOR, PLOT_DIR, format="png"):
    for dataset in ["qm7", "lipo", "delaney", "freesolv", "qm9"]:
        for mode in ["alea", "epis"]:
            plot_tsne_(
                ROOT_DIR,
                RESULT_DIR,
                DATA_DIR,
                COLOR,
                PLOT_DIR,
                dataset,
                mode=mode,
                format=format,
            )
