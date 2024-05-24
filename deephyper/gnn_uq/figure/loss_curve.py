import os
import pickle
import pandas as pd
import numpy as np
import proplot as pplt
from scipy.signal import savgol_filter
from deephyper.gnn_uq.figure import sat


def load_data_(ROOT_DIR, RESULT_DIR, if_simple=False):
    if if_simple:
        loss_file = os.path.join(RESULT_DIR, "loss_simple.pickle")
    else:
        loss_file = os.path.join(RESULT_DIR, "loss.pickle")

    if os.path.exists(loss_file):
        with open(loss_file, "rb") as handle:
            loss_dict = pickle.load(handle)

    else:
        params = [
            ("Delaney", "523"),
            ("Lipo", "523"),
            ("FreeSolv", "523"),
            ("QM7", "523"),
            ("QM9", "811"),
        ]

        loss_dict = {}

        for param in params:
            losses = []
            val_losses = []

            for seed in range(8):
                if if_simple:
                    MODEL_DIR = os.path.join(
                        ROOT_DIR,
                        f"SIMPLE_RE_{param[0].lower()}_random_{seed}_split_{param[1]}/save/model/",
                    )
                else:
                    MODEL_DIR = os.path.join(
                        ROOT_DIR,
                        f"NEW_RE_{param[0].lower()}_random_{seed}_split_{param[1]}/save/model/",
                    )

                arch_path = MODEL_DIR.split("save")[0] + "results.csv"
                df = pd.read_csv(arch_path)
                loss_min = []
                arch_min = []
                id_min = []

                for i in range(len(df)):
                    loss_min_ = np.argsort(df["objective"])[::-1].values[i]
                    arch_min_ = df["p:arch_seq"][loss_min_]
                    id_min_ = df["job_id"][loss_min_]

                    if not any(np.array_equal(arch_min_, x) for x in arch_min):
                        loss_min.append(loss_min_)
                        arch_min.append(arch_min_)
                        id_min.append(id_min_)

                for i in range(10):
                    if if_simple:
                        file = os.path.join(
                            ROOT_DIR,
                            f"NEW_SIMPLE_RESULT/simplepost_model_{param[0].lower()}_random_{seed}_split_{param[1]}/test_{id_min[i]}.pickle",
                        )
                    else:
                        file = os.path.join(
                            ROOT_DIR,
                            f"NEW_POST_RESULT/post_result_{param[0].lower()}_random_{seed}_split_{param[1]}/test_{id_min[i]}.pickle",
                        )
                    with open(file, "rb") as handle:
                        _ = pickle.load(handle)
                        _ = pickle.load(handle)
                        _ = pickle.load(handle)
                        hist = pickle.load(handle)

                    val_loss_temp = hist["val_loss"]
                    loss_temp = hist["loss"]

                    val_loss = np.ones(1000) * val_loss_temp[-1]
                    loss = np.ones(1000) * loss_temp[-1]

                    val_loss[: len(val_loss_temp)] = val_loss_temp
                    loss[: len(loss_temp)] = loss_temp

                    losses.append(loss)
                    val_losses.append(val_loss)

            loss = np.array(losses)
            val_loss = np.array(val_losses)

            loss_dict[param[0].lower()] = [loss, val_loss]

        with open(loss_file, "wb") as handle:
            pickle.dump(loss_dict, handle)

    return loss_dict


def plot_loss_curve(ROOT_DIR, RESULT_DIR, COLOR, PLOT_DIR, if_simple=False, format="pdf"):
    loss_dict = load_data_(ROOT_DIR=ROOT_DIR, RESULT_DIR=RESULT_DIR, if_simple=if_simple)

    fig, ax = pplt.subplots(
        nrows=2,
        ncols=3,
        refwidth=2,
        refheight=2,
        sharey=False,
        wspace=5
    )

    keys = np.array(list(loss_dict.keys()))
    
    keys = keys[[1, 0, 2, 3, 4]]

    params = [
        ("Lipo", "523"),
        ("Delaney", "523"),
        ("FreeSolv", "523"),
        ("QM7", "523"),
        ("QM9", "811"),
    ]

    for i, key in enumerate(keys):
        loss, val_loss = loss_dict[key]
        
        a = []
        
        for j in range(len(val_loss)):
            if np.argmin(val_loss[j]) <= 29:
                a.append(1)
            else:
                a.append((np.abs(val_loss[j, 0] - np.min(val_loss[j, :30]))) / np.abs((val_loss[j, 0] - val_loss[j].min())))

        a = np.array(a)
        b = val_loss.min(axis=1)
        c = val_loss[:, 29]
        
        print(
            f"{key} converge ratio: {a.mean():0.2f} ({a.std():0.2f}) 30 epoch: {b.mean():0.2f} ({b.std():0.2f}) min: {c.mean():0.2f} ({c.std():0.2f})"
        )

        val_loss_mean = val_loss.argmin(axis=1).mean()

        kernel = np.ones(50) / 50

        loss_smooth = savgol_filter(
            loss.mean(axis=0), 11, 3
        )  # np.convolve(loss.mean(axis=0), kernel, mode="valid")
        val_loss_smooth = savgol_filter(
            val_loss.mean(axis=0), 11, 3
        )  # np.convolve(val_loss.mean(axis=0), kernel, mode="valid")

        loss_smooth_std = savgol_filter(
            loss.std(axis=0), 11, 3
        ) # np.convolve(loss.std(axis=0), kernel, mode="valid")
        val_loss_smooth_std = savgol_filter(
            val_loss.std(axis=0), 11, 3
        ) # np.convolve(val_loss.std(axis=0), kernel, mode="valid")

        xx = np.arange(len(loss_smooth))

        vmax = max(loss_smooth)

        ax[i].plot(xx, loss_smooth, c=COLOR[0], label="Training")
        ax[i].fill_between(
            xx,
            loss_smooth - loss_smooth_std,
            loss_smooth + loss_smooth_std,
            c=COLOR[0],
            alpha=0.2,
        )

        ax[i].plot(xx, val_loss_smooth, c=COLOR[1], label="Validation")
        # ax[i].vlines(val_loss_mean, vmax, c=COLOR[3])
        ax[i].fill_between(
            xx,
            val_loss_smooth - val_loss_smooth_std,
            val_loss_smooth + val_loss_smooth_std,
            c=COLOR[1],
            alpha=0.2,
        )

        if i == 4:
            ax[i].legend(loc="upper right", prop={"size": 12}, ncol=1)
            
        if i < 4:
            ylim=[-2, 6]
        else:
            ylim=[None, None]

        ax[i].format(
            xlabel="Epoch",
            ylabel="Loss",
            xlim=[-20, 1000],
            ylim=ylim,
            xticklabelsize=12,
            yticklabelsize=12,
            xlabelsize=15,
            ylabelsize=15,
            xticks=[0, 250, 500, 750, 1000],
        )
        
        if params[i][0] == "Delaney":
            txt = "ESOL"
        else:
            txt = params[i][0]

        ax[i].text(
            0.05,
            0.98,
            txt,
            ha="left",
            va="top",
            transform=ax[i].transAxes,
            fontsize=12,
            color="k",
            weight="bold",
        )

    ax[-1].axis("off")

    if if_simple:
        out_file = os.path.join(PLOT_DIR, f"loss_curve_simple.{format}")
    else:
        out_file = os.path.join(PLOT_DIR, f"loss_curve.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)
