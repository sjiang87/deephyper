import os
import pandas as pd
import numpy as np
import proplot as pplt
from deephyper.gnn_uq.figure import sat


def load_data_(ROOT_DIR):
    params = [
        ("Delaney", "523"),
        ("Lipo", "523"),
        ("FreeSolv", "523"),
        ("QM7", "523"),
        ("QM9", "811"),
    ]
    seeds = 8
    objs = [[[] for _ in range(seeds)] for _ in range(len(params))]

    for seed in range(seeds):
        dfs = [
            pd.read_csv(
                os.path.join(
                    ROOT_DIR,
                    f"NEW_RE_{label[0].lower()}_random_{seed}_split_{label[1]}/results.csv",
                )
            )
            for label in params
        ]
        for i, df in enumerate(dfs):
            obj = df.objective.values
            prev_value = None

            for j in range(len(obj)):
                if obj[j] <= -2:
                    if prev_value is not None:
                        obj[j] = prev_value
                else:
                    prev_value = obj[j]

            obj_smooth = np.convolve(obj, np.ones(50) / 50, mode="valid")
            objs[i][seed] = obj_smooth[:950]

    return objs


def load_data_simple_(ROOT_DIR):
    params = [
        ("Delaney", "523"),
        ("Lipo", "523"),
        ("FreeSolv", "523"),
        ("QM7", "523"),
        ("QM9", "811"),
    ]
    seeds = 8
    objs = [[[] for _ in range(seeds)] for _ in range(len(params))]

    for seed in range(seeds):
        dfs = [
            pd.read_csv(
                os.path.join(
                    ROOT_DIR,
                    f"NEW_RE_{label[0].lower()}_random_{seed}_split_{label[1]}/results.csv",
                )
            )
            for label in params
        ]
        for i, df in enumerate(dfs):
            obj = df.objective.values
            prev_value = None

            for j in range(len(obj)):
                if obj[j] <= -2:
                    if prev_value is not None:
                        obj[j] = prev_value
                else:
                    prev_value = obj[j]

            obj_smooth = np.convolve(obj, np.ones(50) / 50, mode="valid")
            objs[i][seed] = obj_smooth[:950]

    objs_simple = [[[] for _ in range(seeds)] for _ in range(len(params))]

    for seed in range(seeds):
        dfs = [
            pd.read_csv(
                os.path.join(
                    ROOT_DIR,
                    f"SIMPLE_RE_{label[0].lower()}_random_{seed}_split_{label[1]}/results.csv",
                )
            )
            for label in params
        ]
        for i, df in enumerate(dfs):
            obj = df.objective.values
            prev_value = None

            for j in range(len(obj)):
                if obj[j] <= -2:
                    if prev_value is not None:
                        obj[j] = prev_value
                else:
                    prev_value = obj[j]

            obj_smooth = np.convolve(obj, np.ones(50) / 50, mode="valid")
            objs_simple[i][seed] = obj_smooth[:950]

    return objs, objs_simple


def plot_search_reward(ROOT_DIR, LABEL, COLOR, PLOT_DIR, format="pdf"):
    objs = load_data_(ROOT_DIR)

    fig, ax = pplt.subplots(
        nrows=1, ncols=5, sharey=False, refwidth=2, refheight=2, wspace=4.5
    )

    for i, obj in enumerate(objs):
        obj_mean = np.mean(obj, axis=0)
        obj_std = np.std(obj, axis=0)
        xx = np.arange(len(obj_mean))
        if i == 0:
            ylabel = "Reward (LL)"
        else:
            ylabel = None

        if i != 4:
            ylim = [-1.4, -0.6]
            yticks = [-1.4, -1.2, -1, -0.8, -0.6]
        else:
            ylim = [0.0, 1.6]
            yticks = [0, 0.4, 0.8, 1.2, 1.6]

        ax[i].plot(xx, obj_mean, label=LABEL[i], color=COLOR[1])
        ax[i].fill_between(
            xx, obj_mean - obj_std, obj_mean + obj_std, color=sat(COLOR[1], 0.8)
        )
        ax[i].legend(loc=4, prop={"size": 12})
        ax[i].format(
            xlabel="Evaluation Index",
            ylabel=ylabel,
            xlim=[0, 1000],
            xticklabelsize=12,
            yticklabelsize=12,
            xlabelsize=15,
            ylabelsize=15,
            ylim=ylim,
            yticks=yticks,
            xticks=[0, 250, 500, 750, 1000],
        )

    out_file = os.path.join(PLOT_DIR, f"search_reward.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)


def plot_search_reward_simple(ROOT_DIR, LABEL, COLOR, PLOT_DIR, format="pdf"):
    objs, objs_simple = load_data_simple_(ROOT_DIR)

    fig, ax = pplt.subplots(
        nrows=1, ncols=5, sharey=False, refwidth=2, refheight=2, wspace=4.5
    )

    for i, obj in enumerate(objs):
        obj_mean = np.mean(obj, axis=0)
        obj_std = np.std(obj, axis=0)
        xx = np.arange(len(obj_mean))

        ax[i].plot(xx, obj_mean, color=COLOR[1], label="AutoGNNUQ")
        ax[i].fill_between(
            xx, obj_mean - obj_std, obj_mean + obj_std, color=sat(COLOR[1], 0.8)
        )

    for i, obj in enumerate(objs_simple):
        obj_mean = np.mean(obj, axis=0)
        obj_std = np.std(obj, axis=0)
        xx = np.arange(len(obj_mean))
        
        ax[i].plot(xx, obj_mean, color=COLOR[0], label="AutoGNNUQ-Simple")
        ax[i].fill_between(
            xx, obj_mean - obj_std, obj_mean + obj_std, color=sat(COLOR[0], 0.8)
        )

        ax[i].text(
            0.02,
            0.98,
            LABEL[i],
            ha="left",
            va="top",
            transform=ax[i].transAxes,
            fontsize=12,
            color="k",
            weight="bold",
        )

    ax[0].legend(ncol=1, loc=4, prop={"size": 11})

    for i in range(len(objs)):
        if i == 0:
            ylabel = "Reward (LL)"
        else:
            ylabel = None

        if i != 4:
            ylim = [-1.8, -0.6]
            yticks = [-0.6, -0.9, -1.2, -1.5, -1.8]
        else:
            ylim = [-0.8, 1.6]
            yticks = [-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6]
            
        ax[i].format(
            xlabel="Evaluation Index",
            ylabel=ylabel,
            xlim=[0, 1000],
            xticklabelsize=12,
            yticklabelsize=12,
            xlabelsize=15,
            ylabelsize=15,
            ylim=ylim,
            yticks=yticks,
            xticks=[0, 250, 500, 750, 1000],
        )

    out_file = os.path.join(PLOT_DIR, f"search_reward_with_simple.{format}")

    fig.save(out_file, bbox_inches="tight", dpi=600)
