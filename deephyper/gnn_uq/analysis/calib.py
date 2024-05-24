import os
import pickle
import numpy as np
import uncertainty_toolbox as uct
from deephyper.gnn_uq.figure import sat
from deephyper.gnn_uq.analysis import conf_level, miscal_area


def calculate_conf_calib(RESULT_DIR):
    LABELS = [
        "Lipo",
        "ESOL",
        "FreeSolv",
        "QM7" ,
        r"QM9 $\mu$",
        r"QM9 $\alpha$",
        r"QM9 $\epsilon_{\mathrm{HOMO}}$",
        r"QM9 $\epsilon_{\mathrm{LUMO}}$",
        r"QM9 $\Delta \epsilon$",
        r"QM9 $\langle R^2 \rangle$",
        r"QM9 ZPVE" ,
        r"QM9 $c_v$",
        r"QM9 $U_0$" ,
        r"QM9 $U$" ,
        r"QM9 $H$" ,
        r"QM9 $G$" ,
    ]
    result_file = os.path.join(RESULT_DIR, "val_test_result.pickle")
    qm9_result_file = os.path.join(RESULT_DIR, "val_test_result_qm9.pickle")

    with open(result_file, "rb") as handle:
        result = pickle.load(handle)

    with open(qm9_result_file, "rb") as handle:
        result_qm9 = pickle.load(handle)

    output = {}

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

        diff = np.abs(perc_tot - cf1)
        ece = np.mean(diff, axis=1)

        mce = np.max(diff, axis=1)

        output[(LABELS[i], split, "before")] = [ece, mce, MCA]

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

        diff = np.abs(perc_tot - cf1)
        ece = np.mean(diff, axis=1)

        mce = np.max(diff, axis=1)
        output[(LABELS[i], split, "after")] = [ece, mce, MCA]

    out_file = os.path.join(RESULT_DIR, "calib.pickle")

    with open(out_file, "wb") as handle:
        pickle.dump(output, handle)
