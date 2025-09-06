# %% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %% Open files

# 2D - ED
filename = "results/cardiac_acdc_TDAseg_scores_ED_['2D', '2D', 'skip', 2.5, 1, 1.0, 5, 20, 100, 0, inf, 2, inf]_len150.csv"
df_2d_ed = pd.read_csv(filename)

# 2D - ES
filename = "results/cardiac_acdc_TDAseg_scores_ES_['2D', '2D', 'skip', 2.5, 1, 1.0, 5, 20, 0, 0, inf, 2, inf]_len150.csv"
df_2d_es = pd.read_csv(filename)

# 3D - ED
filename = "results/cardiac_acdc_TDAseg_scores_ED_['3D', '3D', 'skip', 2.5, 1, 1.0, 5, 20, 1000, 0, inf, 2, inf]_len150.csv"
df_3d_ed = pd.read_csv(filename)

# 3D - ES
filename = "results/cardiac_acdc_TDAseg_scores_ES_['3D', '3D', 'skip', 2.5, 2, 1.0, 5, 20, 0, 0, inf, 2, inf]_len150.csv"

df_3d_es = pd.read_csv(filename)

# Separate into train and test
df_2d_ed_train, df_2d_ed_test = df_2d_ed.iloc[:100], df_2d_ed.iloc[100:]
df_2d_es_train, df_2d_es_test = df_2d_es.iloc[:100], df_2d_es.iloc[100:]
df_3d_ed_train, df_3d_ed_test = df_3d_ed.iloc[:100], df_3d_ed.iloc[100:]
df_3d_es_train, df_3d_es_test = df_3d_es.iloc[:100], df_3d_es.iloc[100:]
dataframes_train = [df_2d_ed_train, df_3d_ed_train, df_2d_es_train, df_3d_es_train]
dataframes_test = [df_2d_ed_test, df_3d_ed_test, df_2d_es_test, df_3d_es_test]

# %% Plot boxplot scores - train and test

dataframes = dataframes_train + dataframes_test

resize_coeff = 3.5
fig, axs = plt.subplots(
    ncols=3, nrows=1, figsize=(3 * resize_coeff, 1.15 * resize_coeff)
)
segmentation_labels = ["LV", "RV", "Myo"]
for label_i, label in enumerate(segmentation_labels):
    segs = [[df.iloc[i][label] for i in range(len(df))] for df in dataframes]

    axs[label_i].boxplot(segs, notch=True, showmeans=True)
    axs[label_i].set_title(label)
    axs[label_i].set_ylim(0, 1)
    axs[label_i].set_xlim(0.5, 8.5)
    axs[label_i].set_xticklabels(["2D", "3D"] * 4)
    for xtick, color in zip(
        axs[label_i].get_xticklabels(), ["C1", "C1", "C2", "C2"] * 2
    ):
        xtick.set_color(color)
    axs[label_i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    axs[label_i].add_patch(Rectangle((4.5, 0), 4.5, 1, color="skyblue", alpha=0.5))
    for x in [1.55, 5.55]:
        axs[label_i].text(
            x,
            -0.175,
            "ED",
            horizontalalignment="center",
            verticalalignment="bottom",
            color="C1",
        )
    for x in [3.55, 7.55]:
        axs[label_i].text(
            x,
            -0.175,
            "ES",
            horizontalalignment="center",
            verticalalignment="bottom",
            color="C2",
        )
    axs[label_i].text(
        2.5,
        -0.275,
        "Train set",
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    axs[label_i].text(
        6.55,
        -0.275,
        "Test set",
        horizontalalignment="center",
        verticalalignment="bottom",
    )
plt.subplots_adjust(left=None, bottom=0.25, right=None, top=None, wspace=0.2, hspace=0)
plt.savefig(
    "results/scores_ACDC_article.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

# %% Print scores - train and test

data_types = ["training", "testing"]
dataframes_by_type = [dataframes_train, dataframes_test]
method_types = [
    "2D ED",
    "3D ED",
    "2D ES",
    "3D ES",
]

for data_type, dataframes in zip(data_types, dataframes_by_type):
    print(f"\nScores for {data_type.upper()}:")
    segmentation_labels = ["LV", "RV", "Myo"]
    for label_i, label in enumerate(segmentation_labels):
        print(f"\nLabel: {label}")
        segs = [[df.iloc[i][label] for i in range(len(df))] for df in dataframes]
        for i_seg, seg in enumerate(segs):
            mean = np.mean(seg)
            std = np.std(seg)
            median = np.median(seg)
            latex_msg = f"${mean:.2f}\\pm{std:.2f}$"
            method_type = method_types[i_seg]
            print(
                f"  {method_type}: mean={mean:.3f}, std={std:.3f}, median={median:.3f} --- "
                + latex_msg
            )


# Scores for TRAINING:
# Label: LV
#   2D ED: mean=0.725, std=0.183, median=0.779 --- $0.72\pm0.18$
#   3D ED: mean=0.693, std=0.314, median=0.842 --- $0.69\pm0.31$
#   2D ES: mean=0.508, std=0.265, median=0.565 --- $0.51\pm0.27$
#   3D ES: mean=0.378, std=0.379, median=0.413 --- $0.38\pm0.38$
# Label: RV
#   2D ED: mean=0.495, std=0.222, median=0.555 --- $0.50\pm0.22$
#   3D ED: mean=0.429, std=0.338, median=0.586 --- $0.43\pm0.34$
#   2D ES: mean=0.274, std=0.203, median=0.258 --- $0.27\pm0.20$
#   3D ES: mean=0.235, std=0.247, median=0.160 --- $0.23\pm0.25$
# Label: Myo
#   2D ED: mean=0.398, std=0.139, median=0.405 --- $0.40\pm0.14$
#   3D ED: mean=0.393, std=0.212, median=0.456 --- $0.39\pm0.21$
#   2D ES: mean=0.316, std=0.171, median=0.327 --- $0.32\pm0.17$
#   3D ES: mean=0.269, std=0.281, median=0.171 --- $0.27\pm0.28$
# Scores for TESTING:
# Label: LV
#   2D ED: mean=0.693, std=0.149, median=0.720 --- $0.69\pm0.15$
#   3D ED: mean=0.631, std=0.344, median=0.790 --- $0.63\pm0.34$
#   2D ES: mean=0.388, std=0.305, median=0.402 --- $0.39\pm0.31$
#   3D ES: mean=0.221, std=0.352, median=0.000 --- $0.22\pm0.35$
# Label: RV
#   2D ED: mean=0.442, std=0.265, median=0.517 --- $0.44\pm0.26$
#   3D ED: mean=0.363, std=0.315, median=0.447 --- $0.36\pm0.32$
#   2D ES: mean=0.272, std=0.204, median=0.280 --- $0.27\pm0.20$
#   3D ES: mean=0.190, std=0.252, median=0.000 --- $0.19\pm0.25$
# Label: Myo
#   2D ED: mean=0.360, std=0.121, median=0.345 --- $0.36\pm0.12$
#   3D ED: mean=0.357, std=0.213, median=0.401 --- $0.36\pm0.21$
#   2D ES: mean=0.225, std=0.171, median=0.217 --- $0.23\pm0.17$
#   3D ES: mean=0.157, std=0.228, median=0.000 --- $0.16\pm0.23$
