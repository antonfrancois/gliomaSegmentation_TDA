# %% Imports

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %% Functions


def OpenModel():
    # Find last updated file
    if not oracle:
        filename = "results/cardiac_" + modality + "_" + repr(dimension) + "D_scores_"
    elif oracle:
        filename = (
            "results/cardiac_" + modality + "_" + repr(dimension) + "D_oracle_scores_"
        )
    files = glob.glob(filename + "*")
    files = [
        file
        for file in files
        if file[0 : len(filename) + len(str(parameters))] == filename + str(parameters)
    ]
    files_length = [
        int(file[len(filename) + len(str(parameters)) + 4 : -4]) for file in files
    ]
    if len(files) != 0:
        ind = np.argmax(files_length)
        file, i_min = files[ind], files_length[ind] + 1
        print("Found file", file, "- i_min =", i_min)
        return pd.read_csv(file)
    else:
        print("Error! No files found.", filename, filename + str(parameters))
        return None


# %% Build dataframes

DATAFRAMES = []

modality, dimension, oracle = "ED", 2, False
# # Parameters 5
# sigma = 1.5  # Preprocess, Gaussian blur
# radius_ball = 0  # Preprocess, dilation parameter
# dt_threshold = 1  # Step 1, threshold for suggest_t
# H0_features_max = 10  # Step 1, number of H2 bars to consider
# thresh_small_LV = 200  # Step 1, minimal width of LV
# ratio_small_RV = 0.1  # Step 1, minimal width of RV, compared to LV
# ratio_big_RV = 5  # Step 1, maximal width of RV, compared to LV
# add_to_iterations = 2  # Step 1, fill gap between LV and RV
# maximal_distance = np.inf  # Step 1, max Hausdorff distance between RV and LV
# parameters = [
#     sigma,
#     radius_ball,
#     dt_threshold,
#     H0_features_max,
#     thresh_small_LV,
#     ratio_small_RV,
#     ratio_big_RV,
#     add_to_iterations,
#     maximal_distance,
# ]
# Parameters - Step1_Hausdorff - 6
sigma = 1.5  # Preprocess, Gaussian blur
radius_ball = 0  # Preprocess, dilation parameter
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max = 20  # Step 1, number of H2 bars to consider
thresh_small_LV = 500  # Step 1, minimal width of LV
ratio_small_RV = 1 / 4  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = 4  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 2  # Step 1, fill gap between LV and RV
maximal_distance = 50  # Step 1, max Hausdorff distance between RV and LV
parameters = [
    sigma,
    radius_ball,
    dt_threshold,
    H0_features_max,
    thresh_small_LV,
    ratio_small_RV,
    ratio_big_RV,
    add_to_iterations,
    maximal_distance,
]
DATAFRAMES.append(OpenModel())

modality, dimension, oracle = "ES", 2, False
# Parameters 4
sigma = 1.5  # Preprocess, Gaussian blur
radius_ball = 0  # Preprocess, dilation parameter
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max = 20  # Step 1, number of H2 bars to consider
thresh_small_LV = 0  # Step 1, minimal width of LV
ratio_small_RV = 0  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = np.inf  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 5  # Step 1, fill gap between LV and RV
maximal_distance = np.inf  # Step 1, max Hausdorff distance between RV and LV
parameters = [
    sigma,
    radius_ball,
    dt_threshold,
    H0_features_max,
    thresh_small_LV,
    ratio_small_RV,
    ratio_big_RV,
    add_to_iterations,
    maximal_distance,
]
DATAFRAMES.append(OpenModel())

modality, dimension, oracle = "ED", 3, False
# # Parameters 13
# sigma = 2.5                      # Preprocess, Gaussian blur
# radius_ball = 0                # Preprocess, dilation parameter
# dt_threshold = 1.0             # Step 1, threshold for suggest_t
# # H0_features_max = 5           # Step 1, number of H2 bars to consider
# H0_features_max = 10           # Step 1, number of H2 bars to consider
# thresh_small_LV = 0          # Step 1, minimal width of LV
# ratio_small_RV = 0            # Step 1, minimal width of RV, compared to LV
# ratio_big_RV = np.inf               # Step 1, maximal width of RV, compared to LV
# add_to_iterations = 2          # Step 1, fill gap between LV and RV
# parameters = [sigma,radius_ball,dt_threshold,H0_features_max,thresh_small_LV,ratio_small_RV,ratio_big_RV,add_to_iterations]
# Parameters NEW 2
sigma = 2.5  # Preprocess, Gaussian blur
radius_ball = 0  # Preprocess, dilation parameter
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max = 10  # Step 1, number of H2 bars to consider
thresh_small_LV = 1000  # Step 1, minimal width of LV
ratio_small_RV = 1 / 5  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = 3  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 2  # Step 1, fill gap between LV and RV
maximal_distance = 50  # Step 1, max Hausdorff distance between RV and LV
parameters = [
    sigma,
    radius_ball,
    dt_threshold,
    H0_features_max,
    thresh_small_LV,
    ratio_small_RV,
    ratio_big_RV,
    add_to_iterations,
    maximal_distance,
]
DATAFRAMES.append(OpenModel())

modality, dimension, oracle = "ES", 3, False
# Parameters 13
sigma = 2.5  # Preprocess, Gaussian blur
radius_ball = 0  # Preprocess, dilation parameter
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max = 5  # Step 1, number of H2 bars to consider
thresh_small_LV = 0  # Step 1, minimal width of LV
ratio_small_RV = 0  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = np.inf  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 2  # Step 1, fill gap between LV and RV
maximal_distance = np.inf  # Step 1, max Hausdorff distance between RV and LV
parameters = [
    sigma,
    radius_ball,
    dt_threshold,
    H0_features_max,
    thresh_small_LV,
    ratio_small_RV,
    ratio_big_RV,
    add_to_iterations,
    maximal_distance,
]
DATAFRAMES.append(OpenModel())

# Parameters oracle
sigma = 1  # Preprocess, Gaussian blur
radius_ball = 0  # Preprocess, dilation parameter
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max = 20  # Step 1, number of H2 bars to consider
thresh_small_LV = 0  # Step 1, minimal width of LV
ratio_small_RV = 0  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = np.inf  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 5  # Step 1, fill gap between LV and RV
maximal_distance = np.inf  # Step 1, max Hausdorff distance between RV and LV
parameters = [
    sigma,
    radius_ball,
    dt_threshold,
    H0_features_max,
    thresh_small_LV,
    ratio_small_RV,
    ratio_big_RV,
    add_to_iterations,
    maximal_distance,
]

MODELS_oracle = [("ED", 2, True), ("ED", 3, True), ("ES", 2, True), ("ES", 3, True)]
for modality, dimension, oracle in MODELS_oracle:
    # Find last updated file
    if not oracle:
        filename = "results/cardiac_" + modality + "_" + repr(dimension) + "D_scores_"
    elif oracle:
        filename = (
            "results/cardiac_" + modality + "_" + repr(dimension) + "D_oracle_scores_"
        )
    files = glob.glob(filename + "*")
    files = [
        file
        for file in files
        if file[0 : len(filename) + len(str(parameters))] == filename + str(parameters)
    ]
    files_length = [
        int(file[len(filename) + len(str(parameters)) + 4 : -4]) for file in files
    ]
    if len(files) != 0:
        ind = np.argmax(files_length)
        file, i_min = files[ind], files_length[ind] + 1
        print("Found file", file, "- i_min =", i_min)
        DATAFRAMES.append(pd.read_csv(file))
    else:
        print("Error! No files found.", filename)

# %% Plot boxplot scores

fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(15, 4))
segmentation_labels = ["whole", "Myocardium", "LV", "RV"]
for label_i, label in enumerate(segmentation_labels):
    segs = [[df.iloc[i][label] for i in range(len(df))] for df in DATAFRAMES]
    segs = [segs[0], segs[2], segs[1], segs[3], segs[4], segs[6], segs[5], segs[7]]

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
            -0.15,
            "ED",
            horizontalalignment="center",
            verticalalignment="bottom",
            color="C1",
        )
    for x in [3.55, 7.55]:
        axs[label_i].text(
            x,
            -0.15,
            "ES",
            horizontalalignment="center",
            verticalalignment="bottom",
            color="C2",
        )
    axs[label_i].text(
        6.55,
        -0.25,
        "oracle step 1",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # Print scores
    for i_seg, seg in enumerate(segs):
        msg = (
            "$"
            + str(round(np.mean(seg), 3))
            + "\\"
            + "pm"
            + str(round(np.std(seg), 3))
            + "$, median "
            + str(round(np.median(seg), 3))
        )
        msg = repr(i_seg) + " " + msg
        print(msg)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0)
plt.savefig("results/scores_ACDC_article.pdf", format="pdf", bbox_inches="tight")
plt.show()
