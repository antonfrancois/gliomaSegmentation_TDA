import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches

# %% Open our segmentations

# Parameters.
normalize = "max"  # preprocess_brain, divide by max or 255
sigma = 1  # preprocess_brain, Gaussian blur
enhance = (False, False)  # preprocess_brain, apply enhancement or not
radius_enhance = 1  # preprocess_brain, radius of enhancement
dilate = True  # preprocess_brain, apply dilation or not on T1ce
radius_dilation = 1  # preprocess_brain and segment_other_components, radius of dilation
whole_threshold = 1  # segment_whole_object, threshold for suggest_t
max_bars = 2  # segment_geometric_object, number of H2 features to consider
parameters = [
    normalize,
    sigma,
    enhance,
    radius_enhance,
    dilate,
    radius_dilation,
    whole_threshold,
    max_bars,
]

# Find file.
files = glob.glob("results/TDAseg_scores_*")
files = [
    file
    for file in files
    if file[0 : len("results/TDAseg_scores_") + len(str(parameters))]
    == "results/TDAseg_scores_" + str(parameters)
]
files_length = [
    int(file[len("results/TDAseg_scores_") + len(str(parameters)) + 4 : -4])
    for file in files
]
ind = np.argmax(files_length)
file, i_min = files[ind], files_length[ind]
print("Found", file)
df_parameters1 = pd.read_csv(file)

# %% Open U-net

df_parameters2 = pd.read_csv("results/summary_cmp_unet_pp.csv")
names = {name: True for name in df_parameters1["brats_name"]}

# %% Open model

# Parameters of the model.
thresh_smallTC = 50  # upper bound on ratio WT/TC
thresh_WTconnected = 10  # lower bound on ratio first/second largest CC
admissible_argmax_FLAIR = [1, 4]  # values that argmax can take in FLAIR
admissible_argmax_T1ce = [4]  # values that argmax can take in T1ce
RatioComponentsWidth = 1  # lower bound on ratio of ...

# Identify images that satisfy the model.
# df_VerifyModel = pd.read_csv("results/VerifyModel_20250708_dilation3_len1251.csv")
df_VerifyModel = pd.read_csv(
    "results/VerifyModel_[1, 'max', (True, True), 1, True, 3]_len1251.csv"
)
names_verify_model = {name: False for name in df_VerifyModel["brats_name"]}
for i in range(len(df_VerifyModel)):
    names_verify_model[df_VerifyModel["brats_name"][i]] = (
        df_VerifyModel["nonempty"][i] == True
        and df_VerifyModel["smallTC"][i] <= thresh_smallTC
        and df_VerifyModel["WTconnected"][i] >= thresh_WTconnected
        and df_VerifyModel["boundary_TC"][i] > 0.5
        and df_VerifyModel["boundary_ET"][i] < 0.5
        and df_VerifyModel["argmax_FLAIR"][i] in admissible_argmax_FLAIR
        and df_VerifyModel["argmax_T1ce"][i] in admissible_argmax_T1ce
        and (
            df_VerifyModel["RatioComponentsWidth"][i] >= RatioComponentsWidth
            or np.isnan(df_VerifyModel["RatioComponentsWidth"][i])
        )
    )

# Print result.
num_satisfying = np.sum(list(names_verify_model.values()))
percent_satisfying = round(num_satisfying / len(df_VerifyModel) * 100, 3)
print(f"Brains satisfying the model: {num_satisfying} = {percent_satisfying}%")

# %% Plot

fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(15, 4))
segmentation_labels = ["WT", "TC", "ED", "ET"]

for label_i, label in enumerate(segmentation_labels):
    seg_1 = [
        df_parameters1.iloc[i][label]
        for i in range(len(df_parameters1))
        if names[df_parameters1["brats_name"][i]]
    ]
    seg_2 = [
        df_parameters2.iloc[i][label + "_unet"]
        for i in range(len(df_parameters2))
        if names[df_parameters2["brats_name"][i]]
    ]
    seg_1_model = [
        df_parameters1.iloc[i][label]
        for i in range(len(df_parameters1))
        if names[df_parameters1["brats_name"][i]]
        and names_verify_model[df_parameters1["brats_name"][i]]
    ]
    seg_2_model = [
        df_parameters2.iloc[i][label + "_unet"]
        for i in range(len(df_parameters2))
        if names[df_parameters2["brats_name"][i]]
        and names_verify_model[df_parameters2["brats_name"][i]]
    ]

    axs[label_i].boxplot(
        [seg_1, seg_2, seg_1_model, seg_2_model], notch=True, showmeans=True
    )
    axs[label_i].set_title(label)
    axs[label_i].set_ylim(0, 1)
    axs[label_i].set_xlim(0.5, 4.5)
    axs[label_i].set_xticklabels(["PH", "U-Net", "PH", "U-Net"])
    axs[label_i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[label_i].add_patch(
        matplotlib.patches.Rectangle((2.5, 0), 2.5, 1, color="skyblue", alpha=0.5)
    )
    axs[label_i].text(
        1.55,
        -0.15,
        "Whole dataset",
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    axs[label_i].text(
        3.55,
        -0.15,
        "Model satisfied",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # Print scores
    for i_seg, seg in enumerate([seg_1, seg_2, seg_1_model, seg_2_model]):
        msg = (
            "$"
            + str(round(np.mean(seg), 3))
            + "\\"
            + "pm"
            + str(round(np.std(seg), 3))
            + "$, median "
            + str(round(np.median(seg), 3))
        )
        if i_seg == 0:
            msg = label + " - TDA - whole dataset:     " + msg
        elif i_seg == 1:
            msg = label + " - U-net - whole dataset:   " + msg
        elif i_seg == 2:
            msg = label + " - TDA - model satisfied:   " + msg
        elif i_seg == 3:
            msg = label + " - U-net - model satisfied: " + msg
        print(msg)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0)
plt.savefig("results/scores_BraTS_article.pdf", format="pdf", bbox_inches="tight")
plt.show()
