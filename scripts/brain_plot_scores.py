import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches

# %% Open our segmentations

file = "results/brain_brats2025_TDAseg_scores_['max', 1, (False, True), 1, True, 2, 1, 1]_len1251.csv"
df_parameters1 = pd.read_csv(file)

# Print mean score for each class
segmentation_labels = ["WT", "TC", "ED", "ET"]
for label in segmentation_labels:
    seg = [df_parameters1.iloc[j][label] for j in range(len(df_parameters1))]
    msg_scores = f"${round(np.mean(seg), 3)}\\pm{round(np.std(seg), 3)}$, median {round(np.median(seg), 3)}"
    msg = label + " - TDA: " + msg_scores
    print(msg)

# WT - TDA: $0.712\pm0.282$, median 0.828
# TC - TDA: $0.355\pm0.367$, median 0.176
# ED - TDA: $0.525\pm0.269$, median 0.559
# ET - TDA: $0.453\pm0.307$, median 0.514

# %% Open AUCSeg

df_parameters2 = pd.read_csv("results/brain_brats2025_AUCseg_scores.csv")

# Consider the "cc" or "t2" method of AUCSeg, and the best one
df_parameters2_cc = df_parameters2[
    df_parameters2["brats_name"].astype(str).str[-9:-7].eq("cc")
]
df_parameters2_t2 = df_parameters2[
    df_parameters2["brats_name"].astype(str).str[-9:-7].eq("t2")
]
df_parameters2_best = pd.DataFrame()
for name in set(df_parameters2_cc["brats_name"].str[:-17]).intersection(
    set(df_parameters2_t2["brats_name"].str[:-17])
):
    row_cc = df_parameters2_cc[df_parameters2_cc["brats_name"].str[:-17] == name]
    row_t2 = df_parameters2_t2[df_parameters2_t2["brats_name"].str[:-17] == name]
    if (
        row_cc["WT"].values[0] + row_cc["TC"].values[0] + row_cc["ET"].values[0]
        >= row_t2["WT"].values[0] + row_t2["TC"].values[0] + row_t2["ET"].values[0]
    ):
        df_parameters2_best = pd.concat([df_parameters2_best, row_cc])
    else:
        df_parameters2_best = pd.concat([df_parameters2_best, row_t2])
df_parameters2 = df_parameters2_best.reset_index(drop=True)

# Rename the entries "brats_name": remove the "_AUCseg_cc.nii.gz" or "_AUCseg_t2.nii.gz" at the end.
for df in [df_parameters2_cc, df_parameters2_t2, df_parameters2_best]:
    df["brats_name"] = df["brats_name"].str[:-17]

# %% Define names of images to consider.

names = {name: True for name in df_parameters1["brats_name"]}
for name in df_parameters2_cc["brats_name"]:
    name = name
    if name not in names:
        names[name] = False

# # %% Open model -- OLD VERSION
#
# # Parameters of the model.
# thresh_smallTC = 50  # upper bound on ratio WT/TC
# thresh_WTconnected = 10  # lower bound on ratio first/second largest CC
# admissible_argmax_FLAIR = [1, 3]  # values that argmax can take in FLAIR
# admissible_argmax_T1ce = [3]  # values that argmax can take in T1ce
# RatioComponentsWidth = 1  # lower bound on ratio
#
# # Identify images that satisfy the model.
# # df_VerifyModel = pd.read_csv(
# #     "results/brain_brats2025_verifymodel_[1, 'max', (True, True), 1, True, 3]_len1251.csv"
# # )
# df_VerifyModel = pd.read_csv(
#     "results/brain_brats2025_verifymodel_[1, 'max', (False, True), 1, True, 2]_len1251.csv"
# )
# names_verify_model = {name: False for name in df_VerifyModel["brats_name"]}
# for i in range(len(df_VerifyModel)):
#     names_verify_model[df_VerifyModel["brats_name"][i]] = (
#         df_VerifyModel["nonempty"][i] == True
#         and df_VerifyModel["smallTC"][i] <= thresh_smallTC
#         and df_VerifyModel["WTconnected"][i] >= thresh_WTconnected
#         and df_VerifyModel["boundary_TC"][i] > 0.5
#         and df_VerifyModel["boundary_ET"][i] < 0.5
#         and df_VerifyModel["argmax_FLAIR"][i] in admissible_argmax_FLAIR
#         and df_VerifyModel["argmax_T1ce"][i] in admissible_argmax_T1ce
#         and (
#             df_VerifyModel["RatioComponentsWidth"][i] >= RatioComponentsWidth
#             or np.isnan(df_VerifyModel["RatioComponentsWidth"][i])
#         )
#     )
#
# # Print result.
# num_satisfying = np.sum(list(names_verify_model.values()))
# percent_satisfying = round(num_satisfying / len(df_VerifyModel) * 100, 3)
# print(
#     f"Brains satisfying the model: {num_satisfying} = {percent_satisfying}% ({num_satisfying} out of {len(df_VerifyModel)})"
# )

# %% Open model -- NEW VERSION

df_VerifyModel = pd.read_csv(
    "results/brain_brats2025_verifymodel_[1, 'max', (False, True), 1, True, 2]_len1251.csv"
)
names_verify_model = {name: False for name in df_VerifyModel["brats_name"]}
for i in range(len(df_VerifyModel)):
    names_verify_model[df_VerifyModel["brats_name"][i]] = (
        True
        and df_VerifyModel["nonempty"][i] == True
        # and df_VerifyModel["smallTC"][i] <= thresh_smallTC
        # and df_VerifyModel["WTconnected"][i] >= 2
        # and df_VerifyModel["boundary_TC"][i] > 0.5
        # and df_VerifyModel["boundary_ET"][i] < 0.5
        # and df_VerifyModel["argmax_FLAIR"][i] in admissible_argmax_FLAIR
        # and df_VerifyModel["argmax_T1ce"][i] in admissible_argmax_T1ce
        # and (
        #     df_VerifyModel["RatioComponentsWidth"][i] >= RatioComponentsWidth
        #     or np.isnan(df_VerifyModel["RatioComponentsWidth"][i])
        # )
        and df_VerifyModel["h1_median"][i] > 3 / 2
        and df_VerifyModel["h2_median"][i] > 3 / 2
        and df_VerifyModel["h3_dil2"][i] > 0.9
    )

# Print result.
num_satisfying = np.sum(list(names_verify_model.values()))
percent_satisfying = round(num_satisfying / len(df_VerifyModel) * 100, 3)
print(
    f"Brains satisfying the model: {num_satisfying} = {percent_satisfying}% ({num_satisfying} out of {len(df_VerifyModel)})"
)

# %% Plot

fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(15 * 0.9, 4.5 * 0.9))
segmentation_labels = ["WT", "TC", "ED", "ET"]

names_seg = ["TDA", "AUC:cc", "AUC:t2"]
dataframes = [df_parameters1, df_parameters2_cc, df_parameters2_t2]
for label_i, label in enumerate(segmentation_labels):
    segmentations = []
    segmentations_model = []
    for df in dataframes:
        seg = [df.iloc[j][label] for j in range(len(df))]
        seg_model = [
            df.iloc[j][label]
            for j in range(len(df))
            if names[df["brats_name"].iloc[j]]
            and names_verify_model[df["brats_name"].iloc[j]]
        ]
        segmentations.append(seg)
        segmentations_model.append(seg_model)

    axs[label_i].boxplot(
        segmentations + segmentations_model, notch=True, showmeans=True
    )
    axs[label_i].set_title(label)
    axs[label_i].set_ylim(0, 1)
    axs[label_i].set_xlim(0.5, 6.5)
    axs[label_i].set_xticklabels(names_seg + names_seg, rotation=45, ha="right")
    axs[label_i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[label_i].add_patch(
        matplotlib.patches.Rectangle((3.5, 0), 3, 1, color="skyblue", alpha=0.5)
    )
    axs[label_i].text(
        1.9,
        -0.35,
        "Whole dataset",
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    axs[label_i].text(
        5.1,
        -0.35,
        "Model satisfied",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # Print scores
    for model_satisfied in [False, True]:
        if model_satisfied:
            msg_model = label + " - model satisfied    "
            segs = segmentations_model
        else:
            msg_model = label + " - whole dataset      "
            segs = segmentations
        for i_name, name in enumerate(names_seg):
            seg = segs[i_name]
            msg_scores = f"${round(np.mean(seg), 2)}\\pm{round(np.std(seg), 2)}$, median {round(np.median(seg), 2)}"
            msg = msg_model + " - " + name + ": " + msg_scores
            print(msg)

plt.subplots_adjust(left=None, bottom=0.28, right=None, top=None, wspace=0.2, hspace=0)
plt.savefig("results/brain_scores.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %% Print increase between without or with model

for label_i, label in enumerate(segmentation_labels):
    segmentations = []
    segmentations_model = []
    for df in dataframes:
        seg = [df.iloc[j][label] for j in range(len(df))]
        seg_model = [
            df.iloc[j][label]
            for j in range(len(df))
            if names[df["brats_name"].iloc[j]]
            and names_verify_model[df["brats_name"].iloc[j]]
        ]
        segmentations.append(seg)
        segmentations_model.append(seg_model)

    for i_name, name in enumerate(["TDA"]):
        # for i_name, name in enumerate(names_seg):
        seg = segmentations[i_name]
        seg_model = segmentations_model[i_name]
        print(
            f"{label} - {name}: increase of {round(np.mean(seg_model)-np.mean(seg),3)} (from {round(np.mean(seg),3)} to {round(np.mean(seg_model),3)})"
        )
