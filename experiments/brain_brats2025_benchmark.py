import glob
import numpy as np
import pandas as pd
import parseBrats as pB
from segmentations import segment_brain
from utils import (
    get_multiple_dice,
    ChronometerStart,
    ChronometerTick,
)

# %% Full segmentation - Define parameters

# Parameters
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

# %% Full segmentation - Initialize and run

pb = pB.parse_brats(
    brats_list=None, brats_folder="2025", modality="flair", get_template=False
)
i_list = range(len(pb.brats_list))

# Find last updated file with the same parameters
files = glob.glob("results/brain_brats2025_TDAseg_scores_*")
files = [
    file
    for file in files
    if file[0 : len("results/brain_brats2025_TDAseg_scores_") + len(str(parameters))]
    == "results/brain_brats2025_TDAseg_scores_" + str(parameters)
]
files_length = [
    int(
        file[
            len("results/brain_brats2025_TDAseg_scores_")
            + len(str(parameters))
            + 4 : -4
        ]
    )
    for file in files
]
if len(files) != 0:
    ind = np.argmax(files_length)
    file, i_min = files[ind], files_length[ind]
    print("Found file", file, "- i_min =", i_min)
else:
    i_min = 0
    print("No files found")

# Compute scores.
DICEs = dict()

msg = "Compute DICEs... "
start_time = ChronometerStart(msg)
for i in range(i_min, len(i_list)):
    # Open image.
    img_flair, seg_gt = pb(i, to_torch=False, modality="flair", normalize=True)
    img_t1ce, _ = pb(i, to_torch=False, modality="t1ce")

    # Segmentation.
    seg_final = segment_brain(
        img_flair=img_flair,
        img_t1ce=img_t1ce,
        normalize=normalize,
        sigma=sigma,
        enhance=enhance,
        radius_enhance=radius_enhance,
        dilate=dilate,
        radius_dilation=radius_dilation,
        whole_threshold=whole_threshold,
        max_bars=max_bars,
        verbose=False,
        plot=False,
    )

    # Save scores.
    score = get_multiple_dice(seg_final, seg_gt, verbose=False)
    DICEs[pb.brats_list[i_list[i]]] = score
    DICEs[pb.brats_list[i_list[i]]]["brats_name"] = pb.brats_list[i_list[i]]
    ChronometerTick(start_time, i - i_min, len(i_list) - i_min, msg)

# %% Save (the code above can be interrupted at any time)

df = pd.DataFrame.from_dict(DICEs, orient="index")
df = df.rename(columns={0: "WT", 1: "TC", 2: "ED", 3: "ET"}, errors="raise")
if i_min > 0:
    df_old = pd.read_csv(file)
    df = pd.concat(
        [
            df_old[["brats_name", "TC", "ED", "ET", "WT"]],
            df[["brats_name", "TC", "ED", "ET", "WT"]],
        ]
    )
file_name = f"results/brain_brats2025_TDAseg_scores_{str(parameters)}_len{len(df)}.csv"
df.to_csv(file_name, index=False)
