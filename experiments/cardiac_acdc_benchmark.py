# %% Imports

import glob
import numpy as np
import pandas as pd
from segmentations import parseACDC, segment_cardiac
from utils import get_multiple_dice, ChronometerStart, ChronometerTick

# %% Select a modality

modality = "ED"  # "ED" or "ES"

# %% Define parameters

method_whole = "3D"
method_geom = "2D", "3D"
method_other = "skip"
sigma = 3  # Preprocess, Gaussian blur
radius_ball = 0  # Preprocess, dilation parameter
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max_LV = 5  # Step 1, number of H0 bars to consider
H0_features_max_RV = 20  # Step 1, number of H0 bars to consider
thresh_small_LV = 1000 if modality == "ED" else 500  # Step 1, minimal width of LV
ratio_small_RV = 0  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = np.inf  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 2  # Step 1, fill gap between LV and RV
maximal_distance = np.inf  # Step 1, max Hausdorff distance between RV and LV

parameters = [
    method_whole,
    method_geom,
    method_other,
    sigma,
    radius_ball,
    dt_threshold,
    H0_features_max_LV,
    H0_features_max_RV,
    thresh_small_LV,
    ratio_small_RV,
    ratio_big_RV,
    add_to_iterations,
    maximal_distance,
]

# %% Full segmentation - Initialize and run

i_list = range(1, 100 + 1)  # Only training set

# Find last updated file with the same parameters
files = glob.glob("results/cardiac_acdc_TDAseg_scores_*")
files = [
    file
    for file in files
    if file[0 : len("results/cardiac_acdc_TDAseg_scores_") + len(str(parameters))]
    == "results/cardiac_acdc_TDAseg_scores_" + str(parameters)
]
files_length = [
    int(
        file[len("results/cardiac_acdc_TDAseg_scores_") + len(str(parameters)) + 4 : -4]
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
    img, seg_gt, filename = parseACDC(i, end=modality, return_filename=True)

    # Segmentation
    seg_final = segment_cardiac(
        img,
        seg_gt,
        method_whole=method_whole,
        method_geom=method_geom,
        method_other=method_other,
        sigma=sigma,
        radius_dilation=radius_ball,
        dt_threshold=dt_threshold,
        H0_features_max_LV=H0_features_max_LV,
        H0_features_max_RV=H0_features_max_RV,
        thresh_small_LV=thresh_small_LV,
        ratio_small_RV=ratio_small_RV,
        ratio_big_RV=ratio_big_RV,
        add_to_iterations=add_to_iterations,
        maximal_distance=maximal_distance,
        plot=False,
        verbose=False,
    )

    # Save scores.
    score = get_multiple_dice(seg_final, seg_gt, verbose=False)
    name = filename[-28:-10]
    DICEs[name] = score
    DICEs[name]["acdc_name"] = name
    ChronometerTick(start_time, i - i_min, len(i_list) - i_min, msg)

# %% Save (the code above can be interrupted at any time)

df = pd.DataFrame.from_dict(DICEs, orient="index")
df = df.rename(columns={1: "RV", 2: "Myo", 3: "LV"}, errors="raise")
if i_min > 0:
    df_old = pd.read_csv(file)
    df = pd.concat(
        [
            df_old[["acdc_name", "RV", "Myo", "LV"]],
            df[["acdc_name", "RV", "Myo", "LV"]],
        ]
    )
file_name = f"results/cardiac_acdc_TDAseg_scores_{str(parameters)}_len{len(df)}.csv"
df.to_csv(file_name, index=False)
