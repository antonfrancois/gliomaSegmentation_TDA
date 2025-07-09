# %% Imports

import glob
import numpy as np
import pandas as pd
from segmentations import (
    parseACDC,
    preprocess_cardiac,
    segment_whole_object_cardiac,
    segment_geometric_object_cardiac,
    segment_other_components_cardiac,
)
from utils import get_multiple_dice, ChronometerStart, ChronometerTick

# %% Define parameters

parameters_to_test = []
models_to_test = []

modality, dimension, oracle = "ED", 2, False
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
parameters_to_test.append(parameters)
models_to_test.append([(modality, dimension, oracle)])

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
parameters_to_test.append(parameters)
models_to_test.append([(modality, dimension, oracle)])

modality, dimension, oracle = "ED", 3, False
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
parameters_to_test.append(parameters)
models_to_test.append([(modality, dimension, oracle)])

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
parameters_to_test.append(parameters)
models_to_test.append([(modality, dimension, oracle)])

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
parameters_to_test.append(parameters)
models_to_test.append(
    [
        ("ED", 2, True),
        ("ED", 3, True),
        ("ES", 2, True),
        ("ES", 3, True),
    ]
)

# %% Full segmentation - batch ED and ES - one set of parameters at a time

verbose, plot, plot_slice = False, False, False

for parameters, models in zip(parameters_to_test, models_to_test):
    (
        sigma,
        radius_dilation,
        dt_threshold,
        H0_features_max,
        thresh_small_LV,
        ratio_small_RV,
        ratio_big_RV,
        add_to_iterations,
        maximal_distance,
    ) = parameters
    print(f"Parameters: {parameters}")

    for modality, dimension, oracle in models:
        print(f"Modality: {modality}, Dimension: {dimension}, Oracle: {oracle}")
        # Find last updated file.
        if not oracle:
            filename = (
                "results/cardiac_" + modality + "_" + repr(dimension) + "D_scores_"
            )
        elif oracle:
            filename = (
                "results/cardiac_"
                + modality
                + "_"
                + repr(dimension)
                + "D_oracle_scores_"
            )
        files = glob.glob(filename + "*")
        files = [
            file
            for file in files
            if file[0 : len(filename) + len(str(parameters))]
            == filename + str(parameters)
        ]
        files_length = [
            int(file[len(filename) + len(str(parameters)) + 4 : -4]) for file in files
        ]
        if len(files) != 0:
            ind = np.argmax(files_length)
            file, i_min = files[ind], files_length[ind] + 101
            print("Found file", file, "- i_min =", i_min)
            if i_min > 150:
                continue
        else:
            i_min = 101
            print("No files found!", filename)
        # Compute scores.
        DICEs = dict()
        msg = "Compute DICEs... "
        start_time = ChronometerStart(msg)
        for n_image in range(i_min, 150 + 1):
            # Open image
            img, seg_gt = parseACDC(n_image, end=modality)

            # Preprocess.
            img = preprocess_cardiac(
                img=img, sigma=sigma, radius_dilation=radius_dilation
            )

            if dimension == 2:
                seg_final = np.zeros(np.shape(img))
                for z_pos in range(1, np.shape(img)[2]):
                    img_slice, seg_gt_slice = img[:, :, z_pos], seg_gt[:, :, z_pos]
                    if not oracle:
                        seg_whole_slice = segment_whole_object_cardiac(
                            img=img_slice,
                            seg_gt=seg_gt_slice,
                            H0_features_max=H0_features_max,
                            dt_threshold=dt_threshold,
                            thresh_small_LV=thresh_small_LV,
                            ratio_small_RV=ratio_small_RV,
                            ratio_big_RV=ratio_big_RV,
                            radius_dilation=radius_dilation,
                            add_to_iterations=add_to_iterations,
                            maximal_distance=maximal_distance,
                            verbose=verbose,
                            plot=plot,
                        )
                    elif oracle:
                        seg_whole_slice = (seg_gt_slice > 0) * 1
                    seg_geom_slice = segment_geometric_object_cardiac(
                        img_slice, seg_whole_slice, verbose=verbose, plot=plot
                    )
                    seg_final_slice = segment_other_components_cardiac(
                        seg_whole_slice, seg_geom_slice, verbose=verbose
                    )
                    seg_final[:, :, z_pos] = seg_final_slice

            if dimension == 3:
                if not oracle:
                    seg_whole = segment_whole_object_cardiac(
                        img=img,
                        seg_gt=seg_gt,
                        H0_features_max=H0_features_max,
                        dt_threshold=dt_threshold,
                        thresh_small_LV=thresh_small_LV,
                        ratio_small_RV=ratio_small_RV,
                        ratio_big_RV=ratio_big_RV,
                        radius_dilation=radius_dilation,
                        add_to_iterations=add_to_iterations,
                        maximal_distance=maximal_distance,
                        verbose=verbose,
                        plot=plot,
                    )
                elif oracle:
                    seg_whole = (seg_gt > 0) * 1
                seg_geom = segment_geometric_object_cardiac(
                    img, seg_whole, verbose=verbose, plot=plot
                )
                seg_final = segment_other_components_cardiac(
                    seg_whole, seg_geom, verbose=verbose
                )

            # Save dice
            DICEs[n_image] = get_multiple_dice(
                seg_final[:, :, 1:-1],
                seg_gt[:, :, 1:-1],
                labels=(1, 2, 3),
                verbose=False,
            )
            DICEs[n_image]["index"] = n_image
            ChronometerTick(start_time, n_image - i_min, 150 + 1 - i_min, msg)

        # Save
        df = pd.DataFrame.from_dict(DICEs, orient="index")
        df = df.rename(
            columns={0: "whole", 1: "LV", 2: "Myocardium", 3: "RV"}, errors="raise"
        )
        if i_min > 101:
            df_old = pd.read_csv(file)
            df = pd.concat(
                [
                    df_old[["index", "LV", "Myocardium", "RV", "whole"]],
                    df[["index", "LV", "Myocardium", "RV", "whole"]],
                ]
            )
        df.to_csv(filename + str(parameters) + "_len" + repr(len(df)) + ".csv")
