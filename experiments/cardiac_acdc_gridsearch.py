# %% Functions

import pandas as pd
import itertools as it
import sys
import datetime
import time
import numpy as np

from segmentations import parseACDC, segment_cardiac
from utils import (
    get_multiple_dice,
    ChronometerStart,
    ChronometerTick,
    find_resume_index,
    load_previous_partial,
    save_partial,
    aggregate_and_update_summary,
    compute_remaining_work,
)


def evaluate_parameter_set(
    i_list, results_dir, parameters, global_state, modality, column_names
):
    (
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
    ) = parameters

    # Resume support
    i_min, _ = find_resume_index(results_dir, filename_head, parameters)

    # In-memory results dict
    DICEs = {}

    # If resuming, load the last partial and bring already-done rows in
    df_prev = load_previous_partial(results_dir, filename_head, parameters)
    if df_prev is not None and len(df_prev) > 0:
        for _, row in df_prev.iterrows():
            name = row["acdc_name"]
            DICEs[name] = {
                1: row["RV"],
                2: row["Myo"],
                3: row["LV"],
                "acdc_name": name,
            }

    msg = f"Compute DICEs for params {str(parameters)}... "
    start_time = ChronometerStart(msg)

    for idx in range(i_min, len(i_list)):
        i = i_list[idx]
        # Open image
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

        # Save scores
        score = get_multiple_dice(seg_final, seg_gt, verbose=False)
        name = filename[-28:-10]
        DICEs[name] = score
        DICEs[name]["acdc_name"] = name
        ChronometerTick(start_time, idx - i_min, len(i_list) - i_min, msg)

        # Print overall progress
        if global_state is not None:
            global_state["completed_this_run"] += 1
            done = global_state["completed_this_run"]
            total = max(1, global_state["remaining_start"])
            remaining = max(0, total - done)
            elapsed = time.time() - global_state["t0"]
            avg_per_case = elapsed / max(1, done)
            eta_sec = avg_per_case * remaining
            pct = 100.0 * done / total
            msg_global_progess = (
                f"Global progress: {done}/{total} cases ({pct:0.1f}%). "
                f"Elapsed {datetime.timedelta(seconds=int(elapsed))}. "
                f"Overall ETA {datetime.timedelta(seconds=int(eta_sec))}."
            )
            sys.stdout.write(" --- " + msg_global_progess)

    # Final aggregation for this parameter set
    df = pd.DataFrame.from_dict(DICEs, orient="index")
    df = df.rename(columns={1: "RV", 2: "Myo", 3: "LV"}, errors="raise")
    df = df[column_names]
    _ = save_partial(results_dir, filename_head, parameters, df)
    return df


# %% Main, modality ED

modality = "ED"
filename_head = "cardiac_acdc_TDAseg_scores_" + modality + "_"

# Algorithm parameters
method_whole_grid = ["3D"]
method_geom_grid = ["2D", "3D"]
method_other_grid = ["skip"]
sigma_grid = [2.5]
radius_ball_grid = [1]
dt_threshold_grid = [1.0]
H0_features_max_LV_grid = [5]
H0_features_max_RV_grid = [20]
# thresh_small_LV_grid = [0, 1000 if modality == "ED" else 500]
thresh_small_LV_grid = [1000 if modality == "ED" else 500]
ratio_small_RV_grid = [0]
ratio_big_RV_grid = [np.inf]
add_to_iterations_grid = [2]
maximal_distance_grid = [np.inf]

grid = [
    ("2D", "2D", "skip", 2.5, 1, 1.0, 5, 20, 100, 0, np.inf, 2, np.inf),
    ("3D", "2D", "skip", 2.5, 1, 1.0, 5, 20, 1000, 0, np.inf, 2, np.inf),
    ("3D", "3D", "skip", 2.5, 1, 1.0, 5, 20, 1000, 0, np.inf, 2, np.inf),
    ("3D", "3D", "skip", 2.5, 2, 1.0, 5, 20, 1000, 0, np.inf, 2, np.inf),
]

# %% Main, modality ES

modality = "ES"
filename_head = "cardiac_acdc_TDAseg_scores_" + modality + "_"

# Algorithm parameters
method_whole_grid = ["2D"]
method_geom_grid = ["2D"]
method_other_grid = ["2D", "skip"]
sigma_grid = [2]
radius_ball_grid = [0, 1]
dt_threshold_grid = [1.0]
H0_features_max_LV_grid = [5]
H0_features_max_RV_grid = [20]
thresh_small_LV_grid = [0]
# thresh_small_LV_grid = [0, 1000 if modality == "ED" else 500]
ratio_small_RV_grid = [0]
ratio_big_RV_grid = [np.inf]
add_to_iterations_grid = [2]
maximal_distance_grid = [np.inf]

grid = [
    #    ("2D", "2D", "skip", 2, 1, 1.0, 5, 20, 0, 0, np.inf, 2, np.inf),
    #    ("2D", "2D", "skip", 2, 0, 1.0, 5, 20, 0, 0, np.inf, 2, np.inf),
    ("3D", "3D", "skip", 2.5, 1, 1.0, 5, 20, 0, 0, np.inf, 2, np.inf),
    #    ("3D", "3D", "skip", 2.5, 2, 1.0, 5, 20, 500, 0, np.inf, 2, np.inf),
]


# %% Run

# Benchmark parameters
limit_images = 150  # training contains 100 images, testing 50 images
results_dir = "results/gridsearch"
column_names = ["acdc_name", "RV", "Myo", "LV"]

# # Build parameters grid
# grid = list(
#     it.product(
#         method_whole_grid,
#         method_geom_grid,
#         method_other_grid,
#         sigma_grid,
#         radius_ball_grid,
#         dt_threshold_grid,
#         H0_features_max_LV_grid,
#         H0_features_max_RV_grid,
#         thresh_small_LV_grid,
#         ratio_small_RV_grid,
#         ratio_big_RV_grid,
#         add_to_iterations_grid,
#         maximal_distance_grid,
#     )
# )

# Initialize glocal state
i_list = list(range(1, limit_images + 1))
remaining_map, total_remaining = compute_remaining_work(
    results_dir, filename_head, grid, i_list
)
global_state = {
    "t0": time.time(),
    "remaining_start": total_remaining,
    "completed_this_run": 0,
}
print(f"Overall remaining cases to process now (across grid): {total_remaining}")

# Iterate grid
for gi, params in enumerate(grid, start=1):
    print("\n" + "-" * 80)
    print(f"[{gi}/{len(grid)}] Evaluating params: {str(params)}")
    print("-" * 80)
    df = evaluate_parameter_set(
        i_list, results_dir, params, global_state, modality, column_names
    )
    # If the run covered all requested images, update summary
    if len(df) == len(i_list):
        aggregate_and_update_summary(
            results_dir, filename_head, params, df, column_names
        )
    else:
        print(
            f"Warning: parameter set {str(params)} completed only {len(df)}/{len(i_list)} cases."
        )

# %% Print overall best at the end

import os

summary_path = os.path.join(results_dir, filename_head + "grid_summary.csv")
if os.path.exists(summary_path):
    sdf = pd.read_csv(summary_path)
    best = sdf.sort_values("mean_all", ascending=False).head(10)
    print("\nOverall top configurations (by mean_all):")
    print(
        best[["mean_all"] + column_names[1:] + ["n_cases", "parameters"]].to_string(
            index=False
        )
    )
    print("\nBest parameters:")
    if not best.empty:
        print(best.iloc[0]["parameters"])
