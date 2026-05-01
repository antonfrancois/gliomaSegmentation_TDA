# %% Functions

import pandas as pd
import itertools as it
import sys
import datetime
import time

from segmentations import parseSTA, segment_fetal
from utils import (
    get_dice,
    ChronometerStart,
    ChronometerTick,
    find_resume_index,
    load_previous_partial,
    save_partial,
    aggregate_and_update_summary,
    compute_remaining_work,
)


def evaluate_parameter_set(i_list, results_dir: str, parameters, global_state=None):
    (
        sigma,
        radius_dilation,
        zero_boundary,
        min_dist_bars,
        ratio_upper,
        ratio_lower,
        select_bar,
    ) = parameters

    # Resume support
    i_min, _ = find_resume_index(results_dir, filename_head, parameters)

    # In-memory results dict
    DICEs = {}

    # If resuming, load the last partial and bring already-done rows in
    df_prev = load_previous_partial(results_dir, filename_head, parameters)
    if df_prev is not None and len(df_prev) > 0:
        for _, row in df_prev.iterrows():
            name = row["sat_name"]
            DICEs[name] = {
                1: row["CP"],
                "sat_name": name,
            }

    msg = f"Compute DICEs for params {str(parameters)}... "
    start_time = ChronometerStart(msg)

    for idx in range(i_min, len(i_list)):
        # Open image
        img, seg_gt = parseSTA(21 + idx)

        # Segmentation
        seg_final = segment_fetal(
            img=img,
            sigma=sigma,
            radius_dilation=radius_dilation,
            zero_boundary=zero_boundary,
            min_dist_bars=min_dist_bars,
            ratio_upper=ratio_upper,
            ratio_lower=ratio_lower,
            select_bar=select_bar,
            verbose=False,
            plot=False,
        )

        # Save scores
        score = get_dice(seg_final, seg_gt, verbose=False)
        name = f"SAT_{idx}"
        DICEs[name] = {1: score}
        DICEs[name]["sat_name"] = name
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
    df = df.rename(columns={1: "CP"}, errors="raise")
    df = df[["sat_name", "CP"]]
    _ = save_partial(results_dir, filename_head, parameters, df)
    return df


# %% Main

filename_head = "fetal_sat_TDAseg_scores_"
column_names = ["sat_name", "CP"]

# Benchmark parameters
limit_images = None  # set to None for all
results_dir = "results/gridsearch"

# Algorithm parameters
sigma_grid = [0.5]  # Preprocess, Gaussian blur
radius_dilation_grid = [0]  # Preprocess, dilation parameter
zero_boundary_grid = [0.3]  # Preprocess, to suppress the boundary
min_dist_bars_grid = [0.2, 0.3, 0.5]  # Step 2, to consider multiple bars
ratio_upper_grid = [1 / 2]  # Step 2, max relative size of object (circle)
ratio_lower_grid = [1 / 50]  # Step 2, min relative size of interior
select_bar_grid = ["width"]  # Step 2, optimal bar, among "width", "pers", "birth"

n_total = 18
if limit_images is not None:
    i_list = list(range(min(limit_images, n_total)))
else:
    i_list = list(range(n_total))

# Build grid
grid = list(
    it.product(
        sigma_grid,
        radius_dilation_grid,
        zero_boundary_grid,
        min_dist_bars_grid,
        ratio_upper_grid,
        ratio_lower_grid,
        select_bar_grid,
    )
)

# Initialize glocal state
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
    df = evaluate_parameter_set(i_list, results_dir, params, global_state=global_state)
    # If the run covered all requested images, update summary
    if len(df) == len(i_list):
        aggregate_and_update_summary(
            results_dir, filename_head, params, df, column_names
        )
    else:
        print(
            f"Warning: parameter set {str(params)} completed only {len(df)}/{len(i_list)} cases."
        )
