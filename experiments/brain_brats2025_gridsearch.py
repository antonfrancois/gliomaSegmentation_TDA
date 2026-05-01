# %% Functions

import pandas as pd
import itertools as it
import sys
import datetime
import time

import parseBrats as pB
from segmentations import segment_brain
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


def evaluate_parameter_set(pb, i_list, results_dir: str, parameters, global_state=None):
    (
        normalize,
        sigma,
        enhance,
        radius_enhance,
        dilate,
        radius_dilation,
        whole_threshold,
        max_bars,
    ) = parameters

    # Resume support
    i_min, _ = find_resume_index(results_dir, filename_head, parameters)

    # In-memory results dict
    DICEs = {}

    # If resuming, load the last partial and bring already-done rows in
    df_prev = load_previous_partial(results_dir, filename_head, parameters)
    if df_prev is not None and len(df_prev) > 0:
        for _, row in df_prev.iterrows():
            name = row["brats_name"]
            DICEs[name] = {
                0: row["WT"],
                1: row["TC"],
                2: row["ED"],
                3: row["ET"],
                "brats_name": name,
            }

    msg = f"Compute DICEs for params {str(parameters)}... "
    start_time = ChronometerStart(msg)

    for idx in range(i_min, len(i_list)):
        i = i_list[idx]
        # Open image(s)
        img_flair, seg_gt = pb(i, to_torch=False, modality="flair", normalize=True)
        img_t1ce, _ = pb(i, to_torch=False, modality="t1ce")

        # Segmentation
        seg_final = segment_brain(
            img_flair=img_flair,
            img_t1ce=img_t1ce,
            normalize=normalize,
            sigma=sigma,
            enhance=tuple(enhance),
            radius_enhance=radius_enhance,
            dilate=dilate,
            radius_dilation=radius_dilation,
            whole_threshold=whole_threshold,
            max_bars=max_bars,
            verbose=False,
            plot=False,
        )

        # Save scores
        score = get_multiple_dice(seg_final, seg_gt, verbose=False)
        name = pb.brats_list[i]
        DICEs[name] = score
        DICEs[name]["brats_name"] = name
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
    df = df.rename(columns={0: "WT", 1: "TC", 2: "ED", 3: "ET"}, errors="raise")
    df = df[["brats_name", "TC", "ED", "ET", "WT"]]
    _ = save_partial(results_dir, filename_head, parameters, df)
    return df


# %% Silence skimage warning

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Possible precision loss converting image of type .* to uint8 as required by rank filters",
    category=UserWarning,
    module=r"skimage\.filters\.rank\..*",
)

# %% Main

filename_head = "brain_brats2025_TDAseg_scores_"
column_names = ["brats_name", "TC", "ED", "ET", "WT"]

# Benchmark parameters
limit_images = None  # set to None for all
results_dir = "results/gridsearch"

# Algorithm parameters
normalize_grid = ["max"]
sigma_grid = [1, 1.2]
enhance_grid = [(False, True)]
radius_enhance_grid = [1]
dilate_grid = [True]
radius_dilation_grid = [1, 2]
whole_threshold_grid = [1]
max_bars_grid = [1]

# Dataset init once
pb = pB.parse_brats(
    brats_list=None,
    brats_folder="2025",
    modality="flair",
    get_template=False,
)
n_total = len(pb.brats_list)
if limit_images is not None:
    i_list = list(range(min(limit_images, n_total)))
else:
    i_list = list(range(n_total))

# Build grid
grid = list(
    it.product(
        normalize_grid,
        sigma_grid,
        enhance_grid,
        radius_enhance_grid,
        dilate_grid,
        radius_dilation_grid,
        whole_threshold_grid,
        max_bars_grid,
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
    df = evaluate_parameter_set(
        pb, i_list, results_dir, params, global_state=global_state
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
