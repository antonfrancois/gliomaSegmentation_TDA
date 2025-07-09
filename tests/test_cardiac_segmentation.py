# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from segmentations import (
    parseACDC,
    preprocess_cardiac,
    segment_whole_object_cardiac,
    segment_geometric_object_cardiac,
    segment_other_components_cardiac,
    segment_cardiac,
)
from utils import DLT_KW_SEG, get_multiple_dice


# %% Open image

n_image = 103  # between 1 and 150 included
modality = "ED"  # can be 'ED or 'ES' (end diastole, end systole)

img, seg_gt = parseACDC(n_image, end=modality)

# %% Parameters

verbose, plot, plot_slice = False, False, False

sigma = 1  # preprocess_cardiac, Gaussian blur
radius_dilation = 0  # preprocess_cardiac, dilation parameter
dt_threshold = 1.0  # segment_whole_object_cardiac, threshold for suggest_t
H0_features_max = 10  # segment_whole_object_cardiac, number of H2 bars to consider
thresh_small_LV = 200  # segment_whole_object_cardiac, minimal width of LV
ratio_small_RV = 0.1  # segment_whole_object_cardiac, minimal width of RV compared to LV
ratio_big_RV = 5  # segment_whole_object_cardiac, maximal width of RV, compared to LV
add_to_iterations = 2  # segment_whole_object_cardiac, fill gap between LV and RV
maximal_distance = np.inf  # segment_whole_object_cardiac, max dist between LV and RV

# %% Preprocess

img = preprocess_cardiac(img=img, sigma=sigma, radius_dilation=radius_dilation)

# %% Segment 2D (slice by slice)

seg_final = np.zeros(np.shape(img))
for z_pos in range(1, np.shape(img)[2]):
    img_slice, seg_gt_slice = img[:, :, z_pos], seg_gt[:, :, z_pos]
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
    seg_geom_slice = segment_geometric_object_cardiac(
        img_slice, seg_whole_slice, verbose=verbose, plot=plot
    )
    seg_final_slice = segment_other_components_cardiac(
        seg_whole_slice, seg_geom_slice, verbose=verbose
    )
    seg_final[:, :, z_pos] = seg_final_slice

    # Plot and comment
    if plot_slice:
        fig = plt.figure(figsize=(4, 2))
        fig.subplots_adjust(wspace=0.05, hspace=0)
        ax = fig.add_subplot(1, 2, 1)
        ax.axis("off")
        ax.set_title("TDA seg, slice " + repr(z_pos))
        ax.imshow(img_slice, cmap="gray", alpha=0.5, origin="upper")
        ax.imshow(seg_final_slice, **DLT_KW_SEG)

        ax = fig.add_subplot(1, 2, 2)
        ax.axis("off")
        ax.set_title("Ground-truth")
        ax.imshow(img_slice, cmap="gray", alpha=0.5, origin="upper")
        ax.imshow(seg_gt_slice, **DLT_KW_SEG)
        plt.show()

# %% Print results

_ = get_multiple_dice(
    seg_final[:, :, 1:-1], seg_gt[:, :, 1:-1], labels=(1, 2, 3), verbose=True
)

# Sørensen–Dice coefficients: {1: np.float64(0.6), 2: np.float64(0.689), 3: np.float64(0.926)} - Whole: 0.859

# %% Segment 3D (whole CMR)

verbose, plot = False, False

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
seg_geom = segment_geometric_object_cardiac(img, seg_whole, verbose=verbose, plot=plot)
seg_final = segment_other_components_cardiac(seg_whole, seg_geom, verbose=verbose)

# %% Print results

_ = get_multiple_dice(
    seg_final[:, :, 1:-1], seg_gt[:, :, 1:-1], labels=(1, 2, 3), verbose=True
)

# Sørensen–Dice coefficients: {1: np.float64(0.537), 2: np.float64(0.682), 3: np.float64(0.912)} - Whole: 0.803

# %% Do it again, in one function.

# New parameter
maximal_distance = 50  # segment_whole_object_cardiac, max dist between LV and RV

n_image = 89  # between 1 and 150 included
modality = "ED"  # can be 'ED or 'ES' (end diastole, end systole)
img, seg_gt = parseACDC(n_image, end=modality)

# Segment in 2D
method = "2D"
seg_final = segment_cardiac(
    img=img,
    seg_gt=seg_gt,
    method=method,
    sigma=sigma,
    radius_dilation=radius_dilation,
    dt_threshold=dt_threshold,
    H0_features_max=H0_features_max,
    thresh_small_LV=thresh_small_LV,
    ratio_small_RV=ratio_small_RV,
    ratio_big_RV=ratio_big_RV,
    add_to_iterations=add_to_iterations,
    maximal_distance=maximal_distance,
    verbose=False,
    plot=False,
)
_ = get_multiple_dice(
    seg_final[:, :, 1:-1], seg_gt[:, :, 1:-1], labels=(1, 2, 3), verbose=True
)

# Segment in 3D
method = "3D"
seg_final = segment_cardiac(
    img=img,
    seg_gt=seg_gt,
    method=method,
    sigma=sigma,
    radius_dilation=radius_dilation,
    dt_threshold=dt_threshold,
    H0_features_max=H0_features_max,
    thresh_small_LV=thresh_small_LV,
    ratio_small_RV=ratio_small_RV,
    ratio_big_RV=ratio_big_RV,
    add_to_iterations=add_to_iterations,
    maximal_distance=maximal_distance,
    verbose=False,
    plot=False,
)
_ = get_multiple_dice(
    seg_final[:, :, 1:-1], seg_gt[:, :, 1:-1], labels=(1, 2, 3), verbose=True
)

# Sørensen–Dice coefficients: {1: np.float64(0.472), 2: np.float64(0.757), 3: np.float64(0.954)} - Whole: 0.674
# Sørensen–Dice coefficients: {1: np.float64(0.33), 2: np.float64(0.088), 3: np.float64(0.747)} - Whole: 0.249
