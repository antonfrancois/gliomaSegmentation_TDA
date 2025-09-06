# %% Imports

import numpy as np
from segmentations import parseACDC, segment_cardiac
from utils import get_multiple_dice

# %% Compute size of LV in all images

modality = "ED"
lv_sizes = [np.sum(parseACDC(i, end=modality)[1] == 3) for i in range(1, 50 + 1)]
print("ED: Min LV size:", min(lv_sizes))

modality = "ES"
lv_sizes = [np.sum(parseACDC(i, end=modality)[1] == 3) for i in range(1, 50 + 1)]
print("ES: Min LV size:", min(lv_sizes))

# %% Compute number of z slices in all images

n_slices = [parseACDC(i)[0].shape[2] for i in range(1, 150 + 1)]
print("Number of slices:", np.unique(n_slices))
print("Median of slices:", np.median(n_slices))


# %% Open image

modality = "ED"  # "ED" or "ES"
n_image = 1  # in range(1, 150 + 1):
img, seg_gt, filename = parseACDC(n_image, end=modality, return_filename=True)

# %% Define parameters

method_whole = "3D"  # Step 1, "2D" or "3D"
method_geom = "3D"  # Step 2, "2D" or "3D"
method_other = "skip"  # Step 3, "2D", "3D" or "skip"
sigma = 2.5  # Preprocess, Gaussian blur
radius_ball = 2  # Preprocess, dilation parameter
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max_LV = 5  # Step 1, number of H0 bars to consider
H0_features_max_RV = 20  # Step 1, number of H0 bars to consider
thresh_small_LV = 0  # Step 1, minimal width of LV
ratio_small_RV = 0  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = np.inf  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 2  # Step 1, fill gap between LV and RV
maximal_distance = np.inf  # Step 1, max Hausdorff distance between RV and LV

# %% Run the segmentation

seg = segment_cardiac(
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
    verbose=True,
)

# %% Plot scores slice by slice

print("Overall scores:")
get_multiple_dice(seg, seg_gt, labels=(1, 2, 3), verbose=True)

print("Slice by slice:")
for i in range(seg.shape[2]):
    get_multiple_dice(seg[:, :, i], seg_gt[:, :, i], labels=(1, 2, 3), verbose=True)

# %% Plot segmentation

import matplotlib.pyplot as plt

z_pos = seg.shape[2] // 2
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(seg[:, :, z_pos], cmap="nipy_spectral")
axes[0].set_title("Segmentation - First Slice")
axes[1].imshow(seg_gt[:, :, z_pos], cmap="nipy_spectral")
axes[1].set_title("Ground Truth - First Slice")
plt.show()
