# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from segmentations import parseACDC, preprocess_cardiac, segment_whole_object_cardiac
from utils import DLT_KW_SEG, plot_superlevel_sets


# %% Figure article - one coronal slice

n_image = 2
z_pos = 3
img, seg_gt = parseACDC(n_image, end="ED")

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0.1, hspace=0.05)
n = 20
ax = fig.add_subplot(1, 2, 1)
ax.axis("off")
ax.imshow(img[n:, n:, z_pos], cmap="gray", alpha=1, origin="upper")

ax = fig.add_subplot(1, 2, 2)
ax.axis("off")
ax.imshow(img[n:, n:, z_pos], cmap="gray", alpha=0.6, origin="upper")
ax.imshow(seg_gt[n:, n:, z_pos], **DLT_KW_SEG)

plt.savefig("results/coronal_examples.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %% Plot several coronal slices

n_image = 7  # between 1 and 150 included

img, seg_gt = parseACDC(n_image, "ES")
print(np.shape(img), np.shape(seg_gt))

# Plot slices - z
z_pos = [int(i) for i in np.linspace(2, np.shape(img)[2] - 2, 4)]
figsize = (len(z_pos) * 5, 5)
fig = plt.figure(figsize=figsize)
fig.subplots_adjust(wspace=0.05, hspace=0)
for i in range(len(z_pos)):
    ax = fig.add_subplot(1, len(z_pos), i + 1)
    ax.axis("off")
    ax.imshow(img[:, :, z_pos[i]], cmap="gray", alpha=0.9, origin="upper")
    ax.imshow(seg_gt[:, :, z_pos[i]], **DLT_KW_SEG)
ax.set_title("n_image = " + repr(n_image))
plt.show()

# %% Plot superlevel sets of a coronal slice

plot_superlevel_sets(
    img / np.max(img), Times=[0.2, 0.175, 0.15, 0.1, 0.075, 0.05, 0.035]
)

# %% Figure article - Whole object (Module 1) in 2D

# Open image
n_image = 2  # between 1 and 150 included
modality = "ED"  # can be 'ED or 'ES' (end diastole, end systole)
img, seg_gt = parseACDC(n_image, end=modality)
# Parameters
sigma = 1  # Preprocess, Gaussian blur
radius_dilation = 0  # Preprocess, dilation parameter
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max = 10  # Step 1, number of H2 bars to consider
thresh_small_LV = 200  # Step 1, minimal width of LV
ratio_small_RV = 0.1  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = 5  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 2  # Step 1, fill gap between LV and RV
maximal_distance = np.inf  # segment_whole_object_cardiac, max dist between LV and RV

# Segment a slice of the image
img = preprocess_cardiac(img=img, sigma=sigma, radius_dilation=radius_dilation)
z_pos = 3
img_slice, seg_gt_slice = img[:, :, z_pos], seg_gt[:, :, z_pos]
seg_union_slice = segment_whole_object_cardiac(
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
    verbose=False,
    plot=True,
    save=True,
)

# %% Plot consecutive slices

img, seg_gt = parseACDC(2, end="ES")
seg_myo = (seg_gt == 2) * 1

# Get nonempty slices of CC (containing myo)
nonempty_slices = np.where(np.sum(np.sum(seg_gt == 2, 0), 0))[0]
zmin, zmax = nonempty_slices[0], nonempty_slices[-1]

# Plot axial slices
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(wspace=0.05, hspace=0.05)
ax = fig.add_subplot(1, 2, 1)
ax.axis("off")
n = 54
m = 60
ax.imshow(img[n:-n, m:-m, zmax - 1], cmap="gray", alpha=0.5, origin="lower")
CC = seg_myo[n:-n, m:-m, zmax - 1]
ax.imshow(
    np.ma.masked_where(CC == 0, CC),
    alpha=1,
    cmap=ListedColormap(["C3"]),
    origin="lower",
)
CC = seg_myo[n:-n, m:-m, zmax]
ax.imshow(
    np.ma.masked_where(CC == 0, CC),
    alpha=1,
    cmap=ListedColormap(["C6"]),
    origin="lower",
)

# Plot coronal slices
ax = fig.add_subplot(1, 2, 2)
ax.axis("off")
seg_slices = np.zeros((128, 256))
img_slices = np.zeros((128, 256))
vertical_length = 10
xrange = range(80, 140, 10)

xmin = 8
for i, x in enumerate(xrange):
    seg_slices[
        xmin + 2 * i * vertical_length : (xmin + (2 * i + 1) * vertical_length), :
    ] = seg_myo[x, :, :].T
    img_slices[
        xmin + 2 * i * vertical_length : (xmin + (2 * i + 1) * vertical_length), :
    ] = img[x, :, :].T

seg_slices_dilate = np.zeros((256, 256))
img_slices_dilate = np.zeros((256, 256))
seg_slices_dilate[range(0, 256, 2), :] = seg_slices
seg_slices_dilate[range(1, 256 + 1, 2), :] = seg_slices
img_slices_dilate[range(0, 256, 2), :] = img_slices
img_slices_dilate[range(1, 256 + 1, 2), :] = img_slices

m = 15
CC = img_slices_dilate[:, m:-m]
ax.imshow(np.ma.masked_where(CC == 0, CC), cmap="gray", alpha=0.5, origin="upper")
CC = seg_slices_dilate[:, m:-m]
ax.imshow(
    np.ma.masked_where(CC == 0, CC),
    alpha=1,
    cmap=ListedColormap(["C3"]),
    origin="upper",
)

plt.savefig("results/coronal_consecutive_slices.pdf", format="pdf", bbox_inches="tight")
plt.show()
