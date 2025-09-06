# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from segmentations import parseACDC, preprocess_cardiac, segment_whole_object_cardiac
from utils import DLT_KW_IMAGE, DLT_KW_SEG, plot_superlevel_sets

# Redefine color map in DLT_KW_SEG (Myo:orange, LV:red, RV:blue)
DLT_KW_SEG["cmap"] = ListedColormap([[0, 0, 0, 0], "tab:blue", "tab:orange", "tab:red"])

# %% Open and preprocess an image

n_image = 2  # between 1 and 150 included
modality = "ED"  # can be 'ED or 'ES' (end diastole, end systole)
sigma = 2  # Preprocess, Gaussian blur
radius_dilation = 0  # Preprocess, dilation parameter
img, seg_gt = parseACDC(n_image, end=modality)
img = preprocess_cardiac(img=img, sigma=sigma, radius_dilation=radius_dilation)

z_pos = 2

# %% Plot superlevel sets of a coronal slice

plot_superlevel_sets(
    img,
    Times=[0.4, 0.3, 0.2, 0.1, 0.05],
    save_path="results/cardiac_superlevelsets.pdf",
)

# %% Figure article - Whole object (Module 1) in 2D

# Parameters
dt_threshold = 1.0  # Step 1, threshold for suggest_t
H0_features_max = 10  # Step 1, number of H2 bars to consider
thresh_small_LV = 200  # Step 1, minimal width of LV
ratio_small_RV = 0.1  # Step 1, minimal width of RV, compared to LV
ratio_big_RV = 5  # Step 1, maximal width of RV, compared to LV
add_to_iterations = 2  # Step 1, fill gap between LV and RV
maximal_distance = np.inf  # segment_whole_object_cardiac, max dist between LV and RV

# Segment a slice of the image
img_slice, seg_gt_slice = img[:, :, z_pos], seg_gt[:, :, z_pos]
seg_union_slice = segment_whole_object_cardiac(
    img=img_slice,
    seg_gt=seg_gt_slice,
    H0_features_max_LV=10,
    H0_features_max_RV=10,
    add_to_iterations=2,
    verbose=False,
    plot=True,
    save=True,
)


# %% Plot consecutive slices

# Reopen image (without Gaussian blur)
img, seg_gt = parseACDC(n_image, end=modality)
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
ax.imshow(img[n:-n, m:-m, zmax - 1], **DLT_KW_IMAGE)
CC = seg_myo[n:-n, m:-m, zmax - 1]
ax.imshow(
    np.ma.masked_where(CC == 0, CC),
    alpha=1,
    cmap=ListedColormap(["C11"]),
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
ax.imshow(np.ma.masked_where(CC == 0, CC), **DLT_KW_IMAGE)
CC = seg_slices_dilate[:, m:-m]
ax.imshow(
    np.ma.masked_where(CC == 0, CC),
    alpha=1,
    cmap=ListedColormap(["C11"]),
    origin="upper",
)
plt.tight_layout()
plt.savefig("results/cardiac_consecutive_slices.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %% Figure article - one coronal slice

# Reopen image (without Gaussian blur)
img_ed, seg_gt_ed = parseACDC(n_image, end="ED")
img_es, seg_gt_es = parseACDC(n_image, end="ES")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax_list in axes:
    for ax in ax_list:
        ax.axis("off")
axes[0, 0].imshow(img_ed[:, :, z_pos], **DLT_KW_IMAGE)
axes[0, 1].imshow(img_ed[:, :, z_pos], **DLT_KW_IMAGE)
axes[0, 1].imshow(seg_gt_ed[:, :, z_pos], **DLT_KW_SEG)
axes[1, 0].imshow(img_es[:, :, z_pos], **DLT_KW_IMAGE)
axes[1, 1].imshow(img_es[:, :, z_pos], **DLT_KW_IMAGE)
axes[1, 1].imshow(seg_gt_es[:, :, z_pos], **DLT_KW_SEG)


plt.tight_layout()
plt.savefig("results/cardiac_examples.pdf", format="pdf", bbox_inches="tight")
plt.show()
