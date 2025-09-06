# %% Imports

import numpy as np
import persim, cripser, skimage, scipy
import matplotlib.pyplot as plt
from morphology import argmax_image
import parseBrats as pB


def ComputeDiagramAndPlot_img():
    # First ax, brain.
    xmin, ymin = 1, 1
    ax0.imshow(1 - img[xmin:-xmin, ymin:-ymin], cmap="Greys")
    ax0.axis("off")
    ax0.imshow(img_raw[xmin:-xmin, ymin:-ymin], alpha=0.4, cmap="Purples")

    # Second ax, diagram.
    barcode = cripser.computePH(1 - img, maxdim=1)  # Compute diagram
    b = [np.array([bar[1:3] for bar in barcode if bar[0] == i]) for i in range(2)]
    b[0][-1][1] = 1
    persim.plot_diagrams(b, ax=ax, diagonal=False)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.plot([0, 1.1], [1, 1], ls="dashed", c="black")
    ax.plot([0, 1.1], [0, 1.1], ls="dashed", c="black")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])


# %% Open image.

img_idx = 10

pb = pB.parse_brats(
    brats_list=None, brats_folder="2025", modality="flair", get_template=False
)
img_flair, seg_gt = pb(img_idx, to_torch=False, modality="flair", normalize=True)
img_t1ce, _ = pb(img_idx, to_torch=False, modality="t1ce")
pos = argmax_image(img_t1ce)

# Define slice

s = 0
img = img_t1ce[:, :, pos[2] + s] * (seg_gt[:, :, pos[2] + s] > 0)
img_raw = img_t1ce[:, :, pos[2] + s]
img_raw[img_raw == 0] = np.nan
img_raw[seg_gt[:, :, pos[2] + s] > 0] = np.nan

# Plot

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(8, 2 / 3 * 8))
fig.subplots_adjust(wspace=0.05, hspace=-0.1)

# Plot raw image.
ax0 = axs[0, 0]
ax = axs[1, 0]
ComputeDiagramAndPlot_img()
ax0.set_title("Raw image")

# Plot after blur.
sigma = 0.5
pb = pB.parse_brats(
    brats_list=None, brats_folder="2025", modality="flair", get_template=False
)
img_flair, seg_gt = pb(img_idx, to_torch=False, modality="flair", normalize=True)
img_t1ce, _ = pb(img_idx, to_torch=False, modality="t1ce")
img_t1ce = scipy.ndimage.gaussian_filter(img_t1ce, sigma=sigma)
img = img_t1ce[:, :, pos[2] + s] * (seg_gt[:, :, pos[2] + s] > 0)
ax0 = axs[0, 1]
ax = axs[1, 1]
ComputeDiagramAndPlot_img()
ax.set_yticks([])
ax0.set_title(r"After blurring ($\sigma=0.5$)")

# Plot after dilation
radius_ball = 2
img = skimage.morphology.dilation(
    img,
    footprint=skimage.morphology.disk(radius_ball),
    out=None,
)
ax0 = axs[0, 2]
ax = axs[1, 2]
ComputeDiagramAndPlot_img()
ax.set_yticks([])
ax0.set_title(r"After dilation ($r=3$)")
ax.add_patch(plt.Circle((0.6775, 0.7935028), 0.03, color="black", fill=False))

# plt.savefig("results/local_transformations_TDA.pdf", format="pdf", bbox_inches="tight")
plt.show()
