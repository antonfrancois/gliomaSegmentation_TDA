# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from segmentations import parseSTA
from utils import DLT_KW_IMAGE, DLT_KW_SEG

# %% Plot a few images

img, img_seg = parseSTA(img_idx=22)
z_pos = [int(i) for i in np.linspace(60, np.shape(img)[1] - 60, 7)]
z_pos = [int(i) for i in np.linspace(60, np.shape(img)[1] - 70, 5)]
figsize = (len(z_pos) * 5, 3 * 5)
fig = plt.figure(figsize=figsize)
fig.subplots_adjust(wspace=0.05, hspace=0.05)

for i_image, img_idx in enumerate([30, 38]):
    # Open image
    img, img_seg = parseSTA(img_idx)
    # Plot slices
    for i in range(len(z_pos)):
        ax = fig.add_subplot(3, len(z_pos), i + 1 + i_image * len(z_pos))
        ax.axis("off")
        ax.imshow(img[:, z_pos[i], 10:145].T, **DLT_KW_IMAGE)
        ax.imshow(img_seg[:, z_pos[i], 10:145].T, **DLT_KW_SEG)
        ax.invert_yaxis()
plt.tight_layout()
plt.savefig("results/fetal_examples.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %% Plot consecutive slices

n_image = 38
img, seg_gt = parseSTA(n_image)
y_values = [25, 60, 100, 130, 153]

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(wspace=0.05, hspace=0.05)
for i, y in enumerate(y_values):
    # Define slice
    img_slice = img[:, y, :].copy()
    seg_medecin_slice = seg_gt[:, y, :].copy()

    ax = fig.add_subplot(1, len(y_values), i + 1)
    ax.axis("off")
    n = 10
    ax.imshow(img_slice[:, n:-n].T, **DLT_KW_IMAGE)
    CC = seg_medecin_slice[:, n:-n].T
    ax.imshow(CC, **DLT_KW_SEG)
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig("results/fetal_consecutive_slices.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %% Plot all images - z (horizontal plane)

for n_image in range(21, 38 + 1):
    # Open image
    img, img_seg = parseSTA(n_image)
    # Plot slices
    z_pos = [int(i) for i in np.linspace(60, np.shape(img)[2] - 60, 4)]
    figsize = (len(z_pos) * 5, 5)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0)
    for i in range(len(z_pos)):
        ax = fig.add_subplot(1, len(z_pos), i + 1)
        ax.axis("off")
        ax.imshow(img[:, :, z_pos[i]], **DLT_KW_IMAGE)
        ax.imshow(img_seg[:, :, z_pos[i]], **DLT_KW_SEG)
    ax.set_title("n_image = " + repr(n_image))
    plt.tight_layout()
    plt.show()

# %% Plot all images - y (coronal plane)

for n_image in range(21, 38 + 1):
    # Open image
    img, img_seg = parseSTA(n_image)
    # Plot slices
    z_pos = [int(i) for i in np.linspace(60, np.shape(img)[1] - 60, 7)]
    figsize = (len(z_pos) * 5, 5)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0)
    for i in range(len(z_pos)):
        ax = fig.add_subplot(1, len(z_pos), i + 1)
        ax.axis("off")
        ax.imshow(img[:, z_pos[i], :], **DLT_KW_IMAGE)
        ax.imshow(img_seg[:, z_pos[i], :], **DLT_KW_SEG)
    ax.invert_yaxis()
    ax.set_title("n_image = " + repr(n_image))
    plt.tight_layout()
    plt.show()

# %% Plot all images - x (sagittal plane)

for n_image in range(21, 38 + 1):
    # Open image
    img, img_seg = parseSTA(n_image)
    # Plot slices
    z_pos = [int(i) for i in np.linspace(60, np.shape(img)[0] - 50, 5)]
    figsize = (len(z_pos) * 3, 4)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0)
    for i in range(len(z_pos)):
        ax = fig.add_subplot(1, len(z_pos), i + 1)
        ax.axis("off")
        ax.imshow(img[z_pos[i], :, :], **DLT_KW_IMAGE)
        ax.imshow(img_seg[z_pos[i], :, :], **DLT_KW_SEG)
    ax.set_title("n_image = " + repr(n_image))
    plt.tight_layout()
    plt.show()
