# %% Imports

# TDA
import persim
import cripser

# Image
import scipy
import skimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import gudhi

# Anton
import nibabel as nib
import vedo

from PIL import Image
import numpy as np

from utils import ChronometerStart, ChronometerStop

# %% Functions


def PlotImage(image, pos, title=None):
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for i in range(3):
        if i == 0:
            imageslice = image[pos[0], :, :]
        if i == 1:
            imageslice = image[:, pos[1], :]
        if i == 2:
            imageslice = image[:, :, pos[2]]
        axs[i].imshow(imageslice, vmin=0, vmax=1, cmap="gray", origin="lower")
        axs[i].axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=10)
    plt.show()


def PlotMask(mask, image, pos, title=None):
    COLORS = ["white"] + list(mpl.colors.TABLEAU_COLORS) * 10
    cmap = mpl.colors.ListedColormap(COLORS[0 : (int(np.max(mask)) + 1)])
    bounds = list(range(int(np.max(mask)) + 2))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        if i == 0:
            maskslice = mask[pos[0], :, :]
            imageslice = image[pos[0], :, :]
        if i == 1:
            maskslice = mask[:, pos[1], :]
            imageslice = image[:, pos[1], :]
        if i == 2:
            maskslice = mask[:, :, pos[2]]
            imageslice = image[:, :, pos[2]]
        axs[i].imshow(1 - imageslice, cmap="gray", origin="lower", alpha=0.5)
        axs[i].imshow(maskslice, cmap=cmap, origin="lower", alpha=0.75, norm=norm)
        axs[i].axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=10)
    plt.show()


def PersistenceSuperlevel(im):
    if np.max(im) > 1:
        raise ValueError("Image non-comprise entre 0 et 1")
    iminv = 1 - im
    xval = np.arange(0, np.shape(im)[0], 1)
    nx = len(xval)
    yval = np.arange(0, np.shape(im)[0], 1)
    ny = len(yval)
    cc = gudhi.CubicalComplex(
        dimensions=[nx, ny], top_dimensional_cells=np.matrix.flatten(iminv)
    )
    pers = cc.persistence(homology_coeff_field=2)
    return pers


def PersistenceSublevel(im):
    if np.max(im) > 1:
        raise ValueError("Image non-comprise entre 0 et 1")
    xval = np.arange(0, np.shape(im)[0], 1)
    nx = len(xval)
    yval = np.arange(0, np.shape(im)[0], 1)
    ny = len(yval)
    cc = gudhi.CubicalComplex(
        dimensions=[nx, ny], top_dimensional_cells=np.matrix.flatten(im)
    )
    pers = cc.persistence(homology_coeff_field=2)
    return pers


def GetConnectedComponent(img, pos, t):
    """
    Get the connected component of the voxel pos = (x,y,z) at time t.
    The output is a binary image.
    Background value of img must be 0 (as conventional).
    """
    imt = (img >= t) * 1
    if imt[pos[0], pos[1]] == 0:
        raise ValueError("The voxel pos is not active at time t.")

    labels = skimage.measure.label(imt, background=0)
    labeltumor = labels[pos[0], pos[1]]
    imtumor = (labels == labeltumor) * 1

    if labeltumor == 0:
        print("Problem! The label is background :(")

    return imtumor


# %% Filtrations on SRI24 template

image = Image.open("data/sri24/source_80.png")
im = np.array(image)[:, :, 0]
im = im / 255

# Sublevels
fig = plt.figure(figsize=(16, 16))
ax = [fig.add_subplot(1, 5, i + 1) for i in range(5)]
Times = [0, 0.25, 0.5, 0.75, 1]
I = list(range(len(Times)))
for i in I:
    t = Times[i]
    a = ax[i]
    a.axis("off")
    a.set_aspect("equal")
    imt = np.zeros(np.shape(im))
    imt[im >= t] = 1
    imt[im < t] = 0
    a.imshow(imt, origin="lower", vmin=0, vmax=1, cmap="gray")
    a.set_title("t = " + repr(t), fontsize=17.5)
fig.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig("results/tutorial_sri24_sublevels.pdf", bbox_inches="tight")
plt.show()

# Superlevels
fig = plt.figure(figsize=(16, 16))
ax = [fig.add_subplot(1, 5, i + 1) for i in range(5)]
Times = [0, 0.25, 0.5, 0.75, 1]
I = list(range(len(Times)))
ax.reverse()
for i in I:
    t = Times[i]
    a = ax[i]
    a.axis("off")
    a.set_aspect("equal")
    imt = np.zeros(np.shape(im))
    imt[im < t] = 1
    imt[im >= t] = 0
    a.imshow(imt, origin="lower", vmin=0, vmax=1, cmap="gray")
    a.set_title("t = " + repr(1 - t), fontsize=17.5)
fig.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig("results/tutorial_sri24_superlevels.pdf", bbox_inches="tight")
plt.show()


# %% Persistence of SRI24 template

# Plot persistence
pers = PersistenceSublevel(im)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
gudhi.plot_persistence_diagram(pers, axes=axs[0], max_intervals=500)
pers = PersistenceSuperlevel(im)
gudhi.plot_persistence_diagram(pers, max_intervals=500, axes=axs[1])
fig.subplots_adjust(wspace=0.25, hspace=0)
axs[0].set_title("")
axs[1].set_title("")
plt.show()

# %% Plot CC

barcode = cripser.computePH(1 - im, maxdim=2)
H0 = [list(bar[1::]) for bar in barcode if bar[0] == 0]  # Only non-infinite bars
# Sort by persistence
H0 = [bar for _, bar in sorted(zip([bar[1] - bar[0] for bar in H0], H0))[::-1]]

seg = np.zeros(np.shape(im))

i = 0
pos = (int(H0[i][2]), int(H0[i][3]))
t = H0[i][1]
print(pos, t)
seg += GetConnectedComponent(im, pos, 0.5) * 1

i = 1
pos = (int(H0[i][2]), int(H0[i][3]))
t = H0[i][1]
print(pos, t)
# seg += GetConnectedComponent(im,pos,t)*2
seg += GetConnectedComponent(im, pos, 0.8) * 2

i = 2
pos = (int(H0[i][2]), int(H0[i][3]))
t = H0[i][1]
print(pos, t)
seg += GetConnectedComponent(im, pos, t) * 3

# i = 3
# pos = (int(H0[i][2]),int(H0[i][3]))
# t = H0[i][1]
# print(pos, t)
# seg += GetConnectedComponent(im,pos,t)*i

from matplotlib.colors import ListedColormap

cmap_segs = ListedColormap([[0, 0, 0, 0], "tab:red", "tab:blue", "tab:orange"])
DLT_KW_IMAGE = dict(
    cmap="gray",
    # extent=[-1,1,-1,1],
    origin="lower",
    vmin=0,
    vmax=1,
)
DLT_KW_SEG = dict(cmap=cmap_segs, interpolation="nearest", origin="lower")
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.imshow(im, alpha=0.5, **DLT_KW_IMAGE)
axs.imshow(seg, **DLT_KW_SEG)
plt.axis("off")
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.imshow(im, alpha=1, **DLT_KW_IMAGE)
plt.axis("off")
plt.show()
