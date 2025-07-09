"""---------------------------------------------------------------------------------------------------------------------

Train-Free Segmentation in MRI with Cubical Persistent Homology
Anton François & Raphaël Tinarrage
See the repo at https://github.com/antonfrancois/gliomaSegmentation_TDA and article at https://arxiv.org/abs/2401.01160

------------------------------------------------------------------------------------------------------------------------

Global variables:
    CMAP_SEGS
    DLT_KW_SEG
    DLT_KW_GRIDSAMPLE
    DLT_KW_IMAGE

Metrics:
    get_dice
    get_multiple_dice

Plotting:
    plot_superlevel_sets
    set_ticks_off
    image_slice
    make_3d_flat
    plot_segmentation
    plot_comparison_binary_segmentations
    plot_comparison_full_segmentations
    plot_images_and_segmentation
    segCmp
    imCmp

Time:
    ChronometerStart
    ChronometerStop
    ChronometerTick
    format_time
    time_it

--------------------------------------------------------------------------------------------------------------------
"""

# Standard imports.
import sys
import time, datetime

# Third-party imports.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Local imports.
from morphology import argmax_image

"""---------------------------------------------------------------------------------------------------------------------
Global variables
---------------------------------------------------------------------------------------------------------------------"""

CMAP_SEGS = ListedColormap(
    [
        [0, 0, 0, 0],
        "#08B2E3",  # Process Cyan
        "#F6AE2D",  # Hunyardi Yellow
        "#C65CC0",  # French mauve
    ]
)
DLT_KW_SEG = dict(alpha=1, cmap=CMAP_SEGS, interpolation="nearest")
DLT_KW_GRIDSAMPLE = dict(padding_mode="border", align_corners=True)
DLT_KW_IMAGE = dict(
    cmap="gray",
    # extent=[-1,1,-1,1],
    origin="lower",
    vmin=0,
    vmax=1,
)


"""---------------------------------------------------------------------------------------------------------------------
Metrics
---------------------------------------------------------------------------------------------------------------------"""


def get_dice(seg_1, seg_2, verbose=False):
    prod_seg = seg_1 * seg_2
    sum_seg = seg_1 + seg_2
    if sum_seg.sum() == 0:
        dice = 1
    else:
        dice = 2 * prod_seg.sum() / sum_seg.sum()
    if verbose:
        # Non-symmetric scores
        diceleft = prod_seg.sum() / seg_1.sum()
        diceright = prod_seg.sum() / seg_2.sum()
        print(
            "Sørensen–Dice coefficient (sym, left and right):",
            round(dice, 3),
            round(diceleft, 3),
            round(diceright, 3),
        )
    return dice


def get_multiple_dice(seg_1, seg_2, labels=(1, 2, 4), verbose=False):
    dices = {
        l: get_dice((seg_1 == l) * 1, (seg_2 == l) * 1, verbose=False) for l in labels
    }
    dices[0] = get_dice((seg_1 > 0) * 1, (seg_2 > 0) * 1, verbose=False)
    if verbose:
        dices_dic = {l: round(dices[l], 3) for l in labels}
        print(f"Sørensen–Dice coefficients: {dices_dic} - Whole: {round(dices[0], 3)}")
    return dices


"""---------------------------------------------------------------------------------------------------------------------
Plotting
---------------------------------------------------------------------------------------------------------------------"""


def plot_superlevel_sets(image, pos=None, Times=None):
    if Times == None:
        Times = [0, 0.05, 0.3, 0.5, 1]

    if pos == None:
        if image[0, 0, 0] == 0:
            pos = argmax_image(image)
        else:
            pos = argmax_image(1 - image)
    im = image[:, :, pos[2]]

    fig = plt.figure(figsize=(15, 15))  # Notice the equal aspect ratio
    ax = [fig.add_subplot(1, len(Times), i + 1) for i in range(len(Times))]
    for i in range(len(Times)):
        t = Times[i]
        imt = np.zeros(np.shape(im))
        imt[im < t] = 1
        imt[im >= t] = 0
        ax[i].imshow(imt, origin="lower", vmin=0, vmax=1, cmap="gray")
        ax[i].set_title("t = " + repr(Times[i]), fontsize=17.5)
        ax[i].axis("off")
        ax[i].set_aspect("equal")
    fig.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()


def set_ticks_off(ax):
    try:
        ax.tick_params(
            axis="both",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=False,
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False,
        )
    except AttributeError:
        for a in ax.ravel():
            set_ticks_off(a)


def image_slice(I, coord, dim):
    coord = int(coord)
    if dim == 0:
        return I[coord, :, :]
    elif dim == 1:
        return I[:, coord, :]
    elif dim == 2:
        return I[:, :, coord]


def make_3d_flat(img_3D, slice):
    """Take a 'brain' 3D image, take 3 slices and make a long 2D image of it."""
    D, H, W = img_3D.shape

    # im0 = image_slice(img_3D,slice[0],2).T
    # im1 = image_slice(img_3D,slice[1],1).T
    # im2 = image_slice(img_3D,slice[2],0).T
    # adapté pos raph v
    im0 = image_slice(img_3D, slice[2], 2).T
    im1 = image_slice(img_3D, slice[1], 1).T
    im2 = image_slice(img_3D, slice[0], 0).T

    crop = 20
    # print(D-int(1.7*crop),D+H-int(2.7*crop))
    # print(D+H-int(3.2*crop))
    long_img = np.zeros((D, D + H + H - int(3.5 * crop)))
    long_img[:D, : D - crop] = im0[:, crop // 2 : -crop // 2]
    long_img[
        (D - W) // 2 : (D - W) // 2 + W, D - int(1.7 * crop) : D + H - int(2.7 * crop)
    ] = im1[::-1, crop // 2 : -crop // 2]
    long_img[(D - W) // 2 : (D - W) // 2 + W, D + H - int(3 * crop) :] = im2[
        ::-1, crop // 2 :
    ]

    return long_img


def plot_segmentation(name, img, seg=None, pos=None, figsize=(15, 5)):
    if pos is None:
        pos = argmax_image(img)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(make_3d_flat(img, pos), cmap="gray", alpha=0.5, origin="upper")
    if seg is None:
        seg = img * 0
    ax.imshow(make_3d_flat(seg, pos), **DLT_KW_SEG)
    ax.text(235, 15, name, c="white", fontsize=20)
    plt.axis("off")
    plt.show()


def plot_comparison_binary_segmentations(name, img, seg_binary_gt, seg_binary_est):
    pos = argmax_image(img)
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].imshow(make_3d_flat(img, pos), cmap="gray")
    ax[0].text(175, 25, name, c="white", fontsize=20)
    ax[1].imshow(make_3d_flat(img, pos), cmap="gray")
    ax[1].imshow(
        imCmp(
            make_3d_flat(seg_binary_gt, pos),
            make_3d_flat(seg_binary_est, pos),
            method="seg",
        ),
        alpha=0.5,
    )
    ax[1].text(
        200,
        225,
        f"DICE = {get_dice(seg_binary_gt, seg_binary_est):.4f}",
        c="white",
        fontsize=20,
    )
    set_ticks_off(ax)
    plt.show()


def plot_comparison_full_segmentations(name, img, seg_gt, seg_est):
    pos = argmax_image(img)
    seg_contour_true = np.zeros(seg_gt.shape)
    seg_contour_true[seg_gt == 4] = 1

    fig, ax = plt.subplots(4, 1, constrained_layout=True, figsize=(8, 10))
    ax[0].imshow(make_3d_flat(img, pos), cmap="gray")
    ax[0].text(175, 25, name, c="white", fontsize=20)

    ax[1].imshow(make_3d_flat(img, pos), cmap="gray")
    ax[1].imshow(make_3d_flat(seg_gt, pos), **DLT_KW_SEG)
    ax[1].text(250, 225, "True segmentation", c="white", fontsize=20)

    ax[2].imshow(make_3d_flat(img, pos), cmap="gray")
    ax[2].imshow(make_3d_flat(seg_est, pos), **DLT_KW_SEG)
    ax[2].text(250, 225, "our segmentation", c="white", fontsize=20)

    seg_superpose = np.zeros(seg_gt.shape)
    # Good segmentation => Green
    good_bool = np.logical_and((seg_gt == seg_est), (seg_gt + seg_est != 0))
    seg_superpose[good_bool] = 3
    print(
        f"Well labeled pixels {good_bool.sum()}, proportion in image {good_bool.sum()/(seg_gt > 0).sum()}"
    )

    # It is part of the tumour but mislabeled => Orange
    mislabeled_bool = np.logical_and((seg_gt != seg_est), (seg_gt >= 1), (seg_est >= 1))
    seg_superpose[mislabeled_bool] = 2
    print(f"mislabeled pixels: {mislabeled_bool.sum()}")

    # Not part of the tumour => Red
    misseg_bool = np.logical_and((seg_gt != seg_est), (seg_gt == 0))
    not_seg = np.logical_and(seg_gt > 0, seg_est == 0)
    seg_superpose[np.logical_or(misseg_bool, not_seg)] = 1
    print(f"Baddly segmented pixels {misseg_bool.sum()}")

    ax[3].imshow(make_3d_flat(img, pos), cmap="gray")
    ax[3].imshow(
        make_3d_flat(seg_superpose, pos), cmap=CMAP_SEGS, interpolation="nearest"
    )

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()


def plot_images_and_segmentation(
    image_name, img_flair, img_t1ce, seg_gt, figsize=(12, 10)
):
    pos = argmax_image(img_flair)
    fig, ax = plt.subplots(3, 1, figsize=figsize)
    ax[0].imshow(make_3d_flat(img_flair, pos), cmap="gray", vmax=1)
    ax[0].text(175, 25, image_name, c="white", fontsize=20)
    ax[0].text(250, 225, "FLAIR", c="white", fontsize=20)
    ax[1].imshow(make_3d_flat(img_flair, pos), cmap="gray", vmax=1)
    ax[1].text(250, 225, "segmentation", c="white", fontsize=20)
    ax[1].imshow(make_3d_flat(seg_gt, pos), **DLT_KW_SEG)
    ax[2].imshow(make_3d_flat(img_t1ce, pos), cmap="gray", vmax=1)
    ax[2].text(250, 225, "T1ce", c="white", fontsize=20)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()


def segCmp(gt, est, labels_gt, labels_est):
    """
    Compare two segmentation maps and return a color-coded overlay image.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth segmentation (integer-labeled image).
    est : np.ndarray
        Estimated segmentation (integer-labeled image).
    labels_gt : list of int
        List of labels in the ground truth (e.g., [2, 1, 3]).
    labels_est : list of int
        Corresponding labels in the estimated map (e.g., [2, 1, 4] for GT labels).
    Returns
    -------
    color_img : np.ndarray
        An RGB image (H, W, 3) of type float32, with colors:
        - Green: correct label
        - Red: GT present, Est missing :  Missed detection (FN)
        - Yellow: Est present, GT missing : False positive (FP)
        - Blue: Est present but wrong label : Mislabeling (wrong class match)

    Examples:
    -------------
    cmp_seg,legend_patches = segCmp(gt_slice, est_slice, LABEL_GT, LABEL_EST)
    ax.imshow(cmp_seg.transpose((1,0,2)), origin='lower')
    ax.legend(handles=legend_patches, loc='lower right', frameon=True)
    ax.set_title(f"GT vs Prediction")
    """
    gt = np.asarray(gt)
    est = np.asarray(est)
    H, W = gt.shape
    color_img = np.zeros((H, W, 3), dtype=np.float32)

    green = [0, 1, 0]
    red = [1, 0, 0]
    yellow = [1, 1, 0]
    blue = [0, 0, 1]
    gt_null = gt == 0
    gt_smth = gt > 0
    est_null = est == 0
    est_smth = est > 0
    for l_gt, l_est in zip(labels_gt, labels_est):
        gt_mask = gt == l_gt
        est_mask = est == l_est

        match = gt_mask & est_mask  # Green
        wrong = est_mask & ~gt_mask & gt_smth  # Blue (override green)

        color_img[match] += green  # green
        color_img[wrong] += blue  # blue takes priority

    false = gt_null & est_smth  # Yellow
    color_img[false] = yellow  # yellow
    miss = gt_smth & est_null  # Red
    color_img[miss] = red  # red

    color_img = np.clip(color_img, 0, 1)

    # Define legend labels and corresponding colors
    legend_patches = [
        mpatches.Patch(color=green, label="Correct (match)"),
        mpatches.Patch(color=red, label="Missed (in GT only)"),
        mpatches.Patch(color=yellow, label="False positive (in Est only)"),
        mpatches.Patch(color=blue, label="Wrong label (mismatch)"),
    ]
    return color_img, legend_patches


def imCmp(I1, I2, method="supperpose"):
    M, N = I1.shape
    if method == "supperpose":
        return np.concatenate(
            (I2[:, :, None], I1[:, :, None], np.zeros((M, N, 1))), axis=2
        )
        # return np.concatenate((I2[:, :, None],np.zeros((M, N, 1)), I1[:, :, None]), axis=2)
    elif method == "substract":
        return I1 - I2
    elif method == "supperpose weighted":
        abs_diff = np.abs(I1 - I2)[:, :, None]
        return 1 - abs_diff / abs_diff.max() * np.concatenate(
            (I2[:, :, None], I1[:, :, None], np.zeros((M, N, 1))), axis=2
        )
    elif "seg" in method:
        u = I2[:, :, None] * I1[:, :, None]
        if "w" in method:
            return np.concatenate(
                (
                    I1[:, :, None],
                    u + I2[:, :, None] * 0.5,
                    I2[:, :, None],
                    np.ones((M, N, 1)),
                    # np.maximum(I2[:,:,None], I1[:, :, None])
                ),
                axis=2,
            )
        else:
            return np.concatenate(
                (
                    I1[:, :, None] - u,
                    u,
                    I2[:, :, None] - u,
                    np.ones((M, N, 1)),
                    # np.maximum(I2[:,:,None], I1[:, :, None])
                ),
                axis=2,
            )

    else:
        raise ValueError(
            f"method must be in [ 'supperpose','substract','supperpose weighted', 'seg','seg white' ] got {method}"
        )


"""---------------------------------------------------------------------------------------------------------------------
Time
---------------------------------------------------------------------------------------------------------------------"""


def ChronometerStart(msg="Start... "):
    start_time = time.time()
    sys.stdout.write(msg)
    sys.stdout.flush()
    return start_time


def ChronometerStop(start_time, method="ms", linebreak="\n"):
    elapsed_time_secs = time.time() - start_time
    if method == "ms":
        msg = (
            "Execution time: "
            + repr(round(elapsed_time_secs * 1000))
            + " ms."
            + linebreak
        )
    if method == "s":
        msg = "Execution time: " + repr(round(elapsed_time_secs)) + " s." + linebreak
    sys.stdout.write(msg)
    sys.stdout.flush()


def ChronometerTick(start_time, i, i_total, msg):
    elapsed_time_secs = time.time() - start_time
    expected_time_secs = (i_total - i - 1) / (i + 1) * elapsed_time_secs
    msg1 = "It " + repr(i + 1) + "/" + repr(i_total) + ". "
    msg2 = "Duration %s " % datetime.timedelta(seconds=round(elapsed_time_secs))
    msg3 = "Expected remaining time %s." % datetime.timedelta(
        seconds=round(expected_time_secs)
    )
    sys.stdout.write("\r" + msg + msg1 + msg2 + msg3)
    if i >= i_total - 1:
        sys.stdout.write("\n")


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    c = str(seconds - int(seconds))[:5] + "cents"
    return "{:d}:{:02d}:{:02d}s and ".format(int(h), int(m), int(s)) + c


def time_it(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"\nComputation of {func.__name__} done in ", format_time(t2 - t1), " s")
        return result

    return wrap_func


# def update_progress(progress, message=None):
#     # update_progress() : Displays or updates a console progress bar
#     ## Accepts a float between 0 and 1. Any int will be converted to a float.
#     ## A value under 0 represents a 'halt'.
#     ## A value at 1 or bigger represents 100%
#     barLength = 10  # Modify this to change the length of the progress bar
#     status = ""
#     if isinstance(progress, int):
#         progress = float(progress)
#     if not isinstance(progress, float):
#         progress = 0
#         status = "error: progress var must be float\r\n"
#     if progress < 0:
#         progress = 0
#         status = "Halt...\r\n"
#     if progress >= 1:
#         progress = 1
#         status = "Done...\r\n"
#     block = int(round(barLength * progress))
#     text = "\rProgress: [{0}] {1:6.2f}% {2}".format(
#         "#" * block + "-" * (barLength - block), progress * 100, status
#     )
#     if not message is None:
#         text += f" ({message[0]} ,{message[1]:8.2f})."
#     sys.stdout.write(text)
#     sys.stdout.flush()
