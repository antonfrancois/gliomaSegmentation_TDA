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

Grid search:
    find_resume_index
    load_previous_partial
    save_partial
    aggregate_and_update_summary
    compute_remaining_work


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
import os
import glob
import pandas as pd

# Local imports.
from src.morphology import argmax_image

"""---------------------------------------------------------------------------------------------------------------------
Global variables
---------------------------------------------------------------------------------------------------------------------"""

CMAP_SEGS = ListedColormap([[0, 0, 0, 0], "tab:red", "tab:blue", "tab:orange"])
CMAP_COMPARISON = ListedColormap([[0, 0, 0, 0], "tab:red", "tab:orange", "tab:green"])
DLT_KW_SEG = dict(alpha=1, cmap=CMAP_SEGS, interpolation="nearest", origin="lower")
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


def get_multiple_dice(seg_1, seg_2, labels=(1, 2, 3), verbose=False):
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


def plot_superlevel_sets(image, pos=None, Times=None, save_path=None):
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
        ax[i].set_title(f"t = {1-Times[i]}", fontsize=17.5)
        ax[i].axis("off")
        ax[i].set_aspect("equal")
    fig.subplots_adjust(wspace=0.05, hspace=0)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
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


def make_3d_flat(img_3D, slice, crop=20):
    """
    Take a 'brain' 3D image, take 3 slices and make a long 2D image of it.
    Adapted to BraTS 2025 data (shape=(182, 218, 182)).
    """
    D, H, W = img_3D.shape
    im0 = image_slice(img_3D, slice[2], 2).T  # (H, D)
    im1 = image_slice(img_3D, slice[1], 1).T  # (W, D)
    im2 = image_slice(img_3D, slice[0], 0).T  # (W, H)
    hc = crop // 2
    r0 = (H - D) // 2
    im0c = im0[r0 : r0 + D, hc:-hc]
    im1c = im1[:, hc:-hc]
    im2c = im2[:, hc:]
    long_img = np.concatenate([im0c, im1c[::-1], im2c[::-1]], axis=1)
    return long_img


def plot_segmentation(name, img, seg=None, pos=None, figsize=(8, 10 / 4)):
    if pos is None:
        pos = argmax_image(img)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(make_3d_flat(img, pos), cmap="gray", origin="upper")
    if seg is None:
        seg = img * 0
    ax.imshow(make_3d_flat(seg, pos), **DLT_KW_SEG)
    ax.text(175, 25, name, c="white", fontsize=20)
    plt.axis("off")
    plt.show()


def plot_comparison_binary_segmentations(
    name, img, seg_binary_gt, seg_binary_est, figsize=(8, 2 * 10 / 4)
):
    pos = argmax_image(img)
    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=figsize)
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


def plot_comparison_full_segmentations(
    name, img, seg_gt, seg_est, show_ground_truth=True, save_path=None
):
    pos = argmax_image(img)
    seg_contour_true = np.zeros(seg_gt.shape)
    seg_contour_true[seg_gt == 4] = 1

    # Define the superposition segmentation
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
    print(f"Badly segmented pixels {misseg_bool.sum()}")

    if show_ground_truth:
        fig, ax = plt.subplots(4, 1, constrained_layout=True, figsize=(8, 8 * 4 / 3))
        ax[0].imshow(make_3d_flat(img, pos), cmap="gray")
        ax[0].text(300, 25, name, c="white", fontsize=20)

        ax[1].imshow(make_3d_flat(img, pos), cmap="gray")
        ax[1].imshow(make_3d_flat(seg_gt, pos), cmap=CMAP_SEGS, interpolation="nearest")
        ax[1].text(300, 25, "True segmentation", c="white", fontsize=20)

        ax[2].imshow(make_3d_flat(img, pos), cmap="gray")
        ax[2].imshow(
            make_3d_flat(seg_est, pos), cmap=CMAP_SEGS, interpolation="nearest"
        )
        ax[2].text(300, 25, "Our segmentation", c="white", fontsize=20)

        ax[3].imshow(make_3d_flat(img, pos), cmap="gray")
        ax[3].imshow(
            make_3d_flat(seg_superpose, pos),
            cmap=CMAP_COMPARISON,
            interpolation="nearest",
        )
        ax[3].text(300, 25, "Comparison", c="white", fontsize=20)

    else:
        fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(8, 8))
        ax[0].imshow(make_3d_flat(img, pos), cmap="gray")
        ax[0].text(300, 25, name, c="white", fontsize=20)

        ax[1].imshow(make_3d_flat(img, pos), cmap="gray")
        ax[1].imshow(
            make_3d_flat(seg_est, pos), cmap=CMAP_SEGS, interpolation="nearest"
        )
        ax[1].text(300, 25, "Our segmentation", c="white", fontsize=20)

        ax[2].imshow(make_3d_flat(img, pos), cmap="gray")
        ax[2].imshow(
            make_3d_flat(seg_superpose, pos),
            cmap=CMAP_COMPARISON,
            interpolation="nearest",
        )
        ax[2].text(300, 25, "Comparison", c="white", fontsize=20)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def plot_images_and_segmentation(
    image_name, img_flair, img_t1ce, seg_gt, figsize=(10, 10 * 2 / 3)
):
    pos = argmax_image(img_flair)
    fig, ax = plt.subplots(3, 1, figsize=figsize)
    ax[0].imshow(make_3d_flat(img_flair, pos), cmap="gray", vmax=1)
    ax[0].text(175, 25, image_name, c="white", fontsize=20)
    ax[0].text(350, 175, "FLAIR", c="white", fontsize=20)
    ax[1].imshow(make_3d_flat(img_t1ce, pos), cmap="gray", vmax=1)
    ax[1].text(350, 175, "T1ce", c="white", fontsize=20)
    ax[2].imshow(make_3d_flat(img_flair, pos), cmap="gray", vmax=1)
    ax[2].imshow(make_3d_flat(seg_gt, pos), DLT_KW_SEG["cmap"])
    ax[2].text(350, 175, "Segmentation", c="white", fontsize=20)
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


"""---------------------------------------------------------------------------------------------------------------------
Grid search
---------------------------------------------------------------------------------------------------------------------"""


def find_resume_index(results_dir, filename_head, parameters):
    prefix = results_dir + "/" + filename_head + str(parameters)
    files = glob.glob(prefix + "_len*.csv")
    files_len = []
    for f in files:
        try:
            tail = f[len(prefix) + 4 : -4]  # skip '_len' and '.csv'
            files_len.append(int(tail))
        except Exception:
            continue
    if not files_len:
        print("No files found for", parameters)
        return 0, 0
    ind = int(np.argmax(files_len))
    i_min = files_len[ind]
    print("Found partial file:", files[ind], "- resuming at i =", i_min)
    return i_min, i_min


def load_previous_partial(results_dir, filename_head, parameters):
    prefix = results_dir + "/" + filename_head + str(parameters)
    files = glob.glob(prefix + "_len*.csv")
    files_len = []
    for f in files:
        try:
            tail = f[len(prefix) + 4 : -4]
            files_len.append(int(tail))
        except Exception:
            continue
    if not files_len:
        return None
    ind = int(np.argmax(files_len))
    return pd.read_csv(files[ind])


def save_partial(results_dir, filename_head, parameters, df):
    file_name = (
        results_dir + "/" + filename_head + str(parameters) + f"_len{len(df)}.csv"
    )
    df.to_csv(file_name, index=False)
    return file_name


def aggregate_and_update_summary(
    results_dir, filename_head, parameters, full_df, column_names
):
    summary_path = results_dir + "/" + filename_head + "grid_summary.csv"
    means = {
        col_name: float(full_df[col_name].mean(skipna=True))
        for col_name in column_names[1:]
    }
    overall = float(np.nanmean([means[col_name] for col_name in column_names[1:]]))
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "parameters": str(parameters),
        "n_cases": int(len(full_df)),
        **means,
        "mean_all": overall,
    }
    if os.path.exists(summary_path):
        sdf = pd.read_csv(summary_path)
        sdf = pd.concat([sdf, pd.DataFrame([row])], ignore_index=True)
        sdf = sdf.drop_duplicates(subset=["parameters"], keep="last")
    else:
        sdf = pd.DataFrame([row])
    sdf.to_csv(summary_path, index=False)
    # Report best so far
    best = sdf.sort_values("mean_all", ascending=False).head(25)
    print("\nTop 25 configs so far (by mean_all):")
    print(
        best[["mean_all"] + column_names[1:] + ["n_cases", "parameters"]].to_string(
            index=False
        )
    )


def compute_remaining_work(results_dir, filename_head, grid, i_list):
    remaining_by_param = {}
    total_remaining = 0
    for params in grid:
        _, len_done = find_resume_index(results_dir, filename_head, params)
        cases_remaining = max(0, len(i_list) - min(len_done, len(i_list)))
        remaining_by_param[str(params)] = cases_remaining
        total_remaining += cases_remaining
    return remaining_by_param, total_remaining
