"""---------------------------------------------------------------------------------------------------------------------

Train-Free Segmentation in MRI with Cubical Persistent Homology
Anton François & Raphaël Tinarrage
See the repo at https://github.com/antonfrancois/gliomaSegmentation_TDA and article at https://arxiv.org/abs/2401.01160

------------------------------------------------------------------------------------------------------------------------

Brain segmentations:
    suggest_t
    segment_whole_object
    segment_geometric_object
    segment_other_components
    preprocess_brain
    segment_brain

Myocardium segmentations:
    parseACDC
    compute_sphericity
    suggest_t_pos
    preprocess_cardiac
    segment_whole_object_cardiac
    segment_geometric_object_cardiac
    segment_other_components_cardiac
    segment_cardiac

Fetal cortical plate segmentations:
    parseSTA
    preprocess_fetal
    segment_boundary_slices
    segment_fetal

---------------------------------------------------------------------------------------------------------------------
"""

# Standard imports.

# Third-party imports.
import numpy as np
import scipy
import skimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import persim
import cripser
from nibabel import load as nib_load

# Local imports.
from morphology import get_component, get_largest_component
from utils import (
    ChronometerStart,
    ChronometerStop,
    DLT_KW_SEG,
    get_dice,
)
from parseBrats import ROOT_DIRECTORY

"""---------------------------------------------------------------------------------------------------------------------
Brain segmentations
---------------------------------------------------------------------------------------------------------------------"""


def preprocess_brain(
    img_flair,
    img_t1ce,
    sigma,
    normalize="max",
    enhance=(True, True),
    radius_enhance=1,
    dilate=True,
    radius_dilation=0,
):
    # Enhance image.
    mask = img_flair != 0
    footprint = skimage.morphology.ball(radius_enhance)
    if enhance[0]:
        img_flair = skimage.filters.rank.enhance_contrast(
            img_flair, footprint=footprint, mask=mask
        )
    if enhance[1]:
        img_t1ce = skimage.filters.rank.enhance_contrast(
            img_t1ce, footprint=footprint, mask=mask
        )
    # Normalize images.
    if normalize == "max":
        img_flair = img_flair / np.max(img_flair)
        img_t1ce = img_t1ce / np.max(img_t1ce)
    else:
        img_flair = img_flair / 255
        img_t1ce = img_t1ce / 255
    # Gaussian smoothing.
    if sigma > 0:
        img_flair = scipy.ndimage.gaussian_filter(img_flair, sigma=sigma)
        img_t1ce = scipy.ndimage.gaussian_filter(img_t1ce, sigma=sigma)
    # Dilate second image.
    if dilate:
        footprint = skimage.morphology.ball(radius_dilation)
        img_t1ce = skimage.morphology.dilation(
            img_t1ce,
            footprint=footprint,
            out=None,
        )
    return img_flair, img_t1ce


def suggest_t(img, ticks=100, threshold=1, plot=False, ax=None):
    """
    Suggests an optimal threshold value `t` for segmenting the input image `img` by analyzing the change in the number
    of active voxels as the threshold varies. More specifically, one builds the curve of the number of active voxels as
    a function of time, takes its derivative, normalizes it, and then identifies the best threshold as the first one
    where the curve exceeds a specified derivative threshold `dt_threshold`.

    Args:
        img (np.ndarray): Input image, expected to be normalized in [0, 1].
        ticks (int, optional): Number of threshold values to test between 0 and 1. Default is 100.
        threshold (float, optional): Derivative threshold to determine the optimal threshold. Default is 1.
        plot (bool, optional): If True, plots the active voxel curve and its derivative. Default is False.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, a new figure is created.

    Returns:
        float: Suggested threshold value `t`.
    """
    vals = np.linspace(0, 1, ticks)
    # Build suggestion curve.
    active_vxls = np.array([np.sum(img > t) for t in vals])
    # Take derivative via finite differences.
    active_vxls_dt = active_vxls[:-1] - active_vxls[1:]
    # Normalize it so it has integral 1, i.e., np.sum(active_vxls_dt_norm)/len(active_vxls_dt_norm) = 1.
    active_vxls_dt_norm = active_vxls_dt * len(active_vxls_dt) / active_vxls_dt.sum()
    # Find optimal t. The best index is the last index for which active_vxls_dt_norm > dt_threshold.
    best_idx = np.where(active_vxls_dt_norm > threshold)[0][-1]
    best_t = vals[best_idx + 1]
    # Plot if required.
    if plot or not (ax is None):
        c1, c2, c3 = "forestgreen", "firebrick", "goldenrod"
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
        f = ax.plot(vals, active_vxls, "o-", label="f", c=c1)
        ax.plot([best_t, best_t], [0, active_vxls.max()], "--", c=c2)
        ax.text(best_t + 0.01, 0.8 * active_vxls.max(), f"t = {best_t:.3f}", c=c2)
        axt = ax.twinx()
        df = axt.plot(
            vals[:-1], active_vxls_dt_norm, "D--", c=c3, label="df normalized"
        )
        lns = f + df
        labs = [l.get_label() for l in lns]
        ax.set_xlabel("time")
        ax.set_ylabel("f")
        axt.set_ylabel("df")
        ax.yaxis.get_label().set_color(f[0].get_color())
        axt.yaxis.get_label().set_color(df[0].get_color())
        ax.legend(lns, labs, loc="upper right")
        if plot:
            plt.show()
    return best_t


def segment_whole_object(
    img,
    method="suggest_t",
    threshold=1,
    seg_gt=None,
    iterations_binary_closing=0,
    verbose=True,
    plot=True,
):
    """Segments the whole tumor from a FLAIR image. The method 'suggest_t', 'gt' or 'gt_hull'."""
    # Segment via automatic threshold detection.
    if method == "suggest_t":
        if verbose:
            start_time = ChronometerStart("Suggest threshold... ")
        # Find the best threshold.
        t = suggest_t(
            img=img,
            threshold=threshold,
            plot=plot,
        )
        if verbose:
            ChronometerStop(start_time, method="s")
        # Extract the largest component.
        if verbose:
            start_time = ChronometerStart("Extract the largest component... ")
        seg_whole = get_largest_component(img, t, verbose=False)
        if verbose:
            ChronometerStop(start_time, method="s")
        # Fill the holes of the component.
        if verbose:
            start_time = ChronometerStart("Fill the holes... ")
        seg_whole = scipy.ndimage.binary_fill_holes(seg_whole)
        if verbose:
            ChronometerStop(start_time, method="s")
        # Smooth via binary closing.
        if iterations_binary_closing > 0:
            if verbose:
                start_time = ChronometerStart("Smoothing... ")
            seg_whole = scipy.ndimage.morphology.binary_closing(
                seg_whole, iterations=iterations_binary_closing
            )
            if verbose:
                ChronometerStop(start_time, method="s")
    # Segment using the ground-truth segmentation.
    elif method == "gt":
        seg_whole = (seg_gt > 0) * 1
    # Segment using the convex hull of the ground-truth segmentation.
    elif method == "gt_hull":
        if verbose:
            start_time = ChronometerStart("Take convex hull... ")
        seg_whole = (seg_gt > 0) * 1
        seg_whole = skimage.morphology.convex_hull_image(seg_whole)
        seg_whole = seg_whole * 1
        if verbose:
            ChronometerStop(start_time, method="s")
    return seg_whole


def segment_geometric_object(img, seg_whole, max_bars=5, verbose=True, plot=True):
    # Compute persistent homology H2.
    if verbose:
        start_time = ChronometerStart("Compute diagram... ")
    seg_whole_img = img * seg_whole
    barcode = cripser.computePH(1 - seg_whole_img, maxdim=3)
    # Extract (non-infinite) H2 features from the barcode.
    H2 = [list(bar[1::]) for bar in barcode if bar[0] == 2 and bar[2] < 1]
    # Sort list H2 by persistence.
    H2 = [bar for _, bar in sorted(zip([bar[1] - bar[0] for bar in H2], H2))[::-1]]
    if verbose:
        ChronometerStop(start_time, method="s")
    # Print features if required.
    if verbose:
        print("H2 diagram:", H2[0:max_bars])
    # If the diagram is empty, return an empty segmentation.
    if len(H2) == 0:
        return seg_whole_img * 0
    # Plot diagram if required.
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        persim.plot_diagrams(
            [
                np.array([barcode[0][1:3]]),
                np.array([barcode[0][1:3]]),
                np.array([bar[1:3] for bar in barcode if bar[0] == 3 - 1]),
            ]
        )
        plt.title("Persistence diagram of image segmented", fontsize=10)
    # Compute width of the holes of H2-features
    if verbose:
        start_time = ChronometerStart("Identify the largest hole... ")
    idx_max = min(max_bars, len(H2))  # number of top bars to parse
    length_holes = []
    for idx in range(idx_max):
        # Get H2-cycle and its birth voxel.
        bar = H2[idx]
        pos = np.array(bar[2:5]).astype(int)
        t = bar[0] + 0.0001
        seg_geom = get_component(seg_whole_img, pos, 1 - t)
        # Identify inner voxels.
        seg_complement = 1 - seg_geom
        labels = skimage.measure.label(seg_complement, background=0)
        components = [(labels == i) * 1 for i in range(1, np.max(labels) + 1)]
        components_len = [np.sum(component) for component in components]
        # Get the background component and remove it.
        imax_comp = np.argmax(components_len)
        components.pop(imax_comp)
        # Add to list.
        length_holes.append(np.sum([np.sum(component) for component in components]))
    if verbose:
        ChronometerStop(start_time, method="s")
        print("Width of the holes:", {i: length_holes[i] for i in range(idx_max)})
    # Select the largest hole
    idx = np.argmax(length_holes)
    bar = H2[idx]
    pos = np.array(bar[2:5]).astype(int)
    t = bar[0] + 0.0001
    seg_geom = get_component(seg_whole_img, pos, 1 - t)
    # Plot is required.
    if plot:
        patch = plt.Circle((bar[0], bar[1]), 0.01, fill=False)
        ax.add_patch(patch)
        plt.show()
    return seg_geom


def segment_other_components(seg_whole, seg_geom, radius_dilation=0, verbose=True):
    # 3rd step: - Classify components
    # 1 - RED, TC  -> NECROSE INACTIVE, TUMORUS CORE
    # 2 - BLUE, ED -> INFILTRATION, OEDEME
    # 4 - ORANGE, ET -> NECROSE ACTIVE, ENHANCING TUMOR
    # Compute connected components of the complement.
    if verbose:
        start_time = ChronometerStart("Get connected components... ")
    seg_complement = 1 - seg_geom
    labels = skimage.measure.label(seg_complement, background=0)
    if verbose:
        ChronometerStop(start_time, method="s")
    # Get length of the components.
    components = [(labels == i) * 1 for i in range(1, np.max(labels) + 1)]
    components_len = [np.sum(component) for component in components]
    imax_comp = np.argmax(components_len)
    # Classify components: TC or WT.
    seg_final = seg_geom.copy() * 4  # define seg_final
    seg_final[(components[imax_comp] * seg_whole) > 0] = 2  # add ED
    components.pop(imax_comp)  # remove ED (background) from components
    for component in components:
        seg_final[component > 0] = 1  # add TC
    # Undo dilation (from preprocess).
    if radius_dilation > 0:
        # Erode the segmentation to remove dilation effects.
        footprint = skimage.morphology.ball(radius_dilation)
        seg_TC_dilate = (
            skimage.morphology.dilation((seg_final == 1) * 1, footprint=footprint)
            * seg_whole
        )
        seg_ET_erode = (
            skimage.morphology.erosion((seg_final == 4) * 1, footprint=footprint)
            * seg_whole
        )
        # Create final segmentations from ED.
        seg_final = seg_whole.copy() * 2
        # Add TC.
        seg_final[seg_TC_dilate > 0] = 1
        # Add ET.
        seg_final[seg_ET_erode > 0] = 4
    return seg_final


def segment_brain(
    img_flair,
    img_t1ce,
    normalize="max",
    sigma=1,
    enhance=(False, False),
    radius_enhance=1,
    dilate=True,
    radius_dilation=1,
    whole_threshold=1,
    max_bars=2,
    verbose=False,
    plot=False,
):
    # Preprocess image.
    img_flair, img_t1ce = preprocess_brain(
        img_flair,
        img_t1ce,
        sigma=sigma,
        normalize=normalize,
        enhance=enhance,
        radius_enhance=radius_enhance,
        dilate=dilate,
        radius_dilation=radius_dilation,
    )
    # Module 1: Segmentation whole object.
    seg_whole = segment_whole_object(
        img=img_flair,
        threshold=whole_threshold,
        verbose=verbose,
        plot=plot,
    )
    # Module 2: Segmentation geometric object.
    seg_geom = segment_geometric_object(
        img=img_t1ce, seg_whole=seg_whole, max_bars=max_bars, verbose=verbose, plot=plot
    )
    # Module 3: Deduce final segmentation.
    seg_final = segment_other_components(
        seg_whole=seg_whole,
        seg_geom=seg_geom,
        radius_dilation=radius_dilation,
        verbose=verbose,
    )
    return seg_final


"""---------------------------------------------------------------------------------------------------------------------
Myocardium segmentations
---------------------------------------------------------------------------------------------------------------------"""


def parseACDC(n_image, end="ED"):
    """Open an image from the STA dataset. `n_image` must be between 1 and 150 included. (training up to 100, then
    testing ). The parameter `end` can be 'ED or 'ES' (end diastole, end systole)."""
    # Open image and segmentation
    n_image_str = "0" * (3 - len(str(n_image))) + str(n_image)
    if n_image <= 100:
        doc = f"{ROOT_DIRECTORY}/../data/acdc/training/patient{n_image_str}/Info.cfg"
    else:
        doc = f"{ROOT_DIRECTORY}/../data/acdc/testing/patient{n_image_str}/Info.cfg"

    with open(doc) as f:
        lines = f.readlines()
    if end == "ED":
        n_ED = int(lines[0][4:6])
        n_ED_str = "0" * (2 - len(str(n_ED))) + str(n_ED)
        if n_image <= 100:
            filename = f"{ROOT_DIRECTORY}/../data/acdc/training/patient{n_image_str}/patient{n_image_str}_frame{n_ED_str}.nii.gz"
        else:
            filename = f"{ROOT_DIRECTORY}/../data/acdc/testing/patient{n_image_str}/patient{n_image_str}_frame{n_ED_str}.nii.gz"
    elif end == "ES":
        n_ES = int(lines[1][4:6])
        n_ES_str = "0" * (2 - len(str(n_ES))) + str(n_ES)
        if n_image <= 100:
            filename = f"{ROOT_DIRECTORY}/../data/acdc/training/patient{n_image_str}/patient{n_image_str}_frame{n_ES_str}.nii.gz"
        else:
            filename = f"{ROOT_DIRECTORY}/../data/acdc/testing/patient{n_image_str}/patient{n_image_str}_frame{n_ES_str}.nii.gz"
    img = nib_load(filename).get_fdata()
    img /= np.max(img)
    filename = filename[0:-7] + "_gt.nii.gz"
    seg_gt = nib_load(filename).get_fdata()
    return img, seg_gt


def compute_sphericity(CC, verbose=True):
    """CC can be of dimension 2 or 3."""
    if CC.ndim == 2:
        # Get center and radius
        center = scipy.ndimage.center_of_mass(CC)
        center = (int(center[0]), int(center[1]))
        CC_indices = np.where(CC)
        dist_from_center = np.sqrt(
            (CC_indices[0] - center[0]) ** 2 + (CC_indices[1] - center[1]) ** 2
        )
        radius = max(dist_from_center)
        radius = int(radius)

        # Create mask
        disk = skimage.morphology.disk(radius)
        mask = np.zeros(np.shape(CC))

        xmin_overlow = center[0] - radius
        xmin_overlow = -xmin_overlow * (xmin_overlow < 0)
        xmax_overlow = np.shape(mask)[0] - 1 - (center[0] + radius + 1)
        xmax_overlow = -xmax_overlow * (xmax_overlow < 0)
        ymin_overlow = center[1] - radius
        ymin_overlow = -ymin_overlow * (ymin_overlow < 0)
        ymax_overlow = np.shape(mask)[1] - 1 - (center[1] + radius + 1)
        ymax_overlow = -ymax_overlow * (ymax_overlow < 0)
        xmin, xmax = max(0, center[0] - radius), min(
            center[0] + radius + 1, np.shape(CC)[0]
        )
        ymin, ymax = max(0, center[1] - radius), min(
            center[1] + radius + 1, np.shape(CC)[1]
        )

        mask[xmin:xmax, ymin:ymax] = disk[
            xmin_overlow : (np.shape(disk)[0] - xmax_overlow + 1),
            ymin_overlow : (np.shape(disk)[1] - ymax_overlow + 1),
        ]

        # Compute Dice
        dice = get_dice(CC, mask)
        if verbose:
            print("Sphericity Dice:", dice)
        return dice

    elif CC.ndim == 3:
        # Get nonempty slices of CC
        nonempty_slices = np.where(np.sum(np.sum(CC, 0), 0))[0]
        zmin, zmax = nonempty_slices[0], nonempty_slices[-1]
        z_im = range(zmin, zmax + 1)

        # Compute Dices
        dices = [compute_sphericity(CC[:, :, z], verbose=False) for z in z_im]
        return np.mean(dices)


def suggest_t_pos(
    im, pos, t_birth, N=100, dt_threshold=1, direction="left", plot=False
):
    eps = 0.000000001  # erreur machine

    # Compute size of components
    CC_width = dict()
    T = np.linspace(0, 1 - t_birth - eps, N)
    for t in T:
        imt = (im >= t) * 1
        if im.ndim == 2:
            pixel_value = imt[pos[0], pos[1]]
        elif im.ndim == 3:
            pixel_value = imt[pos[0], pos[1], pos[2]]
        else:
            print("Error! Dim", CC.ndim)

        if pixel_value == 0:
            print(t, "Warning! The voxel pos is not active at time t :(")
            CC_width[t] = 0
        else:
            labels = skimage.measure.label(imt, background=0)
            if im.ndim == 2:
                labeltumor = labels[pos[0], pos[1]]
            elif im.ndim == 3:
                labeltumor = labels[pos[0], pos[1], pos[2]]
            CC = (labels == labeltumor) * 1
            CC_width[t] = np.sum(CC)
            if labeltumor == 0:
                print(i_bar, "Warning! The label is background :(")

    # Build suggestion curve
    S = np.array([CC_width[t] for t in T])
    S_dt = S[:-1] - S[1:]  # finite differences
    S_dt_norm = (
        S_dt * len(S_dt) / S_dt.sum()
    )  # S_dt_norm has integral 1, i.e., np.sum(S_dt_norm)/len(S_dt) = 1

    # Find optimal t
    if direction == "left":
        best_i = np.where(S_dt_norm > dt_threshold)[0][
            -1
        ]  # last index for which S_dt_norm>dt_threshold
        best_t = T[best_i + 1]
    elif direction == "right":
        best_i = np.where(S_dt_norm > dt_threshold)[0][
            1
        ]  # first nonzero index for which S_dt_norm>dt_threshold
        best_t = T[best_i - 1]

    # Define component
    imt = (im >= best_t) * 1
    labels = skimage.measure.label(imt, background=0)
    if im.ndim == 2:
        labeltumor = labels[pos[0], pos[1]]
    elif im.ndim == 3:
        labeltumor = labels[pos[0], pos[1], pos[2]]
    CC = (labels == labeltumor) * 1

    # Plot
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2))
        plt.plot(T, S / np.max(S) * np.max(S_dt_norm), c="black", label="_nolegend_")
        plt.plot(T[0:-1], S_dt_norm, c="black", label="_nolegend_")
        plt.scatter(T[0:-1], S_dt_norm, c="black", s=10, label="_nolegend_")
        plt.axhline(dt_threshold, c="blue", label="dt_threshold")
        plt.axvline(best_t, c="pink", label="best t")
        plt.ylim(0, np.max(S_dt_norm[1::]) * 1.1)
        plt.title("Suggest t, dt_threshold = " + repr(dt_threshold))
        plt.legend()
        plt.show()

    return CC


def preprocess_cardiac(img, sigma=1, radius_dilation=0):
    if sigma > 0:
        img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    if radius_dilation > 0:
        img = skimage.morphology.dilation(
            img,
            footprint=skimage.morphology.ball(radius_dilation),
            out=None,
        )
    return img


def segment_whole_object_cardiac(
    img,
    seg_gt,
    H0_features_max=10,
    dt_threshold=1,
    thresh_small_LV=1000,
    ratio_small_RV=0.25,
    ratio_big_RV=5,
    radius_dilation=0,
    add_to_iterations=0,
    maximal_distance=np.inf,
    verbose=False,
    plot=False,
    save=False,
):
    # Compute PH0
    if verbose:
        start_time = ChronometerStart("Compute diagram... ")
    barcode = cripser.computePH(1 - img, maxdim=0)  # Compute diagram
    H0 = [list(bar[1::]) for bar in barcode if bar[0] == 0]
    H0 = [
        bar for _, bar in sorted(zip([bar[1] - bar[0] for bar in H0], H0))[::-1]
    ]  # Sort list H2 by persistence
    H0[0][1] = 1  # correction infinite bar (1 instead of inf)
    if verbose:
        ChronometerStop(start_time, method="s")

    # Compute sphericity
    H0_sphericity = dict()
    H0_CC = dict()
    if verbose:
        start_time = ChronometerStart("Compute sphericity... ")
    for i in range(H0_features_max):
        t_birth, pos = H0[i][0], [int(H0[i][2 + k]) for k in range(img.ndim)]
        CC = suggest_t_pos(
            img, pos, t_birth, dt_threshold=dt_threshold, direction="left"
        )  # Segment via suggest_t_pos
        H0_CC[i] = CC
        H0_sphericity[i] = compute_sphericity(CC, verbose=False)  # Compute sphericity
    if verbose:
        ChronometerStop(start_time, method="s")
    if verbose:
        print("H0_sphericity", H0_sphericity)

    # Segment LV
    H0_sphericity_thresh = {
        i: H0_sphericity[i] * (np.sum(H0_CC[i]) >= thresh_small_LV)
        for i in H0_sphericity
    }
    # Discard small components
    best_i_LV = max(H0_sphericity_thresh, key=lambda k: H0_sphericity_thresh[k])
    seg_LV = H0_CC[best_i_LV]
    if radius_dilation > 0:
        seg_LV = skimage.morphology.binary_erosion(
            seg_LV, footprint=skimage.morphology.ball(radius_dilation)
        )
    if verbose:
        print("Most spherical CC is index", best_i_LV, "- width", np.sum(seg_LV))

    # Segment RV
    if np.isinf(maximal_distance):
        top_features_loc = [
            np.mean(np.where(H0_CC[i]), axis=1) for i in range(H0_features_max)
        ]
        distances = np.linalg.norm(
            np.array(top_features_loc) - top_features_loc[best_i_LV], axis=1
        )
    else:
        seg_LV_dilations = {0: seg_LV}
        if img.ndim == 2:
            footprint = skimage.morphology.disk(1)
        elif img.ndim == 3:
            footprint = skimage.morphology.ball(1)
        for dilation in range(1, maximal_distance + 1):
            seg_LV_dilations[dilation] = skimage.morphology.binary_dilation(
                seg_LV_dilations[dilation - 1], footprint=footprint
            )
        distances = [np.inf for i in range(H0_features_max)]
        for i in range(H0_features_max):
            for dilation in range(0, maximal_distance + 1):
                if np.max(H0_CC[i][np.where(seg_LV_dilations[dilation] > 0)]) > 0:
                    distances[i] = dilation
                    break
    if verbose:
        print("distances before discarding:", distances)
    # Discard small or big components
    for i in range(H0_features_max):
        if np.sum(H0_CC[i]) < ratio_small_RV * np.sum(seg_LV) or np.sum(
            H0_CC[i]
        ) > ratio_big_RV * np.sum(seg_LV):
            distances[i] = np.inf
    if verbose:
        print("distances after discarding:", distances)

    best_i_RV = np.argsort(distances)[
        1
    ]  # second smallest distance (the first one is 0)
    seg_RV = H0_CC[best_i_RV]
    if min(seg_LV[seg_RV > 0]) > 0:
        seg_LV[seg_RV > 0] = 0
        if verbose:
            print("Warning! LV and RV intersect.")
    if not np.isinf(maximal_distance):
        if radius_dilation > 0:
            if img.ndim == 2:
                seg_RV = skimage.morphology.binary_erosion(
                    seg_RV, footprint=skimage.morphology.disk(radius_dilation)
                )
            elif img.ndim == 3:
                seg_RV = skimage.morphology.binary_erosion(
                    seg_RV, footprint=skimage.morphology.ball(radius_dilation)
                )
    if verbose:
        print("Closest CC is index", best_i_RV, "- width", np.sum(seg_RV))
    if np.min(distances) == np.inf:
        print("Error! Min distance is inf.")

    # Dilate LV up to touch RV
    if np.sum(seg_LV) > 0 and np.sum(seg_RV) > 0:
        if img.ndim == 2:
            pairwise_distances = scipy.spatial.distance.cdist(
                np.array(np.where(seg_LV)).T, np.array(np.where(seg_RV)).T
            )
            iterations = int(np.min(pairwise_distances)) + add_to_iterations
        elif img.ndim == 3:
            iterations = add_to_iterations
            for z in range(np.shape(seg_LV)[2]):
                if np.sum(seg_LV[:, :, z]) > 0 and np.sum(seg_RV[:, :, z]) > 0:
                    pairwise_distances = scipy.spatial.distance.cdist(
                        np.array(np.where(seg_LV[:, :, z])).T,
                        np.array(np.where(seg_RV[:, :, z])).T,
                    )
                    iterations = max(
                        iterations, int(np.min(pairwise_distances)) + add_to_iterations
                    )
    else:
        iterations = add_to_iterations
    seg_whole = (
        scipy.ndimage.binary_dilation(seg_LV, iterations=iterations) + seg_RV > 0
    ) * 1
    if verbose:
        print("Number of iterations =", iterations)

    # Fill CC with holes
    if verbose:
        start_time = ChronometerStart("Fill the holes... ")
    seg_whole = scipy.ndimage.binary_fill_holes(seg_whole)
    if verbose:
        ChronometerStop(start_time, method="s")

    # Plot diagram and top features
    if plot:
        fig = plt.figure(figsize=(6, 6 / 2))
        fig.subplots_adjust(wspace=0.05, hspace=0)

        # Plot diagram
        eps = 0.01  # to print text next to the points
        ax = fig.add_subplot(1, 2, 1)
        persim.plot_diagrams(
            [
                np.array([np.clip(bar[1:3], 0, 1) for bar in barcode if bar[0] == i])
                for i in range(1)
            ]
        )
        plot_features = [(H0[i][0], H0[i][1]) for i in range(H0_features_max)]
        for i, (x, y) in enumerate(plot_features):
            ax.scatter(x, y, c="C" + repr(i + 1))
            ax.text(x + eps, y + eps, i, color="black")
        plt.title("Persistence diagram", fontsize=10)

        # Plot connected components
        ax = fig.add_subplot(1, 2, 2)
        ax.axis("off")
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", alpha=0.9, origin="upper")
        elif img.ndim == 3:
            ax.imshow(
                img[:, :, 4], cmap="gray", alpha=0.9, origin="upper"
            )  # 4 is an arbitrary index
        for i in range(H0_features_max):
            CC = H0_CC[i]
            if img.ndim == 3:
                CC = CC[:, :, 4]  # 4 is an arbitrary index
            if np.sum(CC) > 0:
                ax.imshow(
                    np.ma.masked_where(CC == 0, CC),
                    alpha=0.9,
                    cmap=ListedColormap(["C" + repr(i + 1)]),
                    origin="upper",
                )
                center = scipy.ndimage.center_of_mass(CC)
                ax.text(int(center[1]), int(center[0]), i, color="white")

        if save:
            plt.savefig("results/module_1_local.pdf", format="pdf", bbox_inches="tight")
            plt.show()

    # Plot segmentation
    if plot:
        fig = plt.figure(figsize=(10, 10 / 3))
        fig.subplots_adjust(wspace=0.05, hspace=0)

        if img.ndim == 2:
            (
                img_slice,
                seg_medecin_slice,
                seg_LV_slice,
                seg_RV_slice,
                seg_union_slice,
            ) = (img, seg_gt, seg_LV, seg_RV, seg_whole)
        elif img.ndim == 3:
            (
                img_slice,
                seg_medecin_slice,
                seg_LV_slice,
                seg_RV_slice,
                seg_union_slice,
            ) = (
                img[:, :, 4],
                seg_gt[:, :, 4],
                seg_LV[:, :, 4],
                seg_RV[:, :, 4],
                seg_whole[:, :, 4],
            )  # 4 is an arbitrary index

        ax = fig.add_subplot(1, 3, 1)
        ax.axis("off")
        ax.imshow(img_slice, cmap="gray", alpha=0.9, origin="upper")
        ax.imshow(seg_LV_slice + 3 * seg_RV_slice, **DLT_KW_SEG)
        ax.set_title("Segmentation of LV and RV")

        ax = fig.add_subplot(1, 3, 2)
        ax.axis("off")
        ax.imshow(img_slice, cmap="gray", alpha=0.9, origin="upper")
        ax.imshow(seg_union_slice, **DLT_KW_SEG)
        ax.set_title("Segmentation union")

        ax = fig.add_subplot(1, 3, 3)
        ax.axis("off")
        ax.imshow(img_slice, cmap="gray", alpha=0.9, origin="upper")
        ax.imshow(seg_medecin_slice, **DLT_KW_SEG)
        ax.set_title("Segmentation medecin")
        plt.show()

    return seg_whole


def segment_geometric_object_cardiac(img, seg_whole, verbose=False, plot=False):
    img_union = img * seg_whole  # define image restricted to seg_union
    img_union[img_union == 0] = (
        1  # set boundary values to 1 (we will consider sublevel sets)
    )

    # Artificially add boundaries (to create a sphere from the cylinder)
    if img.ndim == 3:
        img_union[:, :, 0], img_union[:, :, -1] = 0, 0

    # Compute barcode
    if verbose:
        start_time = ChronometerStart("Compute diagram... ")
    barcode = cripser.computePH(img_union, maxdim=img.ndim - 1)  # Compute diagram
    if verbose:
        ChronometerStop(start_time, method="s")

    # Get longest bar
    H = [list(bar[1::]) for bar in barcode if bar[0] == img.ndim - 1]  # list bars
    H = [
        bar for _, bar in sorted(zip([bar[1] - bar[0] for bar in H], H))[::-1]
    ]  # sort list H by persistence

    if len(H) >= 1:
        # Get cycle
        bar = H[0]  # most persistent cycle
        pos, t_birth = [int(bar[2 + k]) for k in range(img.ndim)], bar[
            0
        ] + 0.0000001  # erreur machine
        labels = skimage.measure.label((img_union <= t_birth) * 1, background=0)
        if img.ndim == 2:
            labelcycle = labels[pos[0], pos[1]]
        elif img.ndim == 3:
            labelcycle = labels[pos[0], pos[1], pos[2]]
        seg_geom = (labels == labelcycle) * 1

        # Plot diagram and seg
        if plot:
            fig = plt.figure(figsize=(9, 9 / 2))
            ax = fig.add_subplot(1, 3, 1)
            dim_max = int(max([bar[0] for bar in barcode]))
            persim.plot_diagrams(
                [
                    np.array(
                        [np.clip(bar[1:3], 0, 1) for bar in barcode if bar[0] == i]
                    )
                    for i in range(dim_max + 1)
                ],
                ax=ax,
            )
            patch = plt.Circle((H[0][0], H[0][1]), 0.01, fill=False)
            ax.add_patch(patch)  # most persistent cycle
            ax.set_title("Persistence diagram", fontsize=10)

            ax = fig.add_subplot(1, 3, 2)
            ax.axis("off")
            ax.set_title("TDA seg")
            if img.ndim == 2:
                img_slice, seg_contour_slice = img, seg_geom
            elif img.ndim == 3:
                img_slice, seg_contour_slice = img[:, :, 4], seg_geom[:, :, 4]
            ax.imshow(img_slice, cmap="gray", alpha=0.9, origin="upper")
            ax.imshow(2 * seg_contour_slice, **DLT_KW_SEG)
            ax.set_title("Segmentation contour", fontsize=10)
            plt.show()
    else:
        print("Warning! Empty barcode in dimension", img.ndim - 1)
        seg_geom = np.zeros(np.shape(img))

    return seg_geom


def segment_other_components_cardiac(seg_whole, seg_geom, verbose=True):
    "1: LV, 2: Myocardium, 3: RV."
    if verbose:
        start_time = ChronometerStart("Get connected components... ")
    seg_complement = 1 - seg_geom
    labels = skimage.measure.label(seg_complement, background=0)
    if verbose:
        ChronometerStop(start_time, method="s")
    # Get length of the components.
    components = [(labels == i) * 1 for i in range(1, np.max(labels) + 1)]
    components_len = [np.sum(component) for component in components]
    imax_comp = np.argmax(components_len)
    # Classify components.
    seg_final = seg_geom.copy() * 2  # initialize with myocardium
    seg_final[(components[imax_comp] * seg_whole) > 0] = 1  # add LV
    components.pop(imax_comp)  # remove background
    for component in components:
        seg_final[component > 0] = 3  # add RV
    return seg_final


def segment_cardiac(
    img,
    seg_gt,
    method="2D",
    sigma=1,
    radius_dilation=0,
    dt_threshold=1.0,
    H0_features_max=10,
    thresh_small_LV=200,
    ratio_small_RV=0.1,
    ratio_big_RV=5,
    add_to_iterations=2,
    maximal_distance=np.inf,
    verbose=False,
    plot=False,
):
    """The parameter `method` can be '2D' or '3D'."""
    # Preprocess.
    img = preprocess_cardiac(img=img, sigma=sigma, radius_dilation=radius_dilation)
    # Segment 2D (slice by slice)
    if method == "2D":
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
    # Segment 3D (whole CMR)
    elif method == "3D":
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
        seg_geom = segment_geometric_object_cardiac(
            img, seg_whole, verbose=verbose, plot=plot
        )
        seg_final = segment_other_components_cardiac(
            seg_whole, seg_geom, verbose=verbose
        )
    return seg_final


"""---------------------------------------------------------------------------------------------------------------------
Fetal cortical plate segmentations
---------------------------------------------------------------------------------------------------------------------"""


def parseSTA(img_idx, verbose=False):
    """Open an image from the STA dataset. n_image must be between 21 and 38 included."""
    # Open image.
    if img_idx <= 35:
        filename = f"{ROOT_DIRECTORY}/../data/sta/STA{img_idx}.nii.gz"
    else:
        filename = f"{ROOT_DIRECTORY}/../data/sta/STA{img_idx}exp.nii.gz"
    img = nib_load(filename).get_fdata()
    img /= np.max(img)
    # Open tissue.
    if img_idx <= 35:
        filename = f"{ROOT_DIRECTORY}/../data/sta/STA{img_idx}_tissue.nii.gz"
    else:
        filename = f"{ROOT_DIRECTORY}/../data/sta/STA{img_idx}exp_tissue.nii.gz"
    img_tissue = nib_load(filename).get_fdata()
    # Define segmentation of cortical plate.
    seg_gt = ((img_tissue == 112) + (img_tissue == 113)) * 1
    if verbose:
        print("Segmentation size:", np.sum(seg_gt))
    return img, seg_gt


def preprocess_fetal(
    img,
    sigma=0.5,
    radius_dilation=1,
    zero_boundary=0.3,
):
    if radius_dilation > 0:
        img = skimage.morphology.erosion(
            img,
            footprint=skimage.morphology.ball(radius_dilation),
            out=None,
        )
    if sigma > 0:
        img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    if zero_boundary > 0:
        img[np.where(img < zero_boundary)] = 1
    return img


def segment_boundary_slices(
    img,
    min_dist_bars=0.003,
    ratio_upper=3 / 4,
    ratio_lower=1 / 100,
    select_bar="birth",
    y_im=None,
    verbose=False,
    plot=False,
):
    seg_final = np.zeros(np.shape(img))

    # Get nonempty slices of im
    if y_im == None:
        nonempty_slices = np.where(np.sum(np.sum(1 - img, 0), 1))[0]
        ymin_im, ymax_im = nonempty_slices[0], nonempty_slices[-1]
        y_im = range(ymin_im, ymax_im + 1)
        if verbose:
            print("ymin ymax =", ymin_im, ymax_im)

    for y in y_im:
        "Select only one bar"

        # Define slice
        img_slice = img[:, y, :].copy()

        # Compute barcode
        if verbose:
            start_time = ChronometerStart(f"Slice {y} - Compute diagram... ")
        barcode = cripser.computePH(img_slice, maxdim=1)  # Compute diagram
        if verbose:
            ChronometerStop(start_time, method="s")

        # Sort barcode by longest bars
        homology_dim = 1
        H = [
            list(bar[1::]) for bar in barcode if bar[0] == homology_dim
        ]  # Allow infinite bars
        H = [
            bar for _, bar in sorted(zip([bar[1] - bar[0] for bar in H], H))[::-1]
        ]  # Sort list H by persistence

        # Discard bars
        H_CC, H_width_hole_interior = dict(), dict()
        for bar in H:
            # Get CC
            pos, t = np.array(bar[2:4]).astype(int), bar[0] + 0.0000001
            labels = skimage.measure.label((img_slice <= t) * 1, background=0)
            CC = (labels == labels[pos[0], pos[1]]) * 1

            # Compute width hole
            labels = skimage.measure.label(1 - CC, background=0)
            components = [(labels == i) * 1 for i in range(1, np.max(labels) + 1)]
            components_len = [np.sum(component) for component in components]
            width_hole_interior = np.size(img_slice) - max(components_len) - np.sum(CC)
            # width of the hole, without the boundary
            width_hole = np.size(img_slice) - max(components_len)
            # width of the hole, with the boundary

            # Discard (1): CC must not touch the boundary
            CC_filled = scipy.ndimage.binary_fill_holes(CC).astype(int)
            boundary = (
                scipy.ndimage.binary_dilation(CC_filled, iterations=1) - CC_filled
            )
            value_boundary = np.mean(img_slice[np.where(boundary)])
            if value_boundary >= 1:
                continue

            # Discard (2): the whole must not be too large or too small
            size_slice, size_CC = np.sum(img_slice < 1), np.sum(CC)
            if (
                size_CC > ratio_upper * size_slice
                or width_hole_interior < ratio_lower * size_slice
            ):
                continue

            # Save
            H_CC[tuple(bar)], H_width_hole_interior[tuple(bar)] = (
                CC,
                width_hole_interior,
            )

        if verbose:
            print("Saved/discarded bars:", len(H), "/", len(H_CC))

        # Plot diagram
        if plot and len(H) >= 1:
            fig = plt.figure(figsize=(9, 3))
            ax = fig.add_subplot(1, 3, 1)
            dim_max = int(max([bar[0] for bar in barcode]))
            dgms_plt = [
                np.array([np.clip(bar[1:3], 0, 1) for bar in barcode if bar[0] == i])
                for i in range(dim_max + 1)
            ]
            persim.plot_diagrams(
                dgms_plt,
                ax=ax,
            )
            ax.set_title("Persistence diagram of flair", fontsize=10)
            # ax.set_xlim([0, 1.2])
            # ax.set_ylim([0, 1.2])

            if len(H_CC) >= 1:
                ax = fig.add_subplot(1, 3, 2)
                dgms_plt = [
                    np.array([[0, 0]]),
                    np.array(
                        [
                            np.clip(bar[1:3], 0, 1)
                            for bar in barcode
                            if tuple(bar[1::]) in H_CC
                        ]
                    ),
                ]
                persim.plot_diagrams(
                    dgms_plt,
                    ax=ax,
                )
                ax.set_title("Relevant diagram", fontsize=10)
                # ax.set_xlim([0, 1.2])
                # ax.set_ylim([0, 1.2])

        # Get optimal bar
        if len(H_CC) >= 1:
            # Bar with max width
            if select_bar == "width":
                bar = max(H_width_hole_interior, key=H_width_hole_interior.get)

            # Bar with max pers
            elif select_bar == "pers":
                H_pers = {bar: (bar[1] - bar[0]) for bar in H_CC}
                bar = max(H_pers, key=H_pers.get)

                # Bar with min birth
            elif select_bar == "birth":
                H_birth = {bar: bar[0] for bar in H_CC}
                bar = min(H_birth, key=H_birth.get)

            CC = H_CC[bar]
            if plot:
                patch = plt.Circle((bar[0], bar[1]), 0.025, fill=False)
                ax.add_patch(patch)

            # Check if exists a similar bar
            if len(H_CC) >= 2:
                distance_bars = {
                    bar2: np.linalg.norm(np.array(bar[0:2]) - np.array(bar2[0:2]))
                    for bar2 in H_CC
                }
                del distance_bars[bar]
                bar_closest = min(distance_bars, key=distance_bars.get)
                dist = distance_bars[bar_closest]
                if verbose:
                    if dist > min_dist_bars:
                        print("No similar bar:", dist, ">", min_dist_bars)
                    elif dist <= min_dist_bars:
                        print("Exists a similar bar:", dist, "<=", min_dist_bars)
                if dist <= min_dist_bars:
                    # Get CC
                    pos, t = (
                        np.array(bar_closest[2:4]).astype(int),
                        bar_closest[0] + 0.0000001,
                    )
                    CC_closest = H_CC[bar_closest]
                    if plot:
                        patch = plt.Circle(
                            (bar_closest[0], bar_closest[1]), 0.025, fill=False
                        )
                        ax.add_patch(patch)

                    # Join CC
                    CC = (CC + CC_closest > 0) * 1

        else:
            CC = np.zeros(np.shape(img_slice))

        # Save CC
        seg_final[:, y, :] = CC

        # Plot
        if plot:
            ax = fig.add_subplot(1, 3, 3)
            ax.axis("off")
            ax.set_title(f"TDA seg - slice {y}", fontsize=10)
            ax.imshow(img_slice, cmap="gray", origin="lower")
            ax.imshow(CC, **DLT_KW_SEG)
            plt.show()

    return seg_final


def segment_fetal(
    img,
    sigma=0.5,
    radius_dilation=1,
    zero_boundary=0.3,
    min_dist_bars=0.003,
    ratio_upper=3 / 4,
    ratio_lower=1 / 100,
    select_bar="birth",
    verbose=False,
    plot=False,
):
    # Preprocess image.
    img = preprocess_fetal(
        img=img,
        sigma=sigma,
        radius_dilation=radius_dilation,
        zero_boundary=zero_boundary,
    )
    # Module 1: Segmentation whole object.
    seg_final = segment_boundary_slices(
        img=img,
        min_dist_bars=min_dist_bars,
        ratio_upper=ratio_upper,
        ratio_lower=ratio_lower,
        select_bar=select_bar,
        verbose=verbose,
        plot=plot,
    )
    return seg_final
