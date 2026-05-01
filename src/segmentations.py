"""---------------------------------------------------------------------------------------------------------------------

Train-Free Segmentation in MRI with Cubical Persistent Homology
Anton François & Raphaël Tinarrage
See the repo at https://github.com/antonfrancois/gliomaSegmentation_TDA and article at https://arxiv.org/abs/2401.01160

------------------------------------------------------------------------------------------------------------------------

Brain segmentations:
    preprocess_brain
    suggest_t
    threshold_triangle
    segment_whole_object
    seed_superlevel_join_value
    segment_geometric_object
    segment_other_components
    segment_brain

Timing:
    timer
    segment_brain_timed

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
import time
from collections import defaultdict
from contextlib import contextmanager

# Third-party imports.
import numpy as np
import scipy
import skimage
from skimage.morphology import max_tree
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
    DLT_KW_IMAGE,
    DLT_KW_SEG,
    get_dice,
    get_multiple_dice,
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


def suggest_t(img, thresh=0.1, nbins=1000, plot=False, save=False):
    freq, vals = np.histogram(img[img > thresh], bins=nbins)
    vals = (vals[1:] + vals[:-1]) / 2
    mean_freq = freq.sum() / len(freq)
    i = np.where(freq > mean_freq)[0][-1]
    t = vals[i]
    # Plot
    if plot:
        fig, ax = plt.subplots(figsize=(4, 4))
        width = vals[1] - vals[0]
        ax.bar(vals, freq, width=width, align="center", alpha=0.85)
        ax.bar(
            vals[i],
            freq[i],
            width=width,
            align="center",
            alpha=1.0,
            label=f"chosen t = {t:.3g}",
        )
        ax.axhline(mean_freq, linestyle="--", linewidth=1.5, label="mean bin count")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_title("Threshold selection via Module 1")
        ax.legend(frameon=False)
        if save:
            plt.savefig("brain_suggest_t.pdf", dpi=300, bbox_inches="tight")
    return t


def threshold_triangle(image, nbins=1000):
    """Simple wrapper with flip"""
    # nbins is ignored for integer arrays
    # so, we recalculate the effective nbins.
    hist, bin_centers = skimage.exposure.histogram(
        image.reshape(-1), nbins, source_range="image"
    )
    nbins = len(hist)

    # Find peak, lowest and highest gray levels.
    arg_peak_height = np.argmax(hist)
    peak_height = hist[arg_peak_height]
    arg_low_level, arg_high_level = np.flatnonzero(hist)[[0, -1]]

    if arg_low_level == arg_high_level:
        # Image has constant intensity.
        return image.ravel()[0]

    # MODIFICATION HERE: Flip always.
    hist = hist[::-1]
    arg_low_level = nbins - arg_high_level - 1
    arg_peak_height = nbins - arg_peak_height - 1

    # If flip == True, arg_high_level becomes incorrect
    # but we don't need it anymore.
    del arg_high_level

    # Set up the coordinate system.
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = hist[x1 + arg_low_level]

    # Normalize.
    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    # Maximize the length.
    # The ImageJ implementation includes an additional constant when calculating
    # the length, but here we omit it as it does not affect the location of the
    # minimum.
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    # MODIFICATION HERE: Flip always.
    arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]


def segment_whole_object(
    img,
    method="suggest_t",
    threshold=1,
    seg_gt=None,
    iterations_binary_closing=0,
    verbose=True,
    plot=True,
    save=False,
    min_volume_suggest_t=3000,
    # transform_cc=True,
    finetune=True,
):
    """Segments the whole tumor from a FLAIR image. The method 'suggest_t', 'gt' or 'gt_hull'."""
    # Segment via automatic threshold detection.
    if method == "suggest_t":
        if verbose:
            start_time = ChronometerStart("Suggest threshold... ")
        # Find the best threshold.
        t = suggest_t(
            img=img,
            # threshold=threshold,
            plot=plot,
            save=save,
        )
        # Fine-tune t.
        step = 0.01
        while get_largest_component(img, t, verbose=False).sum() < min_volume_suggest_t:
            t -= step
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
        if finetune:
            from utils import argmax_image
            from tests import get_best_component, suggest_t_sphericity

            # Extract the most spherical component.
            min_sphere = 0.5
            min_size_sphericity = 10000
            seg_whole = get_best_component(
                img, seg_whole, t, min_sphere=min_sphere, min_size=min_size_sphericity
            )
            pos = argmax_image(seg_whole * img)
            # Fine-tune.
            offset = 0.02
            min_size_sphericity = 10000
            vmin, vmax = t - offset, t + offset
            t_finetuned = suggest_t_sphericity(
                img, pos, vmin, vmax, min_size_sphericity, ticks=100, method="argmax"
            )
            seg_whole = get_component(img, pos, t_finetuned)
            seg_whole = scipy.ndimage.binary_fill_holes(seg_whole)

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


def seed_superlevel_join_value(image: np.ndarray, seed_zyx) -> np.ndarray:
    """
    For each voxel v, returns the highest threshold t such that v is in the
    seed's connected component within {image >= t} (6-connectivity).

    seed_zyx must match numpy indexing: image[z, y, x].
    Output values are in the same scale as image.
    """
    img = np.asarray(image)
    if img.ndim != 3:
        raise ValueError(f"image must be 3D, got {img.shape}")

    seed = tuple(int(x) for x in np.asarray(seed_zyx).ravel())
    if len(seed) != 3:
        raise ValueError("seed must have 3 coordinates")
    if any(seed[d] < 0 or seed[d] >= img.shape[d] for d in range(3)):
        raise ValueError(f"seed {seed} out of bounds for shape {img.shape}")

    # Build max-tree of superlevel connected components.
    # For 3D, connectivity=1 corresponds to 6-connectivity (face neighbors).
    parent_img, S = max_tree(
        img, connectivity=1
    )  # C-accelerated :contentReference[oaicite:1]{index=1}
    P = parent_img.ravel()
    img_r = img.ravel()
    n = img_r.size

    seed_idx = np.ravel_multi_index(seed, img.shape)

    # Mark all ancestors of the seed in the max-tree
    is_seed_anc = np.zeros(n, dtype=bool)
    cur = seed_idx
    while True:
        is_seed_anc[cur] = True
        nxt = P[cur]
        if nxt == cur:  # root (in skimage canonical max-tree, root is its own parent)
            break
        cur = nxt

    # For each node, store the *first* ancestor (towards root) that lies on seed's ancestor chain.
    # Because S is ordered parent-before-children, we can propagate in one pass. :contentReference[oaicite:2]{index=2}
    seedlink = np.empty(n, dtype=P.dtype)
    root = S[0]
    seedlink[root] = root

    for i in S[1:]:
        seedlink[i] = i if is_seed_anc[i] else seedlink[P[i]]

    out = img_r[seedlink].reshape(img.shape)
    return out


def segment_geometric_object(
    img, seg_whole, max_bars=5, verbose=True, plot=True, save=False
):
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
        plt.title("Persistence diagram of masked image", fontsize=10)
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
        plt.tight_layout()
    if save:
        plt.savefig("results/brain_persistence_module2.pdf")
    if plot:
        plt.show()
    return seg_geom


def segment_other_components(seg_whole, seg_geom, radius_dilation=0, verbose=True):
    # 3rd step: - Classify components
    # 1 - RED, TC  -> NECROSE INACTIVE, TUMORUS CORE
    # 2 - BLUE, ED -> INFILTRATION, OEDEME
    # 3 - ORANGE, ET -> NECROSE ACTIVE, ENHANCING TUMOR
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
    seg_final = seg_geom.copy() * 3  # define seg_final
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
            skimage.morphology.erosion((seg_final == 3) * 1, footprint=footprint)
            * seg_whole
        )
        # Create final segmentations from ED.
        seg_final = seg_whole.copy() * 2
        # Add TC.
        seg_final[seg_TC_dilate > 0] = 1
        # Add ET.
        seg_final[seg_ET_erode > 0] = 3
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
    save=False,
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
        img=img_flair, threshold=whole_threshold, verbose=verbose, plot=plot, save=save
    )
    # Module 2: Segmentation geometric object.
    seg_geom = segment_geometric_object(
        img=img_t1ce,
        seg_whole=seg_whole,
        max_bars=max_bars,
        verbose=verbose,
        plot=plot,
        save=save,
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
Timing
---------------------------------------------------------------------------------------------------------------------"""


@contextmanager
def timer(name, store):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name].append(time.perf_counter() - t0)


def segment_brain_timed(
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
    save=False,
    timings=None,  # pass a dict to collect timings across many cases
):
    if timings is None:
        timings = defaultdict(list)

    # Preprocess image.
    with timer("preprocess", timings):
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
    with timer("module1", timings):
        seg_whole = segment_whole_object(
            img=img_flair,
            threshold=whole_threshold,
            verbose=verbose,
            plot=plot,
            save=save,
        )

    # Module 2: Segmentation geometric object.
    with timer("module2", timings):
        seg_geom = segment_geometric_object(
            img=img_t1ce,
            seg_whole=seg_whole,
            max_bars=max_bars,
            verbose=verbose,
            plot=plot,
            save=save,
        )

    # Module 3: Deduce final segmentation.
    with timer("module3", timings):
        seg_final = segment_other_components(
            seg_whole=seg_whole,
            seg_geom=seg_geom,
            radius_dilation=radius_dilation,
            verbose=verbose,
        )

    return seg_final, timings


"""---------------------------------------------------------------------------------------------------------------------
Myocardium segmentations
---------------------------------------------------------------------------------------------------------------------"""


def parseACDC(n_image, end="ED", return_filename=False):
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
    if return_filename:
        return img, seg_gt, filename
    else:
        return img, seg_gt


def compute_sphericity(CC, verbose=True):
    """CC can be of dimension 2 or 3."""
    if CC.ndim == 2:
        # Return 0 if it touches the boundary
        if (
            np.sum(CC[0, :]) > 0
            or np.sum(CC[-1, :]) > 0
            or np.sum(CC[:, 0]) > 0
            or np.sum(CC[:, -1])
        ):
            if verbose:
                print("Sphericity Dice: 0 (touches boundary)")
            return 0
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
            print(t, "Warning! The voxel pos is not active at time t.")
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
                print("Warning! The label is background.")

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


def preprocess_cardiac(img, sigma=1, radius_dilation=0, method="3D"):
    if method == "2D":
        for z in range(img.shape[2]):
            if sigma > 0:
                img[:, :, z] = scipy.ndimage.gaussian_filter(img[:, :, z], sigma=sigma)
            if radius_dilation > 0:
                img[:, :, z] = skimage.morphology.dilation(
                    img[:, :, z],
                    footprint=skimage.morphology.disk(radius_dilation),
                )
    if method == "3D":
        if sigma > 0:
            img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
        if radius_dilation > 0:
            img = skimage.morphology.dilation(
                img,
                footprint=skimage.morphology.ball(radius_dilation),
            )
    return img


def segment_whole_object_cardiac(
    img,
    seg_gt,
    H0_features_max_LV=10,
    H0_features_max_RV=20,
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
    # (1) Compute PH0
    barcode = cripser.computePH(1 - img, maxdim=0)
    H0 = [list(bar[1::]) for bar in barcode if bar[0] == 0]
    # Sort list H0 by persistence
    H0 = [bar for _, bar in sorted(zip([bar[1] - bar[0] for bar in H0], H0))[::-1]]
    # Correction infinite bar (1 instead of inf)
    H0[0][1] = 1

    # (2) Compute sphericity
    H0_sphericity = dict()
    H0_CC = dict()
    for i in range(H0_features_max_RV):
        t_birth, pos = H0[i][0], [int(H0[i][2 + k]) for k in range(img.ndim)]
        # Segment via suggest_t_pos
        CC = suggest_t_pos(
            img, pos, t_birth, dt_threshold=dt_threshold, direction="left"
        )
        H0_CC[i] = CC
        # Compute sphericity
        if i < H0_features_max_LV:
            H0_sphericity[i] = compute_sphericity(CC, verbose=False)

    if verbose:
        print("H0 sphericity:", H0_sphericity)

    # (3) Segment LV
    H0_sphericity_thresh = {
        i: H0_sphericity[i] * (np.sum(H0_CC[i]) >= thresh_small_LV)
        for i in H0_sphericity
    }
    # Identify LV as the most spherical
    best_i_LV = max(H0_sphericity_thresh, key=lambda k: H0_sphericity_thresh[k])
    seg_LV = H0_CC[best_i_LV]
    # Do dilation (preprocess)
    if radius_dilation > 0:
        if img.ndim == 2:
            footprint = skimage.morphology.disk(radius_dilation)
        elif img.ndim == 3:
            footprint = skimage.morphology.ball(radius_dilation)
        seg_LV = skimage.morphology.binary_erosion(seg_LV, footprint=footprint)
    if verbose:
        print("Dice LV:", get_dice(seg_LV, (seg_gt == 3) * 1))

    # (4) Segment RV
    # Compute distances to LV
    if np.isinf(maximal_distance):
        top_features_loc = [
            np.mean(np.where(H0_CC[i]), axis=1) for i in range(H0_features_max_RV)
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
        distances = [np.inf for _ in range(H0_features_max_RV)]
        for i in range(H0_features_max_RV):
            for dilation in range(0, maximal_distance + 1):
                if np.max(H0_CC[i][np.where(seg_LV_dilations[dilation] > 0)]) > 0:
                    distances[i] = dilation
                    break
    # Discard small or big components
    for i in range(H0_features_max_RV):
        if np.sum(H0_CC[i]) < ratio_small_RV * np.sum(seg_LV) or np.sum(
            H0_CC[i]
        ) > ratio_big_RV * np.sum(seg_LV):
            distances[i] = np.inf
    # Identify RV via the second-smallest distance (the first one is 0)
    best_i_RV = np.argsort(distances)[1]
    seg_RV = H0_CC[best_i_RV]
    # If they intersect, remove the intersection from LV
    if min(seg_LV[seg_RV > 0]) > 0:
        seg_LV[seg_RV > 0] = 0
        if verbose:
            print("Warning! LV and RV intersect.")
    # Undo dilation (preprocess)
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
    if np.min(distances) == np.inf:
        print("Error! Min distance is inf.")
    if verbose:
        dice_LV = get_dice(seg_LV, (seg_gt == 3) * 1)
        dice_RV = get_dice(seg_RV, (seg_gt == 1) * 1)
        print("Dice LV/RV:", dice_LV, dice_RV)

    # (5) Segment whole object: dilate LV up to touch RV
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
    # Fill CC with holes
    seg_whole = scipy.ndimage.binary_fill_holes(seg_whole)

    # Plot diagram and top features
    if plot:
        fig = plt.figure(figsize=(6 / 2, 6))
        fig.subplots_adjust(wspace=0.5, hspace=0.15)
        # Plot diagram
        eps = 0.01
        ax = fig.add_subplot(2, 1, 1)
        persim.plot_diagrams(
            [
                np.array([np.clip(bar[1:3], 0, 1) for bar in barcode if bar[0] == i])
                for i in range(1)
            ]
        )
        plot_features = [(H0[i][0], H0[i][1]) for i in range(H0_features_max_RV)]
        for i, (x, y) in enumerate(plot_features):
            ax.scatter(x, y, c="C" + repr(i + 1))
            ax.text(x + eps, y + eps, i, color="black")
        plt.title("Persistence diagram", fontsize=10)

        # Plot connected components
        ax = fig.add_subplot(2, 1, 2)
        ax.axis("off")
        if img.ndim == 2:
            ax.imshow(img, **DLT_KW_IMAGE, alpha=0.9)
        elif img.ndim == 3:
            ax.imshow(img[:, :, 4], **DLT_KW_IMAGE)  # arbitrary index
        for i in range(H0_features_max_RV):
            CC = H0_CC[i]
            if img.ndim == 3:
                CC = CC[:, :, 4]  # arbitrary index
            if np.sum(CC) > 0:
                ax.imshow(
                    np.ma.masked_where(CC == 0, CC),
                    alpha=0.9,
                    cmap=ListedColormap(["C" + repr(i + 1)]),
                    origin="lower",
                )
                center = scipy.ndimage.center_of_mass(CC)
                ax.text(int(center[1]), int(center[0]), i, color="white")
        # plt.tight_layout()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if save:
            plt.savefig(
                "results/cardiac_module_1_local.pdf", format="pdf", bbox_inches="tight"
            )
        plt.show()

    # Plot segmentation
    if plot:
        if img.ndim == 2:
            slices_to_plot = [(img, seg_gt, seg_LV, seg_RV, seg_whole)]
        elif img.ndim == 3:
            slices_to_plot = [
                (
                    img[:, :, z],
                    seg_gt[:, :, z],
                    seg_LV[:, :, z],
                    seg_RV[:, :, z],
                    seg_whole[:, :, z],
                )
                for z in range(np.shape(img)[2])
            ]
        for (
            img_slice,
            seg_medecin_slice,
            seg_LV_slice,
            seg_RV_slice,
            seg_union_slice,
        ) in slices_to_plot:
            fig = plt.figure(figsize=(6, 3))
            fig.subplots_adjust(wspace=0.05, hspace=0)

            ax = fig.add_subplot(1, 2, 1)
            ax.axis("off")
            ax.imshow(img_slice, **DLT_KW_IMAGE, alpha=0.9)
            ax.imshow(3 * seg_LV_slice + 1 * seg_RV_slice, **DLT_KW_SEG)
            ax.set_title("Segmentation of LV and RV")

            ax = fig.add_subplot(1, 2, 2)
            ax.axis("off")
            ax.imshow(img_slice, **DLT_KW_IMAGE, alpha=0.9)
            ax.imshow(
                seg_union_slice,
                cmap=ListedColormap([[0, 0, 0, 0], "tab:pink"]),
                origin="lower",
            )
            ax.set_title("Segmentation of whole object")

            plt.tight_layout()
            if save:
                plt.savefig(
                    "results/cardiac_whole_segmentation.pdf",
                    format="pdf",
                    bbox_inches="tight",
                )
            plt.show()
    return seg_whole, seg_LV, seg_RV


def segment_geometric_object_cardiac(img, seg_whole, verbose=False, plot=False):
    img_union = img * seg_whole  # define image restricted to seg_union
    img_union[img_union == 0] = (
        1  # set boundary values to 1 (we will consider sublevel sets)
    )

    # Artificially add boundaries (to create a sphere from the cylinder)
    if img.ndim == 3:
        #        img_union[:, :, 0], img_union[:, :, -1] = 0, 0
        img_union_extended = np.zeros(
            (np.shape(img)[0], np.shape(img)[1], np.shape(img)[2] + 2)
        )
        img_union_extended[:, :, 1:-1] = img_union
        img_union = img_union_extended

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

        # Remove artificial boundaries
        if img.ndim == 3:
            seg_geom = seg_geom[:, :, 1:-1]

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
    "1: RV, 2: Myocardium, 3: LV."
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
    seg_final[(components[imax_comp] * seg_whole) > 0] = 1  # add RV
    components.pop(imax_comp)  # remove background
    for component in components:
        seg_final[component > 0] = 3  # add LV
    return seg_final


def segment_cardiac(
    img,
    seg_gt,
    method_whole="2D",
    method_geom="2D",
    method_other="2D",
    sigma=1,
    radius_dilation=0,
    dt_threshold=1.0,
    H0_features_max_LV=10,
    H0_features_max_RV=20,
    thresh_small_LV=200,
    ratio_small_RV=0.1,
    ratio_big_RV=5,
    add_to_iterations=2,
    maximal_distance=np.inf,
    verbose=False,
    plot=False,
):
    """The parameter `method` can be '2D' or '3D'."""
    # (1) Preprocess.
    img = preprocess_cardiac(
        img=img, sigma=sigma, radius_dilation=radius_dilation, method="3D"
    )

    # (2) Segment whole object
    if method_whole == "2D":
        seg_whole = np.zeros(np.shape(img))
        seg_LV = np.zeros(np.shape(img))
        seg_RV = np.zeros(np.shape(img))
        for z_pos in range(0, np.shape(img)[2]):
            img_slice, seg_gt_slice = img[:, :, z_pos], seg_gt[:, :, z_pos]
            seg_whole_slice, seg_LV_slice, seg_RV_slice = segment_whole_object_cardiac(
                img=img_slice,
                seg_gt=seg_gt_slice,
                H0_features_max_LV=H0_features_max_LV,
                H0_features_max_RV=H0_features_max_RV,
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
            seg_whole[:, :, z_pos] = seg_whole_slice
            seg_LV[:, :, z_pos] = seg_LV_slice
            seg_RV[:, :, z_pos] = seg_RV_slice
    elif method_whole == "3D":
        seg_whole, seg_LV, seg_RV = segment_whole_object_cardiac(
            img=img,
            seg_gt=seg_gt,
            H0_features_max_LV=H0_features_max_LV,
            H0_features_max_RV=H0_features_max_RV,
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

    # (3) Segment geometric object
    if method_geom == "2D":
        seg_geom = np.zeros(np.shape(img))
        for z_pos in range(0, np.shape(img)[2]):
            img_slice, seg_whole_slice = img[:, :, z_pos], seg_whole[:, :, z_pos]
            seg_geom_slice = segment_geometric_object_cardiac(
                img_slice, seg_whole_slice, verbose=verbose, plot=plot
            )
            seg_geom[:, :, z_pos] = seg_geom_slice
    elif method_geom == "3D":
        seg_geom = segment_geometric_object_cardiac(
            img, seg_whole, verbose=verbose, plot=plot
        )

    # (4) Deduce final segmentation
    if method_other == "2D":
        seg_final = np.zeros(np.shape(img))
        for z_pos in range(0, np.shape(img)[2]):
            seg_whole_slice, seg_geom_slice = (
                seg_whole[:, :, z_pos],
                seg_geom[:, :, z_pos],
            )
            seg_final_slice = segment_other_components_cardiac(
                seg_whole_slice, seg_geom_slice, verbose=verbose
            )
            seg_final[:, :, z_pos] = seg_final_slice
    elif method_other == "3D":
        seg_final = segment_other_components_cardiac(
            seg_whole, seg_geom, verbose=verbose
        )
    elif method_other == "skip":
        #  Directly combine modules 1 and 2 without module 3
        seg_final = seg_geom.copy() * 2  # initialize with myocardium
        seg_final[seg_LV > 0] = 3  # add LV
        seg_final[seg_RV > 0] = 1  # add RV
    if verbose:
        get_multiple_dice(
            seg_final[:, :, 1:], seg_gt[:, :, 1:], labels=(1, 2, 3), verbose=True
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

            # Compute width hole (without the boundary)
            labels = skimage.measure.label(1 - CC, background=0)
            components = [(labels == i) * 1 for i in range(1, np.max(labels) + 1)]
            components_len = [np.sum(component) for component in components]
            max_components_len = max(components_len) if len(components_len) > 0 else 0
            width_hole_interior = np.size(img_slice) - max_components_len - np.sum(CC)

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
