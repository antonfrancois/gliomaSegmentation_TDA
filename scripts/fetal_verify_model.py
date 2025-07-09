# %% Imports

import numpy as np
import skimage
from segmentations import parseSTA, preprocess_fetal
from utils import ChronometerStart, ChronometerTick

# %% Define poarameters

verbose, plot = False, False

# Parameters
sigma = 0.5  # Preprocess, Gaussian blur
radius_dilation = 1  # Preprocess, dilation parameter
zero_boundary = 0.3  # Preprocess, to suppress the boundary
min_dist_bars = 0.03  # Step 2, to consider multiple bars
ratio_upper = 3 / 4  # Step 2, max relative size of object (circle)
ratio_lower = 1 / 50  # Step 2, min relative size of interior
select_bar = "birth"  # Step 2, optimal bar according to 'width' (of hole), 'pers' (of bar) or 'birth'

# %% Verify model - homology

mean_bool_CC = dict()
max_number_CC = dict()

msg = "Verify model... "
start_time = ChronometerStart(msg)
for n_image in range(21, 38 + 1):
    # Open image.
    img, seg_gt = parseSTA(n_image)
    # Preprocess image.
    img = preprocess_fetal(
        img=img,
        sigma=sigma,
        radius_dilation=radius_dilation,
        zero_boundary=zero_boundary,
    )
    # Verify model slice by slice.
    nonempty_slices = np.where(np.sum(np.sum(seg_gt, 0), 1))[0]
    ymin_im, ymax_im = nonempty_slices[0], nonempty_slices[-1]
    y_im = range(ymin_im, ymax_im + 1)
    if verbose:
        print("ymin ymax =", ymin_im, ymax_im)
    components_number_byslice = dict()
    for y in y_im:
        # Define slice
        img_slice = img[:, y, :].copy()
        seg_medecin_slice = seg_gt[:, y, :].copy()
        if radius_dilation > 0:
            seg_medecin_slice = skimage.morphology.binary_dilation(
                seg_medecin_slice, footprint=skimage.morphology.disk(radius_dilation)
            )
        # Extract CC.
        seg_complement = 1 - seg_medecin_slice
        labels = skimage.measure.label(seg_complement, background=0)
        components = [
            (labels == i) * 1 for i in range(1, np.max(labels) + 1)
        ]  # remove cortical plate
        ratio_big_component = 0.01
        components_number_byslice[y] = len(
            [
                component
                for component in components
                if np.sum(component)
                > np.prod(np.shape(img_slice)) * ratio_big_component
            ]
        )

    mean_bool_CC[n_image] = np.mean(
        [components_number_byslice[y] in [2, 3] for y in components_number_byslice]
    )
    max_number_CC[n_image] = max(components_number_byslice.values())

    ChronometerTick(start_time, n_image - 1, 38, msg)

# %% Print results

print(
    f"Per image, mean number of slices satisfying (H2'): {round(np.mean(list(mean_bool_CC.values())) * 100, 2)}%"
)

print("Max number of CC in all slice:", max(max_number_CC.values()))
