# %% Open image

import parseBrats as pB

# Open Brats folder
brats_list = [
    "BraTS2021_01360",
    "BraTS2021_01053",
    "BraTS2021_01143",
    "BraTS2021_01121",
    "BraTS2021_00318",
    "BraTS2021_01296",
    "BraTS2021_00636",
    "BraTS2021_00557",
    "BraTS2021_01245",
    "BraTS2021_00113",
]
pb = pB.parse_brats(
    brats_list=brats_list, brats_folder="2021", modality="flair", get_template=False
)
print(f"BratS list is {len(pb.brats_list)} long")

# Open a specific image
img_idx = 0
img_flair, seg_gt = pb(img_idx, to_torch=False, modality="flair", normalize=True)
img_t1ce, _ = pb(img_idx, to_torch=False, modality="t1ce")
print("Image name:", pb.brats_list[img_idx])


# %% Plot image

from utils import plot_images_and_segmentation

plot_images_and_segmentation(
    image_name=pb.brats_list[img_idx],
    img_flair=img_flair,
    img_t1ce=img_t1ce,
    seg_gt=seg_gt,
)


# %% Preprocess image

from segmentations import preprocess_brain

normalize = "max"  # preprocess_brain, divide by max or 255
sigma = 1  # preprocess_brain, Gaussian blur
enhance = (False, False)  # preprocess_brain, apply enhancement or not
dilate = (False, True)  # preprocess_brain, apply dilation or not
radius_dilation = 1  # preprocess_brain and segment_other_components, radius of dilation

img_flair, img_t1ce = preprocess_brain(
    img_flair,
    img_t1ce,
    sigma=sigma,
    normalize=normalize,
    enhance=enhance,
    dilate=dilate,
    radius_dilation=radius_dilation,
)

# %% Module 1: Segmentation whole object

from segmentations import segment_whole_object

whole_threshold = 1  # segment_whole_object, threshold for suggest_t

seg_whole = segment_whole_object(
    img=img_flair,
    threshold=whole_threshold,
    verbose=True,
    plot=True,
)


# %% Module 2: Segmentation geometric object

from segmentations import segment_geometric_object

max_bars = 2  # segment_geometric_object, number of H2 features to consider

seg_geom = segment_geometric_object(
    img=img_t1ce, seg_whole=seg_whole, max_bars=max_bars, verbose=True, plot=True
)


# %% Module 3: Deduce final segmentation

from segmentations import segment_other_components

seg_final = segment_other_components(
    seg_whole=seg_whole,
    seg_geom=seg_geom,
    radius_dilation=radius_dilation,
    verbose=True,
)

# %% Compute Dice scores

from utils import get_multiple_dice

get_multiple_dice(seg_final, seg_gt, verbose=True)


# %% Plot final segmentation

from utils import plot_segmentation

plot_segmentation(name=pb.brats_list[img_idx], img=img_flair, seg=seg_final)

# %% Compare with ground-truth segmentation
#
# from utils import segCmp
#
# cmp_seg, legend_patches = segCmp(seg_gt, seg_final, [1, 2, 4], [1, 2, 4])
# ax.imshow(cmp_seg.transpose((1, 0, 2)), origin="lower")
# ax.legend(handles=legend_patches, loc="lower right", frameon=True)
# ax.set_title(f"GT vs Prediction")


# %% Do it all again: Full segmentation

import parseBrats as pB
from segmentations import segment_brain
from utils import get_multiple_dice, plot_images_and_segmentation, plot_segmentation

# Hyperparameters.
img_idx = 15  # index of the image to process
verbose = False  # whether to print messages
plot = False  # whether to plot intermediate results

# Parameters for the algorithm.
normalize = "max"  # preprocess_brain, divide by max or 255
sigma = 1  # preprocess_brain, Gaussian blur
enhance = (False, False)  # preprocess_brain, apply enhancement or not
radius_enhance = 1  # preprocess_brain, radius of enhancement
dilate = True  # preprocess_brain, apply dilation or not on T1ce
radius_dilation = 1  # preprocess_brain and segment_other_components, radius of dilation
whole_threshold = 1  # segment_whole_object, threshold for suggest_t
max_bars = 2  # segment_geometric_object, number of H2 features to consider

# Open an image.
pb = pB.parse_brats(
    brats_list=None, brats_folder="2021", modality="flair", get_template=False
)
img_flair, seg_gt = pb(img_idx, to_torch=False, modality="flair", normalize=True)
img_t1ce, _ = pb(img_idx, to_torch=False, modality="t1ce")

# Plot the image.
plot_images_and_segmentation(
    image_name=pb.brats_list[img_idx],
    img_flair=img_flair,
    img_t1ce=img_t1ce,
    seg_gt=seg_gt,
)

# Segmentation.
seg_final = segment_brain(
    img_flair=img_flair,
    img_t1ce=img_t1ce,
    normalize=normalize,
    sigma=sigma,
    enhance=enhance,
    radius_enhance=radius_enhance,
    dilate=dilate,
    radius_dilation=radius_dilation,
    whole_threshold=whole_threshold,
    max_bars=max_bars,
    verbose=verbose,
    plot=plot,
)

# Compute Dice scores
get_multiple_dice(seg_final, seg_gt, verbose=True)

# Plot final segmentation
plot_segmentation(name=pb.brats_list[img_idx], img=img_flair, seg=seg_final)
