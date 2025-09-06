# %% Imports

import parseBrats as pB
from segmentations import segment_brain
from utils import (
    get_multiple_dice,
    plot_comparison_full_segmentations,
    plot_comparison_binary_segmentations,
    plot_segmentation,
)

# %% Open image

pb = pB.parse_brats(
    brats_list=None,
    brats_folder="2025",
    modality="flair",
    get_template=False,
)
print("Number of images in the collection:", len(pb.brats_list))

n_image = 3
img_flair, seg_gt = pb(n_image, to_torch=False, modality="flair", normalize=True)
img_t1ce, _ = pb(n_image, to_torch=False, modality="t1ce")

# %% Define parameters

normalize = "max"  # preprocess_brain, divide by max or 255
sigma = 1  # preprocess_brain, Gaussian blur
enhance = (False, True)  # preprocess_brain, apply enhancement or not
radius_enhance = 1  # preprocess_brain, radius of enhancement
dilate = True  # preprocess_brain, apply dilation or not on T1ce
radius_dilation = 2  # preprocess_brain and segment_other_components, radius of dilation
whole_threshold = 1  # segment_whole_object, threshold for suggest_t
max_bars = 1  # segment_geometric_object, number of H2 features to consider

# %% Run the segmentation

seg = segment_brain(
    img_flair=img_flair,
    img_t1ce=img_t1ce,
    normalize=normalize,
    sigma=sigma,
    enhance=tuple(enhance),
    radius_enhance=radius_enhance,
    dilate=dilate,
    radius_dilation=radius_dilation,
    whole_threshold=whole_threshold,
    max_bars=max_bars,
    verbose=True,
    plot=True,
)

# %% Plot scores overall

dice = get_multiple_dice(seg, seg_gt, labels=(1, 2, 3), verbose=False)
print("Overall scores:", dice)

# %% Plot our segmentation

plot_comparison_full_segmentations(
    name=pb.brats_list[n_image], img=img_flair, seg_gt=seg_gt, seg_est=seg
)

# %% Plot ground-truth segmentation

plot_segmentation(name=pb.brats_list[n_image], img=img_flair, seg=seg_gt)

# %% Plot our segmentation label by label

# WT
seg_binary = (seg > 0) * 1
seg_gt_binary = (seg_gt > 0) * 1
plot_segmentation(name="Our segmentation", img=img_flair, seg=seg_gt_binary)
plot_comparison_binary_segmentations(
    name=pb.brats_list[n_image],
    img=img_flair,
    seg_binary_gt=seg_gt_binary,
    seg_binary_est=seg_binary,
)

# TC
seg_binary = (seg == 1) * 1
seg_gt_binary = (seg_gt == 1) * 1
plot_segmentation(name="Our segmentation", img=img_flair, seg=seg_gt_binary * 1)
plot_comparison_binary_segmentations(
    name=pb.brats_list[n_image],
    img=img_flair,
    seg_binary_gt=seg_gt_binary,
    seg_binary_est=seg_binary,
)

# ED
seg_binary = (seg == 2) * 1
seg_gt_binary = (seg_gt == 2) * 1
plot_segmentation(name="Our segmentation", img=img_flair, seg=seg_gt_binary * 2)
plot_comparison_binary_segmentations(
    name=pb.brats_list[n_image],
    img=img_flair,
    seg_binary_gt=seg_gt_binary,
    seg_binary_est=seg_binary,
)

# ET
seg_binary = (seg == 3) * 1
seg_gt_binary = (seg_gt == 3) * 1
plot_segmentation(name="Our segmentation", img=img_flair, seg=seg_gt_binary * 3)
plot_comparison_binary_segmentations(
    name=pb.brats_list[n_image],
    img=img_flair,
    seg_binary_gt=seg_gt_binary,
    seg_binary_est=seg_binary,
)
