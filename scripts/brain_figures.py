# %% Imports

import __init__

import src.parseBrats as pB
from src.segmentations import segment_brain
from src.utils import (
    get_multiple_dice,
    plot_comparison_full_segmentations,
)

# %% Open image

pb = pB.parse_brats(
    brats_list=None,
    brats_folder="2025",
    modality="flair",
    get_template=False,
)

n_image = [n for n, name in enumerate(pb.brats_list) if name == "BraTS-GLI-00183-000"][
    0
]
img_flair, seg_gt = pb(n_image, to_torch=False, modality="flair", normalize=True)
img_t1ce, _ = pb(n_image, to_torch=False, modality="t1ce")

# %% Run the segmentation

normalize = "max"  # preprocess_brain, divide by max or 255
sigma = 1  # preprocess_brain, Gaussian blur
enhance = (False, True)  # preprocess_brain, apply enhancement or not
radius_enhance = 1  # preprocess_brain, radius of enhancement
dilate = True  # preprocess_brain, apply dilation or not on T1ce
radius_dilation = 2  # preprocess_brain and segment_other_components, radius of dilation
whole_threshold = 1  # segment_whole_object, threshold for suggest_t
max_bars = 1  # segment_geometric_object, number of H2 features to consider

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
    verbose=False,
    plot=True,
    save=True,
)

# Comparison binary segmentations WT
plot_comparison_full_segmentations(
    pb.brats_list[n_image],
    img_flair,
    (seg_gt > 0) * 1,
    (seg > 0) * 1,
    show_ground_truth=False,
    save_path="results/brain_seg_module1.pdf",
)
# Print Dice score
dice_wt = get_multiple_dice((seg > 0) * 1, (seg_gt > 0) * 1, labels=(1,), verbose=True)

# Comparison binary segmentations ET
plot_comparison_full_segmentations(
    pb.brats_list[n_image],
    img_flair,
    (seg_gt == 3) * 1,
    (seg == 3) * 1,
    show_ground_truth=False,
    save_path="results/brain_seg_module2.pdf",
)
# Print Dice score
dice_et = get_multiple_dice(
    (seg == 3) * 1, (seg_gt == 3) * 1, labels=(1,), verbose=True
)

# Comparison final segmentations
plot_comparison_full_segmentations(
    pb.brats_list[n_image],
    img_flair,
    seg_gt,
    seg,
    show_ground_truth=False,
    save_path="results/brain_seg_module3.pdf",
)
# Print Dice score
dice = get_multiple_dice(seg, seg_gt, labels=(1, 2, 3), verbose=True)

# %% Illustrations with valid model

for brats_index in ["01401", "00780", "01368"]:
    n_image = [
        n
        for n, name in enumerate(pb.brats_list)
        if name == f"BraTS-GLI-{brats_index}-000"
    ][0]
    img_flair, seg_gt = pb(n_image, to_torch=False, modality="flair", normalize=True)
    img_t1ce, _ = pb(n_image, to_torch=False, modality="t1ce")
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
        verbose=False,
        plot=False,
    )
    # Comparison final segmentations
    plot_comparison_full_segmentations(
        pb.brats_list[n_image],
        img_flair,
        seg_gt,
        seg,
        show_ground_truth=True,
        save_path=f"results/brain_seg_BraTS-GLI-{brats_index}-000.pdf",
    )
    # Print Dice score
    dice = get_multiple_dice(seg, seg_gt, labels=(1, 2, 3), verbose=True)

# %% Illustrations with invalid model

# for brats_index in ["00491"]:
for brats_index in ["00606", "00017", "00491"]:
    n_image = [
        n
        for n, name in enumerate(pb.brats_list)
        if name == f"BraTS-GLI-{brats_index}-000"
    ][0]
    img_flair, seg_gt = pb(n_image, to_torch=False, modality="flair", normalize=True)
    img_t1ce, _ = pb(n_image, to_torch=False, modality="t1ce")
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
        verbose=False,
        plot=False,
    )
    # Comparison final segmentations
    plot_comparison_full_segmentations(
        pb.brats_list[n_image],
        img_flair,
        seg_gt,
        seg,
        show_ground_truth=True,
        save_path=f"results/brain_seg_BraTS-GLI-{brats_index}-000.pdf",
    )
    # Print Dice score
    dice = get_multiple_dice(seg, seg_gt, labels=(1, 2, 3), verbose=True)
