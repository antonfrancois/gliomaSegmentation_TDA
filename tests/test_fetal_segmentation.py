# %% Imports

from segmentations import parseSTA, segment_fetal
from utils import get_dice

# %% Segmentation SAT - one image - all slices

img_idx = 22  # between 21 and 38 included
verbose = True
plot = True

# Parameters
normalize = "max"  # preprocess_fetal, divide by max or 255
sigma = 0.5  # preprocess_fetal, Gaussian blur
radius_dilation = 1  # Preprocess, dilation parameter
zero_boundary = 0.3  # Preprocess, to suppress the boundary
min_dist_bars = 0.03  # segment_boundary_slices, to consider multiple bars
ratio_upper = 3 / 4  # segment_boundary_slices, max relative size of object (circle)
ratio_lower = 1 / 50  # segment_boundary_slices, min relative size of interior
select_bar = "birth"  # segment_boundary_slices, optimal bar according to 'width' (of hole), 'pers' (of bar) or 'birth'


# %%  Open image
img, seg_gt = parseSTA(img_idx)

# %%  Segment

seg_final = segment_fetal(
    img=img,
    sigma=sigma,
    radius_dilation=radius_dilation,
    zero_boundary=zero_boundary,
    min_dist_bars=min_dist_bars,
    ratio_upper=ratio_upper,
    ratio_lower=ratio_lower,
    select_bar=select_bar,
    verbose=verbose,
    plot=plot,
)


# %%  Compute dice

dice = get_dice(seg_final, seg_gt, verbose=False)
print("Final Dice score ", round(dice, 3))
