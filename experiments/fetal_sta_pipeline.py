# %% Imports

from segmentations import parseSTA, segment_fetal
from utils import get_dice

# %% Open image

img_idx = 30  # in range(21, 38 + 1)
img, seg_gt = parseSTA(img_idx)

# %% Define parameters

sigma = 0.5  # Preprocess, Gaussian blur
radius_dilation = 0  # Preprocess, dilation parameter
zero_boundary = 0.3  # Preprocess, to suppress the boundary
min_dist_bars = 0.1  # Step 2, to consider multiple bars
ratio_upper = 1 / 2  # Step 2, max relative size of object (circle)
ratio_lower = 1 / 50  # Step 2, min relative size of interior
select_bar = "width"  # Step 2, optimal bar according to 'width' (of hole), 'pers' (of bar) or 'birth'

# %% Run the segmentation

seg = segment_fetal(
    img=img,
    sigma=sigma,
    radius_dilation=radius_dilation,
    zero_boundary=zero_boundary,
    min_dist_bars=min_dist_bars,
    ratio_upper=ratio_upper,
    ratio_lower=ratio_lower,
    select_bar=select_bar,
    verbose=False,
    plot=False,
)

# %% Plot scores overall

get_dice(seg, seg_gt, verbose=True)
