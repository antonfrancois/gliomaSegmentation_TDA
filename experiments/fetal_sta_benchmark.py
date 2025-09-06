# %% Imports

import pandas as pd
from segmentations import parseSTA, segment_fetal
from utils import get_dice, ChronometerStart, ChronometerTick

# %% Define parameters

sigma = 0.5  # Preprocess, Gaussian blur
radius_dilation = 1  # Preprocess, dilation parameter
zero_boundary = 0.3  # Preprocess, to suppress the boundary
min_dist_bars = 0.03  # Step 2, to consider multiple bars
ratio_upper = 3 / 4  # Step 2, max relative size of object (circle)
ratio_lower = 1 / 50  # Step 2, min relative size of interior
select_bar = "birth"  # Step 2, optimal bar according to 'width' (of hole), 'pers' (of bar) or 'birth'
parameters = [
    sigma,
    radius_dilation,
    zero_boundary,
    min_dist_bars,
    ratio_upper,
    ratio_lower,
    select_bar,
]

# %% Run the segmentations

DICEs = dict()

msg = "Compute DICEs... "
start_time = ChronometerStart(msg)
for i in range(21, 38 + 1):
    # Open image
    img, seg_gt = parseSTA(i)

    # Segmentation
    seg_final = segment_fetal(
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

    # Save scores
    score = get_dice(seg_final, seg_gt, verbose=False)
    name = f"SAT_{i}"
    DICEs[name] = {1: score}
    DICEs[name]["sat_name"] = name
    ChronometerTick(start_time, i - 21, 38 - 21, msg)

# %% Save (the code above can be interrupted at any time)

df = pd.DataFrame.from_dict(DICEs, orient="index")
df = df.rename(columns={1: "CP"}, errors="raise")
file_name = f"results/fetal_sat_TDAseg_scores_{str(parameters)}_len{len(df)}.csv"
df.to_csv(file_name, index=False)

# %% Print mean score

print(f"Mean DICE CP: {df['CP'].mean():0.4f}")
