# %% Imports

import numpy as np
import matplotlib.pyplot as plt

from segmentations import parseSTA, segment_fetal
from utils import ChronometerStart, ChronometerTick, get_dice

# %% Define parameters

# Parameters
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

# %% Batch - one set of parameters

dices_batch = dict()
msg = "Compute scores... "
start_time = ChronometerStart(msg)
for img_idx in range(21, 38 + 1):

    # Open image.
    img, seg_gt = parseSTA(img_idx)
    # Segmentation.
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
    # Save dice
    dices_batch[img_idx] = get_dice(seg_final, seg_gt, verbose=False)
    ChronometerTick(start_time, img_idx - 21, 38 - 21 + 1, msg)

# %%  Mean Dice score

print(
    f"Mean dice score: {np.mean(list(dices_batch.values())):.4f} - parameters {parameters}"
)
# Mean dice score: 0.6904 - parameters [0.5, 1, 0.3, 0.03, 0.75, 0.02, 'birth']

# %%  Plot Dice curve

eps = 0.03
plt.figure(figsize=(6, 1.5))
plt.plot(dices_batch.keys(), dices_batch.values(), "black")
plt.scatter(dices_batch.keys(), dices_batch.values(), c="black")
plt.xlim(21 - 0.5, 38 + 0.5)
plt.xticks(range(21, 38 + 1, 2))
plt.ylim(min(dices_batch.values()) - eps, max(dices_batch.values()) + eps)
plt.yticks([0.4, 0.6, 0.8])
plt.xlabel("Gestational week", fontsize=12)
plt.ylabel("Dice", fontsize=12)
# plt.title('Dice score per gestational week');
plt.savefig("results/scores_STA_article.pdf", bbox_inches="tight")
plt.show()
