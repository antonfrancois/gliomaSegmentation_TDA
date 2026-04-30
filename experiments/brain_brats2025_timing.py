import numpy as np
import pandas as pd
import parseBrats as pB
from segmentations import segment_brain_timed
from utils import (
    get_multiple_dice,
    ChronometerStart,
    ChronometerTick,
)
from collections import defaultdict

# %% Full segmentation - Define parameters

# Parameters
normalize = "max"  # preprocess_brain, divide by max or 255
sigma = 1  # preprocess_brain, Gaussian blur
enhance = (False, True)  # preprocess_brain, apply enhancement or not
radius_enhance = 1  # preprocess_brain, radius of enhancement
dilate = True  # preprocess_brain, apply dilation or not on T1ce
radius_dilation = 2  # preprocess_brain and segment_other_components, radius of dilation
whole_threshold = 1  # segment_whole_object, threshold for suggest_t
max_bars = 1  # segment_geometric_object, number of H2 features to consider
parameters = [
    normalize,
    sigma,
    enhance,
    radius_enhance,
    dilate,
    radius_dilation,
    whole_threshold,
    max_bars,
]

# %% Full segmentation - Initialize and run
idx_max = 100
timings_0 = []
timings_1 = []
timings_2 = []
timings_3 = []

pb = pB.parse_brats(
    brats_list=None, brats_folder="2025", modality="flair", get_template=False
)
i_list = range(len(pb.brats_list))

msg = "Run... "
start_time = ChronometerStart(msg)
for i in range(idx_max):
    # Open image.
    img_flair, seg_gt = pb(i, to_torch=False, modality="flair", normalize=True)
    img_t1ce, _ = pb(i, to_torch=False, modality="t1ce")

    # Segmentation.
    seg_final, timings = segment_brain_timed(
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
        verbose=False,
        plot=False,
    )

    # Save scores.
    score = get_multiple_dice(seg_final, seg_gt, verbose=True)
    print("Timings:", timings)
    timings_0.append(timings["preprocess"][0])
    timings_1.append(timings["module1"][0])
    timings_2.append(timings["module2"][0])
    timings_3.append(timings["module3"][0])
    ChronometerTick(start_time, i, idx_max, msg)

# %% Print summary

import numpy as np

print("Preprocess:", np.mean(timings_0), "pm", np.std(timings_0))
print("Module 1  :", np.mean(timings_1), "pm", np.std(timings_1))
print("Module 2  :", np.mean(timings_2), "pm", np.std(timings_2))
print("Module 3  :", np.mean(timings_3), "pm", np.std(timings_3))

# Preprocess: 1.051360964780397 pm 0.12523793075172823
# Module 1  : 11.793848152040082 pm 1.2021415356782572
# Module 2  : 10.005045059939949 pm 1.0385871838251988
# Module 3  : 1.0209064867300186 pm 0.18086960139943847
