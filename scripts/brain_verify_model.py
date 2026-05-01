# %% Imports

import numpy as np
import pandas as pd
import scipy, skimage
import parseBrats as pB
from segmentations import preprocess_brain
from morphology import argmax_image
from utils import (
    get_multiple_dice,
    ChronometerStart,
    ChronometerTick,
)

# %% Functions


def verify_hypotheses(seg_medecin, img_flair):
    verify_model = {"nonempty": None, "WTconnected": None, "argmax": None}
    # Define components
    seg_TC = (seg_medecin == 1) * 1  # TC
    seg_ED = (seg_medecin == 2) * 1  # ED
    seg_ET = (seg_medecin == 3) * 1  # ET
    seg_WT = (seg_medecin > 0) * 1  # WT

    # Components are nonempty
    verify_model["nonempty"] = (
        np.sum(seg_TC) > 0 and np.sum(seg_ED) > 0 and np.sum(seg_ET) > 0
    )

    # Size of TC relative to WT (should be lower than 50)
    if np.sum(seg_TC) == 0:
        verify_model["smallTC"] = np.inf
    else:
        verify_model["smallTC"] = np.sum(seg_WT) / np.sum(seg_TC)

    # CC of WT (ratio of largest CC over second largest, should be at least 10)
    labels = skimage.measure.label(seg_WT, background=0)
    components = [(labels == l) * 1 for l in range(1, np.max(labels) + 1)]
    size_components = np.flip(np.sort([np.sum(component) for component in components]))
    if len(size_components) == 1:
        verify_model["WTconnected"] = np.inf
    else:
        verify_model["WTconnected"] = size_components[0] / size_components[1]

    # Verify if argmax is in ET (should be 1 for ET, >0 for WT)
    verify_model["argmax"] = seg_medecin[argmax_image(img_flair)]

    return verify_model


# def quantify_sphericity(seg_medecin, dilatation=0):
#     # Define components
#     seg_TC = (seg_medecin == 1) * 1  # TC
#     seg_ED = (seg_medecin == 2) * 1  # ED
#     seg_ET = (seg_medecin == 3) * 1  # ET
#     seg_WT = (seg_medecin > 0) * 1  # WT
#
#     # Dilate
#     for i in range(dilatation):
#         seg_ET = scipy.ndimage.binary_dilation(seg_ET, iterations=1)
#     # /!\ modification of seg_medecin here, to get fair results
#     seg_medecin[seg_ET > 0] = 3
#
#     # Extract CC
#     seg_complement = 1 - seg_ET
#     labels = skimage.measure.label(seg_complement, background=0)
#     components = [(labels == i) * 1 for i in range(1, np.max(labels) + 1)]
#
#     # Define ED
#     components_len = [np.sum(component) for component in components]
#     imax_comp = np.argmax(components_len)
#     component_ED = components[imax_comp] * seg_WT
#     components.pop(imax_comp)
#
#     # Define TC
#     component_TC = np.sum(components, 0)
#
#     # Define seg and plot
#     seg_final = seg_ET.copy() * 3  # define seg_final
#     seg_final[component_ED > 0] = 2
#     seg_final[component_TC > 0] = 1
#
#     scores = get_multiple_dice(seg_final, seg_medecin, verbose=False)
#
#     return scores


def check_h3(seg_gt, iterations_dilation=0):
    # Get components.
    seg_tc = (seg_gt == 1) * 1  # TC
    seg_ed = (seg_gt == 2) * 1  # ED
    seg_et = (seg_gt == 3) * 1  # ET
    # seg_wt = (seg_gt > 0) * 1  # WT

    # Dilate if required.
    if iterations_dilation > 0:
        # Dilate ET.
        seg_et = scipy.ndimage.binary_dilation(seg_et, iterations=iterations_dilation)
        # Remove dilated ET in TC and ED.
        seg_tc[seg_et > 0] = 0
        seg_ed[seg_et > 0] = 0

    # Get connected components
    labels = skimage.measure.label(1 - seg_et, background=0)
    components = [(labels == l) * 1 for l in range(1, np.max(labels) + 1)]
    size_components = np.flip(np.sort([np.sum(component) for component in components]))
    # print(size_components)

    # Create new mask
    # 0 -> background
    # 1 -> not background
    idx_background = np.argmax(size_components)
    mask = np.zeros(np.shape(labels))
    mask[components[idx_background] == 0] = 1

    # Compute quantities.
    # TC should be close to 1
    mean_tc = mask[seg_tc > 0].mean() if seg_tc.sum() > 0 else 1
    # ED should be close to 0
    mean_ed = mask[seg_ed > 0].mean() if seg_ed.sum() > 0 else 0
    # TC is exactly 1
    mean_et = mask[seg_et > 0].mean() if seg_et.sum() > 0 else 1

    return mean_tc, mean_ed, mean_et


# %% Benchmark - Verification of the model

# Parameters.
sigma = 1
normalize = "max"
enhance = (False, True)
radius_enhance = 1
dilate = True
radius_dilation = 2
parameters = [sigma, normalize, enhance, radius_enhance, dilate, radius_dilation]

# Open images.
pb = pB.parse_brats(
    brats_list=None, brats_folder="2025", modality="flair", get_template=False
)
i_list = range(len(pb.brats_list))

# Check whether the images satisfy the model.
images_verify_model = dict()
msg = "Verify model... "
start_time = ChronometerStart(msg)
for i in i_list:
    # Open image
    img_flair, seg_gt = pb(i, to_torch=False, modality="flair", normalize=True)
    img_t1ce, _ = pb(i, to_torch=False, modality="t1ce")

    # Preprocess.
    img_flair, img_t1ce = preprocess_brain(
        img_flair,
        img_t1ce,
        sigma,
        normalize=normalize,
        enhance=enhance,
        radius_enhance=radius_enhance,
        dilate=dilate,
        radius_dilation=radius_dilation,
    )

    # Verify model: new quantities.
    img_verify_model = verify_hypotheses(seg_gt, img_flair)
    # img_verify_model["scores"] = quantify_sphericity(seg_gt, dilatation=radius_dilation)
    img_verify_model["brats_name"] = pb.brats_list[i]

    # # Verify model: argmax.
    # img_verify_model["argmax_FLAIR"] = seg_gt[argmax_image(img_flair)]
    # img_verify_model["argmax_T1ce"] = seg_gt[argmax_image(img_t1ce * (seg_gt > 0))]

    # Verify model: old quantities.
    seg_TC = (seg_gt == 1) * 1  # TC
    seg_ED = (seg_gt == 2) * 1  # ED
    seg_ET = (seg_gt == 3) * 1  # ET
    seg_WT = (seg_gt > 0) * 1  # WT
    if np.sum(seg_TC) > 0 and np.sum(seg_ED) > 0 and np.sum(seg_ET) > 0:
        #     # (H2') Compare voxels intensity
        #     mean_TC = np.mean(img_flair[np.where(seg_TC > 0)])
        #     mean_ED = np.mean(img_flair[np.where(seg_ED > 0)])
        #     mean_ET = np.mean(img_flair[np.where(seg_ET > 0)])
        #     img_verify_model["mean_TC_flair"] = mean_TC
        #     img_verify_model["mean_ED_flair"] = mean_ED
        #     img_verify_model["mean_ET_flair"] = mean_ET
        #
        #     mean_TC = np.mean(img_t1ce[np.where(seg_TC > 0)])
        #     mean_ED = np.mean(img_t1ce[np.where(seg_ED > 0)])
        #     mean_ET = np.mean(img_t1ce[np.where(seg_ET > 0)])
        #     img_verify_model["mean_TC_t1ce"] = mean_TC
        #     img_verify_model["mean_ED_t1ce"] = mean_ED
        #     img_verify_model["mean_ET_t1ce"] = mean_ET
        #
        #     # (H3') Boundary of TC (meanvalue should be > 0.5)
        #     componentdilated = scipy.ndimage.binary_dilation(seg_TC, iterations=1)
        #     componentcontour = componentdilated - seg_TC
        #     meanvalue = np.mean(seg_ET[np.where(componentcontour > 0)])
        #     img_verify_model["boundary_TC"] = meanvalue
        #
        #     # (H4') Boundary of ET (meanvalue should be < 0.5)
        #     componentdilated = scipy.ndimage.binary_dilation(seg_ED, iterations=1)
        #     componentcontour = componentdilated - seg_ED
        #     meanvalue = np.mean(seg_ET[np.where(componentcontour > 0)])
        #     img_verify_model["boundary_ET"] = meanvalue

        # New quantities, 16/02/2026
        # H1
        thresh = 0.1
        h1_median = np.median(img_flair[seg_gt > thresh]) / np.median(
            img_flair[img_flair > thresh]
        )
        h1_mean = np.mean(img_flair[seg_gt > thresh]) / np.mean(
            img_flair[img_flair > thresh]
        )
        img_verify_model["h1_median"] = h1_median
        img_verify_model["h1_mean"] = h1_mean

        # H2
        mask_non_et = np.zeros(seg_gt.shape)
        mask_non_et[seg_gt == 1] = 1
        mask_non_et[seg_gt == 2] = 1
        h2_median = np.median(img_t1ce[seg_gt == 3]) / np.median(
            img_t1ce[mask_non_et == 1]
        )
        h2_mean = np.mean(img_t1ce[seg_gt == 3]) / np.mean(img_t1ce[mask_non_et == 1])
        img_verify_model["h2_median"] = h2_median
        img_verify_model["h2_mean"] = h2_mean

        # H3
        h3_dil0 = check_h3(seg_gt, iterations_dilation=0)[0]
        h3_dil1 = check_h3(seg_gt, iterations_dilation=1)[0]
        h3_dil2 = check_h3(seg_gt, iterations_dilation=2)[0]
        h3_dil3 = check_h3(seg_gt, iterations_dilation=3)[0]
        img_verify_model["h3_dil0"] = h3_dil0
        img_verify_model["h3_dil1"] = h3_dil1
        img_verify_model["h3_dil2"] = h3_dil2
        img_verify_model["h3_dil3"] = h3_dil3

    else:
        # img_verify_model["mean_TC_flair"] = 0
        # img_verify_model["mean_ED_flair"] = 0
        # img_verify_model["mean_ET_flair"] = 0
        # img_verify_model["mean_TC_t1ce"] = 0
        # img_verify_model["mean_ED_t1ce"] = 0
        # img_verify_model["mean_ET_t1ce"] = 0
        # img_verify_model["boundary_TC"] = 0
        # img_verify_model["boundary_ET"] = 0
        img_verify_model["h1_median"] = 0
        img_verify_model["h1_mean"] = 0
        img_verify_model["h2_median"] = 0
        img_verify_model["h2_mean"] = 0
        img_verify_model["h3_dil0"] = 0
        img_verify_model["h3_dil1"] = 0
        img_verify_model["h3_dil2"] = 0
        img_verify_model["h3_dil3"] = 0

    # # Verify model: ratio component.
    # if radius_dilation > 0:
    #     seg_ET = skimage.morphology.binary_dilation(
    #         seg_ET, footprint=skimage.morphology.ball(radius_dilation)
    #     )
    #     seg_TC_erode = skimage.morphology.binary_erosion(
    #         seg_TC, footprint=skimage.morphology.ball(radius_dilation)
    #     )
    #
    #     # Verify model
    #     seg_complement = 1 - seg_ET
    #     labels = skimage.measure.label(seg_complement, background=0)
    #     components = [
    #         (labels == j) * 1 for j in range(1, np.max(labels) + 1)
    #     ]  # Remove ET
    #     components_len = [np.sum(component) for component in components]
    #     components_len.pop(np.argmax(components_len))  # Remove background (contains ED)
    #     inner_components_len = sum(components_len)
    #
    #     if np.sum(seg_TC_erode) > 0:
    #         img_verify_model["RatioComponentsWidth"] = inner_components_len / np.sum(
    #             seg_TC_erode
    #         )
    #     else:
    #         img_verify_model["RatioComponentsWidth"] = np.nan
    # else:
    #     img_verify_model["RatioComponentsWidth"] = np.nan

    # Save
    images_verify_model[pb.brats_list[i_list[i]]] = img_verify_model
    ChronometerTick(start_time, i, len(i_list), msg)

# Save results.
pd.DataFrame.from_dict(images_verify_model, orient="index").to_csv(
    f"results/brain_brats2025_verifymodel_{str(parameters)}_len{len(images_verify_model)}.csv"
)

# %% Print results.

# Parameters of the model.
thresh_smallTC = 50  # upper bound on ratio WT/TC
thresh_WTconnected = 10  # lower bound on ratio first/second largest CC
admissible_argmax_FLAIR = [1, 3]  # values that argmax can take in FLAIR
admissible_argmax_T1ce = [3]  # values that argmax can take in T1ce
RatioComponentsWidth = 1  # lower bound on ratio of ...

# Identify images that satisfy the model.
df_VerifyModel = pd.read_csv(
    f"results/brain_brats2025_verifymodel_{str(parameters)}_len{len(images_verify_model)}.csv"
)
names_verify_model = {name: False for name in df_VerifyModel["brats_name"]}
for i in range(len(df_VerifyModel)):
    names_verify_model[df_VerifyModel["brats_name"][i]] = (
        df_VerifyModel["nonempty"][i] == True
        and df_VerifyModel["smallTC"][i] <= thresh_smallTC
        and df_VerifyModel["WTconnected"][i] >= thresh_WTconnected
        and df_VerifyModel["boundary_TC"][i] > 0.5
        and df_VerifyModel["boundary_ET"][i] < 0.5
        and df_VerifyModel["argmax_FLAIR"][i] in admissible_argmax_FLAIR
        and df_VerifyModel["argmax_T1ce"][i] in admissible_argmax_T1ce
        and (
            df_VerifyModel["RatioComponentsWidth"][i] >= RatioComponentsWidth
            or np.isnan(df_VerifyModel["RatioComponentsWidth"][i])
        )
    )

# Print result.
num_satisfying = np.sum(list(names_verify_model.values()))
percent_satisfying = round(num_satisfying / len(df_VerifyModel) * 100, 3)
print(f"Brains satisfying the model: {num_satisfying} = {percent_satisfying}%")
