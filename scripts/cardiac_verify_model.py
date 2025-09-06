# %% Imports

import pandas as pd
import numpy as np
import skimage

from segmentations import parseACDC
from utils import ChronometerStart, ChronometerTick

# %%  Verify model

dilatation_totest = range(0, 3 + 1)

NumberComponents2D = {
    dilatation: {modality: dict() for modality in ["ED", "ES"]}
    for dilatation in dilatation_totest
}
NumberComponents3D = {
    dilatation: {modality: dict() for modality in ["ED", "ES"]}
    for dilatation in dilatation_totest
}

msg = "Verify model... "
start_time = ChronometerStart(msg)
for n_image in range(1, 150 + 1):
    for modality in ["ED", "ES"]:
        for dilatation in dilatation_totest:
            # Open image
            img, seg_medecin, filename = parseACDC(
                n_image, end=modality, return_filename=True
            )
            seg_myo = (seg_medecin == 2) * 1
            acdc_name = filename[-28:-10]
            mean_RV = np.mean(img * (seg_medecin == 1))
            mean_LV = np.mean(img * (seg_medecin == 3))
            mean_Myo = np.mean(img * (seg_medecin == 2))
            satisfy_mean_condition = (mean_RV > mean_Myo) and (mean_LV > mean_Myo)

            " Verif model 2D "

            components_number_byslice = []
            for z_pos in range(np.shape(seg_myo)[2]):
                # Get slice
                seg_myo_slice = seg_myo[:, :, z_pos].copy()
                # Dilation
                if dilatation > 0:
                    seg_myo_slice = skimage.morphology.dilation(
                        seg_myo_slice,
                        footprint=skimage.morphology.disk(dilatation),
                    )
                # Extract CC
                seg_complement = 1 - seg_myo_slice
                labels = skimage.measure.label(seg_complement, background=0)
                # Remove myocardium
                components = [(labels == i) * 1 for i in range(1, np.max(labels) + 1)]
                components_number_byslice.append(len(components))
            # Save scores
            NumberComponents2D[dilatation][modality][
                acdc_name
            ] = components_number_byslice
            # Mean condition
            if not satisfy_mean_condition:
                NumberComponents2D[dilatation][modality][acdc_name] = [0]

            " Verif model 3D "

            # Dilate 3D
            if dilatation > 0:
                seg_myo = skimage.morphology.dilation(
                    seg_myo, footprint=skimage.morphology.ball(dilatation)
                )
            # Extract CC
            seg_complement = 1 - seg_myo
            labels = skimage.measure.label(seg_complement, background=0)
            # Remove myocardium
            components = [(labels == i) * 1 for i in range(1, np.max(labels) + 1)]
            NumberComponents3D[dilatation][modality][acdc_name] = len(components)
            # Mean condition
            if not satisfy_mean_condition:
                NumberComponents2D[dilatation][modality][acdc_name] = 0

    ChronometerTick(start_time, n_image, 150 + 1, msg)

# %% Print results

# 2D - ED and ES separately
for dilatation in NumberComponents2D:
    for modality in NumberComponents2D[dilatation]:
        n = sum(
            [
                list(np.unique(nb)) == [2]
                for nb in NumberComponents2D[dilatation][modality].values()
            ]
        )
        print("2D", modality, dilatation, round(n / 150 * 100, 2), "%")
print()


# 2D - ED and ES together
for dilatation in NumberComponents2D:
    n = sum(
        [
            list(np.unique(nb)) == [2]
            for nb in NumberComponents2D[dilatation]["ED"].values()
        ]
        + [
            list(np.unique(nb)) == [2]
            for nb in NumberComponents2D[dilatation]["ES"].values()
        ]
    )
    print("2D", "ES and ED", dilatation, round(n / (2 * 150) * 100, 2), "%")
print()

# 3D - ED and ES separately
for dilatation in NumberComponents3D:
    for modality in NumberComponents3D[dilatation]:
        n = sum([nb == 2 for nb in NumberComponents3D[dilatation][modality].values()])
        print("3D", modality, dilatation, round(n / 150 * 100, 2), "%")
print()

# 3D - ED and ES together
for dilatation in NumberComponents3D:
    n = sum(
        [nb == 2 for nb in NumberComponents3D[dilatation]["ED"].values()]
        + [nb == 2 for nb in NumberComponents3D[dilatation]["ES"].values()]
    )
    print("3D", "ES and ED", dilatation, round(n / (2 * 150) * 100, 2), "%")
print()


# %% Save results (2D and 3D)

df = pd.DataFrame(
    {
        "acdc_name": list(NumberComponents2D[0]["ED"].keys())
        + list(NumberComponents2D[0]["ES"].keys()),
    }
)

for dilatation in NumberComponents2D:
    images_verify_model_2d = dict()
    images_verify_model_3d = dict()

    for modality in NumberComponents2D[dilatation]:
        for acdc_name in NumberComponents2D[dilatation][modality]:
            cc = NumberComponents2D[dilatation][modality][acdc_name]
            images_verify_model_2d[acdc_name] = (list(np.unique(cc)) == [2]) * 1
            cc = NumberComponents3D[dilatation][modality][acdc_name]
            images_verify_model_3d[acdc_name] = (cc == 2) * 1

    df["2d:dil" + str(dilatation)] = df["acdc_name"].map(images_verify_model_2d)
    df["3d:dil" + str(dilatation)] = df["acdc_name"].map(images_verify_model_3d)


df.to_csv("results/cardiac_acdc_verifymodel.csv", index=False)

# 2D ED 0 69.33 %
# 2D ES 0 8.67 %
# 2D ED 1 72.0 %
# 2D ES 1 12.0 %
# 2D ED 2 66.67 %
# 2D ES 2 10.67 %
# 2D ED 3 58.67 %
# 2D ES 3 9.33 %
# 2D ES and ED 0 39.0 %
# 2D ES and ED 1 42.0 %
# 2D ES and ED 2 38.67 %
# 2D ES and ED 3 34.0 %

# 3D ED 0 0.0 %
# 3D ES 0 6.67 %
# 3D ED 1 41.33 %
# 3D ES 1 54.67 %
# 3D ED 2 60.67 %
# 3D ES 2 73.33 %
# 3D ED 3 81.33 %
# 3D ES 3 84.0 %
# 3D ES and ED 0 3.33 %
# 3D ES and ED 1 48.0 %
# 3D ES and ED 2 67.0 %
# 3D ES and ED 3 82.67 %
