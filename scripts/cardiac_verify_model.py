# %% Imports

import numpy as np
import skimage
import matplotlib.pyplot as plt
from segmentations import parseACDC
from utils import ChronometerStart, ChronometerTick


# %%  Verify model - homology

dilatation_totest = range(0, 3 + 1)
NumberComponents3D = {
    dilatation: {modality: dict() for modality in ["ED", "ES"]}
    for dilatation in dilatation_totest
}
NumberComponents2D = {
    dilatation: {modality: dict() for modality in ["ED", "ES"]}
    for dilatation in dilatation_totest
}

msg = "Verify model... "
start_time = ChronometerStart(msg)
for n_image in range(1, 150 + 1):
    for modality in ["ED", "ES"]:
        for dilatation in dilatation_totest:
            # Open image
            img, seg_medecin = parseACDC(n_image, end=modality)

            # Get nonempty slices of CC (containing myo)
            nonempty_slices = np.where(np.sum(np.sum(seg_medecin == 2, 0), 0))[0]
            zmin, zmax = nonempty_slices[0], nonempty_slices[-1]
            z_im = range(zmin, zmax + 1)

            " Verif model 3D "

            # Get slices in z_im
            img_zrange, seg_medecin_zrange = img[:, :, z_im], seg_medecin[:, :, z_im]
            seg_myo = (seg_medecin_zrange == 2) * 1

            # Dilate 3D (slice by slice)
            if dilatation > 0:
                for z in range(np.shape(seg_myo)[2]):
                    seg_myo[:, :, z] = skimage.morphology.binary_dilation(
                        seg_myo[:, :, z], footprint=skimage.morphology.disk(dilatation)
                    )

            # Add planes of 1's
            seg_myo_augmented = np.zeros(
                (
                    np.shape(img_zrange)[0],
                    np.shape(img_zrange)[1],
                    np.shape(img_zrange)[2] + 2,
                )
            )
            seg_myo_augmented[:, :, 1:-1] = seg_myo
            seg_myo_augmented[:, :, 0], seg_myo_augmented[:, :, -1] = 1, 1
            seg_myo = (seg_myo_augmented.astype(int) == 1) * 1

            # Extract CC
            seg_complement = 1 - seg_myo
            labels = skimage.measure.label(seg_complement, background=0)
            components = [
                (labels == i) * 1 for i in range(1, np.max(labels) + 1)
            ]  # Remove myocardium

            NumberComponents3D[dilatation][modality][n_image] = len(components)

            " Verif model 2D "

            # Get slices in z_im
            img_zrange, seg_medecin_zrange = img[:, :, z_im], seg_medecin[:, :, z_im]
            seg_myo = (seg_medecin_zrange == 2) * 1

            # Dilate 3D (slice by slice)
            if dilatation > 0:
                for z in range(np.shape(seg_myo)[2]):
                    seg_myo[:, :, z] = skimage.morphology.binary_dilation(
                        seg_myo[:, :, z], footprint=skimage.morphology.disk(dilatation)
                    )

            components_number_byslice = []
            for z_pos in range(np.shape(seg_myo)[2]):
                # Get slice
                seg_myo_slice = seg_myo[:, :, z_pos]

                # Extract CC
                seg_complement = 1 - seg_myo_slice
                labels = skimage.measure.label(seg_complement, background=0)
                components = [
                    (labels == i) * 1 for i in range(1, np.max(labels) + 1)
                ]  # Remove myocardium

                components_number_byslice.append(len(components))

            NumberComponents2D[dilatation][modality][n_image] = np.mean(
                components_number_byslice
            )

    ChronometerTick(start_time, n_image, 150 + 1, msg)

# %% Print results - Verify model homology

# 3D
for modality in ["ED", "ES"]:
    for dilatation in dilatation_totest:
        n = np.sum(
            np.array(list(NumberComponents3D[dilatation][modality].values())) == 2
        )
        print("3D", modality, dilatation, round(n / 150 * 100, 2), "%")
    print()

for dilatation in dilatation_totest:
    n = np.sum(
        np.array(
            list(NumberComponents3D[dilatation]["ED"].values())
            + list(NumberComponents3D[dilatation]["ES"].values())
        )
        == 2
    )
    print("3D", "ES and ED", dilatation, round(n / (2 * 150) * 100, 2), "%")
print()

# 2D
for modality in ["ED", "ES"]:
    for dilatation in dilatation_totest:
        n = np.sum(
            np.array(list(NumberComponents2D[dilatation][modality].values())) == 2
        )
        print("2D", modality, dilatation, round(n / 150 * 100, 2), "%")
    print()

for dilatation in dilatation_totest:
    n = np.sum(
        np.array(
            list(NumberComponents2D[dilatation]["ED"].values())
            + list(NumberComponents2D[dilatation]["ES"].values())
        )
        == 2
    )
    print("2D", "ES and ED", dilatation, round(n / (2 * 150) * 100, 2), "%")

# %% Plot - Verify model homology
for modality in ["ED", "ES"]:
    plt.figure(figsize=(10, 2))
    plt.hist(
        [
            list(NumberComponents3D[dilatation][modality].values())
            for dilatation in dilatation_totest
        ]
    )
    plt.legend(["dilatation " + repr(dilatation) for dilatation in dilatation_totest])
    plt.title("3D - " + modality)
    plt.show()

for modality in ["ED", "ES"]:
    plt.figure(figsize=(10, 2))
    plt.hist(
        [
            list(NumberComponents2D[dilatation][modality].values())
            for dilatation in dilatation_totest
        ]
    )
    plt.legend(["dilatation " + repr(dilatation) for dilatation in dilatation_totest])
    plt.title("2D - " + modality)
    plt.show()
