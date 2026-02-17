# %% Imports

import __init__
from matplotlib import pyplot as plt

import src.parseBrats as pB
from src.segmentations import (
    segment_brain, preprocess_brain, segment_whole_object,
    threshold_triangle
)
from src.utils import (
    get_multiple_dice,
    plot_comparison_full_segmentations,
    make_3d_flat,
    CMAP_SEGS
)
import numpy as np
from src.morphology import (
    argmax_image,
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, MiniBatchKMeans
import skimage.filters as filt


#%%
# classification logic

def gmm_segment_pair_subsample(
    flair: np.ndarray,
    t1ce: np.ndarray,
    n_components: int = 4,
    sample_size: int = 200_000,
    mask_zeros: str = "either",   # "either" or "both"
    random_state: int = 0,
    n_init: int = 10,
    max_iter: int = 300,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    return_posteriors: bool = False,
):
    """
    Fit a GMM on a random subset of non-zero voxels using paired features (FLAIR, T1ce),
    then predict labels for all voxels.

    Parameters
    ----------
    flair, t1ce : np.ndarray
        3D volumes of identical shape, already co-registered and on the same grid.
    n_components : int
        Number of mixture components.
    sample_size : int
        Number of voxels to sample for fitting the GMM (subsampling for speed/memory).
    mask_zeros : {"either","both"}
        "either": keep voxels where (flair>0) OR (t1ce>0)
        "both":   keep voxels where (flair>0) AND (t1ce>0)
    random_state : int
        Seed for reproducibility.
    n_init, max_iter, covariance_type, reg_covar :
        Passed to sklearn.mixture.GaussianMixture
    return_posteriors : bool
        If True, also return posterior probabilities (can be large).

    Returns
    -------
    labels : np.ndarray
        3D int label map. Background (excluded voxels) is 0, components are 1..K.
    posteriors : np.ndarray or None
        4D array (Z,Y,X,K) if return_posteriors=True else None.
    gmm : GaussianMixture
        Fitted GMM in normalized feature space.
    norm_params : dict
        Parameters used to normalize features (robust center/scale per modality).
    """
    flair = np.asarray(flair)
    t1ce = np.asarray(t1ce)
    if flair.shape != t1ce.shape or flair.ndim != 3:
        raise ValueError(f"Expected two 3D volumes of same shape, got {flair.shape} and {t1ce.shape}")

    f = flair.astype(np.float32, copy=False)
    t = t1ce.astype(np.float32, copy=False)

    if mask_zeros == "either":
        mask = (f > 0) | (t > 0)
    elif mask_zeros == "both":
        mask = (f > 0) & (t > 0)
    else:
        raise ValueError("mask_zeros must be 'either' or 'both'")

    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        raise ValueError("No voxels selected by the non-zero mask.")

    rng = np.random.default_rng(random_state)
    if sample_size is None or sample_size <= 0 or sample_size >= idx.size:
        sample_idx = idx
    else:
        sample_idx = rng.choice(idx, size=sample_size, replace=False)

    # Build paired features for the sampled voxels: X = [FLAIR, T1ce]
    f_flat = f.ravel()
    t_flat = t.ravel()
    X = np.stack([f_flat[sample_idx], t_flat[sample_idx]], axis=1)  # (N, 2)

    # Robust normalization per modality (helps EM stability across MRI intensity scales)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-8
    Xn = (X - med) / mad

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=reg_covar,
        random_state=random_state,
    )
    gmm.fit(Xn)

    # Predict for all masked voxels (not only the sample)
    X_all = np.stack([f_flat[idx], t_flat[idx]], axis=1)
    X_all_n = (X_all - med) / mad

    y_all = gmm.predict(X_all_n)  # 0..K-1

    labels = np.zeros(f.shape, dtype=np.int16)
    labels.ravel()[idx] = (y_all + 1).astype(np.int16)  # reserve 0 for background

    posteriors = None
    if return_posteriors:
        p_all = gmm.predict_proba(X_all_n)  # (M, K)
        posteriors = np.zeros(f.shape + (n_components,), dtype=np.float32)
        posteriors.reshape(-1, n_components)[idx] = p_all.astype(np.float32)

    norm_params = {"median": med, "mad": mad, "mask_zeros": mask_zeros}

    return labels, posteriors, gmm, norm_params


def segment_pair_subsample(
    flair: np.ndarray,
    t1ce: np.ndarray,
    method: str = "gmm",                 # "gmm", "bgmm", "kmeans", "mbkmeans"
    n_components: int = 4,               # for GMM/BGMM; also used as n_clusters for KMeans
    sample_size: int = 200_000,
    mask_zeros: str = "either",          # "either" or "both"
    random_state: int = 0,
    normalize: str = "robust",           # "robust" or "zscore" or "none"
    return_scores: bool = False,         # posteriors for mixtures; distances for kmeans
    # --- shared-ish hyperparams
    max_iter: int = 300,
    n_init: int = 10,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    # --- BGMM-specific (sensible defaults)
    weight_concentration_prior_type: str = "dirichlet_process",
    weight_concentration_prior: float | None = None,
    # --- MiniBatchKMeans-specific
    batch_size: int = 8192,
):
    """
    Fit a model on a random subset of non-zero voxels using paired features (FLAIR, T1ce),
    then predict labels for all selected voxels.

    Supported methods
    -----------------
    - "gmm"     : sklearn.mixture.GaussianMixture (soft posteriors available)
    - "bgmm"    : sklearn.mixture.BayesianGaussianMixture (soft posteriors available)
    - "kmeans"  : sklearn.cluster.KMeans (hard labels; optional distances)
    - "mbkmeans": sklearn.cluster.MiniBatchKMeans (hard labels; optional distances)

    Returns
    -------
    labels : np.ndarray
        3D int label map. Background/excluded voxels are 0; clusters are 1..K.
    scores : np.ndarray or None
        If method in {"gmm","bgmm"} and return_scores=True:
            posteriors array of shape (Z,Y,X,K)
        If method in {"kmeans","mbkmeans"} and return_scores=True:
            distances array of shape (Z,Y,X,K)
        Else None.
    model : object
        Fitted sklearn model.
    norm_params : dict
        Normalization parameters and mask convention.
    """
    flair = np.asarray(flair)
    t1ce = np.asarray(t1ce)
    if flair.shape != t1ce.shape or flair.ndim != 3:
        raise ValueError(f"Expected two 3D volumes of same shape, got {flair.shape} and {t1ce.shape}")

    f = flair.astype(np.float32, copy=False)
    t = t1ce.astype(np.float32, copy=False)

    if mask_zeros == "either":
        mask = (f > 0) | (t > 0)
    elif mask_zeros == "both":
        mask = (f > 0) & (t > 0)
    else:
        raise ValueError("mask_zeros must be 'either' or 'both'")

    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        raise ValueError("No voxels selected by the non-zero mask.")

    rng = np.random.default_rng(random_state)
    if sample_size is None or sample_size <= 0 or sample_size >= idx.size:
        sample_idx = idx
    else:
        sample_idx = rng.choice(idx, size=sample_size, replace=False)

    f_flat = f.ravel()
    t_flat = t.ravel()

    # Features on the sample: (N,2)
    X = np.stack([f_flat[sample_idx], t_flat[sample_idx]], axis=1)

    # ----- Normalization (recommended for stability)
    if normalize == "robust":
        center = np.median(X, axis=0)
        scale = np.median(np.abs(X - center), axis=0) + 1e-8
    elif normalize == "zscore":
        center = X.mean(axis=0)
        scale = X.std(axis=0) + 1e-8
    elif normalize == "none":
        center = np.zeros(2, dtype=np.float32)
        scale = np.ones(2, dtype=np.float32)
    else:
        raise ValueError("normalize must be 'robust', 'zscore', or 'none'")

    Xn = (X - center) / scale

    # ----- Model selection / construction
    method = method.lower()
    if method == "gmm":
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        model.fit(Xn)

        def predict_labels(Xalln):  # 0..K-1
            return model.predict(Xalln)

        def predict_scores(Xalln):  # (M,K)
            return model.predict_proba(Xalln)

        K = n_components

    elif method == "bgmm":
        model = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=reg_covar,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            random_state=random_state,
        )
        model.fit(Xn)

        def predict_labels(Xalln):
            return model.predict(Xalln)

        def predict_scores(Xalln):
            return model.predict_proba(Xalln)

        K = n_components

    elif method == "kmeans":
        model = KMeans(
            n_clusters=n_components,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        model.fit(Xn)

        def predict_labels(Xalln):
            return model.predict(Xalln)

        def predict_scores(Xalln):
            # Distances to cluster centers: smaller = closer
            return model.transform(Xalln)

        K = n_components

    elif method == "mbkmeans":
        model = MiniBatchKMeans(
            n_clusters=n_components,
            batch_size=batch_size,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        model.fit(Xn)

        def predict_labels(Xalln):
            return model.predict(Xalln)

        def predict_scores(Xalln):
            return model.transform(Xalln)

        K = n_components

    else:
        raise ValueError("method must be one of: 'gmm', 'bgmm', 'kmeans', 'mbkmeans'")

    # ----- Predict for all selected voxels (not only the sample)
    X_all = np.stack([f_flat[idx], t_flat[idx]], axis=1)
    X_all_n = (X_all - center) / scale

    y_all = predict_labels(X_all_n)  # 0..K-1

    labels = np.zeros(f.shape, dtype=np.int16)
    labels.ravel()[idx] = (y_all + 1).astype(np.int16)  # reserve 0 for background

    scores = None
    if return_scores:
        s_all = predict_scores(X_all_n)  # (M,K)
        scores = np.zeros(f.shape + (K,), dtype=np.float32)
        scores.reshape(-1, K)[idx] = s_all.astype(np.float32)

    norm_params = {
        "center": center,
        "scale": scale,
        "normalize": normalize,
        "mask_zeros": mask_zeros,
        "method": method,
    }

    return labels, scores, model, norm_params

def apply_skimage_threshold(image, method, non_zero_min = .2, plot = False):

    non_zero_im = img_flair[img_flair > non_zero_min]

    if method == "isodata":
        thresh = filt.threshold_isodata(non_zero_im)
    elif method == "li":
        thresh = filt.threshold_li(non_zero_im)
    elif method == "otsu":
        thresh = filt.threshold_otsu(non_zero_im)# hist = (counts, bin_centers))
    elif method == "mean":
        thresh = filt.threshold_mean(non_zero_im)
    elif method == "triangle":
        # thresh = filt.threshold_triangle(non_zero_im)
        thresh = threshold_triangle(non_zero_im)
    elif method == "yen":
        thresh = filt.threshold_yen(non_zero_im)
    elif method == "minimum":
        try:
            thresh = filt.threshold_minimum(non_zero_im)
        except RuntimeError:
            thresh = 0
    seg_whole = (img_flair > thresh) * 1

    if plot:
        plot_comparison_full_segmentations(
            f"{method} "+pb.brats_list[i],
            img_flair,
            (seg_gt > 0) * 1,
            (seg_whole > 0) * 1,
            show_ground_truth=False,
            # save_path="results/brain_seg_module1.pdf",
    )
    dice_wt_method = get_multiple_dice((seg_whole > 0) * 1, (seg_gt > 0) * 1, labels=(1,), verbose=False)

    return dice_wt_method, thresh, seg_whole

# %% Open image

pb = pB.parse_brats(
    brats_list=None,
    brats_folder="2025",
    modality="flair",
    get_template=False,
)


# n_list = np.random.choice(range(len(pb.brats_list)), size = 30, replace=False)
n_list = np.arange(len(pb.brats_list))
# n_list = [676,  220,  407, 20,  21,  46 ]
print(n_list)
#%%

normalize = "max"  # preprocess_brain, divide by max or 255
sigma = 1  # preprocess_brain, Gaussian blur
enhance = (False, True)  # preprocess_brain, apply enhancement or not
radius_enhance = 1  # preprocess_brain, radius of enhancement
dilate = True  # preprocess_brain, apply dilation or not on T1ce
radius_dilation = 2  # preprocess_brain and segment_other_components, radius of dilation
whole_threshold = 1  # segment_whole_object, threshold for suggest_t
max_bars = 1  # segment_geometric_object, number of H2 features to consider
verbose = True
plot = False
save = False
# Gather run metadata (your parameters)
run_params = dict(
    normalize=normalize, sigma=sigma, enhance=enhance, radius_enhance=radius_enhance, dilate=dilate, radius_dilation=radius_dilation, whole_threshold=whole_threshold, max_bars=max_bars, verbose=verbose, plot=plot, save=save,
)

import pandas as pd

csv_path = "results/skimage_thresholds/comparison_skimage.csv"
COLUMNS = [
    "subject",
    "method",
    "threshold",
    "dice_wt",
    "parameters_dict"
]
df = pd.DataFrame(columns=COLUMNS)
# df.to_csv(csv_path, index=False)

for i in n_list:
    print(f"\n {i+1}/{len(n_list)} : {pb.brats_list[i]}")
# i = n_list[2]

    img_flair, seg_gt = pb(i, to_torch=False, modality="flair", normalize=True)
    img_t1ce, _ = pb(i, to_torch=False, modality="t1ce")

    # Preprocess image.
    img_flair, img_t1ce = preprocess_brain(
        img_flair,
        img_t1ce,
        sigma=sigma,
        normalize=normalize,
        enhance=enhance,
        radius_enhance=radius_enhance,
        dilate=dilate,
        radius_dilation=radius_dilation,
    )
    # Module 1: Segmentation whole object.
    seg_whole_sugg, t = segment_whole_object(
        img=img_flair.copy(), method="suggest_t", threshold=whole_threshold, verbose=verbose, plot=plot, save=save
    )
    # Print Dice score
    print("\nsuggest_t:")
    dice_wt_sugg = get_multiple_dice((seg_whole_sugg > 0) * 1, (seg_gt > 0) * 1, labels=(1,), verbose=True)
    print()
    # plot_comparison_full_segmentations(
    #     "sugg "+pb.brats_list[i],
    #     img_flair,
    #     (seg_gt > 0) * 1,
    #     (seg_whole_sugg > 0) * 1,
    #     show_ground_truth=False,
    #     # save_path="results/brain_seg_module1.pdf",
    # )
    # plt.show()
    row = {"subject":  pb.brats_list[i],
        "method": "suggest_t" ,
        "threshold":  t ,
        "dice_wt": dice_wt_sugg[1] ,
        "parameters_dict" : run_params,}
    df.loc[len(df)] = row

    # # %% try all thresholds


    methods= ["isodata", "li", "otsu", "triangle", "yen"]
    # dice_wt_otsu = apply_skimage_threshold(img_flair, method = "otsu", plot = True)
    non_zeros_mins = [0, .1, .2, .3]
    nzm = .1
    for m in methods:
        dice_wt_m, threshold, _ = apply_skimage_threshold(img_flair,
                                                          method = m,
                                                          non_zero_min=nzm,
                                                          plot = False)
        print(f"method : {m}, non_zero_min : {nzm}, dice : {dice_wt_m[1]}")
        plt.show()
        row = {"subject":  pb.brats_list[i],
            "method": m ,
            "threshold":  threshold ,
            "dice_wt": dice_wt_m[1] ,
            "parameters_dict" : {"non_zero_min" : nzm}}
        df.loc[len(df)] = row

    df.to_csv(csv_path, index=False)

#%%  Make plot after


import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

methods= [
          "li",
        "isodata",
          "otsu",
          "yen",
          "triangle",
            "suggest_t"
          ]

def _safe_literal_eval(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return {}


def boxplot_methods_by_nonzero_min(
    csv_path: str,
    dice_col: str = "dice_wt",
    params_col: str = "parameters_dict",
    method_col: str = "method",
    subject_col: str = "subject",
    include_suggest_t: bool = True,
    out_dir: str | None = None,
):
    """
    One box plot per non_zero_min.
    Each box = Dice distribution across subjects for one method.
    """

    df = pd.read_csv(csv_path)


    df[dice_col] = pd.to_numeric(df[dice_col], errors="coerce")

    data = [
        df[df[method_col] == m][dice_col].dropna().values
        for m in methods
    ]


    fig, ax = plt.subplots(figsize=(1.4 * len(methods) + 2, 5))

    bp = ax.boxplot(
        data,
        labels=methods ,#+ ["ours"],
        showfliers=True,
        # patch_artist=True,
        showmeans=True,
        # meanline=True,
        notch=True,
    )

    # Aesthetics
    for box in bp["boxes"]:
        box.set(alpha=0.7)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice score", fontsize=20)
    ax.set_title(f"WT Dice distribution by thresholding method", fontsize=25)
    ax.grid(axis="y", linestyle=":", linewidth=0.7)
    ax.set_xticklabels(labels=methods[:-1] + ["Ours"],
                       rotation = 45, fontsize=20)

    # ax.legend()

    fig.tight_layout()
    plt.show()
    if out_dir is not None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(
            f"{out_dir}/boxplot.png",
            dpi=200,
        )
        plt.close(fig)

    return df

boxplot_methods_by_nonzero_min(
    csv_path= csv_path,
    # csv_path="/home/turtlefox/Documents/12_SegmentationTDA/gliomaSegmentation_TDA/results/skimage_thresholds/comparison_skimage_flaironly.csv",
    dice_col="dice_wt",
    out_dir="results/skimage_thresholds",
    include_suggest_t=False,
)


#%%

df = pd.read_csv(csv_path)
method = "suggest_t"
for m in methods:
    df_m = df[df["method"] == m]

    print(f"{m}: mean :{df_m["dice_wt"].mean():.3f} \pm {df_m["dice_wt"].std():.3f}; median : {df_m["dice_wt"].median():.3f}")

#%%  Research archive
# labels, post, gmm, norm = gmm_segment_pair_subsample(
#         img_flair, img_t1ce,
#         n_components=6,
#         sample_size=100_000,
#         mask_zeros="either",
#         return_posteriors=False,
#     )
# labels, dist_km, model_km, _ = segment_pair_subsample(
#         img_flair, img_t1ce, method="mbkmeans", n_components=4, sample_size=80_000, return_scores=True
#     )
#
# pos = argmax_image(seg_gt)
# fig, ax = plt.subplots(3, 1, figsize=(8, 8))
# ax[0].imshow(make_3d_flat(img_t1ce, pos), cmap="gray")
# ax[1].imshow(make_3d_flat(img_flair, pos), cmap="gray")
# ax[2].imshow(make_3d_flat(img_flair, pos), cmap="gray")
# l = ax[2].imshow(
#     make_3d_flat(labels, pos),
#     # cmap=CMAP_SEGS,
#     interpolation="nearest"
# )
# ax[2].legend()
# plt.show()
#
# #%%
# seg_means = []
# for n in np.unique(labels):
#     seg_means.append( img_flair[labels == n].mean() )
#
#     fig, ax = plt.subplots(3, 1, figsize=(8, 8))
#     ax[0].imshow(make_3d_flat(img_t1ce, pos), cmap="gray")
#     ax[1].imshow(make_3d_flat(img_flair, pos), cmap="gray")
#     ax[2].imshow(make_3d_flat(img_flair, pos), cmap="gray")
#     ax[2].imshow(
#         make_3d_flat(labels == n, pos),
#         # cmap=CMAP_SEGS,
#         interpolation="nearest"
#     )
#     ax[0].set_title(f"n = {n}")
#     plt.show()
#
# print(seg_means)
# max_seg = np.argmax(seg_means)
# print(seg_means)
# print("max seg", max_seg)
# seg_whole_method = labels == max_seg
#
# #%%
# dice_wt_method = get_multiple_dice((seg_whole_method > 0) * 1, (seg_gt > 0) * 1, labels=(1,), verbose=True)
# plot_comparison_full_segmentations(
#     f"{method} "+pb.brats_list[i],
#     img_flair,
#     (seg_gt > 0) * 1,
#     (seg_whole_method > 0) * 1,
#     show_ground_truth=False,
#     # save_path="results/brain_seg_module1.pdf",
# )
# plt.show()



    # seg_whole_method = segment_whole_object(
    #     img=img_flair.copy(), method=method, threshold=whole_threshold, verbose=verbose, plot=plot, save=save
    # )
    # # Print Dice score
    # print(f"\n{method}:")
    # dice_wt_method = get_multiple_dice((seg_whole_method > 0) * 1, (seg_gt > 0) * 1, labels=(1,), verbose=True)
    # plot_comparison_full_segmentations(
    #     f"{method} "+pb.brats_list[i],
    #     img_flair,
    #     (seg_gt > 0) * 1,
    #     (seg_whole_method > 0) * 1,
    #     show_ground_truth=False,
    #     # save_path="results/brain_seg_module1.pdf",
    # )
    # plt.show()