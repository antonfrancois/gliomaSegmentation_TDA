"""---------------------------------------------------------------------------------------------------------------------

Train-Free Segmentation in MRI with Cubical Persistent Homology
Anton François & Raphaël Tinarrage
See the repo at https://github.com/antonfrancois/gliomaSegmentation_TDA and article at https://arxiv.org/abs/2401.01160

------------------------------------------------------------------------------------------------------------------------

Functions and classes:
    resize_image
    nib_normalize
    open_nib
    parse_brats

---------------------------------------------------------------------------------------------------------------------
"""

# Standard imports.
import os
from pathlib import Path
from warnings import warn
import csv
import tomllib

# Third-party imports.
import numpy as np
from nibabel import load as nib_load
from skimage.exposure import match_histograms

# import torch
# import nrrd

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# Configuration file is expected at repository root.
CONFIG_PATH = Path(ROOT_DIRECTORY).parent / "config.toml"


def _default_data_paths():
    root = Path(ROOT_DIRECTORY).parent
    return {
        "template": root / "data" / "template" / "sri_spm8" / "templates",
        "brats": root / "data" / "brats",
        "brats_2021": root / "data" / "brats_2021",
        "brats_2025": Path(ROOT_DIRECTORY)
        / "/data/brats/BraTS2025-GLI-PRE-Challenge-TrainingData",
        "bratsreg_2022_train": root / "data" / "bratsreg_2022" / "BraTSReg_Training_Data_v3",
        "bratsreg_2022_valid": root / "data" / "bratsreg_2022" / "BraTSReg_Validation_Data",
        "bratsreg_2022_seg_train": root / "data" / "bratsreg_2022" / "Train_seg",
        "bratsreg_2022_seg_valid": root / "data" / "bratsreg_2022" / "Valid_seg",
    }


def _load_data_paths():
    paths = _default_data_paths()
    if CONFIG_PATH.exists():
        try:
            cfg = tomllib.loads(CONFIG_PATH.read_text())
            user_paths = cfg.get("data_paths", {})
            for key, value in user_paths.items():
                if value:
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = CONFIG_PATH.parent / candidate
                    paths[key] = candidate
        except Exception as exc:  # pragma: no cover - defensive, keep behavior
            warn(f"Could not read config at {CONFIG_PATH}: {exc}")
    return paths


DATA_PATHS = _load_data_paths()


def resize_image(image, scale_factor):
    Ishape = image[0].shape[2:]
    Ishape_D = tuple([int(s * scale_factor) for s in Ishape])
    id_grid = make_regular_grid(Ishape_D, dx_convention="2square").to(image[0].device)
    i_s = []
    for i in image:
        i_s.append(
            torch.nn.functional.grid_sample(
                i.to(image[0].device), id_grid, **DLT_KW_GRIDSAMPLE
            )
        )
    return i_s


def nib_normalize(img, method="mean"):
    if method == "mean" or method is None:
        print("using mean")
        img = (img - img.mean()) / (img.std() + 1e-30)
        img = np.clip(img, a_min=0, a_max=1)
    elif method == "min_max":
        img += img.min()
        img /= img.max()
    else:
        raise ValueError(f"method must be 'mean' or 'min_max' got {method}")
    return img


def open_nib(
    folder_name, irm_type, data_base, format=".nii.gz", normalize=True, to_torch=True
):
    """
    to get a nib, use normalise = False and to_torch = False
    """
    if data_base in ("brats", "brats_2021", "bratsreg_2022", "brats_2025"):
        base_path = DATA_PATHS[data_base]
    else:
        base_path = Path(data_base)

    folder_path = Path(base_path) / folder_name

    if data_base != "brats_2025":
        file_path = folder_path / f"{folder_name}_{irm_type}{format}"
    else:
        if irm_type == "flair":
            irm_type = "t2f"
        elif irm_type == "t1ce":
            irm_type = "t1c"
        file_path = folder_path / f"{folder_name}-{irm_type}{format}"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not open file at '{file_path}'. Please update dataset paths in {CONFIG_PATH}"
        )

    img_nib = nib_load(str(file_path))
    img = img_nib.get_fdata()
    method = None
    if isinstance(normalize, str):
        method = normalize
        normalize = True
    if normalize:
        # print(f">I am normalizinf {normalize}")
        img = nib_normalize(img, method=method)
    if to_torch:
        return torch.Tensor(img)[None, None]
    else:
        return img_nib


class parse_brats:

    def __init__(
        self,
        brats_list=None,
        template_folder=None,
        brats_folder=None,
        get_template=True,
        modality="T1",
        device="cpu",
    ):
        """

        :param brats_list: list of stings containing the name of the folders
        :param template_folder: path to template folder
        :param brats_folder: path to brats db
        :param modality: modality of the IRM ex: `'T1'`,`'T2'`
        """
        self.flag_brats_2021 = False
        self.flag_bratsReg_2022 = False
        self.flag_brats_2025 = False
        if template_folder is None:
            template_folder = DATA_PATHS["template"]
        if brats_folder is not None:
            brats_folder = Path(brats_folder)
        if brats_folder is None or "2021" in str(brats_folder):
            self.brats_folder = DATA_PATHS["brats_2021"]
            self.flag_brats_2021 = True
        elif "2022_valid" in str(brats_folder):
            self.brats_folder = DATA_PATHS["bratsreg_2022_valid"]
            self.flag_bratsReg_2022 = True
        elif "2022" in str(brats_folder):
            self.brats_folder = DATA_PATHS["bratsreg_2022_train"]
            # print(f"\n!!!!!! {self.flag_bratsReg_2022} <<<<<<<<\n")
            self.flag_bratsReg_2022 = True
        elif "2025" in str(brats_folder):
            self.brats_folder = DATA_PATHS["brats_2025"]
            self.flag_brats_2025 = True
        else:
            self.brats_folder = Path(brats_folder)

        # TODO : check that the list is correct by parsing with os.get_dir ...
        if brats_list is None:
            if self.flag_brats_2021:
                warn(
                    "It is not recommended to set brats_list to None with BraTS2021"
                    "database. It can lead to errors because ventricule segmentations "
                    "where not made for all data."
                )
            self._make_brats_list(self.brats_folder)
        else:
            self.brats_list = brats_list
        self.modality = modality
        self.device = device
        self.flag_get_template = get_template
        if not self.flag_bratsReg_2022 and get_template:
            template_folder = Path(template_folder)
            template_brain_path = template_folder / f"{modality}_brain.nii"
            template_seg_path = template_folder / "seg_sri24.mgz"
            if not template_brain_path.exists() or not template_seg_path.exists():
                raise FileNotFoundError(
                    f"Template files missing under '{template_folder}'. Please update dataset paths in {CONFIG_PATH}"
                )
            template_nib = nib_load(str(template_brain_path))
            self.template_affine = template_nib.affine
            self.template_img = template_nib.get_fdata()[:, ::-1, :, 0]

            self.template_seg = nib_load(str(template_seg_path)).get_fdata()

    def _make_brats_list(self, folder):
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Dataset folder '{folder}' not found. Please update dataset paths in {CONFIG_PATH}"
            )
        self.brats_list = []
        for obj in os.listdir(folder):
            if self.flag_bratsReg_2022 and "BraTSReg_" in obj:
                self.brats_list.append(obj)
            if self.flag_brats_2021 and "BraTS2021" in obj:
                self.brats_list.append(obj)
            if self.flag_brats_2025 and "BraTS-GLI" in obj:
                self.brats_list.append(obj)

    def get_template(self, normalised=True):
        if normalised:
            img_norm = (self.template_img - self.template_img.min()) / (
                self.template_img.max() - self.template_img.min()
            )
            return torch.Tensor(img_norm, device=self.device)[None, None]
        else:
            return torch.Tensor(self.template_img.copy(), device=self.device)

    def get_template_vesi(self):
        vesi_seg = torch.zeros(self.template_seg.shape)
        vesi_seg[self.template_seg == 4] = 1
        vesi_seg[self.template_seg == 43] = 1
        return vesi_seg.flip(1)

    def get_template_whiteMatter(self):
        whmtr_seg = torch.zeros(self.template_seg.shape)
        whmtr_seg[self.template_seg == 2] = 1
        whmtr_seg[self.template_seg == 41] = 1
        return whmtr_seg.flip(1)

    def get_vesicule_seg(self, index, mask_correction=None):
        path = Path(DATA_PATHS["brats_2021"])
        brats_name = self.brats_list[index]
        brats_img_size = (240, 240, 155)
        seg_path = path / brats_name / f"{brats_name}_segV.seg.nrrd"
        if not seg_path.exists():
            raise FileNotFoundError(
                f"Could not open file at '{seg_path}'. Please update dataset paths in {CONFIG_PATH}"
            )
        vesi, header = nrrd.read(str(seg_path))
        space_origin = header["space origin"]
        vesi_s = vesi.shape
        if vesi_s == brats_img_size:
            return torch.tensor(vesi)[None, None]

        vesi_pad = np.zeros(brats_img_size)
        vesi_pad[
            int(space_origin[0]) : int(space_origin[0] + vesi_s[0]),
            int(space_origin[1]) : int(space_origin[1] + vesi_s[1]),
            int(space_origin[2]) : int(space_origin[2] + vesi_s[2]),
        ] = vesi
        # vesi = open_nib(brats_name,'segV','brats_2021',
        #                   format='.nii.gz',normalize=False,to_torch=True)
        # img_1 = torch.zeros(vesi.shape)
        # img_1[vesi==4] = 1
        # img_1[vesi==43] = 1
        if not mask_correction is None:
            vesi[mask_correction > 0] = 0
        return torch.tensor(vesi_pad)[None, None]

    def get_edges(self, index, modality):
        nib_im, seg = self.__call__(
            index, to_torch=False, modality=modality, get_nib=True
        )
        edges_im_nib = edge_Dog(nib_im, fwhm=3)
        edges_seg_nib = edge_Dog(seg, fwhm=3)
        return edges_im_nib.get_fdata(), edges_seg_nib.get_fdata()

    def _read_landmarks_csv_(self, file):
        with open(file) as csv_f:
            csv_reader = csv.DictReader(csv_f, delimiter=",")
            landmarks = []
            for row in csv_reader:
                try:
                    listv = [float(row["Z"]), 239 + float(row["Y"]), float(row["X"])]
                except KeyError:
                    listv = [float(row[" Z"]), 239 + float(row[" Y"]), float(row[" X"])]
                landmarks.append(listv)
        return torch.Tensor(landmarks)

    def _get_landmarks(self, path, file_list):
        file_ldk_1 = [f for f in file_list if "_01_" in f and "_landmarks.csv" in f][0]
        # print(file_ldk_1)
        ldk_1 = self._read_landmarks_csv_(path + file_ldk_1)
        if "Training_Data" in path:
            file_ldk_0 = [
                f for f in file_list if "_00_" in f and "_landmarks.csv" in f
            ][0]
            # print(file_ldk_0)
            ldk_0 = self._read_landmarks_csv_(path + file_ldk_0)
            return (ldk_0, ldk_1)
        return (None, ldk_1)

    def _call_brats_2021_(self, index, to_torch, normalize=False, get_nib=False):
        brats_name = self.brats_list[index]
        gliom = open_nib(
            brats_name,
            self.modality.lower(),
            "brats_2021",
            normalize=False if get_nib else "min_max",
            to_torch=False,
        )
        segmentation_tumor = open_nib(
            brats_name, "seg", "brats_2021", normalize=False, to_torch=to_torch
        )
        if get_nib:
            return (gliom, segmentation_tumor)
        if to_torch:
            segmentation_tumor[segmentation_tumor == 2] = 0.5
            segmentation_tumor[segmentation_tumor == 4] = 1
        else:
            segmentation_tumor = segmentation_tumor.get_fdata()
        # gliom = nib.load("/Users/maillard/Downloads/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_00008/BraTS2021_000_T1.nii.gz")
        # histogram normalisation

        gliom_img = gliom.get_fdata()
        if self.flag_get_template and normalize:
            gliom_img[gliom_img > 0] = match_histograms(
                gliom_img[gliom_img > 0], self.template_img[self.template_img > 0]
            )
            gliom_img = (gliom_img - gliom_img.min()) / (
                gliom_img.max() - gliom_img.min()
            )

        if to_torch:
            gliom_img = torch.Tensor(gliom_img, device=self.device)[None, None]
        return (gliom_img, segmentation_tumor)

    def _call_bratsReg_2022(self, index, to_torch, scale=0, rigidly_reg=False):
        """

        :param index: (int)
        :return: The two brains data to get.
        !!! Do not use rigidly_reg it does not work
        """
        path = Path(self.brats_folder)

        folder_name = self.brats_list[index]
        case_path = path / folder_name
        if not case_path.exists():
            raise FileNotFoundError(
                f"Case folder '{case_path}' not found. Please update dataset paths in {CONFIG_PATH}"
            )
        file_list = os.listdir(case_path)
        file_0 = [
            f for f in file_list if "_00_" in f and self.modality.lower() + "." in f
        ][0]
        if rigidly_reg:
            file_1 = [
                f
                for f in file_list
                if "_01_" in f and self.modality.lower() in f and "resampled" in f
            ][0]
        else:
            file_1 = [
                f for f in file_list if "_01_" in f and self.modality.lower() + "." in f
            ][0]
        img_nib_0 = nib_load(str(case_path / file_0))
        self.affine = img_nib_0.affine
        img_0 = img_nib_0.get_fdata()

        img_nib_1 = nib_load(str(case_path / file_1))
        img_1 = img_nib_1.get_fdata()

        # img_1[img_1 > 0] = match_histograms(img_1[img_1 > 0], img_0[img_0 > 0])
        img_0 = (img_0 - img_0.min()) / (img_0.max() - img_0.min())
        img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min())
        # v_min, v_max = min(img_0.min(), img_1.min()), max(img_0.max(),img_1.max() )
        # v_min, v_max = img_0.min(),img_0.max()
        #
        # img_0 = (img_0 - v_min) / (v_max - v_min)
        # img_1 = (img_1 - v_min) / (v_max - v_min)

        landmarks = self._get_landmarks(path, file_list)

        # Segmentation !
        if "Training_Data" in str(path):
            seg_path = Path(DATA_PATHS["bratsreg_2022_seg_train"])
        elif "Validation" in str(path):
            seg_path = Path(DATA_PATHS["bratsreg_2022_seg_valid"])
        else:
            raise ValueError("Something went wrong.")
        seg_file_0 = seg_path / f"{folder_name}_seg_00_.nii.gz"
        seg_file_1 = seg_path / f"{folder_name}_seg_01_.nii.gz"
        for seg_file in (seg_file_0, seg_file_1):
            if not seg_file.exists():
                raise FileNotFoundError(
                    f"Could not open file at '{seg_file}'. Please update dataset paths in {CONFIG_PATH}"
                )

        seg_img_0 = nib_load(str(seg_file_0)).get_fdata()
        seg_img_0[seg_img_0 == 1] = 3
        seg_img_0[seg_img_0 == 2] = 1.5
        seg_img_0 = seg_img_0 / 3

        seg_img_1 = nib_load(str(seg_file_1)).get_fdata()
        seg_img_1[seg_img_1 == 1] = 3
        seg_img_1[seg_img_1 == 2] = 1.5
        seg_img_1 = seg_img_1 / 3

        # TODO : Check if we should normalize
        if to_torch:
            img_0 = torch.Tensor(img_0, device=self.device)[None, None]
            img_1 = torch.Tensor(img_1, device=self.device)[None, None]

            seg_img_0 = torch.Tensor(seg_img_0, device=self.device)[None, None]
            seg_img_1 = torch.Tensor(seg_img_1, device=self.device)[None, None]
            if scale != 1:
                img_0, img_1, seg_img_0, seg_img_1 = resize_image(
                    (img_0, img_1, seg_img_0, seg_img_1), scale
                )
                landmarks = (
                    landmarks[0] * scale if not landmarks[0] is None else None,
                    landmarks[1] * scale,
                )

        return img_0, img_1, seg_img_0, seg_img_1, landmarks

    def _call_brats_2025_(self, index, to_torch, normalize=False, get_nib=False):
        brats_name = self.brats_list[index]
        gliom = open_nib(
            brats_name,
            self.modality.lower(),
            "brats_2025",
            normalize=False if get_nib else "min_max",
            to_torch=False,
        )
        segmentation_tumor = open_nib(
            brats_name, "seg", "brats_2025", normalize=False, to_torch=to_torch
        )
        if get_nib:
            return (gliom, segmentation_tumor)
        if to_torch:
            segmentation_tumor[segmentation_tumor == 2] = 0.5
            segmentation_tumor[segmentation_tumor == 3] = 1
        else:
            segmentation_tumor = segmentation_tumor.get_fdata()

        gliom_img = gliom.get_fdata()
        if self.flag_get_template and normalize:
            gliom_img[gliom_img > 0] = match_histograms(
                gliom_img[gliom_img > 0], self.template_img[self.template_img > 0]
            )
            gliom_img = (gliom_img - gliom_img.min()) / (
                gliom_img.max() - gliom_img.min()
            )

        if to_torch:
            gliom_img = torch.Tensor(gliom_img, device=self.device)[None, None]
        return (gliom_img, segmentation_tumor)

    def __call__(
        self,
        index,
        to_torch=True,
        modality=None,
        scale=0,
        rigidly_reg=False,
        normalize=False,
        get_nib=False,
    ):
        """Open the brats folder in self.brats_list at the desired index

        :param index: must be int < len(brats_list)
        :param to_torch: (bool) return image as torch.Tensor if True, else a numpy array.
        :return: image at the index of the brats_list
        """
        if index >= len(self.brats_list):
            raise ValueError(
                f"You asked for a too high value, index is : {index} and len(brat_list) is : {len(self.brats_list)}"
            )
        if not modality is None:
            self.modality = modality
        if self.flag_brats_2021:
            return self._call_brats_2021_(
                index, to_torch, normalize=normalize, get_nib=get_nib
            )
        if self.flag_bratsReg_2022:
            return self._call_bratsReg_2022(
                index, to_torch, scale, rigidly_reg=rigidly_reg
            )
        if self.flag_brats_2025:
            return self._call_brats_2025_(
                index, to_torch, normalize=normalize, get_nib=get_nib
            )
        # source = nib.Nifti1Image(source_img, self.template_affine)
        # return source.get_fdata()
