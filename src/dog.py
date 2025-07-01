#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Input:
# File from : https://github.com/neurolabusc/PyDog/blob/main/dog.py

#  input NIfTI filename, e.g. DWI.nii
#  full-width half maximum, e.g. 4
#Output
#  Binary NIfTI image with 'z' prefix
#Example Usage
# python dog.py ./DWI.nii 4

import nibabel as nib
from scipy.ndimage import (
    gaussian_filter1d,
    distance_transform_edt
)
import numpy as np
import scipy.stats as st
import os
import sys
import math
#skimage package is "scikit-image"
import skimage
import warnings
# from nil

def clamp(low, high, value):
    """bound an integer to a range
    Parameters
    ----------
    low : int
    high : int
    value : int
    Returns
    -------
    result : int
    """
    return max(low, min(high, value))

def dehaze(img, level, verbose=0):
    """use Otsu to threshold https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html
        n.b. threshold used to mask image: dark values are zeroed, but result is NOT binary
    Parameters
    ----------
    img : Niimg-like object
        Image(s) to run DoG on (see :ref:`extracting_data`
        for a detailed description of the valid input types).
    level : int
        value 1..5 with larger values preserving more bright voxels
        dark_classes/total_classes
            1: 3/4
            2: 2/3
            3: 1/2
            4: 1/3
            5: 1/4
    verbose : :obj:`int`, optional
        Controls the amount of verbosity: higher numbers give more messages
        (0 means no messages). Default=0.
    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image`
    """
    fdata = img.get_fdata()
    level = clamp(1, 5, level)
    n_classes = abs(3 - level) + 2
    dark_classes = 4 - level
    dark_classes = clamp(1, 3, dark_classes)
    thresholds = skimage.filters.threshold_multiotsu(fdata, n_classes)
    thresh = thresholds[dark_classes - 1]
    if verbose > 0:
        print("Zeroing voxels darker than {}".format(thresh))
    fdata[fdata < thresh] = 0
    return fdata

# https://github.com/nilearn/nilearn/blob/1607b52458c28953a87bbe6f42448b7b4e30a72f/nilearn/image/image.py#L164
def _smooth_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):
    """Smooth images by applying a Gaussian filter.
    Apply a Gaussian filter along the three first dimensions of `arr`.
    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        4D array, with image number as last dimension. 3D arrays are also
        accepted.
    affine : :class:`numpy.ndarray`
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
        If `fwhm='fast'`, the affine is not used and can be None.
    fwhm : scalar, :class:`numpy.ndarray`/:obj:`tuple`/:obj:`list`, 'fast' or None, optional
        Smoothing strength, as a full-width at half maximum, in millimeters.
        If a nonzero scalar is given, width is identical in all 3 directions.
        A :class:`numpy.ndarray`, :obj:`tuple`, or :obj:`list` must have 3 elements,
        giving the FWHM along each axis.
        If any of the elements is zero or None, smoothing is not performed
        along that axis.
        If  `fwhm='fast'`, a fast smoothing will be performed with a filter
        [0.2, 1, 0.2] in each direction and a normalisation
        to preserve the local average value.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed).
    ensure_finite : :obj:`bool`, optional
        If True, replace every non-finite values (like NaNs) by zero before
        filtering. Default=True.
    copy : :obj:`bool`, optional
        If True, input array is not modified. True by default: the filtering
        is not performed in-place. Default=True.
    Returns
    -------
    :class:`numpy.ndarray`
        Filtered `arr`.
    Notes
    -----
    This function is most efficient with arr in C order.
    """
    # Here, we have to investigate use cases of fwhm. Particularly, if fwhm=0.
    # See issue #1537
    if isinstance(fwhm, (int, float)) and (fwhm == 0.0):
        warnings.warn("The parameter 'fwhm' for smoothing is specified "
                      "as {0}. Setting it to None "
                      "(no smoothing will be performed)"
                      .format(fwhm))
        fwhm = None
    if arr.dtype.kind == 'i':
        if arr.dtype == np.int64:
            arr = arr.astype(np.float64)
        else:
            arr = arr.astype(np.float32)  # We don't need crazy precision.
    if copy:
        arr = arr.copy()
    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0
    if isinstance(fwhm, str) and (fwhm == 'fast'):
        arr = _fast_smooth_array(arr)
    elif fwhm is not None:
        fwhm = np.asarray([fwhm]).ravel()
        fwhm = np.asarray([0. if elem is None else elem for elem in fwhm])
        affine = affine[:3, :3]  # Keep only the scale part.
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        #n.b. FSL specifies blur in sigma, SPM in FWHM
        # FWHM = sigma*sqrt(8*ln(2)) = sigma*2.3548.
        #convert fwhm to sd in voxels see https://github.com/0todd0000/spm1d
        fwhmvox = fwhm / vox_size
        sd = fwhmvox / math.sqrt(8 * math.log(2))
        for n, s in enumerate(sd):
            if s > 0.0:
                gaussian_filter1d(arr, s, output=arr, axis=n)
    return arr

def binary_zero_crossing(fdata):
    """binarize (negative voxels are zero)
    Parameters
    ----------
    fdata : numpy.memmap from Niimg-like object
    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image`
    """
    edge = np.where(fdata > 0.0, 1, 0)
    edge = distance_transform_edt(edge)
    edge[edge > 1] = 0
    edge[edge > 0] = 1
    edge = edge.astype('uint8')
    return edge

def difference_of_gaussian(fdata, affine, fwhmNarrow, verbose=0):
    """Apply Difference of Gaussian (DoG) filter.
    https://en.wikipedia.org/wiki/Difference_of_Gaussians
    https://en.wikipedia.org/wiki/Marr–Hildreth_algorithm
    D. Marr and E. C. Hildreth. Theory of edge detection. Proceedings of the Royal Society, London B, 207:187-217, 1980
    Parameters
    ----------
    fdata : numpy.memmap from Niimg-like object
    affine : :class:`numpy.ndarray`
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
    fwhmNarrow : int
        Narrow kernel width, in millimeters. Is an arbitrary ratio of wide to narrow kernel.
            human cortex about 2.5mm thick
            Large values yield smoother results
    verbose : :obj:`int`, optional
        Controls the amount of verbosity: higher numbers give more messages
        (0 means no messages). Default=0.
    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image`
    """

    #Hardcode 1.6 as ratio of wide versus narrow FWHM
    # Marr and Hildreth (1980) suggest narrow to wide ratio of 1.6
    # Wilson and Giese (1977) suggest narrow to wide ratio of 1.5
    fwhmWide = fwhmNarrow * 1.6
    #optimization: we will use the narrow Gaussian as the input to the wide filter
    fwhmWide = math.sqrt((fwhmWide*fwhmWide) - (fwhmNarrow*fwhmNarrow))
    if verbose > 0:
        print('Narrow/Wide FWHM {} / {}'.format(fwhmNarrow, fwhmWide))
    imgNarrow = _smooth_array(fdata, affine, fwhmNarrow)
    imgWide = _smooth_array(imgNarrow, affine, fwhmWide)
    img = imgNarrow - imgWide
    img = binary_zero_crossing(img)
    return img

def dog_img(img, fwhm, verbose=0):
    """Find edges of a NIfTI image using the Difference of Gaussian (DoG).
    Parameters
    ----------
    img : Niimg-like object
        Image(s) to run DoG on (see :ref:`extracting_data`
        for a detailed description of the valid input types).
    fwhm : int
    	Edge detection strength, as a full-width at half maximum, in millimeters.
    verbose : :obj:`int`, optional
        Controls the amount of verbosity: higher numbers give more messages
        (0 means no messages). Default=0.
    Returns
    -------
    :class:`nibabel.nifti1.Nifti1Image`
    """

    if verbose > 0:
        print('Input intensity range {}..{}'.format(np.nanmin(img), np.nanmax(img)))
        print('Image shape {}x{}x{}'.format(img.shape[0], img.shape[1], img.shape[2]))

    dog_fdata = dehaze(img, 3, verbose)
    dog = difference_of_gaussian(dog_fdata, img.affine, fwhm, verbose)
    out_img = nib.Nifti1Image(dog, img.affine, img.header)
    #update header
    out_img.header.set_data_dtype(np.uint8)
    out_img.header['intent_code'] = 0
    out_img.header['scl_slope'] = 1.0
    out_img.header['scl_inter'] = 0.0
    out_img.header['cal_max'] = 0.0
    out_img.header['cal_min'] = 0.0
    # pth, nm = os.path.split(fnm)
    # if not pth:
    #     pth = '.'
    # outnm = pth + os.path.sep + 'z' + nm
    # nib.save(out_img, outnm)
    return out_img


if __name__ == '__main__':
    """Apply Gaussian smooth to image
    Parameters
    ----------
    fnm : str
        NIfTI image to convert
    """
    if len(sys.argv) < 2:
        print('No filename provided: I do not know which image to convert!')
        sys.exit()
    fnm = sys.argv[1]
    img = nib.load(fnm)
    img.header.set_data_dtype(np.float32)
    dog_imported_img = dog_img(img, fwhm=3, verbose=1)
    pth, nm = os.path.split(fnm)
    if not pth:
        pth = '.'
    outnm = pth + os.path.sep + 'z' + nm
    nib.save(dog_imported_img, outnm)