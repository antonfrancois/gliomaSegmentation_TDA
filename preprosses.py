import numpy as np
from misc import *

def select_nonoverlapping_patches(image, num_patches_depth, num_patches_height, num_patches_width):
    """
    Selects non-overlapping patches from a 3D image using a vectorized approach, by specifying the number of patches in each dimension.
    :return: torch tensor of dimensions: (num_patches_depth, num_patches_height, num_patches_width, patch_size[0], patch_size[1], patch_size[2])
    """
    image_depth, image_height, image_width = image.shape[:3]

    # Calculate the patch size and stride in each dimension
    patch_size_depth = image_depth // num_patches_depth
    patch_size_height = image_height // num_patches_height
    patch_size_width = image_width // num_patches_width
    stride_depth = patch_size_depth
    stride_height = patch_size_height
    stride_width = patch_size_width
    print(f"stride = {stride_depth,stride_height,stride_width}")

    # Generate the indices for the top-left corner of each patch
    patch_indices = np.meshgrid(
        np.arange(0, image_depth - patch_size_depth + 1, stride_depth),
        np.arange(0, image_height - patch_size_height + 1, stride_height),
        np.arange(0, image_width - patch_size_width + 1, stride_width)
    )
    print(f"patch_indices ,1= {patch_indices[0].shape}")
    # Flatten the indices into a single tensor
    patch_indices = np.stack(patch_indices, axis=-1)
    print(f"patch_indices ,2= {patch_indices.shape}")
    patch_indices = patch_indices.reshape(-1, 3)
    print(f"patch_indices ,3= {patch_indices.shape}")
    print(patch_indices[50])

    # Extract the patches using the indices
    patches = np.empty(
        (num_patches_depth*num_patches_height*num_patches_width,
         patch_size_depth,patch_size_height,patch_size_width)
    )
    for i,index in enumerate(patch_indices):
        patches[i] =  image[
              index[0]:index[0]+stride_depth,
              index[1]:index[1]+stride_height,
              index[2]:index[2]+stride_width
              ]
    print('patches : ',patches.shape)
    # patches = patches.reshape(num_patches_depth, num_patches_height, num_patches_width, patch_size_depth, patch_size_height, patch_size_width)

    return patches,

def reconstruct_image_from_patches(patches,num_patches, image_shape):
    image_depth, image_height, image_width = image_shape
    num_patches_depth, num_patches_height, num_patches_width = num_patches

    # Calculate the patch size and stride in each dimension
    patch_size_depth = image_depth // num_patches_depth
    patch_size_height = image_height // num_patches_height
    patch_size_width = image_width // num_patches_width
    stride_depth = patch_size_depth
    stride_height = patch_size_height
    stride_width = patch_size_width

    # Generate the indices for the top-left corner of each patch
    patch_indices = np.meshgrid(
        np.arange(0, image_depth - patch_size_depth + 1, stride_depth),
        np.arange(0, image_height - patch_size_height + 1, stride_height),
        np.arange(0, image_width - patch_size_width + 1, stride_width)
    )
    patch_indices = np.stack(patch_indices, axis=-1)
    patch_indices = patch_indices.reshape(-1, 3)

    # Create a zero-initialized image of the required size
    image = np.zeros((image_depth, image_height, image_width))

    # Fill the image with the patches
    for i,index in enumerate(patch_indices):
        image[
            index[0]:index[0]+stride_depth,
            index[1]:index[1]+stride_height,
            index[2]:index[2]+stride_width
        ] += patches[i]

    return image

@time_it
def patch_normalisation(image,num_patches = (8,8,8)):
    num_patches_depth,num_patches_height,num_patches_width = num_patches

    # Extract the patches
    patches, = select_nonoverlapping_patches(image, num_patches_depth, num_patches_height, num_patches_width)

    ## Renormalisation
    mean_im, var_im = image[image>0].mean(), image[image>0].std()
    print(mean_im, var_im)
    patches_norm = np.zeros(patches.shape)
    for i,p in enumerate(patches):
        if p.mean() >0:
            patches_norm[i] = (p - mean_im)/var_im

    image_norm = reconstruct_image_from_patches(
        patches_norm,
        (num_patches_depth,num_patches_height,num_patches_width),
        image.shape
    )
    # Put back image between 0 and 1
    print(f'normalise min,max = {image_norm.min()}, {image_norm.max()}')
    image_norm[image_norm == 0] = image_norm.min()
    image_norm = (image_norm - image_norm.min())/(image_norm.max() - image_norm.min())

    return image_norm

def equalize_image(image):
    # Calculate the cumulative distribution function (CDF)
    image_flat = image.flatten()
    im_max = 1 if image.max() < 1 else image.max()
    print(f"IMMAX = {im_max}")
    hist, bins = np.histogram(image_flat[image_flat>0], bins=256, range=(0,im_max))
    cdf = np.cumsum(hist)

    # Normalize the CDF to span the entire dynamic range
    cdf_normalized = cdf / cdf[-1]

    # Create a lookup table for the intensity values
    lut = np.interp(image_flat, bins[:-1], cdf_normalized)

    # Apply the lookup table to the image to equalize it
    image_equalized = lut.reshape(image.shape)

    return image_equalized