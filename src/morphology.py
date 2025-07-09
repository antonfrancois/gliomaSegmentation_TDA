"""---------------------------------------------------------------------------------------------------------------------

Train-Free Segmentation in MRI with Cubical Persistent Homology
Anton François & Raphaël Tinarrage
See the repo at https://github.com/antonfrancois/gliomaSegmentation_TDA and article at https://arxiv.org/abs/2401.01160


------------------------------------------------------------------------------------------------------------------------

Functions:
    argmax_image
    get_component
    get_highest_component
    get_largest_component

---------------------------------------------------------------------------------------------------------------------
"""

# Third-party imports.
import numpy as np
import skimage


def argmax_image(img):
    """Return the position of the maximum value in a 3D image."""
    return np.unravel_index(img.argmax(), img.shape)


def get_component(img, pos, t):
    """
    Get the connected component of the voxel pos = (x,y,z) at time t. The output is a binary image. The background
    value of the image must be 0 (as conventional).
    """
    # Check if the voxel pos is active at time t.
    assert (
        img[pos[0], pos[1], pos[2]] >= t
    ), "In get_component: voxel not active at time t."
    # Threshold image above t.
    img_t = (img >= t) * 1
    # If the voxel pos is not active at time t.
    if img_t[pos[0], pos[1], pos[2]] == 0:
        component = img * 0
    # If the voxel pos is active at time t.
    else:
        labels = skimage.measure.label(img_t, background=0)
        label_pos = labels[pos[0], pos[1], pos[2]]
        # Check if the label is background.
        assert label_pos != 0, "In get_component: voxel is in the background."
        # Get the connected component of the voxel pos.
        component = (labels == label_pos) * 1
    return component


def get_highest_component(img, t):
    """Return the connected component of the image at time t containing the brightest voxel."""
    # Get position of the maximum value in the image.
    pos = argmax_image(img)
    # Get its connected component.
    component = get_component(img, pos, t)
    return component


def get_largest_component(img, t, verbose=False):
    """Return the largest connected component of the image at time t."""
    # Threshold image above t.
    img_t = (img >= t) * 1
    # Get the connected components of the thresholded image.
    labels = skimage.measure.label(img_t, background=0)
    nb_labels = np.max(labels)
    # Compute the size of each connected component.
    components_size = [0] + [np.sum(labels == i) for i in range(1, nb_labels)]
    # Get largest component.
    idx = np.argmax(components_size)
    component = (labels == idx) * 1
    # Print comment if required.
    if verbose:
        print("There are", np.max(components_size), "connected components at time", t)
    return component
