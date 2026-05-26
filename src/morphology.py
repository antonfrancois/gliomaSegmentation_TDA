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
    get_highest_component
    sphere_dice
    get_best_component
    suggest_t_sphericity
    get_most_spherical_component
    _component_surface_areas_from_labels
    sphericity_from_mask

---------------------------------------------------------------------------------------------------------------------
"""

# Third-party imports.
import numpy as np
import skimage
import scipy.ndimage as ndi
import math
import scipy


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

def sphere_dice(mask, spacing=(1, 1, 1)):
    mask = mask.astype(bool)
    dz, dy, dx = map(float, spacing)

    V = mask.sum() * dz * dy * dx
    r = (3 * V / (4 * np.pi)) ** (1 / 3)

    com_zyx = np.array(ndi.center_of_mass(mask))
    com_xyz = com_zyx[::-1] * np.array([dx, dy, dz])

    Z, Y, X = mask.shape
    zz, yy, xx = np.indices(mask.shape)
    pts = np.stack([xx * dx, yy * dy, zz * dz], axis=-1)
    d2 = np.sum((pts - com_xyz) ** 2, axis=-1)

    sphere = d2 <= r * r

    inter = np.logical_and(mask, sphere).sum()
    if mask.sum() + sphere.sum() > 0:
        return 2 * inter / (mask.sum() + sphere.sum())
    else:
        return 0


def get_best_component(img_flair, seg_whole_largest, t, min_sphere=0.5, min_size=10000):
    # Compute dice sphere.
    ds = sphere_dice(seg_whole_largest)
    if ds < min_sphere:
        seg_whole_sphere = get_most_spherical_component(img_flair, t, min_size=min_size)
        seg_whole_sphere = scipy.ndimage.binary_fill_holes(seg_whole_sphere)
        # If the sphericity is still low, go back to previous segmentation.
        ds = sphere_dice(seg_whole_sphere)
        if ds >= min_sphere:
            return seg_whole_sphere
        else:
            return seg_whole_largest
    else:
        return seg_whole_largest


def suggest_t_sphericity(img, pos, vmin, vmax, min_size, ticks=100, method="argmax"):
    vals = np.linspace(vmin, vmax, ticks)
    # Build suggestion curve.
    sphericities = []
    voxels_count = []
    for t in vals:
        if img[pos[0], pos[1], pos[2]] < t:
            sphericities.append(0)
            voxels_count.append(0)
        else:
            mask = get_component(img, pos, t)
            sphericities.append(sphericity_from_mask(mask))
            voxels_count.append(np.sum(mask))
    sphericities = np.array(sphericities)
    sphericities /= sphericities.max()
    voxels_count = np.array(voxels_count)

    # Derivative
    sphericities_dt = sphericities[:-1] - sphericities[1:]
    sphericities_dt = np.insert(sphericities_dt, 0, 0, axis=0)
    sphericities_dt -= sphericities_dt.min()  # just to plot
    sphericities_dt /= sphericities_dt.max()  # just to plot

    # Also, the component must have at least min_size voxels (if possible).
    if voxels_count[0] < min_size:
        return vals[0]
    sphericities[voxels_count < min_size] = -np.inf
    sphericities_dt[voxels_count < min_size] = np.inf

    # Criteria.
    if method == "argmax":
        sphere_i = np.argmax(sphericities)
    elif method == "argmin_dt":
        sphere_i = np.argmin(sphericities_dt)
    elif method == "both":
        sphere_i = int((np.argmax(sphericities) + np.argmin(sphericities_dt)) / 2)
    sphere_t = vals[sphere_i]
    return sphere_t


def get_most_spherical_component(
    img,
    t,
    min_size=1,
    connectivity=1,
    return_phi=False,
    relax_step=5000,
    min_size_floor=1,
):
    """
    Return the most spherical connected component in a 3D image thresholded at t.

    If no component satisfies vol >= min_size, relax the constraint:
        min_size, min_size-relax_step, min_size-2*relax_step, ...
    until at least one component qualifies or min_size_floor is reached.

    Uses sphericity: phi = π^(1/3) (6V)^(2/3) / A, with A from exposed voxel faces.
    connectivity=1 -> 6-neighborhood, connectivity=2 -> 18, connectivity=3 -> 26
    """
    img_t = img >= t

    struct = ndi.generate_binary_structure(3, connectivity)
    labels, nb = ndi.label(img_t, structure=struct)
    if nb == 0:
        out = np.zeros_like(img_t, dtype=np.uint8)
        return (out, -np.inf) if return_phi else out

    # Volume per label
    vol = np.bincount(labels.ravel()).astype(np.int64)
    vol[0] = 0

    # Surface area per label (vectorized)
    surf = _component_surface_areas_from_labels(labels).astype(np.float64)
    surf[0] = np.inf  # exclude background

    const = math.pi ** (1.0 / 3.0)
    phi = np.full_like(surf, -np.inf, dtype=np.float64)

    # Relax min_size until something is valid
    cur_min = int(min_size)
    floor = int(min_size_floor)

    while True:
        valid = (vol >= cur_min) & (surf > 0) & np.isfinite(surf)
        if np.any(valid):
            phi[valid] = const * (6.0 * vol[valid]) ** (2.0 / 3.0) / surf[valid]
            break

        if cur_min <= floor:
            out = np.zeros_like(img_t, dtype=np.uint8)
            return (out, -np.inf) if return_phi else out

        cur_min = max(floor, cur_min - int(relax_step))

    best_label = int(np.argmax(phi))
    comp = (labels == best_label).astype(np.uint8)

    return (comp, float(phi[best_label])) if return_phi else comp


def _component_surface_areas_from_labels(labels: np.ndarray) -> np.ndarray:
    """
    Vectorized surface area (exposed voxel faces) per label for 3D labels (0=background).
    Unit voxel faces, 6-connectivity notion.
    """
    labels = np.asarray(labels)
    assert labels.ndim == 3
    nlab = int(labels.max())
    if nlab == 0:
        return np.zeros(1, dtype=np.int64)

    surf = np.zeros(nlab + 1, dtype=np.int64)

    # Internal boundaries (between neighboring voxels)
    def add_axis_faces(a, b):
        diff = a != b
        la = a[diff]
        lb = b[diff]
        if la.size:
            la = la[la > 0]
            if la.size:
                surf[:] += np.bincount(la, minlength=surf.size)
        if lb.size:
            lb = lb[lb > 0]
            if lb.size:
                surf[:] += np.bincount(lb, minlength=surf.size)

    # x axis (0)
    add_axis_faces(labels[:-1, :, :], labels[1:, :, :])
    # y axis (1)
    add_axis_faces(labels[:, :-1, :], labels[:, 1:, :])
    # z axis (2)
    add_axis_faces(labels[:, :, :-1], labels[:, :, 1:])

    # Outer boundary faces (volume touching the image boundary)
    for sl in (
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    ):
        v = sl.ravel()
        v = v[v > 0]
        if v.size:
            surf[:] += np.bincount(v, minlength=surf.size)

    return surf


def sphericity_from_mask(mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    m = np.asarray(mask).astype(bool)
    Vvox = int(m.sum())
    if Vvox == 0:
        return 0.0

    dz, dy, dx = map(float, spacing)

    # Physical volume
    V = Vvox * dz * dy * dx

    # Count shared faces (adjacent voxel pairs) along each axis
    adj_x = np.logical_and(m[:-1, :, :], m[1:, :, :]).sum()
    adj_y = np.logical_and(m[:, :-1, :], m[:, 1:, :]).sum()
    adj_z = np.logical_and(m[:, :, :-1], m[:, :, 1:]).sum()
    shared_faces = int(adj_x + adj_y + adj_z)

    # Exposed faces count (unit cubes): 6*Vvox - 2*shared_faces
    exposed_faces = 6 * Vvox - 2 * shared_faces

    # Convert exposed faces to physical surface area
    # Faces perpendicular to x have area dy*dz, perpendicular to y have dx*dz, perpendicular to z have dx*dy.
    # We need counts per axis of exposed faces, not just total. Compute directly:

    # Exposed faces perpendicular to x (between x-neighbors and volume boundary)
    # Count boundaries where occupancy changes across x
    ex_x = (m[:-1, :, :] != m[1:, :, :]).sum()
    # Add outer boundary faces at x=0 and x=end
    ex_x += m[0, :, :].sum() + m[-1, :, :].sum()

    # Exposed faces perpendicular to y
    ex_y = (m[:, :-1, :] != m[:, 1:, :]).sum()
    ex_y += m[:, 0, :].sum() + m[:, -1, :].sum()

    # Exposed faces perpendicular to z
    ex_z = (m[:, :, :-1] != m[:, :, 1:]).sum()
    ex_z += m[:, :, 0].sum() + m[:, :, -1].sum()

    A = float(ex_x) * (dy * dz) + float(ex_y) * (dx * dz) + float(ex_z) * (dx * dy)
    if A <= 0:
        return 0.0

    phi = (math.pi ** (1.0 / 3.0)) * ((6.0 * V) ** (2.0 / 3.0)) / A
    return float(phi)