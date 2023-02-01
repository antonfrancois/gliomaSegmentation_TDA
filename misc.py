from time import time
import os
import numpy as np
import scipy.ndimage as scipynd
from matplotlib.colors import ListedColormap

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

DLT_EDGES_THRSLD = .0625

# ---------- Metric -------------------------------------------

def DICE(seg_1,seg_2):
    prod_seg = seg_1 * seg_2
    sum_seg = seg_1 + seg_2
    return 2*prod_seg.sum() / sum_seg.sum()


# __________ matplotlib _______________________________________
cmap_segs = ListedColormap(
    [[0,0,0,0],
     'tab:red',
     'tab:blue',
     'tab:orange'
     ])
DLT_KW_GRIDSAMPLE = dict(padding_mode="border",
                         align_corners=True
                         )
DLT_KW_IMAGE = dict(cmap='gray',
                      # extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=0,vmax=1)
DLT_KW_SEG= dict(alpha=1,
                 cmap=cmap_segs,
                 interpolation='nearest'
                 )

def set_ticks_off(ax):
    try:
        ax.tick_params(
                axis='both',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                right=False,
                left=False,
                labelbottom=False, # labels along the bottom edge are off
                labelleft=False,
            )
    except AttributeError:
        for a in ax.ravel():
            set_ticks_off(a)

def imCmp(I1, I2, method='supperpose'):
    M, N = I1.shape
    if method == 'supperpose':
        return np.concatenate((I2[:, :, None], I1[:, :, None], np.zeros((M, N, 1))), axis=2)
        # return np.concatenate((I2[:, :, None],np.zeros((M, N, 1)), I1[:, :, None]), axis=2)
    elif method == 'substract':
        return I1 - I2
    elif method == 'supperpose weighted':
        abs_diff = np.abs(I1 - I2)[:,:,None]
        return 1- abs_diff/abs_diff.max() * np.concatenate((I2[:, :, None], I1[:, :, None], np.zeros((M, N, 1))), axis=2)
    else:
        raise ValueError(f"method must be in [ 'supperpose','substract','supperpose weighted' ] got {method}")

def image_slice(I,coord,dim):
    coord = int(coord)
    if dim == 0:
        return I[coord,:,:]
    elif dim == 1:
        return I[:,coord,:]
    elif dim == 2:
        return I[:,:,coord]

def make_3d_flat(img_3D,slice):
    """ Take a 'brain' 3D image, take 3 slices and make a long 2D image of it.

    """
    D,H,W = img_3D.shape

    # im0 = image_slice(img_3D,slice[0],2).T
    # im1 = image_slice(img_3D,slice[1],1).T
    # im2 = image_slice(img_3D,slice[2],0).T
    # adapté pos raph v
    im0 = image_slice(img_3D,slice[2],2).T
    im1 = image_slice(img_3D,slice[1],1).T
    im2 = image_slice(img_3D,slice[0],0).T

    crop = 20
    # print(D-int(1.7*crop),D+H-int(2.7*crop))
    # print(D+H-int(3.2*crop))
    long_img = np.zeros((D,D+H+H-int(3.5*crop)))
    long_img[:D,:D-crop] = im0[:,crop//2:-crop//2]
    long_img[(D-W)//2:(D-W)//2 + W,D-int(1.7*crop):D+H-int(2.7*crop)] = im1[::-1,crop//2:-crop//2]
    long_img[(D-W)//2:(D-W)//2 + W,D+H-int(3*crop):] = im2[::-1,crop//2:]

    # long_img[long_img== 0] =1
    return long_img

# ________________ Time management decorator ___________________________
def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    c = str(seconds - int(seconds))[:5]+"cents"
    return "{:d}:{:02d}:{:02d}s and ".format(int(h), int(m), int(s)) + c


def time_it(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"\nComputation of {func.__name__} done in ",format_time(t2 -t1)," s")
        return result
    return wrap_func


def update_progress(progress,message = None):
    # update_progress() : Displays or updates a console progress bar
    ## Accepts a float between 0 and 1. Any int will be converted to a float.
    ## A value under 0 represents a 'halt'.
    ## A value at 1 or bigger represents 100%
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}] {1:6.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    if not message is None:
        text += f' ({message[0]} ,{message[1]:8.2f}).'
    sys.stdout.write(text)
    sys.stdout.flush()