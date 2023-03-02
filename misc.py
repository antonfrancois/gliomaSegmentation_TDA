from time import time
import os,sys
import numpy as np
import scipy.ndimage as scipynd
from matplotlib.colors import ListedColormap
from nibabel import Nifti1Image
import dog

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# DLT_EDGES_THRSLD = .0625
DLT_EDGES_THRSLD = .1
# ---------- Metric -------------------------------------------

def DICE(seg_1,seg_2):
    if seg_1.max() == 0 or seg_2.max() == 0:
        return 0
    prod_seg = seg_1 * seg_2
    sum_seg = seg_1 + seg_2
    return 2*prod_seg.sum() / sum_seg.sum()

def edgesSobel(img,thrsld = DLT_EDGES_THRSLD):
    sx = scipynd.sobel(img, axis=0, mode='constant')
    sy = scipynd.sobel(img, axis=1, mode='constant')
    sz = scipynd.sobel(img, axis=2, mode='constant')
    sobel = np.sqrt(sx**2 + sy**2 + sz**2 )

    if thrsld is None:
        return sobel
    edges = np.zeros(sobel.shape)
    mask = (sobel > sobel.max()*thrsld)
    edges[mask ] = 1
    return edges

def edgesDog(img,fwhm= 6,dilat =1):
    # convert img to nifty
    nib_im = Nifti1Image(img,affine=np.eye(4))
    try:
        edges_im_nib = dog.dog_img(nib_im, fwhm=fwhm).get_fdata()
    except ValueError:
        return np.zeros(img.shape)
    if dilat >0:
        edges_im_nib = scipynd.binary_dilation(edges_im_nib,iterations=dilat).astype(float)
    return edges_im_nib

# @time_it
class EdgesMatch_Dice:
    """ Classe meusurant à quel point les bords de la
    segmentation sont inclus dans ceux de l'image.
    Pour l'optimisation, on ne calcule qu'une fois les bords
    de l'image. Les bords sont obtenus à partir d'un filtre
    de sobel 3D.

    Usage:
    `EdgesMatch_Dice(img,None)(seg)`

    """
    def __init__(self,img):
        """
        img : numpy array
        thrsld : valeur entre 0 et 1
        """
        self.img_edges = edgesDog(img,fwhm=6,dilat=0)

    def __call__(self, seg):
        self.seg_edges = edgesDog(seg,fwhm=6,dilat=0)
        intersect = self.img_edges * self.seg_edges
        return DICE(intersect,self.seg_edges)

# @time_it
def edges_DiceMatch(img,seg):
    return  EdgesMatch_Dice(img)(seg)



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
    elif 'seg' in method:
        u = I2[:,:,None] * I1[:, :, None]
        if 'w' in method:
            return np.concatenate(
                (
                    I1[:, :, None],
                    u + I2[:,:,None]*.5,
                    I2[:,:,None],
                    np.ones((M,N,1))
                    # np.maximum(I2[:,:,None], I1[:, :, None])
                ), axis=2
            )
        else:
            return np.concatenate(
                    (
                        I1[:, :, None] - u,
                        u,
                        I2[:,:,None] - u,
                        np.ones((M,N,1))
                        # np.maximum(I2[:,:,None], I1[:, :, None])
                    ), axis=2
                )

    else:
        raise ValueError(f"method must be in [ 'supperpose','substract','supperpose weighted', 'seg','seg white' ] got {method}")

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