from time import time
import os
import numpy as np

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DLT_KW_GRIDSAMPLE = dict(padding_mode="border",
                         align_corners=True
                         )
DLT_KW_IMAGE = dict(cmap='gray',
                      # extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=0,vmax=1)

# ---------- Metric -------------------------------------------

def DICE(seg_1,seg_2):
    prod_seg = seg_1 * seg_2
    sum_seg = seg_1 + seg_2
    return 2*prod_seg.sum() / sum_seg.sum()


# __________ matplotlib _______________________________________
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