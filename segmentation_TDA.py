import warnings

import scipy
import scipy.ndimage as ndimage
import skimage
import vedo
import numpy as np
# import my_torchbox as tb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from misc import *
import parseBrats as pb
import sys
import time

# TDA
import persim
import cripser

# Comments
import sys
import time


# def ChronometerStart(msg='Start... '):
#     start_time = time.time()
#     sys.stdout.write(msg); sys.stdout.flush()
#     return start_time
#
# def ChronometerStop(start_time, method='ms', linebreak='\n'):
#     elapsed_time_secs = time.time() - start_time
#     if method == 'ms':
#         msg = 'Execution time: '+repr(round(elapsed_time_secs*1000))+' ms.'+linebreak
#     if method == 's':
#         msg = 'Execution time: '+repr(round(elapsed_time_secs))+' s.'+linebreak
#     sys.stdout.write(msg); sys.stdout.flush()

def getConnectedComponent(img, pos, t):
    '''
    Get the connected component of the voxel pos = (x,y,z) at time t.
    The output is a binary image.
    Background value of img must be 0 (as conventional).
    '''
    imt = (img>=t)*1
    if imt[pos[0],pos[1],pos[2]]==0: raise ValueError('The voxel pos is not active at time t.')
    labels = skimage.measure.label(imt, background=0)
    labeltumor = labels[pos[0],pos[1],pos[2]]
    imtumor = (labels == labeltumor)*1
    if labeltumor==0: print('Problem! The label is background :(')
    return imtumor

def argmax_image(img,sign=-1):
    N = len(np.shape(img))
    s = img.shape
    new_shp = s[:sign*N] + (np.prod(s[sign*N:]),)
    max_idx = img.reshape(new_shp).argmax(-1)
    return np.unravel_index(max_idx, s[sign*N:])

def get_highest_connectedComponent(img,t):
    pos = argmax_image(img)
    # Filtration
    Segmentation = getConnectedComponent(img, pos, t)
    return Segmentation

def get_largest_CC(img, t, verbose=False):
    imgt = (img>=t)*1
    Labels = skimage.measure.label(imgt, background=0)
    nlabels = np.max(Labels)

    CardinalLabels = [0]+[np.sum(Labels==i) for i in range(1,nlabels)]
    ilabel = np.argmax(CardinalLabels)
    CC = (Labels==ilabel)*1

    if verbose: print('There are', np.max(CardinalLabels), 'labels')
    return CC

@time_it
def suggest_t(img,pos=None, N= 25,plot=True,dt_threshold=.5,verbose= True,ax=None):
    """

    :param img:
    :param pos: Save time to give pos
    :param N:
    :param plot:
    :param dt_threshold:
    :param verbose:
    :return:
    """
    if verbose : print('suggest_t : Compute curve... ')

    tmax = 1  if pos is None else img[pos[0],pos[1],pos[2]]
    t_list = np.linspace(0.01,tmax,N+1)
    if pos is None:
        filtr = np.array([np.sum(img>t) for t in t_list])
    else:
        filtr = np.zeros(t_list.shape)
        for i,t in enumerate(t_list):
            cc = getConnectedComponent(img, pos, t)
        filtr[i] = np.sum(cc==1)
    filtr_dt = (filtr[:-1] - filtr[1:])*(N/(tmax-0.01))
    filtr_dt_norm = len(filtr_dt)*filtr_dt/filtr_dt.sum()

    # best_t = -1
    # index = N-2
    # while best_t < 0:
    #     dt = filtr_dt_norm[index]
    #     if dt > dt_threshold:
    #         best_dt = dt
    #         best_i = index+2
    #         best_t = t_list[index+2]
    #         print('best_t :',best_t)
    #         break
    #     elif index == 0:
    #         warnings.warn("No best t found, ajdust dt_threshold")
    #         return 0
    #     else:
    #         index -= 1

    best_i = np.where(filtr_dt_norm>dt_threshold)[0][-1]
    best_t = t_list[best_i+1]

    # Plot
    if plot or not ax is None:
        c1,c2,c3 = 'forestgreen','firebrick','goldenrod'
        if ax is None:
            fig, ax = plt.subplots(1, 1,
                                    figsize = (6,6),
                                    constrained_layout=True)
        f = ax.plot(t_list,filtr,'o-',label='f',c=c1)
        ax.plot([best_t,best_t],[0,filtr.max()],'--',c=c2)
        ax.text(best_t + .01,.8*filtr.max(),f"t = {best_t:.3f}",c=c2)
        # plt.scatter(t_list,filtr)
        axt = ax.twinx()
        df = axt.plot(t_list[:-1],filtr_dt_norm,
                 'D--',
                c=c3,
                label='df normalized')
        lns = f+df
        labs = [l.get_label() for l in lns]

        ax.set_xlabel("time")
        ax.set_ylabel("f")
        axt.set_ylabel("df")

        ax.yaxis.get_label().set_color(f[0].get_color())
        axt.yaxis.get_label().set_color(df[0].get_color())

        ax.legend(lns, labs,loc='upper right')
        if ax is None: plt.show()
    return best_t,filtr,filtr_dt_norm,t_list

@time_it
def Segmentation(img_flair,img_t1ce,n_H2=1, plot=False,verbose=True):
    # Plot images
    if plot:
        fig, axs = plt.subplots(2, 3, figsize = (9,6))
        hauteurs = [80-10, 80, 80+10]
        for i in range(3):
            imslice = img_flair[:,:,hauteurs[i]]
            axs[0,i].imshow(imslice,  vmin=0, vmax = 1, cmap='gray', origin ='lower')
            axs[0,i].axis('off')
            imslice = img_t1ce[:,:,hauteurs[i]]
            axs[1,i].imshow(imslice,  vmin=0, vmax = 1, cmap='gray', origin ='lower')
            axs[1,i].axis('off')
        fig.suptitle('flair and t1ce',fontsize=10)

    # 1 - Choose a parameter t for segmentation
    t,_,_,_ = suggest_t(img_flair, N=25, dt_threshold=1, plot=plot, verbose=verbose)

    # 2 - Get segmentation
    pos = argmax_image(img_flair)
    Segmentation = getConnectedComponent(img_flair, pos, t)

    # Plot segmentation
    if plot:
        fig, axs = plt.subplots(1, 3, figsize = (9,3))
        hauteurs = [pos[2], pos[2]+10, pos[2]-10]
        for i in range(3):
            imslice = img_flair[:,:,hauteurs[i]]
            Segmentationslice = Segmentation[:,:,hauteurs[i]]
            implt = np.concatenate((imslice[:,:,None],Segmentationslice[:,:,None],Segmentationslice[:,:,None]*0),2)
            axs[i].imshow(implt,  vmin=0, vmax = 1, cmap='gray', origin ='lower')
            axs[i].axis('off')
        fig.suptitle('Segmentation flair',fontsize=10)


    # 3 - Compute PH2
    SegmentationColors = (img_t1ce)*Segmentation
    maxdim = 3
    # if verbose: start_time = ChronometerStart('Compute diagram... ')
    barcode = cripser.computePH(1-SegmentationColors,maxdim=maxdim) # Compute diagram
    H2 = [list(bar[1::]) for bar in barcode if bar[0]==2]
    H2 = [bar for _,bar in sorted(zip([bar[1]-bar[0] for bar in H2],H2))[::-1]] # Sort list H2
    # if verbose: ChronometerStop(start_time, method='s')
    if verbose: print('There are '+repr(len(barcode))+' bars.')

    # Plot diagram
    if plot:
        plt.figure(figsize=(4,4))
        persim.plot_diagrams([np.array([bar[1:3] for bar in barcode if bar[0]==i]) for i in range(maxdim)])
        plt.title('Persistence diagram of t1ce segmented',fontsize=10)

    # 3 - Segmentation nécrotique
#    n_H2 = 1 #choose the number of H2 cycles
    SegmentationNecrotique = []
    for i in range(n_H2):
        bar = H2[i]
        pos = [int(bar[2]),int(bar[3]),int(bar[4])]
        if verbose: print('The', i, 'highest bar is', bar)

        # Segmentation
        t = bar[0]+0.0001
        Segmentation_t1ce = getConnectedComponent(SegmentationColors, pos, 1 - t)

        # Plot
        if plot:
            fig, axs = plt.subplots(1, 3, figsize = (9,3))
            hauteurs = [pos[2], pos[2]-5, pos[2]+5]
            for i in range(3):
                imslice = img_t1ce[:,:,hauteurs[i]]
                Segmentationslice = Segmentation_t1ce[:,:,hauteurs[i]]
                implt = np.concatenate((imslice[:,:,None],Segmentationslice[:,:,None],Segmentationslice[:,:,None]*0),2)
                axs[i].imshow(1-implt,  vmin=0, vmax = 1, cmap='gray', origin ='lower')
                axs[i].axis('off')
            fig.suptitle('Segmentation tumeur', fontsize=10)

        # Save
        SegmentationNecrotique.append(Segmentation_t1ce)

    # 4 - Segmentation infiltration
    # Define masque nécrotique
    MasqueNecrotique = Segmentation*(1-np.sum([seg for seg in SegmentationNecrotique],0))
#    print('The following value should be 0: '+repr(MasqueNecrotique[0,0,0]))
#    print('The following set should be {0,1}: '+repr(set([a[0] for a in MasqueNecrotique.reshape(-1,1)])))

    # Define segmentatation
    N = 100 #minimal number of voxels in a connected component

    # Get labels with at least N voxels
    Segmentationt1ce = []
    cardinalofsegmentations = []
    im = img_t1ce*MasqueNecrotique
    im = (im!=0)*1
    labels = skimage.measure.label(im, background=0)
    nlabels = np.max(labels)
    for l in range(1,nlabels):
        imlabel = (labels == l)*1
        cardinalofsegmentations.append(np.sum(imlabel))
        if np.sum(imlabel) >= N:
            Segmentationt1ce.append(imlabel)

    if verbose: print(len(Segmentationt1ce), 'connected components have been selected, among', nlabels-1)
    if verbose: print('Cardinal of segmentations:',set(cardinalofsegmentations))

    # Plot the segmentation
    if plot:
        for seg in Segmentationt1ce:
            pos = np.mean(seg.nonzero(),1)
            pos = [int(pos[0]),int(pos[1]),int(pos[2])]
            fig, axs = plt.subplots(1, 3, figsize = (9,3))
            hauteurs = [pos[2]-3, pos[2], pos[2]+3]
            for i in range(3):
                imslice = seg[:,:,hauteurs[i]]
                Segmentationslice = img_t1ce[:,:,hauteurs[i]]
                implt = np.concatenate((imslice[:,:,None],Segmentationslice[:,:,None],Segmentationslice[:,:,None]*0),2)
                axs[i].imshow(implt,  vmin=0, vmax = 1, cmap='gray', origin ='lower')
                axs[i].axis('off')
            fig.suptitle('Segmentation nécrotique', fontsize=10)


    # 5 - Final step: gather the segmentation in one matrix
    # Define FinalSegmentation
    FinalConnectedComponents = SegmentationNecrotique + Segmentationt1ce
    FinalSegmentation = FinalConnectedComponents[0]*0
    for i in range(len(FinalConnectedComponents)):
        FinalSegmentation = FinalSegmentation + (i+1)*FinalConnectedComponents[i]
    print('FinalSegmentation contains',np.max(FinalSegmentation),'classes.')
    # Plot
    if plot:
        N_classes = np.max(FinalSegmentation)
        cmaplist = [plt.cm.jet(i) for i in range(plt.cm.jet.N)]; cmaplist[0] = (1, 1, 1, 1.0)
        cmaplist = [cmaplist[int(i)] for i in np.linspace(0,len(cmaplist)-1,N_classes)]
        cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, plt.cm.jet.N)

        fig, axs = plt.subplots(1, 3, figsize = (9,3))
        pos = argmax_image(img_flair)
        hauteurs = [pos[2]-10, pos[2], pos[2]+10]
        for i in range(3):
            imslice = FinalSegmentation[:,:,hauteurs[i]]
            flairslice = img_flair[:,:,hauteurs[i]]
            axs[i].imshow(1-flairslice, cmap='gray', origin ='lower',alpha=0.5)
            axs[i].imshow(imslice, cmap=cmap, origin ='lower',alpha=0.5,interpolation=None)
            axs[i].axis('off')
        fig.suptitle('Segmentation finale en '+repr(N_classes)+' classes', fontsize=10);
        plt.show()
    return FinalSegmentation

if __name__ == '__main__':

    ' Get segmentation at time t '

    sigma = 1 #smoothing parameter
    t = 0.5 #filtration value. Plus t est petit, plus on aura de points

    # Open image

    pb = pb.parse_brats(brats_list=None,brats_folder='2022',modality='flair')
    print(f"BratS list is {len(pb.brats_list)} item long")
    img_1,img_2,_,_ = pb(0,to_torch=False)
    img = img_1

    # Smooth image

    img = scipy.ndimage.gaussian_filter(img, sigma=sigma)

    # Get most luminous point

    pos = argmax_image(img)

    # Filtration

    segmentation = getConnectedComponent(img, pos, t)

    # Plot

    fig, axs = plt.subplots(1, 3, figsize = (12,4))
    hauteurs = [pos[2], 40, 60]
    for i in range(3):
        imslice = img[:,:,hauteurs[i]]
        segmentationslice = segmentation[:,:,hauteurs[i]]
        implt = np.concatenate((imslice[:,:,None],segmentationslice[:,:,None],segmentationslice[:,:,None]*0),2)
        axs[i].imshow(implt,  vmin=0, vmax = 1, cmap='gray', origin ='lower')
        axs[i].axis('off')

    ' Suggest a parameter t '

    S = suggest_t(img)

