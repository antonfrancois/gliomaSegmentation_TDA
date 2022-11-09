import torch

import __init__
import my_torchbox as tb
import my_toolbox as tlb
from constants import *
import csv
import pandas as pd
from persistent_homology.segmentation_TDA import *
import scipy
import image_3d_visualisation as i3v
import matplotlib as mpl
import nibabel as nib

# ==============================================================

def PlotImage(image, pos, title=None):
    fig, axs = plt.subplots(1, 3, figsize = (9,3))
    for i in range(3):
        if i==0: imageslice = image[pos[0],:,:]
        if i==1: imageslice = image[:,pos[1],:]
        if i==2: imageslice = image[:,:,pos[2]]
        axs[i].imshow(imageslice,vmin=0, vmax = 1, cmap='gray', origin ='lower')
        axs[i].axis('off')
    if title is not None: fig.suptitle(title,fontsize=10)
    plt.show()

def PlotMask(mask, image, pos, title=None):
    COLORS = ['white']+list(mpl.colors.TABLEAU_COLORS)*10
    cmap = mpl.colors.ListedColormap(COLORS[0:(int(np.max(mask))+1)])
    bounds = list(range(int(np.max(mask))+2))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 3, figsize = (12,4))
    for i in range(3):
        if i==0: maskslice = mask[pos[0],:,:]; imageslice = image[pos[0],:,:]
        if i==1: maskslice = mask[:,pos[1],:]; imageslice = image[:,pos[1],:]
        if i==2: maskslice = mask[:,:,pos[2]]; imageslice = image[:,:,pos[2]]
        axs[i].imshow(1-imageslice, cmap='gray', origin ='lower',alpha=0.5)
        axs[i].imshow(maskslice, cmap=cmap, origin ='lower',alpha=0.75,norm=norm)
        axs[i].axis('off')
    if title is not None: fig.suptitle(title,fontsize=10)
    plt.show()

@time_it
def Segmentations(img_flair,img_t1ce,n_H2=1,dt_threshold=1,plot=False,verbose =False):
    # Plot images
    if plot:
        pos = argmax_image(img_flair)
        PlotImage(img_flair, pos, title='flair')
        PlotImage(img_t1ce, pos, title='t1ce')

    # 0 - Select high intensity zone
    pos = argmax_image(img_flair)

    # 1 - Choose a parameter t for segmentation
    t,_,_,_ = suggest_t(img_flair, pos=pos, N=100, dt_threshold=dt_threshold, plot=plot, verbose=verbose)

    # 2 - Get segmentation
    Segmentation = GetConnectedComponent(img_flair, pos, t)

    # 2' - Test for topology - add isolated pixels
    labels, nlabels = scipy.ndimage.label(1-Segmentation)
    print('Number of isolated labels:',nlabels)
    backgroundsegmentation = []
    cardinalofsegmentations = []
    for l in range(nlabels):
        imlabel = (labels == l)*1
        backgroundsegmentation.append(imlabel.copy())
        cardinalofsegmentations.append(np.sum(imlabel))
#    if verbose: print('Cardinal of segmentations:',set(cardinalofsegmentations))
    for i in range(nlabels):
        if cardinalofsegmentations[i]<max(cardinalofsegmentations):
            Segmentation[backgroundsegmentation[i]==1]=1

    if plot:
        PlotMask(Segmentation, img_flair, pos, title='Global segmentation - after filling')

    # 3 - Compute PH2
    SegmentationColors = (img_t1ce)*Segmentation
    maxdim = 3
    if verbose: start_time = ChronometerStart('Compute diagram... ')
    barcode = cripser.computePH(1-SegmentationColors,maxdim=maxdim) # Compute diagram
    H2 = [list(bar[1::]) for bar in barcode if bar[0]==2 and bar[2]<1] # Only non-infinity bars
    H2 = [bar for _,bar in sorted(zip([bar[1]-bar[0] for bar in H2],H2))[::-1]] # Sort list H2
    if verbose: ChronometerStop(start_time, method='s')
    if verbose: print('There are '+repr(len(barcode))+' bars.')

    # Plot diagram
    if plot:
        if len(H2)>0:
            fig, ax = plt.subplots(1,1, figsize=(4,4))
            persim.plot_diagrams([np.array([barcode[0][1:3]]), np.array([barcode[0][1:3]]), np.array([bar[1:3] for bar in barcode if bar[0]==maxdim-1])])
            plt.title('Persistence diagram of t1ce segmented',fontsize=10)
        else:
            print('H2 is empty!')

    # 3 - Segmentation nécrotique
    FinalSegmentations = []
    param_vals = []
    M = min(6, len(H2))
    for m in range(M):
        n_H2 = 1
        n_H2_index = m
        param_vals.append(dict(h2i = m, h2N = 1))  # <<< update val param
        SegmentationNecrotique = []
        for i in range(n_H2_index,n_H2_index+n_H2):
#            print('i bar:', i)
            bar = H2[i]
            pos = [int(bar[2]),int(bar[3]),int(bar[4])]
            if plot: patch = plt.Circle((bar[0],bar[1]), 0.01,fill=False); ax.add_patch(patch)
            if verbose: print('The '+repr(i)+'st highest bar is', bar)

            # Segmentation
            t = bar[0]+0.0001
            Segmentation_t1ce = GetConnectedComponent(SegmentationColors, pos, 1-t)

            # Plot
            if plot: PlotMask(Segmentation_t1ce, img_t1ce, pos, title='Segmentation tumeur')

            # Save
            SegmentationNecrotique.append(Segmentation_t1ce)

        # 4 - Segmentation infiltration
        # Define masque nécrotique
        MasqueNecrotique = Segmentation*(1-np.sum([seg for seg in SegmentationNecrotique],0))

        # Define segmentation - Get labels with at least N voxels
        N = 100 #minimal number of voxels in a connected component
        Segmentationt1ce = []
        cardinalofsegmentations = []
        im = MasqueNecrotique
        labels, nlabels = scipy.ndimage.label(im)
#        print('nlabels in MasqueNecrotique:',nlabels)
        for l in range(1,nlabels):
            imlabel = (labels == l)*1
            cardinalofsegmentations.append(np.sum(imlabel))
            if np.sum(imlabel) >= N:
                Segmentationt1ce.append(imlabel)
#        if verbose: print(len(Segmentationt1ce), 'connected components have been selected, among', nlabels-1)
    #    if verbose: print('Cardinal of segmentations:',set(cardinalofsegmentations))

        FinalSegmentation = Segmentation.copy()*0
        # 1 - BLEU, TC  -> NECROSE INACTIVE, TUMORUS CORE
        # 2 - ORANGE, WT -> INFILTRATION, OEDEME, WHOLE TUMOR
        # 4 - ROUGE, ET -> NECROSE ACTIVE, ENHENCING TUMOR
        FinalSegmentation[SegmentationNecrotique[0]>0] = 4

        for c in range(len(Segmentationt1ce)):
            Class = Segmentationt1ce[c]
            ClassDilated = scipy.ndimage.binary_dilation(Class,iterations=1)
            Contour = ClassDilated - Class
            meanvalue = np.mean(Segmentation[np.where(Contour>0)])
            if meanvalue<1/2: #sort du masque Segmentation, label 2
                FinalSegmentation[Class>0] = 2
            else: #label 1
                FinalSegmentation[Class>0] = 1

#         MeanValuesOutOfSeg = []
#         for c in range(len(Segmentationt1ce)):
#             Class = Segmentationt1ce[c]
#             ClassDilated = scipy.ndimage.binary_dilation(Class,iterations=1)
#             Contour = ClassDilated - Class
#             MeanValuesOutOfSeg.append(np.mean(Segmentation[np.where(Contour>0)]))

#         MeanValuesOutOfSeg_min = min(MeanValuesOutOfSeg)
#         for c in range(len(Segmentationt1ce)):
#             Class = Segmentationt1ce[c]
#             if MeanValuesOutOfSeg[c]==MeanValuesOutOfSeg_min: #sort du masque Segmentation, label 2
#                 FinalSegmentation[Class>0] = 2
#             else: #label 1
#                 FinalSegmentation[Class>0] = 1

        # Plot
        if plot: PlotMask(FinalSegmentation, img_flair, pos, title='Segmentation finale')

        FinalSegmentations.append(FinalSegmentation)
    return FinalSegmentations,param_vals

# ==============================================================

def compute_dices(seg_list,seg_real):
    # 1 - BLEU, TC  -> NECROSE INACTIVE, TUMORUS CORE
    # 2 - ORANGE, WT -> INFILTRATION, OEDEME, WHOLE TUMOR
    # 4 - ROUGE, ET -> NECROSE ACTIVE, ENHENCING TUMOR
    dice_list=[]
    for sl in seg_list:
        # dice_1
        dice_TC = tlb.dice((sl==1).astype(int),(seg_real==1).astype(int))
        # dice_2
        dice_WT = tlb.dice((sl==2).astype(int),(seg_real==2).astype(int))
        # dice_3
        dice_ET = tlb.dice((sl==4).astype(int),(seg_real==4).astype(int))
        # dice_4
        dice_union = tlb.dice((sl>=1).astype(int),(seg_real>=1).astype(int))

        dice_list.append([dice_TC,dice_WT,dice_ET,dice_union])
    return dice_list

# def make_brats_list(csv_file,folder):
#     """Will take all name list that are not in the csv_file """
#     nogo_list = []
#     with open(csv_file) as result_file:
#         csv_reader = csv.DictReader(result_file, delimiter=';')
#         for row in csv_reader:
#             nogo_list.append(row['brats_name'])
#
#     brats_list = []
#     for obj in os.listdir(folder):
#         if ('BraTSReg_' in obj or 'BraTS2021' in obj) and not obj in nogo_list:
#             brats_list.append(obj)

def make_new_brast_list(n_img_to_test=100):
    pb = tb.parse_brats(brats_list=None, brats_folder='2021',get_template=False)
    i_list = torch.arange(len(pb.brats_list))
    i_list  = torch.randperm(len(pb.brats_list))[:n_img_to_test]
    new_brats_list = []
    for i in i_list:
        new_brats_list.append(pb.brats_list[i])
    print(new_brats_list)
    return new_brats_list

def make_brats_list_from_csv(csv_file):
    brast_list = []
    with open(csv_file) as result_file:
        csv_reader = csv.DictReader(result_file, delimiter=';')
        for row in csv_reader:
            brast_list.append(row['brats_name'])

BRATS_LIST_DFLT = ['BraTS2021_01410', 'BraTS2021_00336', 'BraTS2021_01573', 'BraTS2021_00016', 'BraTS2021_01250', 'BraTS2021_00418', 'BraTS2021_01227', 'BraTS2021_00433', 'BraTS2021_01137', 'BraTS2021_01413', 'BraTS2021_00196', 'BraTS2021_00419', 'BraTS2021_01077', 'BraTS2021_01304', 'BraTS2021_01149', 'BraTS2021_01044', 'BraTS2021_00543', 'BraTS2021_01480', 'BraTS2021_00275', 'BraTS2021_00747', 'BraTS2021_00025', 'BraTS2021_00572', 'BraTS2021_00105', 'BraTS2021_00227', 'BraTS2021_01316', 'BraTS2021_00292', 'BraTS2021_01563', 'BraTS2021_01314', 'BraTS2021_01496', 'BraTS2021_00327', 'BraTS2021_01632', 'BraTS2021_01140', 'BraTS2021_01450', 'BraTS2021_00413', 'BraTS2021_01555', 'BraTS2021_00127', 'BraTS2021_01338', 'BraTS2021_00338', 'BraTS2021_01119', 'BraTS2021_00480', 'BraTS2021_00446', 'BraTS2021_01033', 'BraTS2021_01167', 'BraTS2021_00820', 'BraTS2021_00429', 'BraTS2021_01026', 'BraTS2021_01520', 'BraTS2021_01651', 'BraTS2021_00485', 'BraTS2021_00737', 'BraTS2021_00431', 'BraTS2021_01489', 'BraTS2021_01329', 'BraTS2021_00106', 'BraTS2021_00589', 'BraTS2021_00369', 'BraTS2021_00329', 'BraTS2021_01228', 'BraTS2021_01010', 'BraTS2021_01585', 'BraTS2021_00402', 'BraTS2021_00625', 'BraTS2021_00149', 'BraTS2021_01310', 'BraTS2021_00528', 'BraTS2021_01264', 'BraTS2021_00366', 'BraTS2021_01417', 'BraTS2021_00414', 'BraTS2021_00823', 'BraTS2021_00502', 'BraTS2021_01201', 'BraTS2021_00518', 'BraTS2021_01177', 'BraTS2021_00344', 'BraTS2021_00216', 'BraTS2021_01407', 'BraTS2021_01052', 'BraTS2021_00356', 'BraTS2021_01326', 'BraTS2021_01512', 'BraTS2021_00250', 'BraTS2021_00687', 'BraTS2021_01164', 'BraTS2021_00554', 'BraTS2021_01219', 'BraTS2021_00035', 'BraTS2021_00810', 'BraTS2021_01078', 'BraTS2021_01061', 'BraTS2021_00816', 'BraTS2021_01401', 'BraTS2021_00577', 'BraTS2021_01194', 'BraTS2021_00412', 'BraTS2021_01084', 'BraTS2021_01253', 'BraTS2021_00469', 'BraTS2021_00477', 'BraTS2021_01147', 'BraTS2021_01042', 'BraTS2021_00134', 'BraTS2021_01173', 'BraTS2021_01116', 'BraTS2021_00203', 'BraTS2021_01604', 'BraTS2021_00680', 'BraTS2021_01103', 'BraTS2021_01428', 'BraTS2021_01572', 'BraTS2021_01056', 'BraTS2021_00221', 'BraTS2021_00377', 'BraTS2021_01534', 'BraTS2021_01641', 'BraTS2021_00586', 'BraTS2021_00758', 'BraTS2021_01196', 'BraTS2021_01188', 'BraTS2021_01648', 'BraTS2021_01075', 'BraTS2021_00800', 'BraTS2021_01422', 'BraTS2021_01151', 'BraTS2021_00317', 'BraTS2021_00238', 'BraTS2021_01162', 'BraTS2021_01128', 'BraTS2021_01608', 'BraTS2021_01642', 'BraTS2021_01009', 'BraTS2021_00269', 'BraTS2021_00675', 'BraTS2021_01342', 'BraTS2021_00639', 'BraTS2021_00400', 'BraTS2021_00313', 'BraTS2021_01359', 'BraTS2021_00088', 'BraTS2021_00811', 'BraTS2021_01464', 'BraTS2021_01624', 'BraTS2021_01273', 'BraTS2021_01031', 'BraTS2021_00709', 'BraTS2021_01601', 'BraTS2021_00544', 'BraTS2021_00756', 'BraTS2021_00305', 'BraTS2021_00836', 'BraTS2021_00024', 'BraTS2021_01487', 'BraTS2021_00839', 'BraTS2021_01034', 'BraTS2021_00840', 'BraTS2021_00449', 'BraTS2021_00222', 'BraTS2021_00691', 'BraTS2021_01550', 'BraTS2021_00147', 'BraTS2021_01659', 'BraTS2021_01542', 'BraTS2021_00251', 'BraTS2021_00621', 'BraTS2021_01400', 'BraTS2021_00236', 'BraTS2021_01242', 'BraTS2021_01415', 'BraTS2021_01234', 'BraTS2021_00260', 'BraTS2021_00569', 'BraTS2021_00046', 'BraTS2021_00742', 'BraTS2021_00280', 'BraTS2021_00074', 'BraTS2021_00077', 'BraTS2021_01175', 'BraTS2021_00322', 'BraTS2021_01427', 'BraTS2021_01611', 'BraTS2021_00212', 'BraTS2021_01040', 'BraTS2021_00655', 'BraTS2021_01017', 'BraTS2021_00359', 'BraTS2021_01037', 'BraTS2021_01013', 'BraTS2021_00376', 'BraTS2021_00694', 'BraTS2021_01468', 'BraTS2021_01657', 'BraTS2021_01387', 'BraTS2021_00132', 'BraTS2021_01508', 'BraTS2021_00751', 'BraTS2021_01630', 'BraTS2021_01424', 'BraTS2021_00312', 'BraTS2021_01258', 'BraTS2021_01471', 'BraTS2021_00443', 'BraTS2021_01610', 'BraTS2021_00801', 'BraTS2021_00540', 'BraTS2021_01322', 'BraTS2021_00441', 'BraTS2021_00187', 'BraTS2021_01528', 'BraTS2021_00563', 'BraTS2021_00102', 'BraTS2021_01649', 'BraTS2021_00453', 'BraTS2021_01582', 'BraTS2021_00579', 'BraTS2021_01144', 'BraTS2021_00058', 'BraTS2021_01525', 'BraTS2021_01334', 'BraTS2021_00668', 'BraTS2021_01455', 'BraTS2021_00348', 'BraTS2021_00787', 'BraTS2021_01145', 'BraTS2021_01261', 'BraTS2021_01254', 'BraTS2021_00273', 'BraTS2021_01522', 'BraTS2021_01336', 'BraTS2021_01236', 'BraTS2021_00830', 'BraTS2021_01065', 'BraTS2021_00750', 'BraTS2021_01074', 'BraTS2021_00649', 'BraTS2021_00267', 'BraTS2021_01540', 'BraTS2021_00395', 'BraTS2021_00353', 'BraTS2021_01589', 'BraTS2021_00588', 'BraTS2021_00797', 'BraTS2021_01266', 'BraTS2021_00802', 'BraTS2021_00831', 'BraTS2021_00008', 'BraTS2021_01423', 'BraTS2021_00547', 'BraTS2021_00211', 'BraTS2021_00444', 'BraTS2021_00062', 'BraTS2021_00210', 'BraTS2021_00506', 'BraTS2021_00456', 'BraTS2021_01049', 'BraTS2021_01602', 'BraTS2021_00258', 'BraTS2021_01218', 'BraTS2021_01148', 'BraTS2021_01625', 'BraTS2021_00097', 'BraTS2021_01214', 'BraTS2021_00594', 'BraTS2021_01240', 'BraTS2021_01574', 'BraTS2021_00320', 'BraTS2021_01395', 'BraTS2021_01607', 'BraTS2021_01179', 'BraTS2021_01477', 'BraTS2021_00729', 'BraTS2021_00089', 'BraTS2021_01345', 'BraTS2021_01045', 'BraTS2021_00507', 'BraTS2021_00440', 'BraTS2021_00778', 'BraTS2021_01438', 'BraTS2021_00206', 'BraTS2021_01365', 'BraTS2021_00159', 'BraTS2021_00146', 'BraTS2021_01282', 'BraTS2021_00061', 'BraTS2021_00533', 'BraTS2021_01277', 'BraTS2021_01063', 'BraTS2021_01440', 'BraTS2021_01380', 'BraTS2021_01200', 'BraTS2021_01291', 'BraTS2021_01098', 'BraTS2021_00294', 'BraTS2021_01156', 'BraTS2021_01532', 'BraTS2021_00383', 'BraTS2021_01138', 'BraTS2021_01036', 'BraTS2021_01364', 'BraTS2021_00230', 'BraTS2021_00723', 'BraTS2021_00031', 'BraTS2021_01072', 'BraTS2021_01369', 'BraTS2021_01460', 'BraTS2021_01560', 'BraTS2021_01579', 'BraTS2021_00107', 'BraTS2021_01126', 'BraTS2021_00789', 'BraTS2021_00191', 'BraTS2021_01071', 'BraTS2021_01289', 'BraTS2021_00051', 'BraTS2021_01501', 'BraTS2021_01299', 'BraTS2021_01504', 'BraTS2021_01656', 'BraTS2021_01307', 'BraTS2021_00022', 'BraTS2021_01203', 'BraTS2021_00126', 'BraTS2021_00090', 'BraTS2021_00819', 'BraTS2021_01150', 'BraTS2021_00548', 'BraTS2021_01279', 'BraTS2021_00650', 'BraTS2021_01280', 'BraTS2021_00774', 'BraTS2021_00773', 'BraTS2021_01370', 'BraTS2021_01593', 'BraTS2021_01020', 'BraTS2021_00498', 'BraTS2021_01557', 'BraTS2021_01599', 'BraTS2021_01189', 'BraTS2021_01192', 'BraTS2021_01076', 'BraTS2021_01154', 'BraTS2021_00130', 'BraTS2021_01102', 'BraTS2021_01521', 'BraTS2021_01643', 'BraTS2021_00340', 'BraTS2021_00150', 'BraTS2021_01498', 'BraTS2021_01067', 'BraTS2021_00176', 'BraTS2021_00479', 'BraTS2021_00999', 'BraTS2021_00558', 'BraTS2021_01665', 'BraTS2021_00249', 'BraTS2021_00545', 'BraTS2021_01469', 'BraTS2021_00538', 'BraTS2021_00693', 'BraTS2021_01216', 'BraTS2021_00677', 'BraTS2021_01371', 'BraTS2021_01274', 'BraTS2021_01001', 'BraTS2021_01087', 'BraTS2021_01198', 'BraTS2021_01652', 'BraTS2021_00539', 'BraTS2021_00239', 'BraTS2021_00523', 'BraTS2021_01545', 'BraTS2021_00379', 'BraTS2021_00296', 'BraTS2021_00688', 'BraTS2021_00772', 'BraTS2021_01323', 'BraTS2021_01104', 'BraTS2021_01185', 'BraTS2021_00120', 'BraTS2021_00676', 'BraTS2021_01028', 'BraTS2021_01505', 'BraTS2021_00204', 'BraTS2021_00690', 'BraTS2021_00530', 'BraTS2021_01448', 'BraTS2021_01559', 'BraTS2021_00166', 'BraTS2021_00117', 'BraTS2021_00063', 'BraTS2021_01278', 'BraTS2021_01083', 'BraTS2021_01097', 'BraTS2021_00289', 'BraTS2021_01247', 'BraTS2021_00070', 'BraTS2021_01357', 'BraTS2021_00706', 'BraTS2021_01353', 'BraTS2021_01457', 'BraTS2021_00645', 'BraTS2021_01199', 'BraTS2021_01220', 'BraTS2021_01215', 'BraTS2021_01453', 'BraTS2021_01068', 'BraTS2021_00504', 'BraTS2021_01402', 'BraTS2021_01066', 'BraTS2021_00599', 'BraTS2021_00409', 'BraTS2021_01021', 'BraTS2021_00291', 'BraTS2021_01503', 'BraTS2021_01394', 'BraTS2021_01262', 'BraTS2021_01635', 'BraTS2021_00122', 'BraTS2021_00707', 'BraTS2021_01107', 'BraTS2021_00576', 'BraTS2021_01296', 'BraTS2021_01011', 'BraTS2021_01081', 'BraTS2021_01168', 'BraTS2021_01552', 'BraTS2021_01463', 'BraTS2021_00552', 'BraTS2021_01231', 'BraTS2021_01526', 'BraTS2021_00583', 'BraTS2021_00390', 'BraTS2021_00300', 'BraTS2021_01546', 'BraTS2021_01324', 'BraTS2021_00828', 'BraTS2021_01303', 'BraTS2021_01486', 'BraTS2021_01110', 'BraTS2021_00626', 'BraTS2021_01048', 'BraTS2021_00109', 'BraTS2021_00455', 'BraTS2021_00019', 'BraTS2021_01241', 'BraTS2021_01160', 'BraTS2021_01050', 'BraTS2021_00000', 'BraTS2021_01494', 'BraTS2021_01226', 'BraTS2021_01418', 'BraTS2021_00284', 'BraTS2021_00725', 'BraTS2021_01583', 'BraTS2021_00561', 'BraTS2021_01421', 'BraTS2021_01430', 'BraTS2021_01654', 'BraTS2021_00026', 'BraTS2021_00104', 'BraTS2021_00570', 'BraTS2021_01637', 'BraTS2021_01027', 'BraTS2021_01434', 'BraTS2021_00325', 'BraTS2021_00032', 'BraTS2021_00684', 'BraTS2021_00214', 'BraTS2021_00297', 'BraTS2021_01398', 'BraTS2021_00162', 'BraTS2021_01143', 'BraTS2021_00360', 'BraTS2021_00170', 'BraTS2021_01551', 'BraTS2021_00575', 'BraTS2021_01239', 'BraTS2021_01035', 'BraTS2021_00517', 'BraTS2021_00495', 'BraTS2021_01260', 'BraTS2021_00240', 'BraTS2021_01202', 'BraTS2021_01213', 'BraTS2021_01578', 'BraTS2021_00620', 'BraTS2021_01404', 'BraTS2021_00703', 'BraTS2021_01064', 'BraTS2021_00364', 'BraTS2021_00656', 'BraTS2021_00193', 'BraTS2021_00759', 'BraTS2021_01114', 'BraTS2021_01429', 'BraTS2021_01384', 'BraTS2021_00616', 'BraTS2021_00139', 'BraTS2021_00708', 'BraTS2021_01644', 'BraTS2021_00375']
BRATS_LIST_DFLT = [b for b in BRATS_LIST_DFLT if b[11] =='0']


class seg_parse_brats:

    def __init__(self,brats_list,smooth_flair = 1,smooth_t1ce = 1):
        self.pb_flair = tb.parse_brats(brats_list=brats_list,brats_folder='2021',modality='flair',get_template=False)
        self.pb_t1ce = tb.parse_brats(brats_list=brats_list,brats_folder='2021',modality='t1ce',get_template=False)
        self.sigma_t1ce = smooth_flair
        self.sigma_flair = smooth_t1ce
        print(f"BraTS_list is {len(brats_list)} long ")

    def __getitem__(self, item):
        if item == self.__len__():
            raise IndexError("item too big")
        img_flair, real_seg = self.pb_flair(item,to_torch=False,normalize=False)
        img_t1ce, _ = self.pb_t1ce(item,to_torch=False,normalize=False)
        img_t1ce = scipy.ndimage.gaussian_filter(img_t1ce, sigma=self.sigma_t1ce)
        img_flair = scipy.ndimage.gaussian_filter(img_flair, sigma=self.sigma_flair)
        brats_name = self.pb_flair.brats_list[item]
        return  img_flair,img_t1ce,real_seg,brats_name

    def __len__(self):
        return len(self.pb_flair.brats_list)

def format_colName_csv(param):
    key = param.keys()
    name = ""
    for k in param.keys():
        name += f"{k}:{param[k]}_"
    return name[:-1]

def write_csv(csv_file,brats_name,column_name,dice):
    db = pd.read_csv(csv_file,sep=';')
    if not brats_name in db['brats_name'].unique():
        new_row = [-1]*len(db.keys())
        new_row[0] = brats_name
        db.loc[len(db.index)] = new_row

    dice_name = [" TC"," WT"," ET"," union"]
    for d,dn in zip(dice,dice_name):
        bratsindex = db.index[db['brats_name'] == brats_name]
        db.loc[bratsindex,["TDA_"+column_name+dn]] = d
    db.to_csv(csv_file,index=False,sep=';')

@time_it
def get_evaluate_segUnet(brats_name):
    path = "/home/turtlefox/Documents/Doctorat/data/brats2021_seg/"
    file = path + brats_name +'_seg__.nii.gz'

    seg_unet = nib.load(file).get_fdata()
    seg_unet[seg_unet==3] = 4

    return [seg_unet],dict(U='net')
# 1 - BLEU, TC  -> NECROSE INACTIVE, TUMORUS CORE
# 2 - ORANGE, WT -> INFILTRATION, OEDEME, WHOLE TUMOR
# 4 - ROUGE, ET -> NECROSE ACTIVE, ENHENCING TUMOR
if __name__ == '__main__':

    csv_file = ROOT_DIRECTORY + "/persistent_homology/results/"
    csv_file += "benchmark_segment_TDA_IPNP.csv"

    # make_new_brast_list(500)
    brats_list = BRATS_LIST_DFLT
    # brats_list = make_brats_list_from_csv(csv_file)

    spb = seg_parse_brats(brats_list,smooth_flair =1,smooth_t1ce=1)


    for i, img_data in enumerate(spb):
        I_flair,I_t1ce,seg_real,brats_name = img_data
        print("\n ==========================================")
        print(f"{brats_name} oppened {i+1}/{len(brats_list)}")
        i3v.imshow_3d_slider(I_flair)
        plt.show()
        # Apply segmentation TDA
        # plot = False
        # verbose = True
        # n_H0_index = 0
        # n_H2_index = 2
        FinalSegmentations_TDA,params_vals = Segmentations(I_flair,I_t1ce,n_H2=1,dt_threshold=1)
        dice_list_TDA = compute_dices(FinalSegmentations_TDA,seg_real)


        FinalSegmentations_unet,params_vals = get_evaluate_segUnet(brats_name)
        # i3v.compare_3D_images_vedo(FinalSegmentations,seg_real)
        dice_list_unet = compute_dices(FinalSegmentations_unet,seg_real)
        print(dice_list_TDA,dice_list_unet)

        i3v.compare_3D_images_vedo(FinalSegmentations_TDA,seg_real)
        i3v.compare_3D_images_vedo(FinalSegmentations_unet,seg_real)

        # params_vals = [dict(pr=3,ffr=5),dict(gt=4,ft=1)]
        # dice_list = [[9,10,11,12],[5,6,7,8]]

        # for pv,dices in zip(params_vals,dice_list):
        #     # column_name = format_colName_csv(pv)
        #     column_name = 'Unet'
        #     write_csv(csv_file,brats_name,column_name,dices)



    # i3v.compare_3D_images_vedo(img_flair,img_t1ce)
    # # Apply segmentation
    # tda_segmentation = Segmentation(img_flair,img_t1ce,plot=True,verbose=True)
    # tda_segmentation = tda_segmentation/tda_segmentation.max()
    #
    # i3v.compare_3D_images_vedo(tda_segmentation,img_t1ce)
    # i3v.compare_3D_images_vedo(real_seg,img_flair)
    #
    # i3v.compare_3D_images_vedo(tda_segmentation,real_seg)

#