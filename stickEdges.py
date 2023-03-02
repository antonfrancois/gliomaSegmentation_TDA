
import numpy as np
import matplotlib.pyplot as plt


from misc import *
import segmentation_TDA as sTDA
def respect_edges(img_flair,true_seg,t_suggested,pos,
                  n_test = 20,ovrlap=.1, plot= False,ax= None,edgeDice=None):
    edgesDICE_stock = np.zeros(n_test)
    if plot or (not ax is None): DICE_stock = np.zeros(n_test)
    t_list= np.linspace(t_suggested - ovrlap, t_suggested + 1.5*ovrlap,n_test)

    if edgeDice is None:
        edgeDice = EdgesMatch_Dice(img_flair)
    best_ed = -1

    for i,t in enumerate(t_list):
        try:
            seg_union = sTDA.getConnectedComponent(img_flair,pos,t)
        except ValueError:
            seg_union = np.zeros(true_seg.shape)
        # seg_union = get_largest_CC(img_flair, t, verbose=False)
        edgesDICE_stock[i]= edgeDice(seg_union)
        if plot or (not ax is None):
            DICE_stock[i] = DICE(seg_union,true_seg)
        # # select the new best t
        if edgesDICE_stock[i] > best_ed:
            best_ed = edgesDICE_stock[i]
            best_seg = seg_union
            new_t = t

    # # select the new best t
    # new_t_idx = np.argmax(edgesDICE_stock)
    # new_t = t_list[new_t_idx]

    if plot or (not ax is None):
        if ax is None:
            fig,ax = plt.subplots(figsize=(5,5))
        ax.plot(t_list,edgesDICE_stock,'o-',label="edges_DiceMatch")
        ax.plot(t_list,DICE_stock,'D-',label="Dice")
        ax.plot([t_suggested,t_suggested],
                [
                    min(DICE_stock.min(),edgesDICE_stock.min()),
                    max(DICE_stock.max(),edgesDICE_stock.max())
                ],
                '--', label="t_sugg")
        ax.plot([new_t,new_t],
                [
                    min(DICE_stock.min(),edgesDICE_stock.min()),
                    max(DICE_stock.max(),edgesDICE_stock.max())
                ],
                '--', label="new_t")
        ax.set_xlabel("t")
        ax.set_ylabel("Dices")


        ax.legend()

    return new_t,best_seg,ax

def plot_respectEdges(img_flair,
                        seg,
                        t_suggested,
                        pos,
                        n_test=11,
                        brats_name=None,
                        ovrlap=.1,
                        save_fig= False):
    img_flair_b = scipynd.gaussian_filter(img_flair,sigma=1)
    edgeDice = EdgesMatch_Dice(img_flair)
    fig, axs = plt.subplots(2,2,
                            figsize = (10,5),
                            gridspec_kw={'width_ratios': [1, 2]},
                            constrained_layout=True)
    gs = axs[0, 0].get_gridspec()
    # remove the underlying axes
    axs[0,0].remove()
    axs[1,0].remove()
    axbig = fig.add_subplot(gs[:, 0])
    new_t,new_seg,axbig = respect_edges(img_flair_b,seg,t_suggested,pos,
                                        ax=axbig, n_test=n_test,ovrlap=ovrlap,
                                        edgeDice=edgeDice)
    edgeDice(new_seg)
    img_flat = make_3d_flat(img_flair,pos)
    axs[0,1].imshow(img_flat,alpha = 1, **DLT_KW_IMAGE)
    axs[0,1].imshow(imCmp(
        make_3d_flat(new_seg,pos),
        make_3d_flat(seg,pos),
        method='seg'
    ),
        alpha = .5
    )
    if not brats_name is None:
        axs[0,1].text(175,25,brats_name,c='white',fontsize=20)
    axs[0,1].text(200,225,f"DICE = {DICE(new_seg,seg):.4f}",c='white',fontsize=20)
    set_ticks_off(axs[0,1])
    axs[1,1].imshow(
        imCmp(
        make_3d_flat(edgeDice.seg_edges,pos),
        make_3d_flat(edgeDice.img_edges,pos),
        method='seg'
        )
    )
    set_ticks_off(axs[1,1])

    if save_fig:
        path = ROOT_DIRECTORY
        fig.savefig(path +f"edges_refine_plot_{brats_name}.pdf")
    return new_seg