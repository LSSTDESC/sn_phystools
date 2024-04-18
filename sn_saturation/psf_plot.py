#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:08:23 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
from . import plt


def plot_pixel(seeing,
               psf_type,
               varxsel,
               xp,
               varysel,
               yp,
               varxplot,
               varyplot,
               titleadd,
               type_plot='imshow'):
    """
    Display of the pixels fraction distribution

    Parameters
    ----------
    seeing : float
        seeing used for the plot.
    psf_type : str
        PSF type.
    varxsel : str
        x variable to display.
    xp : float
        x of the flux center or of the pixel center.
    varysel : str
        y variable to display.
    yp : float
        y of the flux center or of the pixel center.
    varxplot : TYPE
        DESCRIPTION.
    varyplot : TYPE
        DESCRIPTION.
    titleadd : str
        Title to add.
    type_plot : str, optional
        Type of plot (imshow/contour). The default is 'imshow'.

    Returns
    -------
    None.

    """

    res = np.load('PSF_pixel_{}.npy'.format(psf_type))
    print(res.dtype)
    fontsize = 20
    idx = np.abs(res['seeing']-seeing) < 1.e-3
    print(len(res[idx]))
    idx &= np.abs(res[varxplot]-xp) < 1.e-5
    idx &= np.abs(res[varyplot]-yp) < 1.e-5
    sel = res[idx]
    print('here man', varxplot, varyplot, seeing, len(sel))
    # print('selection',len(sel),np.unique(res[[varxsel,varysel]]))
    fig, ax = plt.subplots(figsize=(10, 10))
    seeing_pix = seeing/0.2  # seeing in arcsec - pixel LSST = 0.2"
    sigma = seeing_pix/2.355
    # titleform = 'seeing: {} - sigma: {} pixel'.format(np.round(seeing,2),np.round(sigma,2))
    titleform = '{} - seeing: {}"'.format(titleadd, np.round(seeing, 2))
    # fig.suptitle(titleform,fontsize=fontsize)
    # ax.set_title(titleform,fontsize=fontsize)
    ax.set_title(titleform)
    dim = int(np.sqrt(len(sel)))
    xcm = np.reshape(sel[varxsel], (dim, dim))
    ycm = np.reshape(sel[varysel], (dim, dim))
    Zc = np.reshape(sel['pixel_frac'], (dim, dim))

    print('hhh', np.min(xcm), np.max(xcm), dim, len(sel))
    if type_plot == 'contour':
        CS = ax.contourf(xcm, ycm, Zc, 20, cmap=plt.cm.viridis)
    if type_plot == 'imshow':
        CS = ax.imshow(Zc, extent=[np.min(xcm), np.max(
            xcm), np.min(ycm), np.max(ycm)])
    # ax.plot(xxc_m, yyc_m, 'k.')
    vmin = np.min(sel['pixel_frac'])
    vmax = np.max(sel['pixel_frac'])
    vmin = np.round(vmin, 2)
    vmax = np.round(vmax, 2)
    print(vmin, vmax, np.arange(vmin, vmax, 0.05))
    shrink = 0.82
    # shrink = 1.
    cbar = fig.colorbar(CS, ax=ax, ticks=np.arange(
        vmin, vmax, 0.01), shrink=shrink)
    """
    cbar.set_label('Flux fraction', rotation=270,
                   fontsize=fontsize,labelpad=30)# position=(12.,0.5))
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(r'x [pixel]',fontsize=fontsize)
    ax.set_ylabel(r'y [pixel]',fontsize=fontsize)
    ax.tick_params(labelsize = fontsize)
    """
    cbar.set_label('Flux fraction', rotation=270,
                   labelpad=30)  # position=(12.,0.5))
    # cbar.ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(r'x [pixel]')
    ax.set_ylabel(r'y [pixel]')
    # ax.tick_params(labelsize = fontsize)
    ax = plt.gca()
    ax.set_aspect('equal')
    # ax.set_xlim([-3.,3.])
    # ax.set_ylim([-3.,3.])
    # cbar.set_clim(np.round(vmin,1), np.round(vmax,1))
    # plt.show()
    plt.savefig('flux_dist_center_position.png')
