#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:16:59 2023

@author: philippe.gris@clermont.in2p3.fr
"""
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


class Calc_zlim:
    def __init__(self, effic, frac=0.95):
        """
        class to estimate and plot the redshift limit

        Parameters
        ----------
        effic : array
            array of efficiencies.
        frac : float, optional
            cumsum value for zlim estimator. The default is 0.95.

        Returns
        -------
        None.

        """

        from sn_tools.sn_rate import SN_Rate

        rateSN = SN_Rate(rate='Perrett', H0=72, Om0=0.3)

        zz, rate, err_rate, nsn, err_nsn = rateSN(zmin=0.01,
                                                  zmax=1.1,
                                                  dz=0.001,
                                                  duration=180.,
                                                  survey_area=9.6,
                                                  account_for_edges=True)

        effic = effic.fillna(0.)
        effic = effic.to_records(index=False)
        interp = interpolate.interp1d(effic['z'], effic['effi'],
                                      bounds_error=False, fill_value=0.)

        self.nsn = rate*interp(zz)
        self.zz = zz
        self.frac = frac
        self.zlim = self.grab_zlim()

    def grab_zlim(self):
        """
        Method where zlim is estimated

        Returns
        -------
        float
            redshift limit value.

        """

        norm = self.nsn.cumsum()[-1]
        nsn_cum = self.nsn.cumsum()/norm

        val = interpolate.interp1d(nsn_cum, self.zz,
                                   bounds_error=False, fill_value=0.)

        return np.round(val(self.frac), 5)

    def plot_zlim(self, zlim):
        """
        Method to plot cumsum vs z and zlim value

        Parameters
        ----------
        zlim : float
            redshift limit value.

        Returns
        -------
        None.

        """

        norm = self.nsn.cumsum()[-1]
        nsn_cum = self.nsn.cumsum()/norm
        fig, ax = plt.subplots()

        ax.plot(self.zz, nsn_cum)

        xmin = 0.01
        xmax = 1.0
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([0., 1.01])
        ax.plot([xmin, zlim], [self.frac]*2, color='k', linestyle='dashed')
        ax.plot([zlim]*2, [0.01, self.frac], color='k', linestyle='dashed')
        ax.set_xlabel('$z$')
        ax.set_ylabel(r'frac($N_{SN}^{z<}$)')
        ax.grid()


def effi(resa, resb, xvar='z', bins=np.arange(0.01, 1.1, 0.02)):

    groupa = resa.groupby(pd.cut(resa[xvar], bins))
    groupb = resb.groupby(pd.cut(resb[xvar], bins))

    effi = groupb.size()/groupa.size()

    bin_centers = (bins[: -1] + bins[1:])/2

    effi_err = np.sqrt(groupb.size()*(1.-effi))/groupa.size()

    df = pd.DataFrame(bin_centers, columns=[xvar])
    df['effi'] = effi.to_list()
    df['effi_err'] = effi_err.to_list()

    return df


def plot_effi(effival, xvar='z', leg='',
              bins=np.arange(0.01, 1.1, 0.01), fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    #effival = effi(resa, resb, xvar=xvar, bins=bins)

    x = effival[xvar]
    y = effival['effi']
    yerr = effival['effi_err']
    ax.errorbar(x, y, yerr=yerr, label=leg, marker='o', color='k')


def select(res, list_sel):
    """
    Function to select a pandas df

    Parameters
    ----------
    res : pandas df
        data to select.

    Returns
    -------
    pandas df
        selected df.

    """
    idx = True
    for vals in list_sel:
        idx &= vals[1](res[vals[0]], vals[2])

    return res[idx]


def histSN_params(data, vars=['x1', 'color', 'z', 'daymax']):
    """
    Function to plot SN parameters

    Parameters
    ----------
    data : pandas df
        data to plot.
    vars : list(str), optional
        params to plot. The default is ['x1', 'color', 'z', 'daymax'].

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ipos = [(0, 0), (0, 1), (1, 0), (1, 1)]

    jpos = dict(zip(vars, ipos))

    for key, vals in jpos.items():
        ax[vals].hist(data[key])
        ax[vals].set_xlabel(key)
        ax[vals].set_ylabel('Number of Entries')


def zlimit(tab, covcc_col='Cov_colorcolor', z_col='z', sigmaC=0.04):
    """
    Function to estimate zlim for sigmaC value

    Parameters
    ---------------
    tab: astropy table
      data to process: columns covcc_col and z_col should be in this tab
    covcc_col: str, opt
        name of the column corresponding to the cov_colorcolor value (default: Cov_colorcolor)
    z_col: str, opt
       name of the column corresponding to the redshift value (default: z)
    sigmaC: float, opt
      sigma_color value to estimate zlimit from (default: 0.04)

    Returns
    ----------
    The zlimit value corresponding to sigmaC

    """
    interp = interp1d(np.sqrt(tab[covcc_col]),
                      tab[z_col], bounds_error=False, fill_value=0.)

    interpv = interp1d(tab[z_col], np.sqrt(tab[covcc_col]),
                       bounds_error=False, fill_value=0.)

    zmin = tab[z_col].min()
    zmax = tab[z_col].max()
    zvals = np.arange(zmin, zmax, 0.005)

    colors = interpv(zvals)
    ii = np.argmin(np.abs(colors-sigmaC))
    res = interp(sigmaC)
    # print(colors)
    return np.round(res, 4)
