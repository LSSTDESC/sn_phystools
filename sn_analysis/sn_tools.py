#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 6 13:23:18 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import matplotlib.pyplot as plt
import numpy as np
from sn_tools.sn_io import loopStack_params
from sn_tools.sn_utils import multiproc
import pandas as pd
import glob
import operator
from astropy.cosmology import w0waCDM
from scipy.interpolate import interp1d
from scipy.integrate import quad


def dmu_over_dz(Om=0.3, w0=-1.0, wa=0., plot=False):
    """
    Function to estimate distance modulus derivative vs z

    Parameters
    ----------
    Om : str, optional
        Omega_m parameter. The default is 0.3.
    w0 : str, optional
        w0 DE eq. state parameter. The default is -1.0.
    wa : str, optional
        wa DE eq. state parameter. The default is 0..
    plot : bool, optional
        to display the results. The default is False.

    Returns
    -------
    res : pandas df
        columns: z, dmu_over_dz.

    """

    cosmo = CosmoDist()

    zstep = 0.01
    zmin = 0.01
    zmax = 1.2+zstep
    z = np.arange(zmin, zmax, zstep)
    h = 1.e-8
    zh = np.arange(zmin+h, zmax+h, zstep)
    cref = cosmo.mu_astro(z, Om, w0, wa)
    # ctest = cosmo.mu(z, Om, w0, wa)
    ch = cosmo.mu_astro(zh, Om, w0, wa)
    deriv_mu = (ch-cref)/h
    res = pd.DataFrame(z, columns=['z'])
    res['dmu_over_dz'] = deriv_mu

    if plot:
        fig, ax = plt.subplots()
        ax.plot(res['z'], res['dmu_over_dz'])

        ax.grid()
        ax.set_xlabel('$z$')
        ax.set_ylabel('$dmu_dz$')
        plt.show()

    return res


class CosmoDist:
    """
    class to estimate cosmology parameters

    Parameters
    ---------------
    H0 : float,opt
      Hubble cte (default: 72.  # km.s-1.Mpc-1)
    c: float, opt
     speed of the light (default: = 299792.458  # km.s-1)

    """

    def __init__(self, H0=70, c=2.99792e5):

        self.H0 = H0
        self.c = c

    def dL(self, z, Om=0.3, w0=-1., wa=0.0):

        cosmology = w0waCDM(H0=self.H0,
                            Om0=Om,
                            Ode0=1.-Om,
                            w0=w0, wa=wa)

        return cosmology.luminosity_distance(z).value*1.e6

    def cosmo_func(self, z, Om=0.3, w0=-1.0, wa=0.0):
        """
        Method to estimate the integrand for the luminosity distance

        Parameters
        ---------------
        z: float
          redshift
        Om: float, opt
          Omega_m parameter (default: 0.3)
        w0: float, opt
         w0 DE parameter (default: -1.0)
        wa: float, opt
          wa DE parameter (default: 0.)

        Returns
        -----------
        the integrand (float)

        """
        wp = w0+wa*z/(1.+z)
        # wp = w0

        H = Om*(1+z)**3+(1.-Om)*(1+z)**(3*(1.+wp))
        # H = Om*(1+z)**3+(1.-Om)*(1+z)

        fu = np.sqrt(H)

        return 1/fu

    def dL_old(self, z, Om=0.3, w0=-1., wa=0.0):
        """
        Method to estimate the luminosity distance

        Parameters
        ---------------
      z: float
           redshift
        Om: float, opt
          Omega_m parameter (default: 0.3)
        w0: float, opt
         w0 DE parameter (default: -1.0)
       wa: float, opt
         wa DE parameter (default: 0.)

        Returns
        ----------
        luminosity distance
        """
        norm = self.c/self.H0
        norm *= 1.e6

        def integrand(x): return self.integrand(x, norm, Om, w0, wa)

        if (hasattr(z, '__iter__')):
            s = np.zeros(len(z))
            for i, t in enumerate(z):
                s[i] = (1+t)*quad(integrand, 0.0, t, limit=100)[0]
            return s
        else:
            return (1+z)*quad(integrand, 0.0, z, limit=100)[0]

    def integrand(self, z, norm, Om, w0, wa):

        return norm*self.cosmo_func(z, Om, w0, wa)

    def mu_old(self, z, Om=0.3, w0=-1.0, wa=0.0):
        """
        Method to estimate distance modulus

        Parameters
        ---------------
        z: float
           redshift
        Om: float, opt
          Omega_m parameter (default: 0.3)
        w0: float, opt
          w0 DE parameter (default: -1.0)
        wa: float, opt
            wa DE parameter (default: 0.)

        Returns
        -----------
        distance modulus (float)

        """

        if (hasattr(z, '__iter__')):
            return np.log10(self.dL(z, Om, w0, wa))*5-5
        else:
            return (np.log10(self.dL([z], Om, w0, wa))*5-5)[0]

        # return 5.*np.log10(self.dL(z, Om, w0, wa))+25. #if dL in Mpc

    def mu(self, z, Om=0.3, w0=-1.0, wa=0.0):
        """
        Method to estimate distance modulus

        Parameters
        ---------------
        z: float
           redshift
        Om: float, opt
          Omega_m parameter (default: 0.3)
        w0: float, opt
          w0 DE parameter (default: -1.0)
        wa: float, opt
            wa DE parameter (default: 0.)

        Returns
        -----------
        distance modulus (float)

        """

        if (hasattr(z, '__iter__')):
            return np.log10(self.dL(z, Om, w0, wa))*5-5
        else:
            return (np.log10(self.dL([z], Om, w0, wa))*5-5)[0]

    def mu_astro(self, z, Om, w0, wa):

        cosmology = w0waCDM(H0=self.H0,
                            Om0=Om,
                            Ode0=1.-Om,
                            w0=w0, wa=wa)

        return cosmology.distmod(z).value

    def mufit(self, z, alpha, beta, Mb, x1, color, mbfit,
              Om=0.3, w0=-1.0, wa=0.0):

        return mbfit+alpha*x1-beta*color-self.mu(z, Om, w0, wa)-Mb


def loadData(theDir, dbName, inDir, field='COSMOS'):
    """
    Funtion to load data

    Parameters
    ----------
    theDir : str
        location dir.
    dbName : str
        dbName.
    field : str, opt
        field to process

    Returns
    -------
    res : astropytable
        loaded data.

    """

    searchname = '{}/{}/{}/SN*{}*.hdf5'.format(theDir, dbName, inDir, field)
    print('searching for', searchname)
    files = glob.glob(searchname)

    if len(files) == 0:
        return pd.DataFrame()

    #restot = pd.DataFrame()
    params = dict(zip(['objtype'], ['astropyTable']))
    restot = multiproc(files, params, loopStack_params, 8)
    restot.convert_bytestring_to_unicode()

    """
    for fi in files:
        res = loopStack([fi], objtype='astropyTable').to_pandas()
        restot = pd.concat((restot, res))
        """
    return restot.to_pandas()


def load_complete_dbSimu(dbDir, dbName, inDir, alpha=0.4, beta=3,
                         listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb'):
    """


    Parameters
    ----------
    dbDir : TYPE
        DESCRIPTION.
    dbName : TYPE
        DESCRIPTION.
    inDir : TYPE
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.4.
    beta : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """

    res = pd.DataFrame()
    fields = listDDF.split(',')
    for field in fields:
        ll = loadData(dbDir, dbName, inDir, field)
        ll['field'] = field
        res = pd.concat((res, ll))
    print('loaded', len(res), len(res['healpixID'].unique()))
    res = complete_df(res, alpha, beta)

    return res


def complete_df(res, alpha=0.4, beta=3, Mb=-19.1):
    """
    Function to complete df infos

    Parameters
    ----------
    res : pandas df
        df to complete.
    alpha : floar, optional
        alpha parameter for the estimation of mu,sigma_mu. The default is 0.4.
    beta : float, optional
        beta parameter for the estimation of mu,sigma_mu. The default is 3.
    Mb: float, optional
        Mb parameter for the estimation of mu. The default is -19.1

    Returns
    -------
    res : pandas df
        completed df.

    """

    # get dmu_over_dz vs z
    """
    dmu_dz = dmu_over_dz(plot=True)
    # make 1d interpolator out of it
    interp_dmudz = interp1d(dmu_dz['z'], dmu_dz['dmu_over_dz'],
                            bounds_error=False, fill_value=0.)
    res['deriv_mu_z'] = interp_dmudz(res['z'])
    
    print(res.columns)
    """
    res['sigmaC'] = np.sqrt(res['Cov_colorcolor'])
    res['sigmat0'] = np.sqrt(res['Cov_t0t0'])
    res['sigmax1'] = np.sqrt(res['Cov_x1x1'])
    res['Cov_mbmb'] = (
        2.5 / (res['x0_fit']*np.log(10)))**2*res['Cov_x0x0']
    res['Cov_x1mb'] = -2.5*res['Cov_x0x1'] / \
        (res['x0_fit']*np.log(10))
    res['Cov_colormb'] = -2.5*res['Cov_x0color'] / \
        (res['x0_fit']*np.log(10))
    if 'Cov_x0z' in res.columns:
        res['Cov_mbz'] = -2.5*res['Cov_x0z'] / \
            (res['x0_fit']*np.log(10))

    res['sigma_mu'] = res.Cov_mbmb\
        + (alpha**2)*res.Cov_x1x1\
        + (beta**2)*res.Cov_colorcolor\
        + 2*alpha*res.Cov_x1mb-2*beta*res.Cov_colormb\
        - 2*alpha*beta*res.Cov_x1color
    if 'Cov_x0z' in res.columns:
        res['sigma_mu'] += res.deriv_mu_z**2*res.Cov_zz\
            + 2.*res.deriv_mu_z*res.Cov_mbz\
            + 2.*alpha*res.deriv_mu_z*res.Cov_x1z\
            - 2.*beta*res.deriv_mu_z*res.Cov_colorz

    res['sigma_mu'] = np.sqrt(res['sigma_mu'])
    res['mb_fit'] = -2.5*np.log10(res['x0_fit']) + 10.635
    res['mu'] = res['mb_fit']+alpha * \
        res['x1_fit']-beta*res['color_fit']-Mb

    return res


def plotSN_2D(data, varx='z', legx='z', vary='sigma_mu', legy='$\sigma_{\mu}$'):
    """
    function to perform 2D plots

    Parameters
    ----------
    data : pandas df
        data to plot.
    varx : str, optional
        x-axis var. The default is 'z'.
    legx : str, optional
        x-axis label. The default is 'z'.
    vary : str, optional
        y-axis var. The default is 'sigma_mu'.
    legy : str, optional
        y-axis label. The default is '$\sigma_{\mu}$'.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data[varx], data[vary], 'k.')

    ax.set_xlabel(legx)
    ax.set_ylabel(legy)


def plotSN_2D_binned(data, varx='z', legx='z', bins=np.arange(0.5, 0.6, 0.01),
                     vary='sigma_mu', legy='$\sigma_{\mu}$'):
    """
    function to perform 2D plots

    Parameters
    ----------
    data : pandas df
        data to plot.
    varx : str, optional
        x-axis var. The default is 'z'.
    legx : str, optional
        x-axis label. The default is 'z'.
    vary : str, optional
        y-axis var. The default is 'sigma_mu'.
    legy : str, optional
        y-axis label. The default is '$\sigma_{\mu}$'.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    group = data.groupby(pd.cut(data[varx], bins))
    bin_centers = (bins[: -1] + bins[1:])/2
    y = group[vary].mean()
    yerr = group[vary].std()
    ax.errorbar(bin_centers, y, yerr=yerr, color='k', marker='.')

    ax.set_xlabel(legx)
    ax.set_ylabel(legy)


def plotSN_effi(data, xvar='n_epochs_aft', bins=range(1, 20, 1),
                var_cut='sigmaC', var_sel=0.04, op=operator.le):
    """
    Function to estimate and plot efficiency

    Parameters
    ----------
    data : pandas df
        data to process.
    xvar : str, optional
        x-axis var. The default is 'n_epochs_aft'.
    bins : list(int), optional
        bins for efficiency estimation. The default is range(0, 20, 1).
    var_cut : str, optional
        selection var. The default is 'sigmaC'.
    var_sel : float, optional
        selection val. The default is 0.04.
    op : operator, optional
        operator for sel. The default is operator.le.

    Returns
    -------
    None.

    """

    group = data.groupby(pd.cut(data[xvar], bins))
    idx = op(data[var_cut], var_sel)
    sel_data = data[idx]
    group_sel = sel_data.groupby(pd.cut(sel_data[xvar], bins))

    # estimate efficiency here
    effi = group_sel.size()/group.size()

    print(effi)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bins[:-1], effi, 'ko')


def loadData_fakeSimu(theDir, theFile=''):

    if theFile == '':
        searchname = '{}/SN*.hdf5'.format(theDir)
        print('searching for', searchname)
        files = glob.glob(searchname)
    else:
        files = ['{}/{}'.format(theDir, theFile)]

    #restot = pd.DataFrame()
    params = dict(zip(['objtype'], ['astropyTable']))
    restot = multiproc(files, params, loopStack_params, 1)
    restot.convert_bytestring_to_unicode()

    return restot.to_pandas()
