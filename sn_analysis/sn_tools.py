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


def loadData(theDir, dbName, inDir, field='COSMOS', seasons='*', nproc=8,
             dataType='pandasDataFrame', suffix=''):
    """
    Funtion to load data

    Parameters
    ----------
    theDir : str
        location dir.
    dbName : str
        dbName.
    field : str, opt
        field to process. The default is 'COSMOS'.
    seasons: str, opt
        list of seasons to process. The default is * (all seasons). 
    nproc: int, optional.
     number of procs for multiprocessing. The default is 8.
    dataType: str, opt.
      data type to process. The default is 'pandasDataFrame'
    suffix: str, opt
      suffix for file name. The default is ''

    Returns
    -------
    res : dataType
        loaded data.

    """

    seas = seasons.split(',')

    files = []
    for sea in seas:
        """
        if field == 'WFD':
            searchname = '{}/{}/{}/SN*{}*_{}_0.01_0.7.hdf5'.format(
                theDir, dbName, inDir, field, sea)
        else:
            searchname = '{}/{}/{}/SN*{}*_{}.hdf5'.format(
                theDir, dbName, inDir, field, sea)
        """
        searchname = '{}/{}/{}/SN*{}*_{}{}.hdf5'.format(
            theDir, dbName, inDir, field, sea, suffix)

        print('searching for', searchname)
        files += glob.glob(searchname)

    if len(files) == 0:
        return pd.DataFrame()

    # restot = pd.DataFrame()
    params = dict(zip(['objtype'], [dataType]))
    # params = dict(zip(['objtype'], ['pandasDataFrame']))

    restot = multiproc(files, params, loopStack_params, nproc)
    # restot.convert_bytestring_to_unicode()

    # resfi = restot.to_pandas()
    """
    for fi in files:
        res = loopStack([fi], objtype='astropyTable').to_pandas()
        restot = pd.concat((restot, res))
        """
    return restot


def load_complete_dbSimu(dbDir, dbName, inDir, alpha=0.13, beta=3.1,
                         listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                         seasons='*', nproc=8,
                         dataType='pandasDataFrame', suffix=''):
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
    seasons: str, optional.
        list of seasons to process. The default is * (all seasons).
    nproc: int, optional.
        number of procs for multiprocessing. The default is 8.
    dataType: str, opt.
      data type to process. The default is 'pandasDataFrame'
    suffix: str, opt.
     suffix for file name to process. The default is ''

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    from astropy.table import Table
    res = pd.DataFrame()
    fields = listDDF.split(',')
    for field in fields:
        ll = loadData(dbDir, dbName, inDir, field,
                      seasons=seasons, nproc=nproc,
                      dataType=dataType, suffix=suffix)
        ll['field'] = field
        if isinstance(ll, Table):
            ll.convert_bytestring_to_unicode()
            ll = ll.to_pandas()

        res = pd.concat((res, ll))

    if not res.empty:
        print('loaded', len(res), len(res['healpixID'].unique()))
        res = complete_df(res, alpha, beta)

    return res


def complete_df(res, alpha=0.13, beta=3.1, Mb=-19.1):
    """
    Function to complete df infos

    Parameters
    ----------
    res : pandas df
        df to complete.
    alpha : floar, optional
        alpha parameter for the estimation of mu,sigma_mu. The default is 0.13
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
    if 'Cov_zz' in res.columns:
        dmu_dz = dmu_over_dz(plot=False)
        # make 1d interpolator out of it
        interp_dmudz = interp1d(dmu_dz['z'], dmu_dz['dmu_over_dz'],
                                bounds_error=False, fill_value=0.)
        res['deriv_mu_z'] = interp_dmudz(res['z'])

    res['sigmaC'] = np.sqrt(res['Cov_colorcolor'])
    res['sigmat0'] = np.sqrt(res['Cov_t0t0'])
    res['sigmax1'] = np.sqrt(res['Cov_x1x1'])
    res['Cov_mbmb'] = (
        2.5 / (res['x0_fit']*np.log(10)))**2*res['Cov_x0x0']
    res['Cov_x1mb'] = -2.5*res['Cov_x0x1'] / \
        (res['x0_fit']*np.log(10))
    res['Cov_colormb'] = -2.5*res['Cov_x0color'] / \
        (res['x0_fit']*np.log(10))
    if 'Cov_zx0' in res.columns:
        res['Cov_zmb'] = -2.5*res['Cov_zx0'] / \
            (res['x0_fit']*np.log(10))

    res['sigma_mu'] = res.Cov_mbmb\
        + (alpha**2)*res.Cov_x1x1\
        + (beta**2)*res.Cov_colorcolor\
        + 2*alpha*res.Cov_x1mb-2*beta*res.Cov_colormb\
        - 2*alpha*beta*res.Cov_x1color

    if 'Cov_zz' in res.columns:
        res['sigma_mu'] += res.deriv_mu_z**2*res.Cov_zz\
            + 2.*res.deriv_mu_z*res.Cov_zmb\
            + 2.*alpha*res.deriv_mu_z*res.Cov_zx1\
            - 2.*beta*res.deriv_mu_z*res.Cov_zcolor

    res['sigma_mu'] = np.sqrt(res['sigma_mu'])
    res['mb_fit'] = -2.5*np.log10(res['x0_fit']) + 10.635
    res['mu'] = res['mb_fit']+alpha * \
        res['x1_fit']-beta*res['color_fit']-Mb

    res['mb'] = -2.5*np.log10(res['x0']) + 10.635
    res['mu_exp'] = res['mb']+alpha * \
        res['x1']-beta*res['color']-Mb
    res['diff_mu'] = res['mu_exp']-res['mu']
    res['diff_mb'] = (res['mb']-res['mb_fit'])

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

    # restot = pd.DataFrame()
    params = dict(zip(['objtype'], ['astropyTable']))
    restot = multiproc(files, params, loopStack_params, 1)
    restot.convert_bytestring_to_unicode()

    return restot.to_pandas()


def transform(dicta):
    """
    Function to transform a dict of var to a dict of list(var)

    Parameters
    ----------
    dicta : dict
        input dict.

    Returns
    -------
    dictb : dict
        output dict.

    """

    dictb = {}

    for key, vals in dicta.items():
        dictb[key] = [vals]

    return dictb


def load_cosmo_data(theDir, dbName, cols_group, spectro_config,
                    cols=['MoM', 'WFD_TiDES', 'all_Fields']):
    """

    Function to load cosmo data and estimate (mean, std)

    Parameters
    ----------
    theDir : str
        Data dir.
    dbName : str
        dbName of interest.
    timescale : str
        Time scale (season/year).
    spectro_config : str
        survey spectro config.
    cols : list(int), optional
        List of columns to estimate (mean,std). 
        The default is ['MoM', 'WFD_TiDES', 'all_Fields'].

    Returns
    -------
    dfb : TYPE
        DESCRIPTION.

    """

    fName = '{}/cosmo_*{}*.hdf5'.format(theDir, dbName)
    fis = glob.glob(fName)
    if len(fis) == 0:
        print('Problem here: file not found in path', fName)

    df = pd.DataFrame()

    for fi in fis:
        dd = pd.read_hdf(fi)
        if 'nsn_z_0.8_sigma_mu' in dd.columns:
            dd['nsn_rat_highz'] = dd['nsn_z_0.8_sigma_mu'] / dd['nsn_z_0.8']
        df = pd.concat((df, dd))

    # re-calculate SMoM here if necessary
    if 'wa_fit' not in df.columns:
        df = recalc(df)

    for vv in ['Om0', 'w0', 'wa']:
        vvb = 'Cov_{}_{}_fit'.format(vv, vv)
        if vvb in df.columns:
            df['sigma_{}'.format(vv)] = df[vvb]**0.5

    dictagg = {}

    colsb = set(df.columns).intersection(set(cols))
    colsb = list(colsb)

    for ccol in colsb:
        dictagg[ccol] = ['mean', 'std']

    dfb = df.groupby(cols_group).agg(dictagg).reset_index()

    cols_fi = cols_group.copy()
    for vv in colsb:
        cols_fi += ['{}_mean'.format(vv), '{}_std'.format(vv)]

    dfb.columns = cols_fi

    diffcol = set(cols).difference(set(colsb))
    if diffcol:
        for vv in diffcol:
            dfb['{}_mean'.format(vv)] = 0
            dfb['{}_std'.format(vv)] = 0
    dfb['dbName'] = dbName
    dfb['spectro_config'] = spectro_config

    return dfb


def get_cov_name(a, b, df):
    """
    Function to estimate the covariance(a,b) name

    Parameters
    ----------
    a : str
        first tag for the name.
    b : str
        second tag for the name.
    df : pandas df
        Data to process.

    Returns
    -------
    str
      the Cov(a,b) name in df.

    """

    vva = 'Cov_{}_{}_fit'.format(a, b)
    vvb = 'Cov_{}_{}_fit'.format(b, a)

    if vva in df.columns:
        return vva
    else:
        return vvb


def recalc(df, cova='Cov_Om0_Om0_fit',
           covb='Cov_w0_w0_fit',
           covab='Cov_Om0_w0_fit',
           delta_chi=6.17):
    """
    Function to recalc the SMoM metric

    Parameters
    ----------
    df : pandas df
        Data to process.
    cova : str, optional
        first var cov. The default is 'Cov_Om0_Om0_fit'.
    covb : str, optional
        second var cov. The default is 'Cov_w0_w0_fit'.
    covab : str, optional
        cov(a,b). The default is 'Cov_Om0_w0_fit'.
    delta_chi : float, optional
        Chisquare (C.L.). The default is 6.17.

    Returns
    -------
    df : pandas df
        original df plus SMoM value.

    """

    sigma_Om0 = df[cova]**0.5
    sigma_w0 = df[covb]**0.5
    covab = get_cov_name('Om0', 'w0', df)
    rho = df[covab]/(sigma_Om0*sigma_w0)
    smom_inv = delta_chi*sigma_w0*sigma_Om0*(1.-rho**2)**0.5

    df['MoM'] = 1./smom_inv
    df['sigma_w0'] = sigma_w0

    return df


def get_spline(df, xvar, yvar):

    from scipy.interpolate import make_interp_spline
    xnew = np.linspace(np.min(df[xvar]), np.max(df[xvar]), 100)
    spl = make_interp_spline(df[xvar], df[yvar], k=3)  # type: BSpline
    spl_smooth = spl(xnew)

    return xnew, spl_smooth


def get_pulls(data):
    """
    Function to estimate the pulls

    Parameters
    ----------
    data : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        processed data.

    """
    from scipy import stats

    data = pull_it(data)

    r = []
    cols = []

    vvals = ['x1', 'color', 'mb', 'mu']
    # vvals = ['color']
    for vv in vvals:
        pullvar = 'pull_{}'.format(vv)
        """
        histo_fit(data, pullvar)
        plt.show()
        """
        selb = sel_for_pull(data, pullvar)
        vala = -1.0
        valb = -1.0
        mymean = 0.0
        mystd = 0.0
        pval = 0.
        kurtosis = 0.

        if len(selb) >= 10:
            rr = fit_pull(selb, pullvar)
            vala = rr[1]
            valb = rr[2]
            mymean = selb[pullvar].mean()
            mystd = selb[pullvar].std()

            res = stats.kurtosistest(selb[pullvar].to_list())
            pval = res.pvalue
            kurtosis = stats.kurtosis(selb[pullvar], fisher=True)

        cols += ['mu_{}'.format(vv), 'sigma_{}'.format(vv),
                 'mean_{}'.format(vv), 'std_{}'.format(vv),
                 'kurtosis_{}'.format(vv), 'pvalue_kurtosis_{}'.format(vv)]
        r += [vala, valb, mymean, mystd, kurtosis, pval]

    res = pd.DataFrame([r], columns=cols)

    return res


def fit_pull(sel, pullvar):
    """
    Function to fit the pulls using a gaussian fit

    Parameters
    ----------
    sel : pandas df
        Data to fit.
    pullvar : str
        variable to fit.

    Returns
    -------
    coeff : list(float)
        fitted values.

    """
    from scipy.optimize import curve_fit
    hist, bins = np.histogram(sel[pullvar], bins=50)
    bin_centres = (bins[:-1] + bins[1:])/2
    p0 = [np.max(hist), 0., 1.]

    try:
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
    except Exception:
        coeff = [-1, -1, -1]

    return coeff


def gauss(x, *p):
    """
    gaussian function 

    Parameters
    ----------
    x : float
        x values.
    *p : list(float)
        gaussian parameters.

    Returns
    -------
    list(float)
        function values.

    """
    A, mu, sigma = p
    return A/np.sqrt(sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))


def sel_for_pull(data, pullvar, nstd=3):
    """
    function to select data for pull estimation

    Parameters
    ----------
    data : pandas df
        Data to process.
    pullvar : str
        variable.
    nstd : int, optional
        window for pull estimation (nubmer of std). The default is 3.

    Returns
    -------
    selb : pandas df
        selected df.

    """

    idx = data[pullvar] >= -5.
    idx &= data[pullvar] <= 5.
    selb = data[idx]

    mystd = selb[pullvar].std()
    mymean = selb[pullvar].mean()

    idx = data[pullvar] >= mymean-nstd*mystd
    idx &= data[pullvar] <= mymean+nstd*mystd
    selb = data[idx]

    return selb


def pull_it(dfa):
    """
    Function to estimate the pulls

    Parameters
    ----------
    dfa : pandas df
        data to process.

    Returns
    -------
    df : pandas df
        original df+pull variables added.

    """

    df = pd.DataFrame(dfa)
    df['pull_x1'] = (df['x1']-df['x1_fit'])/df['sigmax1']
    df['pull_color'] = (df['color']-df['color_fit'])/df['sigmaC']
    df['pull_daymax'] = (df['daymax']-df['t0_fit'])/df['sigmat0']
    df['pull_mb'] = df['diff_mb']/np.sqrt(df['Cov_mbmb'])
    df['pull_mu'] = df['diff_mu']/df['sigma_mu']

    return df


def histo_fit(sel, pullvar, fitgauss=True):

    fig, ax = plt.subplots()
    figtitle = pullvar
    fig.suptitle(pullvar)
    print('fitting', pullvar, sel[pullvar])

    # selb = pd.DataFrame(sel)
    selb = sel_for_pull(sel, pullvar, nstd=3.)

    ax.hist(selb[pullvar], histtype='step', bins=50)

    # Get the fitted curve
    if fitgauss:
        coeff = fit_pull(selb, pullvar)
        xmin = selb[pullvar].min()
        xmax = selb[pullvar].max()
        newbins = np.arange(xmin, xmax, 0.01)
        hist_fit = gauss(newbins, *coeff)
        mean = np.round(coeff[1], 2)
        sigma = np.round(coeff[2], 2)
        leg = 'pull= {} +- {}'.format(mean, sigma)
        ax.plot(newbins, hist_fit, label=leg)
        print('bbb', coeff[0], coeff[1], coeff[2])
    print(figtitle, np.mean(selb[pullvar]), np.std(selb[pullvar]))

    ax.grid(visible=True)
    ax.legend()
