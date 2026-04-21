#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 08:56:26 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

def fit_hist(sel, pullvar,bins='auto',fit_with_errors=False):
    """
    Function to fit an histogram

    Parameters
    ----------
    sel : array
        Data to fit.
    pullvar : str
        variable to fit.
    bins : array(float)
        histogram bins.
    fit_with_errors: bool, opt.
        to fit the histogram with errors

    Returns
    -------
    coeff : list(float)
        fit parameter values (A, mu, sigma) (gauss).

    """

    hist, bins_h = np.histogram(sel[pullvar], bins=bins)
    bin_centres = (bins_h[:-1] + bins_h[1:])/2
    max_hist = np.max(hist)*np.sqrt(2.*np.pi)
    p0 = [max_hist, 0., 1.]
    nevts = np.sum(hist)
    yerr = None
    if fit_with_errors:
        effi = hist/nevts
        #yerr = 1./np.sqrt(nevts*effi*(1.-effi))
        yerr = np.sqrt(hist*(1.-effi))/np.sqrt(nevts)
        #empty bins: high errors
        yerr[yerr == 0.] = 9999.

    ndof = nevts-3
    chi_square = 9999.
    stat = [sel[pullvar].mean(),sel[pullvar].std()]
    try:
        
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0,
                                      sigma=yerr)
        err_coeff = np.sqrt(np.diag(var_matrix))
        
        exp = gauss(bin_centres,*coeff)
        diff = hist-exp
        vv = 1
        if yerr is not None:
            vv = yerr
        chi_square = np.sum((diff/vv)**2)
    
    except Exception:
        coeff = [-1, -1, -1]
        err_coeff = [-1.,-1.,-1.]
        chi_square=9999.
        
    outres = make_df(pullvar,list(coeff),list(err_coeff),chi_square,ndof,stat)
    return outres

def fit_linear(x,y):
    
    p0=[0,1.]
    coeff, var_matrix = curve_fit(lin,x,y, p0=p0)
    
    
    return coeff, var_matrix

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
    return A/(np.sqrt(2.*np.pi)*sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))

def lin(x,*p):
    """
    Linear func

    Parameters
    ----------
    x : float
        x-axis values.
    *p : list(float)
        parameters.

    Returns
    -------
    float
        Result.

    """
    
    a,b=p
    
    return a*x+b

def make_df(pullvar,coeff,err_coeff,chi_square,ndof,stat):
    """
    Function to transform a set of result to a pandas df

    Parameters
    ----------
    pullvar : str
        var name.
    coeff : list(float)
        Fit coeffs.
    err_coeff : list(float)
        Fit coeff errors.
    chi_square : float
        chisq value.
    ndof : int
        ndof.
    stat : list(float)
        pull mean and std.

    Returns
    -------
    res : pandas df
        Output data.

    """
    
    ra = [pullvar]+coeff+err_coeff+[chi_square,ndof]+stat

    r = []
    r.append(ra)    
    res = pd.DataFrame(r,columns=['sn_param',
                                  'A','mu','sigma',
                                  'err_A','err_mu','err_sigma',
                                  'chisq','ndof','pull_mean','pull_std'])
    
    return res
    
def load_fit_atmos_data(theDir,atmos_params):
    """
    Function to load data (zp, mean_wave) vs sigma_atmos_params

    Parameters
    ----------
    theDir : str
        Data dir.
    atmos_params : list(str)
        List of atmospheric parameters.

    Returns
    -------
    df_zp : pandas df
        zp data.
    df_wave : pandas df
        mean wave data.

    """
    df_zp = pd.DataFrame()
    df_wave = pd.DataFrame()
    for atm in atmos_params:
        fName = '{}/zp_atmos_{}.hdf5'.format(theDir,atm)
        df = pd.read_hdf(fName)
        for b in 'grizy':
            df['std_zp_{}'.format(b)] *= 1000
        df= df.round({'mean_airmass':2})
        dfa = linfit_atmos(df,varxp=atm,
                           vary_prefix='zp',
                           airmass=[1.2,2.0],bands='grizy')
        dfb = linfit_atmos(df,varxp=atm,
                           vary_prefix='mean_wave',
                           airmass=[1.2,2.0],bands='grizy')
        df_zp = pd.concat((df_zp,dfa))
        df_wave = pd.concat((df_wave,dfb))

    return df_zp,df_wave

def linfit_atmos(df,varxp='pwv',
                 vary_prefix='zp',airmass=[1.2,2.0],bands='grizy'):
    """
    Function to perform a linear fit of zp/mean_wave vs atmos params

    Parameters
    ----------
    df : pandas df
        Data to fit.
    varxp : str, optional
        atmos parameter. The default is 'pwv'.
    vary_prefix : str, optional
        prefix obs (zp/mean_wave). The default is 'zp'.
    airmass : list(float), optional
        List of airmass values o consider. The default is [1.2,2.0].
    bands : str, optional
        Filters to consider. The default is 'grizy'.

    Returns
    -------
    dfn : pandas df
        output data.

    """
    
    ro = []
    #print(df.columns)
    varx = 'sigma_{}'.format(varxp)
    from sn_analysis.sn_tools import fit_lin
    for airm in airmass:
        idx = df['mean_airmass'] == airm
        sel = pd.DataFrame(df[idx])
        for b in bands:
            yvar = 'std_{}_{}'.format(vary_prefix,b)
            yvar_rel = 'std_{}_{}_rel'.format(vary_prefix,b)
            sel[yvar_rel] = sel[yvar]/sel['mean_{}'.format(varxp)]
            #print('fitting',sel[[varx,yvar,yvar_rel]])
            res = list(fit_lin(sel,varx,yvar))
            res += [b,airm]
            ro.append(res)
    dfn = pd.DataFrame(ro,columns=['slope','intercept','band','airmass'])
    dfn['atmos_param'] = varxp
    dfn['obs_param'] = vary_prefix
    dfn['atmos_param_value'] = df['mean_{}'.format(varxp)].mean()
    dfn['sigma_max'] = df['{}'.format(varx)].max()
    return dfn
