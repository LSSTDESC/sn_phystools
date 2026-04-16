#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:15:11 2026

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def get_atmos_data(df,atmos_params=['airmass','ozone','aerosol','pwv']):
    """
    Function to grab atmos data (interpolator)

    Parameters
    ----------
    df : pandas df
        Data to process.
    atmos_params : list(str), optional
        List of atmos parameters. 
        The default is ['airmass','ozone','aerosol','pwv'].

    Returns
    -------
    dd : pandas df
        Output data.

    """
    
    cols = ['band','airmass',
                     'atmos_param','obs_param']
    dd = df.groupby(cols).apply(lambda x:get_interp(x),include_groups=False).reset_index()
    return dd

def get_interp(grp):
    """
    Function to grab interpolator of atmos params

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    pandas df
        table of interpolators.

    """
    
    
    a = grp['slope'].values[0]
    b = grp['intercept'].values[0]
    xmin = 0.
    xmax = grp['sigma_max'].values[0]
    xvals = np.linspace(xmin,xmax,100)
    yvals = a*xvals+b
    interp = interp1d(xvals,yvals)

    dd = {}
    dd['interp'] = [interp]
    return pd.DataFrame.from_dict(dd)
    
    
    
def get_values(df,sigmas=dict(zip(['airmass','ozone','aerosol','pwv'],
                                  [3e-3,20,5e-3,0.2]))):
    """
    Function to estimate sigma_zp and sigma_mean_wave 
    for a set of sigmas of atmos params

    Parameters
    ----------
    df : pandas df
        Data to process.
    sigmas : dict, optional
        Atmos parameters. 
        The default is dict(zip(['airmass','ozone','aerosol','pwv'],                                  [3e-3,20,5e-3,0.2])).

    Returns
    -------
    db : pandas df
        Output data.

    """
    
    cols = ['band','airmass',
                    'atmos_param','obs_param']
    da = df.groupby(cols).apply(lambda x: get_val(x,sigmas),include_groups=False).reset_index()
     
    db = da.groupby(['band','airmass']).apply(lambda x: calc_combi(x),include_groups=False).reset_index()
   
    return db

def get_val(grp, thedict):
    """
    Function to grab interp values

    Parameters
    ----------
    grp : pandas df
        Data to process.
    thedict : dict
        sigma values.

    Returns
    -------
    pandas df
        output data.

    """
    
    atmos_param = grp.name[2]
    sigma = thedict[atmos_param]
    
    dd = {}
    myinterp = grp['interp'].values[0]
    dd['sigma_value'] = [myinterp(sigma)]
    
    return pd.DataFrame.from_dict(dd)
    
def calc_combi(grp):
    """
    Function to estimate combination of sigmas

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    res : pandas df
        Output data.

    """
    
    atmos_params = grp['atmos_param'].to_list()
    sigma_values = grp['sigma_value'].to_list()
    what = grp['obs_param'].unique()[0]
    
    rr = np.sqrt(np.sum(np.array(sigma_values)**2))
    
    atmos_params.append('total')
    po = 'sigma_{}_'.format(what)
    atmos_params = list(map(lambda x: po+x,atmos_params))
    sigma_values.append(rr)
    
    res = pd.DataFrame([sigma_values],columns=atmos_params)
    res = res.astype(float)
    res =res.round(decimals=4)
   
    return res

def get_obs_values(grp,sigma):
    """
    To estimate sigmas of obs params (zp or mean_wave) 
    for sigma values of atmos params

    Parameters
    ----------
    grp : pandas df
        Data to process.
    sigma : dict(str,array(float))
        sigma_atmos_param value.

    Returns
    -------
    df : pandas df
        Output data.

    """
    
    print(grp.name)
    atmos_param = grp.name[2]
    sigmas = sigma[atmos_param]
    
    interp = grp['interp'].values[0]
    
    res = interp(sigmas)
    
    df = pd.DataFrame(res,columns=['sigma_obs_param'])
    
    df['sigma_atmos_param'] = sigmas
    
    return df

def rename(dfa,atmos_params):
    """
    function to perform some renaming

    Parameters
    ----------
    dfa : pandas df
        Data to process.
    atmos_params: list(str)
        List of atmos parameters.

    Returns
    -------
    df : pandas df
        Output data.

    """
    
    df = pd.DataFrame(dfa)
    obs_param = df['obs_param'].unique()[0]
    for atm in atmos_params:
        vvara = 'sigma_obs_param_{}'.format(atm)
        vvarb = 'sigma_{}_{}'.format(obs_param,atm)
        df = df.rename(columns={vvara:vvarb})
    
    df = df.rename(columns={'sigma_tot':'sigma_{}_tot'.format(obs_param)})
    
    return df