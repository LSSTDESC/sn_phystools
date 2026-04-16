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
    if 'config' in sigma.keys():
        df['config'] = sigma['config']
    
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

def process_obs_data(df_zp,sigma,atmos_params,do_combi=False):
    """
    Data processing (per obs)

    Parameters
    ----------
    df_zp : pandas df
        Data to process.
    sigma : dict(str,array(float))
        List of sigma for atmos params (key).

    Returns
    -------
    combis : pandas df
        Processed data.

    """
    
    interp_zp = get_atmos_data(df_zp,atmos_params=atmos_params)
    cols = ['band', 'airmass', 'atmos_param', 'obs_param']
    df_values = interp_zp.groupby(cols).apply(lambda x: get_obs_values(x,sigma),include_groups=False).reset_index()

    print(df_values)

    if do_combi:
        combis = df_values.groupby(['band','airmass','obs_param']).apply(lambda x:make_combi(x),include_groups=False).reset_index()
    else:
        ccols = ['band','airmass','config']
        combis = df_values.groupby(ccols).apply(lambda x:transform(x),include_groups=False).reset_index()
        #return combis
    
    print(combis)

    #estimate sigma_tot

    combis['sigma_tot'] = 0

    for atm in atmos_params:
        combis['sigma_tot'] += combis['sigma_obs_param_{}'.format(atm)]**2
    
    combis['sigma_tot'] = np.sqrt(combis['sigma_tot'])

    print(combis)

    return combis

def make_combi(grp,atmos_params=['airmass','ozone','aerosol','pwv']):
    """
    Function to make a combination of pandas sf

    Parameters
    ----------
    grp : pandas df
        Data to process.
    atmos_params : list(str), optional
        List of atmos params. The default is ['airmass','ozone','aerosol','pwv'].

    Returns
    -------
    df_combi : pandas df
        output data.

    """
    
    cols = ['sigma_atmos_param','sigma_obs_param']
    df_combi = pd.DataFrame()
    for atm in atmos_params:
        idx = grp['atmos_param'] == atm
        sel = pd.DataFrame(grp[idx][cols])
        sigma_atm = 'sigma_{}'.format(atm)
        sigma_obs = 'sigma_obs_param_{}'.format(atm)
        sel = sel.rename(columns={'sigma_atmos_param':sigma_atm,
                                  'sigma_obs_param':sigma_obs})
        if df_combi.empty:
            df_combi = pd.DataFrame(sel)
        else:
            df_combi = df_combi.merge(sel, how='cross')
        
    return df_combi

def merge_zp_wave(combi_zp,combi_wave,atmos_params):
    """
    Merge two df to form a single one

    Parameters
    ----------
    combi_zp : pandas df
        first df to merge.
    combi_wave : pandas df
        second df to merge.
    atmos_params : list(str)
        List of atmos params.

    Returns
    -------
    combi_tot : pandas df
        merged df.

    """
    
    combi_zp = rename(combi_zp,atmos_params)
    combi_wave = rename(combi_wave,atmos_params)
    
    combi_zp = combi_zp.drop(columns=['obs_param'])
    combi_wave = combi_wave.drop(columns=['obs_param'])
    
    ccols = ['band','airmass']
    
    if 'config' in combi_zp.columns.to_list():
        ccols += ['config']
    for atm in atmos_params:
        ccols += ['sigma_{}'.format(atm)]
        
    print('alalala',ccols)
    combi_tot = combi_zp.merge(combi_wave,left_on=ccols,right_on=ccols)
        
    print(combi_tot.columns)    
    
    return combi_tot

def transform(grp):
    """
    Function to transform a grp

    Parameters
    ----------
    grp : pandas df
        Data to transform.

    Returns
    -------
    res : pandas df
        Output data.

    """
    
    obs_param = grp['obs_param'].unique()[0]
    sigma_obs = grp['sigma_obs_param'].to_list()
    
    cola = grp['atmos_param'].to_list()
    
    cola = list(map(lambda x: 'sigma_obs_param_{}'.format(x),cola))
    
    sigma_atm = grp['sigma_atmos_param'].to_list()
    
    colb = grp['atmos_param'].to_list()
    
    colb = list(map(lambda x: 'sigma_{}'.format(x),colb))
    
    print(sigma_obs,cola)
    
    values = sigma_obs+sigma_atm
    cols = cola+colb
    
    res = pd.DataFrame([values],columns=cols)
    res['obs_param'] = obs_param
    return res