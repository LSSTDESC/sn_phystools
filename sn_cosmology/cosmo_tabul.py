#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:35:41 2026

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import numpy as np
from sn_tools.sn_utils import multiproc
from sn_cosmology.cosmo_tools import load_cosmo_params_from_script
from sn_tools.sn_cosmo_model import cosmo_wrapper

__all__ = ['Cosmo_tabul','check_interp_new']

class Cosmo_tabul:
    def __init__(self,params):
        """
        class to estimate a grid of distmod values according to cosmo parameters

        Parameters
        ----------
        params : dict
            parameters.

        Returns
        -------
       None.

        """
        
        self.params = params
    
    def __call__(self):
        """
        Mein method

        Returns
        -------
        df_tot : pandas df
            tabulated grid of parameters.

        """
        
        df_tot = self.build_sample()    
        
        return df_tot
        
    
    def build_sample(self):
        """
        Function to build the sample
    
        Parameters
        ----------
        opts : opts parser
            parameters of the script.
        outName : str
            output file name.
    
        Returns
        -------
        None.
    
        """
        
        fitparams = self.params['cosmofitparams']
        fitparams_min = self.params['cosmofitparams_min']
        fitparams_max = self.params['cosmofitparams_max']
        fitparams_delta = self.params['cosmofitparams_delta']
        
        
        par_name = fitparams.split(',')
        par_min = fitparams_min.split(',')
        par_max = fitparams_max.split(',')
        par_delta = fitparams_delta.split(',')
        
        par_min = list(map(float,par_min))
        par_max = list(map(float,par_max))
        par_delta = list(map(float,par_delta))
        
        df_params = pd.DataFrame()
        
        for i,vv in enumerate(par_name):
            df_ = self.get_par_values(vv,par_min[i],par_max[i],par_delta[i])
            
            if len(df_params) == 0:
                df_params = df_
            else:
                df_params = df_params.merge(df_,how='cross')
        
        df_params['num'] = df_params.index
        
        pp = {}
        
        pp['data'] = df_params
        pp['cosmodict'] = self.params
        pp['par_name'] = par_name
        
        toproc = df_params['num'].to_list()
        
        df_tot = multiproc(toproc,pp,self.build_multi,nproc=8)    
        
        #outName = '{}_{}.hdf5'.format(params['outName'],params['demodel'])
        
        outName = self.params['outName']
        if outName != '':
            df_tot.to_hdf(outName,key='distmod')

        return df_tot

    def build_multi(self,toproc, pp, j=0, output_q=None):
        """
        Method to estimate distance moduli 
    
        Parameters
        ----------
        toproc : list(int)
            index to process.
        pp : dict
            parameter dict.
        j : int, optional
            internal tag for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            where to copy the results. The default is None.
    
        Returns
        -------
        pandas df
            Output data.
    
        """
        
        df_params = pp['data']
        params = pp['cosmodict']
        par_name = pp['par_name']
        
        idx = df_params['num'].isin(toproc)
        
        df_params = df_params[idx]
        
        df_tot = pd.DataFrame()
        for i, row in df_params.iterrows():
            de_values = '{},{}'.format(row[par_name[0]],row[par_name[1]])
            params['devalues'] = de_values
            cosmo_params = load_cosmo_params_from_script(params)
            vv = dist_modulus(cosmo_params)
            for i in range(len(par_name)):
                vv[par_name[i]] = row[par_name[i]]
            df_tot = pd.concat((df_tot,vv))
        
        if output_q is not None:
                return output_q.put({j: df_tot})
        else:
                return df_tot
            
    def get_par_values(self,col,xmin,xmax,delta):
        """
        Method to estimate parameter values
    
        Parameters
        ----------
        col : str
            column name.
        xmin : float
            min value.
        xmax : float
            max value.
        delta : float
            step value.
    
        Returns
        -------
        res : TYPE
            DESCRIPTION.
    
        """
        
        vv = np.arange(xmin,xmax+delta,delta)
        
        res = pd.DataFrame(vv,columns=[col])
        
        return res      
      
def dist_modulus(cosmo_params,z=np.arange(0.01,1.11,0.01)):
    """
    Function to estimate the distance moduli from cosmo estimation

    Parameters
    ----------
    cosmo_params : dict
        cosmo parameters.
    z : array, optional
        List of redshifts. The default is np.arange(0.01,1.11,0.01).

    Returns
    -------
    res : pandas df
        distmod vs z.

    """
    cosmo = cosmo_wrapper(cosmo_params)
        
    distmod = cosmo.distmod(z).value
    
    res = pd.DataFrame(z,columns=['z'])
    
    res['distmod'] = distmod
    
    return res
 
def check_interp(interp,df_tot,ccols,params):
    """
    Function to check interpolator

    Parameters
    ----------
    opts : parser opts
        script parameters.
    df_tot : pandas df
        tabulated data.

    Returns
    -------
    None.

    """
    
    dmin={}
    dmax={}
    for vv in ccols:
        dmin[vv] = df_tot[vv].min()
        dmax[vv] = df_tot[vv].max()
    
    nrand = 5

    to = pd.DataFrame()
    for vv in ccols:
        rrand= np.random.uniform(dmin[vv],dmax[vv],nrand)
        if len(to) == 0:
            to = pd.DataFrame(rrand.tolist(),columns=[vv])
        else:
            to[vv] = rrand.tolist()
    
    print('random check')
    print(to)
    
    res = list(interp(to[ccols]))
    
    real_val = []
        
    deparams = params['deparams'].split(',')
    for i,row in to.iterrows():
        rb  =[]
        for j in range(len(deparams)):
            rb.append(row[deparams[j]])
        rb = list(map(str,rb))
        rb = ','.join(rb)
        params['devalues'] = rb
        if 'Om0' in ccols:
            params['Om0'] = row['Om0']
        cosmo_params = load_cosmo_params_from_script(params)
        estim_val =  dist_modulus(cosmo_params,[row['z']])['distmod'].values[0]
        real_val.append(estim_val)
        
   
    for i in range(len(res)):
        print(i,res[i],real_val[i],res[i]/real_val[i])
    
    