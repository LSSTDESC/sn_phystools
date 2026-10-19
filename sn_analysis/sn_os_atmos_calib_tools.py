#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:06:15 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
import pandas as pd
from sn_tools.sn_obs import get_fields
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u

def load_OS(dbDir,dbName,dbExtens):
    """
    function to load os

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        OS name.
    dbExtens : str
        OS extens.

    Returns
    -------
    df : pandas df
        loaded data.

    """
    
    fName = '{}/{}.{}'.format(dbDir,dbName,dbExtens)
    
    tt = np.load(fName)
    
    df = pd.DataFrame.from_records(tt)
    
    return df

def get_ebv(df,ebv_map):
    """
    Function to grab ebvofMW

    Parameters
    ----------
    df : pandas df
        input data.
    ebv_map : str
        file of ebv map.

    Returns
    -------
    res : array
        original array+ebvofMW value.

    """

    #load ebv map
    ebv = pd.read_hdf(ebv_map)
    
    #transform obs to df
    #df = pd.DataFrame.from_records(obs)
    
    #merge with ebv file
    
    cols = ['healpixID','ebvofMW']
    ebv = ebv[cols]
    df = df.merge(ebv, left_on=[cols[0]],right_on=[cols[0]])
    
    #return as record arrar
    
    res = df.to_records(index=False)
    del df
    
    return res

def ddf_wfd(obs,nside=64):
    """
    Function to get ddf and wfd obs

    Parameters
    ----------
    obs : array
        observations.
    nside : int, optional
        healpix nside parameter. The default is 64.

    Returns
    -------
    ddf : array
        DDF obs.
    wfd : array
        WFD obs.

    """
    #get pixels
    obs=get_pixels(obs,nside)
    
    #get ebvofMW
    obs = get_ebv(obs,
                 ebv_map='reference_files/dustmap_64_delta_mag_dust.hdf5')
    
    # get DDFs
    ddf = get_fields(obs,'input/simulation/lookup_ddf.csv')  
    
    cols = 'observationId'  
    obsids = ddf[cols].tolist()
    
    idx = np.isin(obs[cols],obsids)
    
    wfd = obs[~idx]
    
    return ddf,wfd

def get_pixels(obs,nside):
    """
    Function to get pixels

    Parameters
    ----------
    obs : array
        Data to process.
    nside : int
        healpix nside parameter.

    Returns
    -------
    obs : pandas df
        original array+healpixID,pixRA, pixDec columns.

    """
    
    obs = pd.DataFrame.from_records(obs)
    #set pixelIds
    healpixIDs = hp.ang2pix(nside, obs['RA'],
                               obs['Dec'], nest=True, lonlat=True)
    
    # get pixel coordinates
    coords = hp.pix2ang(nside, healpixIDs,
                        nest=True, lonlat=True)
    pixRA, pixDec = coords[0], coords[1]
    
    obs['healpixID'] = healpixIDs
    obs['pixRA'] = pixRA
    obs['pixDec'] = pixDec
    """
    This is too slow
    import numpy.lib.recfunctions as rf
    obs = rf.append_fields(obs, 'healpixID',healpixIDs)
    obs = rf.append_fields(obs, 'pixRA',pixRA)
    obs = rf.append_fields(obs, 'pixDec',pixDec)   
    """
    return obs

def load_star_calib(fName,star_type='calspec'):
    """
    star calib loading

    Parameters
    ----------
    fName : str
        star filename.
    star_type : str, optional
        star type. The default is 'calspec'.

    Returns
    -------
    df : pandas df
        output data.

    """
    
    #load star calib file
    
    df = pd.read_csv(fName)
    
    
    if star_type == 'calspec':
        df = df.fillna(-999)
    
        r = []
        for i, row in df.iterrows():
            stname = row['Star_name']
            if row['RA']==-999:
                r.append((stname,-999,-999))
            else:
                coords = '{} {}'.format(row['RA'], row['Decl'])
                c = SkyCoord(coords, frame='fk4', unit=(u.hourangle, u.deg))
                r.append((stname,c.ra.degree,c.dec.degree))
            
        df_ra_dec = pd.DataFrame(r,columns=['Star_name','RA_deg','Dec_deg'])
    
        df = df.merge(df_ra_dec,right_on=['Star_name'],
                                  left_on=['Star_name'])   
        
        #drop old RA,Dec col
        df = df.drop(columns=['RA','Decl'])
        
        #rename to get RA, Dec
        
        #df = df.rename(columns={'RA_deg':'RA','Dec_deg':'Dec'})

        idx = df['Dec_deg'] <= 10
        
        df = df[idx]

    return df