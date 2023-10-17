#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:45:27 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
import pandas as pd


def doInt(df, cols):
    """
    Function to transform col to int

    Parameters
    ----------
    df : pandas df
        Data to process.
    cols : list(str)
        List of columns to transform to int.

    Returns
    -------
    df : pandas df
        Output df.

    """

    for cc in cols:
        df[cc] = df[cc].astype(int)

    return df


def nmax(list_visit):
    """

    Function to estimate the max of total number of visits 
    from a (str) filter alloc.

    Parameters
    ----------
    list_visit : list(str)
        filter list distribution.

    Returns
    -------
    int
        Max numvber of visits.

    """

    ro = []
    for ll in list_visit:
        vv = ll.split('/')
        vv = list(map(int, vv))
        ro.append(np.sum(vv))

    return np.max(ro)


def gime_combi(filter_alloc, nvisits_band):
    """
    Function to extract params (filter allo, ...)

    Parameters
    ----------
    filter_alloc : list(str=)
        filter allocation.
    nvisits_band : dict
        Number of visits per band.

    Returns
    -------
    sf : TYPE
        DESCRIPTION.
    nvis : TYPE
        DESCRIPTION.
    combi1 : TYPE
        DESCRIPTION.
    combi2 : TYPE
        DESCRIPTION.

    """

    dictout = {}
    for ia, ff in enumerate(filter_alloc):
        fa = ff.split('/')
        nv = nvisits_band[ia].split('/')
        nv = list(map(int, nv))

        for i, vv in enumerate(fa):
            if vv not in dictout.keys():
                dictout[vv] = []
            dictout[vv].append(nv[i])

    rb = []
    for key, vals in dictout.items():
        rb.append((key, np.max(vals)))

    res = np.rec.fromrecords(rb, names=['band', 'visits_band'])

    rf = []
    rb = []
    combi1 = []
    combi2 = []
    for b in 'ugrizy':
        idx = res['band'] == b
        rf.append(b)
        nv = int(np.mean(res[idx]['visits_band']))
        rb.append(nv)
        if b == 'u' or b == 'g' or b == 'r':
            combi1.append('{}{}'.format(nv, b))
        else:
            combi2.append('{}{}'.format(nv, b))

    sf = '/'.join(rf)
    nvis = '/'.join(map(str, rb))
    combi1 = '/'.join(combi1)
    combi2 = '/'.join(combi2)

    return sf, nvis, combi1, combi2


def coadd_night(grp, colsum=['numExposures'],
                colmean=['season', 'observationStartMJD', 'moonPhase'],
                combi_filt=['ugriz', 'grizy'],
                visitExposureTime_single=30.):
    """
    Function to coadd obs per night

    Parameters
    ----------
    grp : pandas df
        data to ,process.
    colsum : list(str), optional
        List of col for which to perform a sum. The default is ['numExposures'].
    colmean : list(str), optional
        List of cols for wwhich to estimate the mean.
        The default is ['season', 'observationStartMJD'].
    visitExposureTime_single: float, optional.
       visit exposure time single visit. The default is 30 s.

    Returns
    -------
    pandas df
        output res.

    """

    grp['Nvisits'] = grp['visitExposureTime']/visitExposureTime_single
    grp['Nvisits'] = grp['Nvisits'].astype(int)

    dictout = {}
    for vv in colsum:
        dictout[vv] = [grp[vv].sum()]

    for vv in colmean:
        dictout[vv] = [grp[vv].mean()]

    filters = list(grp['filter'].unique())
    filters.sort()

    for cc in combi_filt:
        vv = list(cc)
        vv.sort()
        if vv == filters:
            dictout['filter_alloc'] = ['/'.join(cc)]
            nf = []
            for ff in list(cc):
                idx = grp['filter'] == ff
                nf.append(grp[idx]['Nvisits'].sum())

            dictout['Nvisits'] = [np.sum(nf)]
            nf = map(str, nf)
            dictout['visits_band'] = ['/'.join(nf)]

    return pd.DataFrame.from_dict(dictout)


def m5_coadd_grp(data, varx='dbName', vary='diff_m5_y2_y10'):
    """
    Function to perform coadds depending on seasons

    Parameters
    ----------
    data : pandas df
        Data to process.
    varx : str, optional
        x-axis var. The default is 'dbName'.
    vary : str, optional
        y-axis var. The default is 'diff_m5_y2_y10'.

    Returns
    -------
    dd : pandas df
        Output data.

    """

    idx = data['season'] == 1
    dd_y1 = data[idx].groupby('filter').apply(
        lambda x: m5_coadd(x, m5Colout='m5_y1')).reset_index()
    dd_y2_y10 = data[~idx].groupby('filter').apply(
        lambda x: m5_coadd(x, m5Colout='m5_y2_y10')).reset_index()

    dd = dd_y1.merge(dd_y2_y10, left_on=['filter'], right_on=['filter'])

    return dd


def m5_coadd(grp, m5Col='fiveSigmaDepth', m5Colout='m5'):
    """
    Function to perform coadds

    Parameters
    ----------
    grp : pandas df
        Data to process.
    m5Col : str, optional
        col name to coadd. The default is 'fiveSigmaDepth'.
    m5Colout : str, optional
        output coadded col name. The default is 'm5'.

    Returns
    -------
    pandas df
        Coadded data.

    """

    coadded = 1.25*np.log10(np.sum(10**(0.8*grp[m5Col])))

    return pd.DataFrame({m5Colout: [coadded]})


def translate(grp):
    """
    Method to perform some translation wrt season min, max.

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    dfi : pandas df
        Output data.

    """

    grp = grp.sort_values(by=['season'])
    seasons = grp['season'].unique()

    val = 'observationStartMJD'
    Tmax = grp[grp['season'] == 1][val].max()
    night_max = grp[grp['night'] == 1][val].max()
    dfi = pd.DataFrame()
    for seas in seasons:
        idx = grp['season'] == seas
        sel = pd.DataFrame(grp[idx])
        Tmin = sel[val].min()
        night_min = sel['night'].min()
        deltaT = Tmin-Tmax
        deltaN = night_min-night_max
        if seas == 1:
            deltaT = 0.
            deltaN = 0
        sel.loc[:, val] -= deltaT
        sel.loc[:, 'night'] -= deltaN

        Tmax = sel[val].max()
        night_max = sel['night'].max()
        TTmin = Tmin-deltaT
        NNmin = night_min-deltaN
        sel['MJD_season'] = (sel[val]-TTmin)/(Tmax-TTmin)+(seas-1)
        sel['night_season'] = (sel[val]-NNmin)/(night_max-NNmin)+(seas-1)
        dfi = pd.concat((dfi, sel))

    return dfi
