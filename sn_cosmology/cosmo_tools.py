#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:46:30 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd


def get_surveys(name, data):
    """
    Function to build a list of surveys from the name

    Parameters
    ----------
    name : str
        Name to process.
    data : pandas df
        array of relation nickname <-> survey.

    Returns
    -------
    r : list(str)
        List of surveys corresponding to name.

    """

    r = []

    nname = name.split('_')

    for nn in nname:
        idx = data['nickname'] == nn
        sel = data[idx]
        surveys = sel['survey'].values[0].split('+')
        nickname = sel['nickname'].values[0]
        for surv in surveys:
            r.append('{}'.format(surv))

    return r


def get_nickname(ll, data):
    """
    Function to build a nickname from a list of surveys

    Parameters
    ----------
    ll : list(str)
        List of surveys to process.
    data : pandas df
        array of relation nickname <-> survey.

    Returns
    -------
    res : str
        Nickname corresponding to the list of surveys.

    """

    part = {}
    for vv in ['desi_', 'crs_']:
        res = list(filter(lambda x: vv in x, ll))
        if len(res) >= 2:
            res.sort()
            part[vv] = res
            ll = list(set(ll) ^ set(res))

    for key, vals in part.items():
        ro = '+'.join(vals)
        ll += [ro]

    idx = data['survey'].isin(ll)
    sel = data[idx]
    assert len(sel) == len(ll), 'Problem when building survey nickname'
    res = '_'.join(sel['nickname'].to_list())

    return res


def get_survey_nickname(tagsurvey, surveys, data):
    """
    Function to get tagsurvey and surveys

    Parameters
    ----------
    tagsurvey : str
        Tag of the survey.
    surveys : list(str)
        List of surveys.
    data : pandas df
        lookup table.

    Returns
    -------
    tagsurvey : str
        output tagsurvey.
    surveys : list(str)
        output list of surveys.

    """

    if tagsurvey == 'notag':
        tagsurvey = get_nickname(surveys, data)
    else:
        surveys = get_surveys(tagsurvey, data)

    return tagsurvey, surveys


def host_effi_1D(lista, listb):
    """
    Function to build a dict of 1D interpolators

    Parameters
    ----------
    lista : list(str)
        List of csv files with z,effi as columns.
    listb : list(str)
        List of keys for the output dict.

    Returns
    -------
    dict_out : dict
        Output data.

    """

    from scipy.interpolate import interp1d
    dict_out = {}
    for i, vv in enumerate(lista):
        dd = pd.read_csv(vv, comment='#')
        nn = listb[i]
        dict_out[nn] = interp1d(dd['z'], dd['effi'],
                                bounds_error=False, fill_value=0.)

    return dict_out


def load_host_effi(dataDir, llist):
    """
    Function to load all effi(z) csv files in a dict

    Parameters
    ----------
    dataDir : str
        Data dir.

    Returns
    -------
    dictout : dict
        key=name; val=interp1d(z,effi).

    """

    dictout = {}
    for ll in llist:
        fName = '{}/{}.csv'.format(dataDir, ll)
        rr = host_effi_1D([fName], [ll])
        dictout.update(rr)

    return dictout


def load_footprints(dataDir):
    """
    Function to load footprints

    Parameters
    ----------
    dataDir : str
        Location dir of the footprint files.

    Returns
    -------
    df : pd.Dataframe
        Footprints (two cols: footprint, healpixID).

    """

    df = pd.DataFrame()

    import glob
    fis = glob.glob('{}/*.hdf5'.format(dataDir))

    for fi in fis:
        dfa = pd.read_hdf(fi)
        df = pd.concat((df, dfa))

    return df
