#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:46:30 2024

@author: philippe.gris@clermont.in2p3.fr
"""


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
