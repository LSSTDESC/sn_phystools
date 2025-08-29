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


def load_data_season(fieldTypes, dataDir, dbName, seas, timescale,
                     select_WFD=True, select_DDF=False,
                     vardf=['z_fit', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',
                            'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
                            'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu',
                            'mu_SN']):
    """
    Function to load data per season

    Parameters
    ----------
    fieldTypes : list(str,str)
        list of (fieldType, zType)
    dataDir : dict
        Data dirs
    dbName : dict
        Dbnames.
    seas : list(int)
        seasons.
   timescale : str
        Timescale to use (year/seson).
    select_WFD : bool, optional
        to select the WFD sample. The default is True.
    select_DDF : bool, optional
        to select the DDF sample. The default is False.
    vardf : list(str), optional
        List of var to use. The default is
        ['z_fit', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',
         'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb,
         'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu','mu_SN'].
    Returns
    -------
    data_survey : dict
        Data.
    nsn_survey : dict
        nsn.

    """

    data_survey = {}
    nsn_survey = {}

    for field in fieldTypes:
        ztype = field[0]
        ftype = field[1]
        name = '{}_{}'.format(ftype, ztype)

        if name not in data_survey.keys():
            data_survey[name] = {}
            nsn_survey[name] = {}

        data_ = load_data(
            dataDir[ftype], dbName[ftype], name, ftype, [seas], timescale)

        if ftype == 'WFD' and select_WFD:
            data_ = select_SN_WFD(data_)

        if ftype != 'WFD' and select_DDF:
            data_ = select_SN_DDF(data_)

        print('data loaded', seas, name, ftype, len(data_))
        data_ = data_[vardf]

        # nsn_ = self.load_nsn_summary(dataDir[ftype], dbName[ftype],
        #                             '{}_{}'.format(ftype, ztype), [seas])

        # nsn_ = self.estimate_nsn_z_allfields_sigma_mu(data_)

        # nsn_ = self.get_nsn_from_survey(nsn_, self.survey, ftype, ztype)

        data_survey[name] = data_

        # nsn_survey[name] = nsn_

    return data_survey


def load_data(dataDir, dbName, runType, fieldType, seasons, timescale):
    """
    Function to load data (SN)

    Parameters
    ----------
    dataDir : str
        Data directory.
    dbName : str
        dbName.
    runType : str
        run type.
    fieldType : str
        fieldtype.
    seasons : list(int)
        list of seasons.
    timescale: str
       Time scale to use (year/season)

    Returns
    -------
    df : pandas df
        Data.

    """
    import glob
    search_dir = '{}/{}/{}'.format(dataDir, dbName, runType)

    files = []
    df = pd.DataFrame()
    for seas in seasons:
        search_path = '{}/SN_{}_{}_{}_{}.hdf5'.format(
            search_dir, fieldType, dbName, timescale, seas)
        # print('search path', search_path)
        fis = glob.glob(search_path)
        if len(fis) == 0:
            print('pb here: no file found in path', search_path)
        files += fis

        for fi in files:
            da = pd.read_hdf(fi)
            df = pd.concat((df, da))

    return df


def select_SN_WFD(dd):
    """
    function to select SN

    Parameters
    ----------
    dd : pandas df
        Data to process.

    Returns
    -------
    pandas df
        Selected data.

    """
    idx = dd['sigma_c'] <= 0.04
    idx &= dd['n_epochs_bef'] >= 5
    idx &= dd['n_epochs_aft'] >= 10
    idx &= dd['n_epochs_m10_p5'] >= 5
    idx &= dd['n_epochs_phase_minus_10'] >= 2

    sel = pd.DataFrame(dd[idx])

    return sel


def select_SN_DDF(dd):
    """
    function  to select SN

    Parameters
    ----------
    dd : pandas df
        Data to process.

    Returns
    -------
    pandas df
        Selected data.

    """
    # idx = dd['sigma_c'] <= 99999999.
    """
    idx = dd['n_epochs_bef'] >= 5
    idx &= dd['n_epochs_aft'] >= 10
    idx &= dd['n_epochs_m10_p5'] >= 5
    idx &= dd['n_epochs_phase_minus_10'] >= 2
    """
    idx = dd['Nfilt_2'] >= 3
    sel = pd.DataFrame(dd[idx])

    return sel


def random_LSST(sn_simu_seas, simu_norm_factor, test_mode=False):
    """
    Method to build a realization of the survey

    Parameters
    ----------
    sn_simu_seas : dict
        dict of data (pandas df)
   simu_norm_factor : dict
      normalization factor for the simulation.
    test_mode : bool, optional
      To activate the test mode. The default is False.
    Returns
    -------
    dd : dict
        output surveys.

    """

    dd = {}
    for key, vals in sn_simu_seas.items():
        idx = simu_norm_factor['survey'] == key
        norm = simu_norm_factor[idx]['norm_factor'].values[0]
        sn_survey = pd.DataFrame()
        for field in vals['field'].unique():
            idx = vals['field'] == field
            sel = vals[idx]
            nsn = int(len(sel)/norm)
            samp_ = sel.sample(nsn)
            if test_mode:
                print('field sample', key, field, nsn, norm)
            sn_survey = pd.concat((sn_survey, samp_))

        dd[key] = sn_survey

    return dd


def clean_survey(data, var='SNID', test_mode=False):
    """
    Method to remove duplicate in SNID

    Parameters
    ----------
    data : pandas df
        Data to process.
    var : str, optional
        col to use to remove duplicates. The default is 'SNID'.
    test_mode : bool, optional
        To activate the test mode. The default is False.

    Returns
    -------
    res : pandas df
        Data with duplicates dropped.

    """
    dfdup = data[data[var].duplicated(keep=False)]

    snids_dup = dfdup[var].to_list()

    idx = data[var].isin(snids_dup)
    df_dup = data[idx]
    df_dup = df_dup.groupby([var]).apply(lambda x: add_survey(x))

    if len(df_dup) > 0 and test_mode:
        print("duplicate", df_dup[[var, 'survey']])

    df_res = pd.DataFrame(data[~idx])
    if len(df_dup) > 0:
        df_res = pd.concat((df_res, df_dup))

    res = df_res.drop_duplicates(subset='SNID')

    return res


def add_survey(grp):
    """
    Method to concatenate surveys that have common SNIDs

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    df : pandas df
        Output df.

    """

    df = pd.DataFrame(grp)

    rr = ''
    for i, vv in grp.iterrows():
        rr += '{}+'.format(vv['survey'])

    rr = '/'.join(rr.split('+')[:-1])

    df['survey'] = rr

    return df


def dump_survey(data, year_min, year_max, nn, surveyDir,
                dbName_DD,
                dbName_WFD, add_str=''):
    """
    Method to dump a survey on disk

    Parameters
    ----------
    data: pandas df
         data to store
    year_min : int
        min year of the survey.
    year_max : int
        year max of the survey.
    nn : int
        number to tag the realization of the survey.
    surveyDir : str
        dir to save the data.
    dbName_DD : str
        OS for the DD fields.
    dbName_WFD : str
        OS for the WFD fields.
    add_str: str, optional
      to add a tag name.The default is ''.

    Returns
    -------
    None.

    """

    outName = '{}/survey_sn_{}_{}_{}_{}_{}{}.hdf5'.format(surveyDir,
                                                          dbName_DD,
                                                          dbName_WFD,
                                                          year_min,
                                                          year_max,
                                                          nn, add_str)
    data.to_hdf(outName, key='sn')


def dump_survey_season(data, seas, nn, surveyDir, add_str=''):
    """
    Method to dump a survey on disk

    Parameters
    ----------
    data: pandas df
         data to store
   seas : int
        season of the survey.
    nn : int
        number to tag the realization of the survey.
    surveyDir : str
        dir to save the data.
    dbName_DD : str
        OS for the DD fields.
    dbName_WFD : str
        OS for the WFD fields.
    add_str: str, optional
      to add a tag name.The default is ''.

    Returns
    -------
    None.

    """

    outName = '{}/survey_sn_{}_{}{}.hdf5'.format(surveyDir,
                                                 seas,
                                                 nn, add_str)

    data.to_hdf(outName, key='sn')


def analyze_survey(sn_sample):
    """
    Function to analyze the survey

    Parameters
    ----------
    sn_sample : pandas df
        Data to process.

    Returns
    -------
    None.

    """

    fields = sn_sample['field'].unique()
    print('Analyze survey', fields)
    for field in fields:
        idx = sn_sample['field'] == field
        print(field, len(sn_sample[idx]))


def get_seasons(seasons):
    """
    Function to get the list of seasons

    Parameters
    ----------
    seasons : str
        list of seasons.

    Returns
    -------
    seasons : list(int)
        list of seasons.

    """

    if '-' in seasons:
        seas = seasons.split('-')
        seas_min = int(seas[0])
        seas_max = int(seas[1])
        seasons = list(range(seas_min, seas_max+1))
    else:
        seas = seasons.split(',')
        seasons = list(map(int, seas))

    return seasons
