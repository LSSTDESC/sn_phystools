#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:52:27 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import os
import pandas as pd
from sn_analysis.sn_calc_plot import select
from sn_analysis.sn_tools import load_complete_dbSimu
from sn_analysis.sn_calc_plot import bin_it
import numpy as np


class sn_load_select:
    def __init__(self, dbDir, dbName, prodType,
                 listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                 fDir='', norm_factor=1):
        """
        class to load and select sn data and make some stats

        Parameters
        ----------
        dbDir : str
            loc dir of the data to process.
        dbName : str
            db name.
        prodType : str
            production type (DDF_spectroz, DDF_photoz).
        listDDF : str, optional
            List of DDF to process.
            The default is 'COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb'.
        fDir: str, opt.
            Location dir of the files. The default is ''
        norm_factor: float, optional
            normalization factor in the simulation. The default is 1.

        Returns
        -------
        None.

        """

        outName_stack = '{}/SN_{}.hdf5'.format(fDir, dbName)
        # load the data
        if not os.path.exists(outName_stack):
            data = load_complete_dbSimu(
                dbDir, dbName, prodType,
                listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb')
            data = data[data.columns.drop(list(data.filter(regex='mask')))]
            data = data.drop(columns=['selected'])
            # data.info(verbose=True)

            data.to_hdf(outName_stack, key='SN')
        else:
            data = pd.read_hdf(outName_stack, key='SN', mode='r')

        # selectc only fitted LC
        # idx = data['fitstatus'] == 'fitok'
        res = data
        self.norm_factor = norm_factor
        # self.printRes(res)

        res['dbName'] = dbName
        res = res[res['field'].isin(listDDF.split(','))]
        self.data = res

    def printRes(self, data):
        """
        Method to print results

        Parameters
        ----------
        res : pandas df
            data to print.

        Returns
        -------
        None.

        """

        res = pd.DataFrame(data)
        print(data['survey_area'].unique())
        survey_area = data['survey_area'].mean()
        res['NSN'] = res.groupby(['field', 'season', 'healpixID'])[
            'field'].transform('size')

        res['season'] = res['season'].astype(int)
        res['healpixID'] = res['healpixID'].astype(int)

        print('res', res)

        rr = res.groupby(['field', 'season']).apply(
            lambda x: self.sn_stat(x, survey_area)).reset_index()
        rr = rr.to_records(index=False)
        print(rr[['field', 'season', 'NSN', 'NSN_explod', 'survey_area']])

    def sn_stat(self, grp, survey_area):

        # nsn = grp['NSN'].sum()
        nsn = len(grp)
        nsn_exp = nsn/self.norm_factor
        nheal = len(grp['healpixID'].unique())

        return pd.DataFrame({'NSN': [nsn],
                             'NSN_explod': [int(nsn_exp)],
                             'survey_area': [nheal*survey_area]})

    def sn_selection(self, dict_sel={}):
        """
        Method to select data

        Parameters
        ----------
        dict_sel : dict, optional
            Selection criteria. The default is {}.

        Returns
        -------
        sel : dict
            Selected data.

        """

        sel = {}
        # select data here
        for selvar in dict_sel.keys():
            sel[selvar] = select(self.data, dict_sel[selvar])

        return sel


def get_nsn(data_dict, grpby=['field'], norm_factor=1):
    """
    Function to grab the number of SN

    Parameters
    ----------
    data_dict : dict
        data to process.
    grpby : list(str), optional
        Used for the groupby df analysis. The default is ['field'].
    norm_factor : int, optional
         Normalization factor. The default is 1.

    Returns
    -------
    nsn_fields : pandas df
        result with NSN col added (in addition to grpby).

    """

    nsn_fields = pd.DataFrame()
    for key, vals in data_dict.items():
        # dd['NSN'] = vals.groupby(['field']).transform('size').reset_index()
        dd = vals.groupby(grpby).size().to_frame('NSN').reset_index()
        dd['NSN'] /= norm_factor
        dd['selconfig'] = key
        nsn_fields = pd.concat((nsn_fields, dd))

    # nsn_fields['dbName'] = self.dbName
    return nsn_fields


def processNSN(dd, dbDir, prodType, listDDF, dict_sel, fDir, norm_factor):
    """
    Function to process Data

    Parameters
    ----------
    dd : pandas df
        list of DB to process.
    dbDir : str
        location dir of OS.
    prodType : str
        production type.
    listDDF : str
        list of DDF to process.
    dict_sel : dict
        Selection dict.
    fDir: str
       location dir of the files
    norm_factor: float
      normalization factor

    Returns
    -------
    None.

    """
    sn_field = pd.DataFrame()
    sn_field_season = pd.DataFrame()

    for io, row in dd.iterrows():
        dbName = row['dbName']

        dd = sn_load_select(dbDir, dbName, prodType,
                            listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                            fDir=fDir)
        data_dict = dd.sn_selection(dict_sel)

        # estimate the number of SN
        fa = get_nsn(data_dict,
                     grpby=['dbName', 'field'], norm_factor=norm_factor)
        sn_field = pd.concat((sn_field, fa))
        fb = get_nsn(data_dict, grpby=['dbName', 'field', 'season'],
                     norm_factor=norm_factor)

        sn_field_season = pd.concat((sn_field_season, fb))

    # save in hdf5 files
    sn_field.to_hdf('{}/sn_field.hdf5'.format(fDir), key='sn')
    sn_field_season.to_hdf('{}/sn_field_season.hdf5'.format(fDir), key='sn')

    print(sn_field)
    print(sn_field_season)


def nsn_vs_sel(dd, dbDir, prodType, listDDF, dict_sel, fDir, norm_factor):
    """
    Function to estimate the number of SN vs selection criteria

    Parameters
    ----------
    dd : pandas df
        data to process.
    dbDir : str
        location of the file to process.
    prodType : str
        production type.
    listDDF : list(str)
        list of DDFs to process.
    dict_sel : dict
        selection criteria.
    fDir : str
        output dir.
    norm_factor : float
        normalization factor.

    Returns
    -------
    None.

    """

    strDDF = '_'.join(listDDF.split(','))
    outName = '{}/nsn_selcriteria_{}.hdf5'.format(fDir, strDDF)

    if os.path.exists(outName):
        res = pd.read_hdf(outName)
        return res

    res = pd.DataFrame()
    for io, row in dd.iterrows():
        dbName = row['dbName']

        dd = sn_load_select(dbDir, dbName, prodType,
                            listDDF=listDDF,
                            fDir=fDir).data

        nsn_all = dd.groupby(['field', 'season']).apply(
            lambda x: nsn_per_sel(x, dict_sel)).reset_index()
        nsn_all = dd.groupby(['field', 'season']).apply(
            lambda x: nsn_per_sel(x, dict_sel)).reset_index()
        nsn_all = nsn_all.groupby(['seldict', 'sel', 'cutnum'])[
            'NSN'].sum().reset_index()

        print('processing', dbName, dd['field'].unique())
        print(dd.groupby(['field']).size())
        nsn_all = nsn_stat(nsn_all, norm_factor, grpcol=[])
        nsn_all['name'] = dbName
        res = pd.concat((res, nsn_all))

    res.to_hdf(outName, key='SN')

    return res


def nsn_per_sel(grp, dict_sel):
    """
    Function to estimate the number of SN passing selection cuts

    Parameters
    ----------
    grp : pandas df
        data to process.
    dict_sel : dict
        selection criteria.

    Returns
    -------
    res : pandas df
        estimated data.

    """

    r = []
    for key, vals in dict_sel.items():
        idx = True
        for valsb in vals:
            idx &= valsb[1](grp[valsb[0]], valsb[2])
            r.append((key, valsb[0], len(grp[idx]), valsb[3]))

    res = pd.DataFrame(r, columns=['seldict', 'sel', 'NSN', 'cutnum'])
    res['NSN'] = res['NSN'].astype(int)

    return res


def nsn_stat(grp, norm_factor, grpcol=['field', 'season']):
    """
    Function to estimate the number of SN and associated error

    Parameters
    ----------
    grp : pandas df
        data to process.
    norm_factor : float
        normalization factor.
    grpcol : list(str), optional
        List of cols for NSN estimation. The default is ['field', 'season'].

    Returns
    -------
    df : pandas df
        output result.

    """

    import numpy as np

    idx = grp['seldict'] == 'nosel'

    df_ref = pd.DataFrame(grp[idx])

    df = pd.DataFrame(grp[~idx])

    if grpcol:
        df = df.merge(df_ref, left_on=grpcol,
                      right_on=grpcol, suffixes=['', '_ref'])
    else:
        df.loc[:, 'NSN_ref'] = df_ref['NSN'].sum()

    print('kooo', df_ref['NSN'])
    df['err_NSN'] = np.sqrt(
        df['NSN']*(1.-df['NSN']/df['NSN_ref']))/df['NSN_ref']
    df['err_NSN'] *= df['NSN_ref']/norm_factor
    df['NSN'] /= norm_factor
    df['NSN_ref'] /= norm_factor
    df['NSN'] = df['NSN'].astype(int)
    df['err_NSN'] = df['err_NSN'].astype(int)
    print(df.columns)
    return df


def processNSN_z(dd, dbDir, prodType, listDDF, dict_sel, fDir,
                 norm_factor, seldict='G10_JLA'):
    """
    Function to process Data

    Parameters
    ----------
    dd : pandas df
        list of DB to process.
    dbDir : str
        location dir of OS.
    prodType : str
        production type.
    listDDF : str
        list of DDF to process.
    dict_sel : dict
        Selection dict.
    fDir: str
       location dir of the files
    norm_factor: float
      normalization factor

    Returns
    -------
    None.

    """
    sn_field = pd.DataFrame()
    sn_field_season = pd.DataFrame()

    df = pd.DataFrame()
    for io, row in dd.iterrows():
        dbName = row['dbName']

        ddb = sn_load_select(dbDir, dbName, prodType,
                             listDDF=listDDF,
                             fDir=fDir)

        data_dict = ddb.sn_selection(dict_sel)

        vv = data_dict[seldict]

        print(vv['field'].unique())

        res = bin_it(vv, bins=np.arange(0.01, 1.1, 0.05),
                     norm_factor=norm_factor)

        res['dbName'] = dbName

        df = pd.concat((df, res))
        # break

    return df
