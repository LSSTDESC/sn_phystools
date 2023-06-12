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


class sn_load_select:
    def __init__(self, dbDir, dbName, prodType,
                 listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb'):
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

        Returns
        -------
        None.

        """

        outName_stack = 'SN_{}.hdf5'.format(dbName)
        # load the data
        if not os.path.exists(outName_stack):
            data = load_complete_dbSimu(
                dbDir, dbName, prodType, listDDF=listDDF)
            data.to_hdf(outName_stack, key='SN')
        else:
            data = pd.read_hdf(outName_stack, key='SN', mode='r')

        # selectc only fitted LC
        # idx = data['fitstatus'] == 'fitok'
        res = data
        self.printRes(res)

        res['dbName'] = dbName
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

        rr = res.groupby(['field', 'season']).apply(
            lambda x: self.sn_stat(x, survey_area)).reset_index()
        rr = rr.to_records(index=False)
        print(rr[['field', 'season', 'NSN', 'survey_area']])

    def sn_stat(self, grp, survey_area):

        nsn = grp['NSN'].sum()
        nheal = len(grp['healpixID'].unique())

        return pd.DataFrame({'NSN': [nsn], 'survey_area': [nheal*survey_area]})

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
