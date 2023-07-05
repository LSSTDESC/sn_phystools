#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:58:35 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import numpy as np
from sn_cosmology.random_hd import HD_random, Random_survey, analyze_data
from sn_tools.sn_utils import multiproc


class Fit_seasons:
    def __init__(self, fitconfig, dataDir_DD, dbName_DD,
                 dataDir_WFD, dbName_WFD, dictsel, survey, prior):
        """
        Class to perform fits for sets of season

        Parameters
        ----------
        fitconfig : dict
            Fit confoguration.
        dataDir_DD : str
            DD data dir.
        dbName_DD : str
            DD db name.
        dataDir_WFD : str
            WFD data dir.
        dbName_WFD : str
            WFD db name.
        dictsel : list(str)
            Selection criteria.
        survey : pandas df
            Survey.
        prior : pandas df
            prior for the fit.

        Returns
        -------
        None.

        """

        self.fitconfig = fitconfig
        self.dataDir_DD = dataDir_DD
        self.dbName_DD = dbName_DD
        self.dataDir_WFD = dataDir_WFD
        self.dbName_WFD = dbName_WFD
        self.dictsel = dictsel
        self.survey = survey
        self.prior = prior

    def __call__(self):
        """
        Main method to perform the cosmological fits for a set of seasons

        Returns
        -------
        dict_fi : dict
            Fit parameter result.

        """
        # dict_fi = {}
        resfi = pd.DataFrame()
        n_season_max = 12
        # n_season_max = 3
        for seas_max in range(2, n_season_max):
            seasons = range(1, seas_max)

            params = {}
            params['seasons'] = seasons
            nrandom = range(50)
            # res = self.fit_seasons(seasons, params, self.fit_seasons, nproc=8)
            res = multiproc(nrandom, params, self.fit_seasons, nproc=8)
            resfi = pd.concat((resfi, res))
            """
            keys = dict_res.keys()
            for key in keys:
                if key not in dict_fi.keys():
                    dict_fi[key] = pd.DataFrame()
                dict_fi[key] = pd.concat((dict_fi[key], dict_res[key]))
            """
        return resfi

    def fit_seasons(self, nrandom, params, j=0, output_q=None):
        """
        Method to make fits on random samples of data

        Parameters
        ----------
        seasons : list(int)
            List of seasons to process.

        Returns
        -------
        dict_res : pandas df
            Fit parameter results.

        """
        seasons = params['seasons']
        hd_fit = HD_random(fitconfig=self.fitconfig, prior=self.prior)

        dict_res = {}

        # print('process', j, nrandom)
        for i in nrandom:

            data = Random_survey(self.dataDir_DD, self.dbName_DD,
                                 self.dataDir_WFD, self.dbName_WFD,
                                 self.dictsel, seasons,
                                 survey=self.survey).data

            # print('nsn', len(data))
            dict_ana = analyze_data(data)
            dict_ana['season'] = np.max(seasons)+1
            # print(dict_ana)

            res = hd_fit(data)

            for key, vals in res.items():
                vals.update(dict_ana)
                res = pd.DataFrame.from_dict(transform(vals))
                if key not in dict_res.keys():
                    dict_res[key] = pd.DataFrame()
                dict_res[key] = pd.concat((res, dict_res[key]))

        # print('sequence', time.time()-time_ref)

        resdf = pd.DataFrame()
        for key, vals in dict_res.items():
            resdf = pd.concat((resdf, vals))

        resdf = resdf.fillna(-999.)

        if output_q is not None:
            output_q.put({j: resdf})
        else:
            return resdf

        # return dict_res
        # return resdf


def transform(dicta):
    """
    Function to transform a dict of var to a dict of list(var)

    Parameters
    ----------
    dicta : dict
        input dict.

    Returns
    -------
    dictb : dict
        output dict.

    """

    dictb = {}

    for key, vals in dicta.items():
        dictb[key] = [vals]

    return dictb
