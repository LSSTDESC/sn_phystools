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
                 dataDir_WFD, dbName_WFD, dictsel, survey,
                 prior, host_effi, frac_WFD_low_sigmaC=0.8,
                 max_sigmaC=0.04, test_mode=0, lowz_optimize=0.1,
                 sigmaInt=0.12, dump_data=False, timescale='year'):
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
        host_effi: dict
            dict of 1D interpolators for host effi vs z.
        frac_WFD_low_sigmaC : float, optional
             fraction of WFD SNe Ia with low sigmaC. The default is 0.8.
         max_sigmaC : float, optional
             Max sigmaC value defining the low sigmaC sample.
             The default is 0.04.
        test_mode: int, optional
          to run the program in test mode. The default is 0.
        lowz_optimize: float, opt.
           z-value where the number of SN should be maximized.
        sigmaInt : float, optional
           SNe Ia intrinsic dispersion. The default is 0.12.
        dump_data : bool, optional
           To dump the cosmology realizations. The default is False.
        timescale : str, optional
           Time scale to estimate the cosmology. The default is 'year'.

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
        self.host_effi = host_effi
        self.frac_WFD_low_sigmaC = frac_WFD_low_sigmaC
        self.max_sigmaC = max_sigmaC
        self.test_mode = test_mode
        self.lowz_optimize = lowz_optimize
        self.sigmaInt = sigmaInt
        self.dump_data = dump_data
        self.timescale = timescale

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
        n_random = 50
        if self.test_mode:
            n_season_max = 3
            n_random = 1

        for seas_max in range(2, n_season_max):
            seasons = range(1, seas_max)

            params = {}
            params[self.timescale] = seasons
            nrandom = range(n_random)
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
        seasons = params[self.timescale]
        hd_fit = HD_random(fitconfig=self.fitconfig,
                           prior=self.prior, test_mode=self.test_mode)

        dict_res = {}

        # print('process', j, nrandom)
        for i in nrandom:

            # get the data
            data = Random_survey(self.dataDir_DD, self.dbName_DD,
                                 self.dataDir_WFD, self.dbName_WFD,
                                 self.dictsel, seasons,
                                 survey=self.survey, sigmaInt=self.sigmaInt,
                                 host_effi=self.host_effi,
                                 frac_WFD_low_sigmaC=self.frac_WFD_low_sigmaC,
                                 max_sigmaC=self.max_sigmaC,
                                 test_mode=self.test_mode,
                                 lowz_optimize=self.lowz_optimize,
                                 timescale=self.timescale).data

            if self.dump_data:
                outName = 'SN_{}_{}_{}.hdf5'.format(
                    self.dbName_DD, self.dbName_WFD, i)
                data.to_hdf(outName, key='sn')

            if self.test_mode:
                print('nsn for this run', len(data))
            # analyze the data
            dict_ana = analyze_data(data)
            # get Nsn with sigmaC <= 0.04
            idx = data['sigmaC'] <= self.max_sigmaC
            dict_ana_b = analyze_data(data[idx], add_str='_sigmaC')
            if self.test_mode:
                print(dict_ana)
                print(dict_ana_b)
            dict_ana.update(dict_ana_b)
            # print(dict_ana)
            dict_ana['season'] = np.max(seasons)+1
            # print(dict_ana)

            # fit the data
            res = hd_fit(data)

            # fitted values in a df
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
