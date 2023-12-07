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
                 sigmaInt=0.12, dump_data=False,
                 timescale='year', outName=''):
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
        outName: str, optional
           output file name. The default is ''.

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
        self.outName = outName

        if outName != '':
            import os
            if os.path.isfile(outName):
                os.remove(outName)

    def __call__(self):
        """
        Main method to perform the cosmological fits for a set of seasons

        Returns
        -------
        dict_fi : dict
            Fit parameter result.

        """
        # dict_fi = {}

        # grab random data

        import time
        time_ref = time.time()
        seasons = range(1, 11)
        nproc = 8
        nrandom = 50
        if self.test_mode:
            nproc = 1
            nrandom = 5
            seasons = [1, 2]

        params = dict(zip(['nrandom'], [nrandom]))
        data = multiproc(seasons, params, self.get_random_sample, nproc=nproc)

        print('elapse time', time.time()-time_ref)
        configs = []
        for seas in seasons:
            configs.append([1, seas])
        params = {}
        params['data'] = data

        restot = pd.DataFrame()
        for key, vals in self.prior.items():
            params['prior'] = key
            params['prior_params'] = vals
            res = multiproc(configs, params, self.fit_time, nproc=nproc)
            restot = pd.concat((restot, res))

        if self.outName != '':
            restot.to_hdf(self.outName, key='cosmofit')

        return 0

    def get_random_sample(self, seasons, params, j=0, output_q=None):
        """
        Method to get random realizations of surveys

        Parameters
        ----------
        seasons : list(int)
            List of seasons (years) to process..
        params : dict
            parameters.
        j : int, optional
            Internal tag for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            Queue for multiprocessing. The default is None.

        Returns
        -------
        df : pandas df
            The random samples.

        """

        nrandom = params['nrandom']

        df = pd.DataFrame()
        for seas in seasons:

            data = Random_survey(self.dataDir_DD, self.dbName_DD,
                                 self.dataDir_WFD, self.dbName_WFD,
                                 self.dictsel, [seas],
                                 survey=self.survey, sigmaInt=self.sigmaInt,
                                 host_effi=self.host_effi,
                                 frac_WFD_low_sigmaC=self.frac_WFD_low_sigmaC,
                                 max_sigmaC=self.max_sigmaC,
                                 test_mode=self.test_mode,
                                 lowz_optimize=self.lowz_optimize,
                                 timescale=self.timescale, nrandom=nrandom).data
            df = pd.concat((df, data))

        if output_q is not None:
            output_q.put({j: df})
        else:
            return df

    def fit_time(self, configs, params, j=0, output_q=None):
        """
        Method to perform cosmological fits on a set of configurations

        Parameters
        ----------
        configs : list(list(int))
            List of years (seasons) to process.
        params : dict
            parameters to use.
        j : int, optional
            Tag for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            Queue for multiprocessing. The default is None.

        Returns
        -------
        resdf : pandas df
            fitted cosmological parameters.

        """

        data = params['data']
        prior = params['prior']
        prior_params = params['prior_params']
        hd_fit = HD_random(fitconfig=self.fitconfig,
                           prior=prior_params, test_mode=self.test_mode)

        resfi = pd.DataFrame()
        for config in configs:
            year_min = config[0]
            year_max = config[1]

            # select the data corresponding to these years
            idx = data['year'] >= year_min
            idx &= data['year'] <= year_max

            sel_data = data[idx]

            resdf = pd.DataFrame()
            # loop on the realizations

            nsurvey = sel_data['survey_real'].unique()

            for nn in nsurvey:
                idc = sel_data['survey_real'] == nn
                sel_data_fit = sel_data[idc]

                print('fitting', year_min, year_max, nn, len(sel_data_fit))
                if self.dump_data:
                    outName = 'SN_{}_{}_{}_{}_{}.hdf5'.format(
                        self.dbName_DD, self.dbName_WFD,
                        year_min, year_max, nn)
                    sel_data_fit.to_hdf(outName, key='sn')

                if self.test_mode:
                    print('nsn for this run', len(sel_data_fit))

                # analyze the data
                dict_ana = analyze_data(sel_data_fit)
                # get Nsn with sigmaC <= 0.04
                idx = sel_data_fit['sigmaC'] <= self.max_sigmaC
                dict_ana_b = analyze_data(sel_data_fit[idx], add_str='_sigmaC')
                if self.test_mode:
                    print(dict_ana)
                    print(dict_ana_b)
                dict_ana.update(dict_ana_b)
                # print(dict_ana)
                dict_ana[self.timescale] = year_max+1
                # print(dict_ana)

                # fit the data
                res = hd_fit(sel_data_fit)
                dict_res = {}
                # fitted values in a df
                for key, vals in res.items():
                    vals.update(dict_ana)
                    res = pd.DataFrame.from_dict(transform(vals))
                    if key not in dict_res.keys():
                        dict_res[key] = pd.DataFrame()
                    dict_res[key] = pd.concat((res, dict_res[key]))

                # print('sequence', time.time()-time_ref)
                resdfb = pd.DataFrame()
                for key, vals in dict_res.items():
                    resdfb = pd.concat((resdfb, vals))

                resdfb['dbName_DD'] = self.dbName_DD
                resdfb['dbName_WFD'] = self.dbName_WFD
                resdfb['real_survey'] = nn
                resdfb['prior'] = prior
                resdfb = resdfb.fillna(-999.)

                resdf = pd.concat((resdf, resdfb))

            resfi = pd.concat((resfi, resdf))
            if self.test_mode:
                print('final result', resdf)
                cols = ['w0_fit', 'Om0_fit', 'MoM',
                        'prior', 'dbName_DD', 'dbName_WFD']
                print(resdf[cols])

            # if self.outName != '':
            #    resdf.to_hdf(self.outName, key='cosmofit', append=True)

        if output_q is not None:
            output_q.put({j: resfi})
        else:
            return resdf

    def fit_seasons_deprecated(self, nrandom, params, j=0, output_q=None):
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
