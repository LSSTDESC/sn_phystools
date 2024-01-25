#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:58:35 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import numpy as np
from sn_cosmology.random_hd import HD_random, Random_survey, analyze_data_new
from sn_tools.sn_utils import multiproc


class Fit_seasons:
    def __init__(self, fitconfig, dataDir_DD, dbName_DD,
                 dataDir_WFD, dbName_WFD, dictsel, survey,
                 prior, host_effi, frac_WFD_low_sigma_mu=0.8,
                 max_sigma_mu=0.12, test_mode=0, plot_test=0,
                 lowz_optimize=0.1,
                 sigmaInt=0.12, surveyDir='',
                 timescale='year', outName='',
                 fields_for_stat=['COSMOS', 'XMM-LSS', 'ELAISS1', 'CDFS',
                                  'EDFSa', 'EDFSb'],
                 seasons=range(1, 11), nrandom=50, nproc=8):
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
        frac_WFD_low_sigma_mu : float, optional
             fraction of WFD SNe Ia with low sigma_mu. The default is 0.8.
         max_sigma_mu : float, optional
             Max sigma_mu value defining the low sigma_mu sample.
             The default is 0.12.
        test_mode: int, optional
          to run the program in test mode. The default is 0.
        lowz_optimize: float, opt.
           z-value where the number of SN should be maximized.
        sigmaInt : float, optional
           SNe Ia intrinsic dispersion. The default is 0.12.
        surveyDir : str, optional
           the dir where to dump the surveys. The default is ''.
        timescale : str, optional
           Time scale to estimate the cosmology. The default is 'year'.
        outName: str, optional
           output file name. The default is ''.
        fields_for_stat : list(str), optional
            List of fields for stat. The default is
            ['COSMOS', 'XMM-LSS', 'ELAISS1', 'CDFS','EDFSa', 'EDFSb'].
        seasons : list(int), optional
            List of seasons for which cosmology is be estimated.
            The default is range(1,11).
        nrandom: int, optional.
            Number of random survey for season/year. The default is 50. 
        nproc: int, optional.
            Number of procs for processing. The default is 8.
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
        self.frac_WFD_low_sigma_mu = frac_WFD_low_sigma_mu
        self.max_sigma_mu = max_sigma_mu
        self.test_mode = test_mode
        self.plot_test = plot_test
        self.lowz_optimize = lowz_optimize
        self.sigmaInt = sigmaInt
        self.surveyDir = surveyDir
        self.timescale = timescale
        self.outName = outName
        self.fields_for_stat = fields_for_stat
        self.seasons = seasons
        self.nrandom = nrandom
        self.nproc = nproc

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

        params = dict(zip(['nrandom'], [self.nrandom]))
        seasons_data = list(range(1, np.max(self.seasons)+1))

        data = multiproc(seasons_data, params,
                         self.get_random_sample, nproc=self.nproc)

        print('elapse time', time.time()-time_ref,
              len(np.unique(data[['survey_real', 'season']])))
        """
        configs = []
        for seas in self.seasons:
            configs.append([1, seas])
        """
        configs = pd.DataFrame()
        ll = list(range(1, self.nrandom+1))
        df = pd.DataFrame(ll, columns=['survey_real'])
        for seas in self.seasons:
            df[self.timescale] = seas
            configs = pd.concat((configs, df))

        params = {}
        params['data'] = data

        restot = pd.DataFrame()
        for key, vals in self.prior.items():
            params['prior'] = key
            params['prior_params'] = vals
            res = multiproc(configs, params, self.fit_time, nproc=self.nproc)
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
                                 frac_WFD_low_sigma_mu=self.frac_WFD_low_sigma_mu,
                                 max_sigma_mu=self.max_sigma_mu,
                                 test_mode=self.test_mode,
                                 plot_test=self.plot_test,
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

        year_min = 1
        resdf = pd.DataFrame()

        for i, row in configs.iterrows():

            year_max = row[self.timescale]
            nsurvey = row['survey_real']

            # select the data corresponding to these years
            idx = data[self.timescale] >= year_min
            idx &= data[self.timescale] <= year_max
            idx &= data['survey_real'] == nsurvey
            sel_data_fit = data[idx]

            # loop on the realizations

            if self.surveyDir != '':
                self.dump_survey(sel_data_fit, year_min, year_max, nsurvey)

            if self.test_mode:
                print('nsn for this run', len(sel_data_fit))

            # analyze the data
            dict_ana = self.analyze_survey_data(sel_data_fit, year_max)

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
            resdfb['real_survey'] = nsurvey
            resdfb['prior'] = prior
            resdfb = resdfb.fillna(-999.)

            resdf = pd.concat((resdf, resdfb))

        if self.test_mode:
            print('final result', resdf)
            cols = ['w0_fit', 'Om0_fit', 'MoM',
                    'prior', 'dbName_DD', 'dbName_WFD']
            print(resdf[cols])

        if output_q is not None:
            output_q.put({j: resdf})
        else:
            return resdf

    def analyze_survey_data(self, sel_data_fit, year_max):
        """
        Method to analyze the survey (nsn per field)

        Parameters
        ----------
        sel_data_fit : pandas df
            Data to analyze.
        year_max : int
            year_max of the survey.

        Returns
        -------
        dict_ana : dict
            Dict containing an analysis of the survey.

        """

        # analyze the data
        dict_ana = analyze_data_new(sel_data_fit,
                                    fields=self.fields_for_stat)
        # get Nsn with sigmaC <= 0.04
        idx = sel_data_fit['sigma_mu'] <= self.max_sigma_mu
        dict_ana_b = analyze_data_new(
            sel_data_fit[idx], add_str='_sigma_mu',
            fields=self.fields_for_stat)

        if self.test_mode:
            print(dict_ana)
            print(dict_ana_b)
        dict_ana.update(dict_ana_b)
        # print(dict_ana)
        dict_ana[self.timescale] = year_max+1
        # print(dict_ana)

        return dict_ana

    def dump_survey(self, data, year_min, year_max, nn):
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

        Returns
        -------
        None.

        """

        outName = '{}/survey_sn_{}_{}_{}_{}_{}.hdf5'.format(self.surveyDir,
                                                            self.dbName_DD,
                                                            self.dbName_WFD,
                                                            year_min,
                                                            year_max,
                                                            nn)
        data.to_hdf(outName, key='sn')

    def fit_time_deprecated(self, configs, params, j=0, output_q=None):
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
            idx = data[self.timescale] >= year_min
            idx &= data[self.timescale] <= year_max

            sel_data = data[idx]

            resdf = pd.DataFrame()
            # loop on the realizations

            nsurvey = sel_data['survey_real'].unique()

            for nn in nsurvey:
                idc = sel_data['survey_real'] == nn
                sel_data_fit = sel_data[idc]

                if self.surveyDir != '':
                    outName = '{}/survey_sn_{}_{}_{}_{}_{}.hdf5'.format(
                        self.surveyDir,
                        self.dbName_DD, self.dbName_WFD,
                        year_min, year_max, nn)
                    sel_data_fit.to_hdf(outName, key='sn')

                if self.test_mode:
                    print('nsn for this run', len(sel_data_fit))

                # analyze the data
                dict_ana = analyze_data_new(sel_data_fit,
                                            fields=self.fields_for_stat)
                # get Nsn with sigmaC <= 0.04
                idx = sel_data_fit['sigma_mu'] <= self.max_sigma_mu
                dict_ana_b = analyze_data_new(
                    sel_data_fit[idx], add_str='_sigma_mu',
                    fields=self.fields_for_stat)
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
