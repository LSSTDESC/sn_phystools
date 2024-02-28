#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:53:33 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from sn_cosmology.cosmo_fit import MyFit, fom
import numpy as np
from sn_analysis.sn_calc_plot import bin_it_mean
from sn_analysis.sn_tools import transform
from sn_tools.sn_utils import multiproc


class HD_random:
    def __init__(self,
                 vardf=['z_fit', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',
                        'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
                        'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu',
                        'mu_SN'],
                 dataNames=['z', 'x1', 'color', 'mb', 'Cov_x1x1',
                            'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
                            'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu',
                            'mu_SN'],
                 fitconfig={},
                 par_protect_fit=['Om0'],
                 prior=pd.DataFrame({'varname': ['Om0'],
                                     'refvalue': [0.3], 'sigma': [0.0073]}),
                 test_mode=0):
        """
        Class to perform cosmological fits

        Parameters
        ----------
        vardf : list(str), optional
            List of data variables 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',
            'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
            'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu'].
        dataNames : list(str), optional
            list of variable names (corresponding to vardf).
            The default is ['z', 'x1', 'color', 'mb', 'Cov_x1x1',
                            'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
                            'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu'].
        fitconfig : dict, optional
            configuration dict for the fit. The default is {}.
        par_protect_fit : list(str), optional
            List of fit parameters to protect. The default is ['Om0'].
        prior : pandas df, optional
            Prior to apply to the Chisquare.
            The default is
            pd.DataFrame({'varname': ['Om0'],'refvalue':[0.3],
                          'sigma':[0.0073]}).
         test_mode: int, optional.
          To activate the program in test mode. The default is 0.

        Returns
        -------
        None.

        """

        self.vardf = vardf
        self.dataNames = dataNames
        self.fitconfig = fitconfig
        self.par_protect_fit = par_protect_fit
        self.prior = prior
        self.test_mode = test_mode

    def __call__(self, data):
        """
        Method to perform cosmological fits.

        Parameters
        ----------
        data : pandas df
            Data to fit.

        Returns
        -------
        dict_fits : dict
            result with fit parameters.

        """

        dataValues = [data[key] for key in self.vardf]
        par_protect_fit = ['Om0']
        par_protect_fit = []
        r = []
        # r = [('Om0', 0.3, 0.0073)]
        r.append(('sigmaInt', 0.12, 0.01))
        prior = pd.DataFrame(
            {'varname': ['Om0'], 'refvalue': [0.3], 'sigma': [0.0073]})
        # prior = pd.DataFrame()

        dict_fits = {}
        idx = data['zType'] == 'spectroz'
        data_sigmaInt = data[idx]
        dataValues_sigmaInt = [data_sigmaInt[key] for key in self.vardf]

        for key, vals in self.fitconfig.items():

            fitparNames = list(vals.keys())
            fitparams = list(vals.values())
            myfit = MyFit(dataValues, dataValues_sigmaInt, self.dataNames,
                          fitparNames=fitparNames, prior=self.prior,
                          par_protect_fit=self.par_protect_fit)
            # get sigmaInt

            sigmaInt = myfit.get_sigmaInt()

            # sigmaInt = 0.12
            # set sigmaInt
            myfit.set_sigmaInt(sigmaInt)

            # myfit.set_sigmaInt(0.0)

            dict_fit = myfit.minuit_fit(fitparams)
            fitpars = []
            for pp in fitparNames:
                fitpars.append(dict_fit['{}_fit'.format(pp)])
            dict_fit['Chi2_fit'] = myfit.xi_square(*fitpars)
            dict_fit['NDoF'] = len(data)-len(fitparNames)
            dict_fit['Chi2_fit_red'] = dict_fit['Chi2_fit']/dict_fit['NDoF']
            dict_fit['sigmaInt'] = myfit.sigmaInt
            if 'wa_fit' in dict_fit.keys():
                cov_a = dict_fit['Cov_w0_w0_fit']
                cov_b = dict_fit['Cov_wa_wa_fit']
                cov_ab = dict_fit['Cov_wa_w0_fit']
                dict_fit['MoM'] = fom(cov_a, cov_b, cov_ab)
            if self.test_mode:
                print(dict_fit)
            # fisher estimation
            # fisher_cov = myfit.covariance_fisher(fitparams)
            # print('Fisher', fisher_cov)
            # print('')
            dict_fits[key] = dict_fit

        return dict_fits


class Fit_surveys:
    def __init__(self, dataDir_DD, dbName_DD,
                 dataDir_WFD, dbName_WFD, sellist, seasons,
                 survey=pd.DataFrame([('COSMOS', 1.1, 1.e8, 1, 10)],
                                     columns=['field', 'zmax', 'sigmaC',
                                              'season_min', 'season_max']),
                 sigmaInt=0.12, host_effi={}, footprints=pd.DataFrame(),
                 low_z_optimize=True, max_sigma_mu=0.12,
                 test_mode=0, plot_test=0, lowz_optimize=0.1,
                 timescale='year', nrandom=50, hd_fit=None,
                 fields_for_stat=['COSMOS', 'XMM-LSS', 'ELAISS1', 'CDFS',
                                  'EDFSa', 'EDFSb'],
                 simu_norm_factor=pd.DataFrame(),
                 nproc=8,
                 vardf=['z_fit', 'mu', 'sigma_mu', 'field', 'healpixID'],
                 surveyDir=''):
        """
        Class to build a complete (WFD+DDF) random survey

        Parameters
        ----------
        dataDir_DD : str
           Location dir of DDF data.
        dbName_DD : str
           dbName of DD data.
        dataDir_WFD : str
           Location dir of WFD data.
        dbName_WFD : str
           dbName of WFD data.
        sellist : list(str)
           selection criteria.
        seasons : list(int)
            List of seasons to process.
        survey : pandas df, optional
            DESCRIPTION. The default is
            pd.DataFrame([('COSMOS', 1.1, 1.e8, 1, 10)],
                         columns=['field', 'zmax', 'sigmaC',
                                  'season_min', 'season_max']).
        sigmaInt: float, opt.
          SN intrinsic dispersion. The default is 0.12.
        host_effi: dict, opt
          1D interpolators of host_effi vs z. The default is {}.
        footprints: pandas df,opt.
          footprints used for spectroz samples. The default is pd.DataFrame().
       low_z_optimize: bool, optional
         to maximize low-z NSN. The default is True
        max_sigma_mu : float, optional
             Max sigmaC value defining the low sigma_mu sample.
             The default is 0.12.
        test_mode: int, optional
            to run the program in test mode. The default is 0.
        lowz_optimize: float, opt.
           z-value where the number of SN should be maximized.
        timescale : str, optional
          Time scale to estimate the cosmology. The default is 'year'.
        nrandom: int, opt.
          number of random survey. The default is 50.


        Returns
        -------
        None.

        """

        self.dataDir_DD = dataDir_DD
        self.dbName_DD = dbName_DD
        self.dataDir_WFD = dataDir_WFD
        self.dbName_WFD = dbName_WFD
        self.sellist = sellist
        self.seasons = seasons
        self.survey = survey
        self.sigmaInt = sigmaInt
        self.host_effi = host_effi
        self.footprints = footprints
        self.low_z_optimize = low_z_optimize
        self.max_sigma_mu = max_sigma_mu
        self.test_mode = test_mode
        self.plot_test = plot_test
        self.lowz_optimize = lowz_optimize
        self.timescale = timescale
        self.nrandom = nrandom
        self.hd_fit = hd_fit
        self.fields_for_stat = fields_for_stat
        self.simu_norm_factor = simu_norm_factor
        self.nproc = nproc
        self.vardf = vardf+['SNID']+[self.timescale]
        self.surveyDir = surveyDir

    def fit_sn_samples(self):
        """
        Method to build random samples SN of DDF+WFD and fit

        Returns
        -------
        sn_sample : pandas df
            The sn random samples (nrandom realization of the survey).

        """

        nsn = pd.DataFrame()
        # sn_sample = pd.DataFrame()

        dataDir = {}
        dbName = {}
        dataDir['DDF'] = self.dataDir_DD
        dataDir['WFD'] = self.dataDir_WFD
        dbName['DDF'] = self.dbName_DD
        dbName['WFD'] = self.dbName_WFD

        fieldTypes = np.unique(
            self.survey[['zType', 'fieldType']].to_records(index=False))

        # sort to have spectroz first
        fieldTypes = sorted(fieldTypes.tolist())[::-1]

        # load the data for all the needed seasons

        global sn_simu_season
        sn_simu_season = {}
        for seas in self.seasons:
            # load the data corresponding to this seas
            sn_simu_season[seas] = self.load_data_season(
                fieldTypes, dataDir, dbName, seas)

        print('data loaded')

        params = {}
        # params['sn_simu_season'] = sn_simu_season
        randomlist = list(range(1, self.nrandom+1, 1))
        resdf = multiproc(randomlist, params, self.fit_parallel, self.nproc)

        return resdf

    def fit_parallel(self, randlist, params, j=0, output_q=None):
        """
        Method to perform HD fits

        Parameters
        ----------
        randlist : list(int)
            list of nsurvey.
        params : dict
            parameters.
        j : int, optional
            multiprocessing tag. The default is 0.
        output_q : multiprocessing queue, optional
            where to put the results. The default is None.

        Returns
        -------
        resdf : pandas df
            Output results.

        """

        # sn_simu_season = params['sn_simu_season']

        resdf = pd.DataFrame()
        for rr in randlist:
            res_fit = self.fit_random_sample(rr)
            res_fit['real_survey'] = rr
            resdf = pd.concat((resdf, res_fit))

        if output_q is not None:
            output_q.put({j: resdf})
        else:
            return resdf

    def fit_random_sample(self, sreal):
        """
        Method to fit a random sample

        Parameters
        ----------
        sn_simu_season : dict
            available sn (key=season) for random HD.

        Returns
        -------
        resdf : pandas df
            Fit result.

        """

        # build random samples for each season
        df_tot = pd.DataFrame()
        sn_sample = pd.DataFrame()

        rand_survey = Random_survey(self.survey,
                                    self.footprints, self.timescale,
                                    self.sigmaInt, self.host_effi,
                                    self.low_z_optimize,
                                    self.plot_test, self.test_mode)

        footprints = self.survey['survey'].to_list()
        footprints = list(map(lambda x: '{}_footprint'.format(x), footprints))
        footprints = set(footprints)

        for seas in self.seasons:
            # res = self.build_sample(sn_simu_season[seas], seas)
            sn_simu_seas = sn_simu_season[seas]

            # make a realization of this survey
            rand_LSST = self.random_LSST(sn_simu_seas)

            # make a random survey for the season
            res, res_foot = rand_survey(rand_LSST, seas)

            sn_sample = pd.concat((sn_sample, res))

            year_max = sn_sample[self.timescale].max()

            # clean the survey to remove duplicate

            sn_sample = self.clean_survey(sn_sample)

            # fit this sample
            res, sel_data_fit = self.fit_data_iterative(sn_sample)

            # analyze the data

            dict_ana = self.analyze_survey_data(sel_data_fit, year_max)

            # merge survey data and fitted params
            dict_res = self.merge_values(res, dict_ana)

            prior = 'noprior'
            if len(self.hd_fit.prior) > 0:
                prior = 'prior'

            resdf = self.complete_data(dict_res, prior)

            # add nsn footprint here
            ccols = set(res_foot.columns.to_list())

            diffcol = list(footprints ^ ccols)
            if diffcol:
                res_foot[diffcol] = 0

            resdf = pd.concat((resdf, res_foot), axis=1)

            df_tot = pd.concat((resdf, df_tot))
            del resdf

        #del sn_sample
        del sn_sample

        if self.surveyDir != '':
            year_min = df_tot[self.timescale].min()
            year_max = df_tot[self.timescale].max()
            self.dump_survey(df_tot, year_min, year_max, sreal)

        return df_tot

    def fit_random_sample_deprecated(self):
        """
        Method to fit a random sample

        Parameters
        ----------
        sn_simu_season : dict
            available sn (key=season) for random HD.

        Returns
        -------
        resdf : pandas df
            Fit result.

        """

        # build random samples for each season
        sn_sample = pd.DataFrame()
        rand_survey = Random_survey(self.survey,
                                    self.footprints, self.timescale,
                                    self.sigmaInt, self.host_effi,
                                    self.low_z_optimize,
                                    self.plot_test, self.test_mode)
        for seas in self.seasons:
            # res = self.build_sample(sn_simu_season[seas], seas)
            sn_simu_seas = sn_simu_season[seas]

            # make a realization of this survey
            rand_LSST = self.random_LSST(sn_simu_seas)

            res = rand_survey(rand_LSST, seas)
            sn_sample = pd.concat((sn_sample, res))

        year_max = sn_sample[self.timescale].max()

        # clean the survey to remove duplicate

        sn_sample = self.clean_survey(sn_sample)

        # fit this sample
        res, sel_data_fit = self.fit_data_iterative(sn_sample)

        del sn_sample

        # analyze the data

        dict_ana = self.analyze_survey_data(sel_data_fit, year_max)

        # merge survey data and fitted params
        dict_res = self.merge_values(res, dict_ana)

        prior = 'noprior'
        if len(self.hd_fit.prior) > 0:
            prior = 'prior'
        resdf = self.complete_data(dict_res, prior)

        return resdf

    def random_LSST(self, sn_simu_seas):
        """
        Method to buil a realization of the survey

        Parameters
        ----------
        sn_simu_seas : dict
            dict of data (pandas df)

        Returns
        -------
        dd : dict
            output surveys.

        """

        dd = {}
        for key, vals in sn_simu_seas.items():
            idx = self.simu_norm_factor['survey'] == key
            norm = self.simu_norm_factor[idx]['norm_factor'].values[0]
            sn_survey = pd.DataFrame()
            for field in vals['field'].unique():
                idx = vals['field'] == field
                sel = vals[idx]
                nsn = int(len(sel)/norm)
                samp_ = sel.sample(nsn)
                sn_survey = pd.concat((sn_survey, samp_))

            dd[key] = sn_survey

        return dd

    def clean_survey(self, data, var='SNID'):
        """
        Method to remove duplicate in SNID

        Parameters
        ----------
        data : pandas df
            Data to process.

        Returns
        -------
        res : pandas df
            Data with duplicates dropped.

        """
        dfdup = data[data[var].duplicated(keep=False)]

        snids_dup = dfdup[var].to_list()

        idx = data[var].isin(snids_dup)
        df_dup = data[idx]
        df_dup = df_dup.groupby([var]).apply(lambda x: self.add_survey(x))

        if len(df_dup) > 0 and self.test_mode:
            print("duplicate", df_dup[[var, 'survey']])

        df_res = pd.DataFrame(data[~idx])
        df_res = pd.concat((df_res, df_dup))

        res = df_res.drop_duplicates(subset='SNID')

        return res

    def add_survey(self, grp):
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

    def fit_data_iterative(self, data):
        """
        Method to perform an iterative cosmo fit (outliers removal)

        Parameters
        ----------
        data : pandas df
            Data to fit.

        Returns
        -------
        res : dict
            Fit parameters.
        sel_data_fit : pandas df
            Clipped data.

        """

        frac_out = 1.
        nsigma = 5.
        frac_outliers = 0.05

        dd = pd.DataFrame(data)
        while frac_out >= frac_outliers:
            res, dd, frac_out = self.fit_data_cleaned(dd, nsigma)

        return res, dd

    def fit_data_cleaned(self, data, nsigma):
        """
        Method to estimate data at nsigma level of the cosmo fit

        Parameters
        ----------
        data : pandas df
            Data to process.
        nsigma : int
            nsigma fit cut.

        Returns
        -------
        data_no_out : pandas df
            clipped data.
        frac_out : float
            Outlier fraction.

        """

        from astropy.cosmology import w0waCDM
        H0 = 70.
        idx = data['sigma_mu'] <= 0.25
        data = pd.DataFrame(data[idx])
        res_fit = self.hd_fit(data)

        keys = list(res_fit.keys())

        res = res_fit[keys[0]]
        w0 = res['w0_fit']
        wa = res['wa_fit']
        Om = res['Om0_fit']

        cosmology = w0waCDM(H0=H0,
                            Om0=Om,
                            Ode0=1.-Om,
                            w0=w0, wa=wa)

        z_fit = data['z_fit'].to_list()
        data['mu_fit'] = cosmology.distmod(z_fit).value
        data['delta_mu'] = (data['mu_SN']-data['mu_fit'])/data['sigma_mu']
        idx = np.abs(data['delta_mu']) <= nsigma
        data_no_out = data[idx]
        frac_out = 1.-len(data_no_out)/len(data)

        if self.test_mode:
            self.plot_fit_results(data, data_no_out)

        return res_fit, data_no_out, frac_out

    def load_data_season(self, fieldTypes, dataDir, dbName, seas):
        """
        Method to load data per season

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

            data_ = self.load_data(
                dataDir[ftype], dbName[ftype], name, ftype, [seas])

            data_ = data_[self.vardf]
            # nsn_ = self.load_nsn_summary(dataDir[ftype], dbName[ftype],
            #                             '{}_{}'.format(ftype, ztype), [seas])

            # nsn_ = self.estimate_nsn_z_allfields_sigma_mu(data_)

            # nsn_ = self.get_nsn_from_survey(nsn_, self.survey, ftype, ztype)

            data_survey[name] = data_
            # nsn_survey[name] = nsn_

        return data_survey

    def load_data(self, dataDir, dbName, runType, fieldType, seasons):
        """
        Method to load data (SN)

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
                search_dir, fieldType, dbName, self.timescale, seas)
            # print('search path', search_path)
            files += glob.glob(search_path)

            for fi in files:
                da = pd.read_hdf(fi)
                df = pd.concat((df, da))

        return df

    def load_nsn_summary(self, dataDir, dbName, runType, seas):
        """
        Method to load nsn stat.

        Parameters
        ----------
        dataDir : str
            Data location dir.
        dbName : str
            OS name.
        runType : str
            Run type (ie DDF_spectroz, WFD_spectroz, DDF_photoz, WFD_photoz).

        Returns
        -------
        res : pandas df
            nsn stat.

        """

        theDir = '{}/{}/{}'.format(dataDir, dbName, runType)

        theName = '{}/nsn_{}_{}.hdf5'.format(theDir, dbName, self.timescale)

        res = pd.read_hdf(theName)

        idx = res[self.timescale].isin(seas)

        return res[idx]

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
        dict_ana = analyze_data_sample(sel_data_fit,
                                       fields=self.fields_for_stat)
        # get Nsn with sigmaC <= 0.04
        idx = sel_data_fit['sigma_mu'] <= self.max_sigma_mu
        dict_ana_b = analyze_data_sample(
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

    def complete_data(self, dict_res, prior):
        """
        Method to finalize data

        Parameters
        ----------
        dict_res : dict
            Data to transform in pandas df.
        prior : str
            prior.

        Returns
        -------
        resdfb : pandas df
            Output data.

        """

        resdfb = pd.DataFrame()
        for key, vals in dict_res.items():
            resdfb = pd.concat((resdfb, vals))

        resdfb['dbName_DD'] = self.dbName_DD
        resdfb['dbName_WFD'] = self.dbName_WFD
        resdfb['prior'] = prior
        resdfb = resdfb.fillna(-999.)

        return resdfb

    def merge_values(self, res, dict_ana):
        """
        Method to merge results (from survey info and fit params)

        Parameters
        ----------
        res : pandas df
            fit results.
        dict_ana : dict
            survey infos.

        Returns
        -------
        dict_res : dict
            Merged data.

        """

        dict_res = {}
        # fitted values in a df
        for key, vals in res.items():
            vals.update(dict_ana)
            res = pd.DataFrame.from_dict(transform(vals))
            if key not in dict_res.keys():
                dict_res[key] = pd.DataFrame()
            dict_res[key] = pd.concat((res, dict_res[key]))

        return dict_res

    def plot_fit_results(self, data, data_no_out):
        """
        Method to plot cosmo fit resuls on top of data

        Parameters
        ----------
        data : pandas df
            Data to plot.
        data_no_out : pandas df
            Cleaned data.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        data = data.sort_values(by=['z_fit'])
        ax.plot(data['z_fit'], data['mu_SN'], 'k.')
        ax.plot(data['z_fit'], data['mu_fit'], color='r', marker='o')

        figb, axb = plt.subplots()
        axb.hist(data_no_out['delta_mu'], histtype='step', bins=80)
        plt.show()


class Random_survey:
    def __init__(self, survey, footprints,
                 timescale, sigmaInt, host_effi, low_z_optimize=True,
                 plot_test=False, test_mode=False):
        """
        class to generate random surveys

        Parameters
        ----------
        survey : pandas df
            Survey parameter.
        footprints : pandas df
            Footprints.
        timescale : str
            Time scale (year/season).
        sigmaInt : float
            sigma_int parameter.
        host_effi : interp1D
            Host-z efficiency.
        plot_test : int, optional
            To plot in test mode. The default is False.
        test_mode : int, optional
            To activate the test mode. The default is False.

        Returns
        -------
        None.

        """

        self.survey = survey
        self.footprints = footprints
        self.timescale = timescale
        self.sigmaInt = sigmaInt
        self.host_effi = host_effi
        self.low_z_optimize = low_z_optimize
        self.plot_test = plot_test
        self.test_mode = test_mode

    def __call__(self, data_survey, seas):
        """
        Method to build sn sample

        Parameters
        ----------
        seas : int
            season num.

        Returns
        -------
        res : pandas df
            sn_sample.

        """

        # get the data for all the surveys (footprints)
        # data_survey, nsn_survey = self.get_data_surveys(sn_simu_seas)

        """
        for key, val in data_survey.items():
            print(key, len(val), nsn_survey[key][['nsn', 'nsn_survey']])

        if self.test_mode:
            for key, vals in nsn_survey.items():
                print(key)
                print(vals[['nsn', 'nsn_survey']])
        """
        # now get the random surveys

        res, res_foot = self.instance_random_survey(data_survey, seas)

        del data_survey
        res = self.correct_mu(res)
        res[self.timescale] = seas

        return res, res_foot

    def get_data_surveys_deprecated(self, data_simu):
        """
        Method to grab data for all the surveys (defined in self.survey)

        Parameters
        ----------
        data_simu : dict
            Simu data. key=(zType_fieldType), vals=pd.DataFrame

        Returns
        -------
        data_survey : dict
            Data to be used to build the survey.
            (key,val)=(survey name, pd.Dataframe)
        nsn_survey : dict
            nsn for the survey . (key, val)=(survey name, pd.DataFrame)

        """

        data_survey = {}
        nsn_survey = {}

        for i, row in self.survey.iterrows():
            survey = row['survey']
            fType = row['fieldType']
            zType = row['zType']
            field = row['field']
            footprint = row['footprint']
            surveyName = row['survey']

            data_, nsn_ = self.get_data_survey(data_simu, surveyName,
                                               zType, fType,
                                               field, footprint)

            data_survey[survey] = data_
            nsn_survey[survey] = nsn_

        return data_survey, nsn_survey

    def get_data_survey_deprecated(self, data_simu, surveyName,
                                   zType, fType, field, footprint):
        """
        Method to grab data for a survey

        Parameters
        ----------
        data_simu : pandas df
            Data to select.
        zType : str
            z measurement (spectroz/photz).
        fType : str
            Field type (DDF/WFD).
        field : str
            Field name.
        footprint : str
            Foot print to apply.

        Returns
        -------
        data_ : pandas df
            sn data corresponding to the footprint.
        nsn_ : pandas df
            number of expected SNe Ie (vs redshift).

        """

        data_type = '{}_{}'.format(zType, fType)
        data_ = data_simu[data_type]
        idx = data_['field'] == field
        data_ = pd.DataFrame(data_[idx])
        data_['survey'] = surveyName
        if self.test_mode:
            print('before foot', len(data_))
        # apply footprint
        data_ = self.apply_footprint(data_, footprint)

        if self.test_mode:
            print('after foot', len(data_))

        # get numbers for this survey
        nsn_ = self.estimate_nsn_z_survey(data_)

        if self.test_mode:
            for ccol in nsn_.columns:
                print(ccol, nsn_[ccol].values)

        return data_, nsn_

    def instance_random_survey(self, survey_lsst, seas):
        """
        Method to perform a random realization of the survey

        Parameters
        ----------
        survey_lsst : dict
            Data to select. (key, val)=(surveyName,SN[pandas df])
        seas: int
          data season.

        Returns
        -------
        sn_survey : pandas df
            A realization of a (full) survey with footprints

        """

        sn_sample = pd.DataFrame()
        sn_foot = pd.DataFrame()
        for i, vv in self.survey.iterrows():
            idx = seas >= vv['season_min']
            idx &= seas <= vv['season_max']
            if not idx:
                continue
            sname = '{}_{}'.format(vv['fieldType'], vv['zType'])
            data = survey_lsst[sname]
            # get the field in this data

            df_samp = self.sn_sample_survey(data, vv)

            sn_foot['{}_footprint'.format(vv['survey'])] = [df_samp.nsn_foot]
            sn_sample = pd.concat((sn_sample, df_samp))

            del df_samp

        return sn_sample, sn_foot

    def sn_sample_survey(self, data, vv):
        """
        Method to build a SNe Ia sample for a survey

        Parameters
        ----------
        data : pandas df
            Data to use.
        vv : pandas series
            survey parameters.

        Returns
        -------
        res_host : pandas df
            SNe Ia sample.

        """

        idxc = data['field'] == vv['field']
        idxc &= data['z_fit'] <= vv['zmax']
        dataf = data[idxc]
        # apply footprint
        datafoot = self.apply_footprint(dataf, vv['footprint'])
        nsn_foot = len(datafoot)

        if vv['survey'] != 'WFD_TiDES':
            nsn_sample = np.min([nsn_foot, vv['nsn_max_season']])

            # grab the sample
            df_samp = datafoot.sample(int(nsn_sample))

        else:
            if self.low_z_optimize:
                nsn_z = self.estimate_nsn_z_survey(datafoot)
                nsn_z['survey'] = vv['survey']
                nsn_z['nsn_survey'] = vv['nsn_max_season']
                df_samp = self.sn_sample_z(datafoot, nsn_z)
            else:
                nsn_sample = np.min([nsn_foot, vv['nsn_max_season']])

                # grab the sample
                df_samp = datafoot.sample(int(nsn_sample))

        # apply effi host
        res_host = self.effi_zhost(df_samp, vv['host_effi'])
        res_host['fieldType'] = vv['fieldType']
        res_host['zType'] = vv['zType']
        res_host['footprint'] = vv['footprint']
        res_host['survey'] = vv['survey']

        del dataf
        del datafoot
        del df_samp

        res_host.nsn_foot = nsn_foot
        return res_host

    def instance_random_survey_deprecated(self, data, nsn):
        """
        Method to perform a random realization of the survey

        Parameters
        ----------
        data : dict
            Data to select. (key, val)=(surveyName,SN[pandas df])
        nsn : dict
            nsn to get (key,val)=(surveyName,NSN[pandas df]).

        Returns
        -------
        sn_survey : pandas df
            A realization of a (full) survey

        """

        sn_survey = pd.DataFrame()
        for key, vals in data.items():
            nsn_survey = nsn[key]['nsn_survey'].unique()[0]
            if nsn_survey > 0:
                df = self.random_sample(key, vals, nsn[key], sn_survey)
                sn_survey = pd.concat((sn_survey, df))
                del df

        return sn_survey

    def analyze_random_sample(self, data, nsn_season=None):
        """
        Method to analyze the random sample

        Parameters
        ----------
        data : pandas df
            Data to analyze.
        nsn_season : pandas df, optional
            Reference values. The default is None.

        Returns
        -------
        None.

        """

        surveys = np.unique(
            data[['zType', 'fieldType']].to_records(index=False))

        dd = data.groupby(['zType', 'fieldType']).apply(
            lambda x: self.estimate_nsn_z_allfields_sigma_mu(x)).reset_index()

        print(dd)
        if nsn_season is not None:
            print(nsn_season)

    def get_nsn_wfd_corr(self, nsn_wfd_photz, nsn_wfd_spectroz):
        """
        Method to estimate nsn from WFD (photz) from WFD spectroz measurements

        Parameters
        ----------
        nsn_wfd_photz : pandas df
            array with the number of photz WFD SNe Ia.
        nsn_wfd_spectroz :  pandas df
            array with the number of spectro WFD SNe Ia..

        Returns
        -------
        tt : pandas df
            array with the corrected number of photz WFD SNe Ia.

        """

        tt = nsn_wfd_photz.merge(nsn_wfd_spectroz[['season', 'nsn_survey']],
                                 left_on=['season'], right_on=['season'],
                                 suffixes=('', '_spectroz'))

        tt['nsn_survey'] -= tt['nsn_survey_spectroz']
        tt = tt.drop(columns=['nsn_survey_spectroz'])
        tt['nsn_survey'] = tt['nsn_survey'].clip(lower=0)

        return tt

    def get_nsn_from_survey(self, nsn):
        """
        Method to grab NSN from the survey

        Parameters
        ----------
        nsn : pandas df
            Initial values.
        survey : pandas df
            survey to consider.
        fieldType: str
          Type of field (DDF/WFD)
        zType: str
          z type (spectroz/photz)

        Returns
        -------
        nsn_new : pandas df
            new nsn values according to the survey.

        """

        fields = nsn['field'].unique()

        # get survey info
        surveyName = nsn['survey'].unique()[0]
        idx = self.survey['survey'] == surveyName
        sel_survey = self.survey[idx]

        simu_factor = sel_survey['simuFactor'].values[0]
        seas_min = sel_survey['season_min']
        seas_max = sel_survey['season_max']
        nsn_survey = sel_survey['nsn_max_season'].values[0]

        idxc = nsn[self.timescale] >= seas_min.values[0]
        idxc &= nsn[self.timescale] <= seas_max.values[0]
        nsn_sel = pd.DataFrame(nsn[idxc])

        if len(nsn_sel) == 0:
            nsn_sel = pd.DataFrame(nsn)
            nsn_sel['nsn_survey'] = 0
        else:
            ccols = list(nsn_sel.filter(like='nsn').columns)
            nsn_sel[ccols] = np.rint(nsn_sel[ccols]/simu_factor)
            nsn_sel['nsn_survey'] = nsn_survey
            nsn_sel['nsn_survey'] = nsn_sel[[
                'nsn', 'nsn_survey']].min(axis=1)
            ccolsb = list(nsn_sel.filter(like='nsn').columns)
            nsn_sel = nsn_sel.fillna(0.)
            nsn_sel[ccolsb] = nsn_sel[ccolsb].astype(int)

        nsn_new = pd.DataFrame(nsn_sel)
        del nsn_sel
        return nsn_new

    def plot_mu(self, data, yvar='mu', H0=70, Om0=0.3, Ode0=0.7, w0=-1., wa=0.0):
        """
        Method to plot mu vs z and compare to a cosmology

        Parameters
        ----------
        data : pandas df
            Data to plot.
        yvar : str, optional
            y-axis variable. The default is 'mu'.
        H0 : float, optional
            H0 parameter. The default is 70.
        Om0 : float, optional
            Om0 parameter. The default is 0.3.
        Ode0 : float, optional
            Ode0 parameter. The default is 0.7.
        w0 : float, optional
            w0 parameter. The default is -1..
        wa : float, optional
            wa parameter. The default is 0.0.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        df = bin_it_mean(data, xvar='z_fit', yvar=yvar)
        print(df)
        ax.errorbar(df['z_fit'], df[yvar], yerr=df['{}_std'.format(yvar)],
                    marker='o', color='k', mfc='None', ms=5)

        from astropy.cosmology import w0waCDM
        cosmo = w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa)

        bins = np.arange(df['z_fit'].min(), 1.1, 0.02)
        f = cosmo.distmod(bins).value

        ax.plot(bins, f, 'bs', ms=5)

        df['diff'] = df[yvar]-f
        figb, axb = plt.subplots()

        axb.plot(df['z_fit'], df['diff'], 'ko')

        idx = df['z_fit'] >= 0.5
        print('mean diff', np.mean(df[idx]['diff']))

        """
        idx = data['sigmaC'] <= 0.04
        seldata = data[idx]
        ax.plot(seldata['z'], seldata['mu'], 'r*', ms=5)

        figb, axb = plt.subplots()
        axb.plot(data['z'], data['sigma_mu'], 'ko', mfc='None', ms=5)
        """
        plt.show()

    def plot_nsn_z(self, data):
        """
        Method to plot the number of supernovas vs z

        Parameters
        ----------
        data : pandas df
            Data to plot.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        from sn_analysis.sn_calc_plot import bin_it

        fig, ax = plt.subplots()

        fields = data['field'].unique()

        for field in fields:
            idx = data['field'] == field
            sel = data[idx]
            selbin = bin_it(sel)
            ax.plot(selbin['z_fit'], selbin['NSN'])

        plt.show()

    def random_sample(self, surveyName, sn_data, nsn_z, ongoing_survey):
        """
        Method to get a random sn sample corresponding to a survey (surveyName)

        Parameters
        ----------
        surveyName : str
            Name of the survey to consider.
        sn_data : pandas df
            sn data sample to use for random choice.
        nsn_z : pandas df
            nsn per zbin.

        Returns
        -------
        res_host : pandas df
            The (randomly) selected SNe Ia sample.

        """

        if self.test_mode:
            print(surveyName, len(sn_data), len(nsn_z))

        idf = self.survey['survey'] == surveyName

        host_effi_key = self.survey[idf]['host_effi'].values[0]
        fieldType = self.survey[idf]['fieldType'].values[0]
        zType = self.survey[idf]['zType'].values[0]
        footprint = self.survey[idf]['footprint'].values[0]

        if self.test_mode:
            print('rr', host_effi_key, fieldType, zType, footprint)
        # check whether this field has already been observed
        sn_data, nsn_z = self.check_survey(
            surveyName, footprint, sn_data, nsn_z, ongoing_survey)

        res = self.sn_sample(sn_data, nsn_z, fieldType)

        res_host = self.effi_zhost(res, host_effi_key)
        res_host['fieldType'] = fieldType
        res_host['zType'] = zType
        res_host['footprint'] = footprint
        res_host['survey'] = surveyName

        if self.test_mode:
            print(res_host.columns)
            print(len(res_host))

        return res_host

    def check_survey(self, surveyName, footprint,
                     sn_data, nsn_z, ongoing_survey):
        """
        Method to check whether a field has already been observed
        (and correct for data and nsn_z accordingly)

        Parameters
        ----------
        surveyName : str
            survey name.
        footprint : str
            footprint.
        sn_data : pandas df
            Data to proces.
        nsn_z : pandas df
            NSN(z).
        ongoing_survey : pandas df
            SN of the (being built) survey.

        Returns
        -------
        sn_data : pandas df
            corrected data set.
        nsn_z : pandas df
            Corrected NSN(z).

        """

        if len(ongoing_survey) == 0:
            return sn_data, nsn_z

        field = sn_data['field'].unique()
        idx = ongoing_survey['fieldType'].isin(field)

        sel_survey = ongoing_survey[idx]

        if len(sel_survey) > 0:
            if self.test_mode:
                print('this field has already been observed', field)
            # on the same area as the current survey:
            # need to estimate how many nsn observed and where
            # apply the footprint on the (already measured) data
            common_survey = self.apply_footprint(sel_survey, footprint)

            # get the number of sn in this common_survey
            nsn_common = self.estimate_nsn_z_survey(common_survey)

            # correct nsn

            """
            print('before')
            columns = nsn_z.columns
            for ccol in columns:
                print(ccol, nsn_z[ccol].values, nsn_common[ccol].values)
            """

            nsn_z = self.correct_nsn_survey(nsn_z, nsn_common)
            """
            print('after')
            for ccol in columns:
                print(ccol, nsn_z[ccol].values)
            """
            # remove sn already observed
            snids = common_survey['SNID'].to_list()
            idg = sn_data['SNID'].isin(snids)
            sn_data = pd.DataFrame(sn_data[~idg])

        return sn_data, nsn_z

    def correct_mu(self, data, H0=70, Om0=0.3, Ode0=0.7,
                   w0=-1., wa=0.0, alpha=0.13, beta=3.1):
        """
        Method to re-estimate distance moduli (MB correction)
        and add sigmaInt to the distance modulus error.

        Parameters
        ----------
        data : pandas df
            Data to process.
        H0 : float, optional
            H0 parameter. The default is 70.
        Om0 : float, optional
            Om0 parameter. The default is 0.3.
        Ode0 : float, optional
            Ode0 parameter. The default is 0.7.
        w0 : float, optional
            w0 parameter. The default is -1..
        wa : float, optional
            wa parameter. The default is 0.0.
        alpha: float, optional.
            nuisance parameter for SN. The default is 0.13
        beta: float, optional.
            nuisance parameter for SN. The default is 3.1

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """

        from random import gauss
        from astropy.cosmology import w0waCDM
        cosmo = w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa)

        sigmu = data['sigma_mu'].to_list()
        """
        var_mu = data['Cov_mbmb']\
            + (alpha**2)*data['Cov_x1x1']\
            + (beta**2)*data['Cov_colorcolor']\
            + 2*alpha*data['Cov_x1mb']\
            - 2*beta*data['Cov_colormb']\
            - 2*alpha*beta*data['Cov_x1color']
        """
        bins = data['z_fit'].to_list()
        dist_mu = cosmo.distmod(bins).value

        # sigmu = np.sqrt(var_mu).to_list()
        # sigma_mu = [np.sqrt(sigmu[i]**2+self.sigmaInt**2)
        #            for i in range(len(sigmu))]

        sigmaInt = [self.sigmaInt]*len(sigmu)
        sigma_mu = np.array(sigmu)**2+np.array(sigmaInt)**2
        sigma_mu = np.sqrt(sigma_mu)

        # mu = [gauss(dist_mu[i], sigma_mu[i]) for i in range(len(dist_mu))]
        mu_shift = np.random.normal(0., sigma_mu)

        data['mu_SN'] = dist_mu+mu_shift
        # data['sigma_mu_SN'] = sigmu
        data['sigma_mu'] = sigmu

        # idx = data['zType'] == 'photz'
        # z_shift = shift = np.random.normal(0., 0.02*(1.+data[idx]['z']))
        # data.loc[idx, 'z'] += z_shift

        return data

    def correct_nsn_survey(self, dfa, dfb, nsn_survey_corr=False):
        """
        Method to correct for nsn_zmin_zmax

        Parameters
        ----------
        dfa : pandas df
            Data to correct.
        dfb : pandas df
            Data for correction.

        Returns
        -------
        dfc : pandas df
            Resulting data.

        """

        """
        zlim = np.arange(0.0, 1.1, 0.1)
        bbin = []


        for i in range(len(zlim)-1):
            zmin = zlim[i]
            zmax = zlim[i+1]
            bbin.append('z_{}_{}'.format(zmin, zmax))
         """

        surveyName = dfa['survey'].unique()[0]

        # dfa = dfa.drop(columns=['survey'])
        # dfb = dfb.drop(columns=['survey'])

        dfc = dfa.merge(dfb, left_on=['field'], right_on=[
            'field'], suffixes=['', '_y'])

        ccolsy = list(dfc.filter(like='_y').columns)
        ccols = list(map(lambda x: x.replace('_y', ''), ccolsy))

        dfc[ccols] -= dfc[ccolsy].values

        dfc = dfc.drop(columns=ccolsy)

        dfc[ccols] = dfc[ccols].clip(0)

        # special treatment for nsn_survey

        if not nsn_survey_corr:
            dft = pd.DataFrame(dfa['nsn'].to_list(), columns=['nsn'])
            dft['nsn_survey'] = dfa['nsn_survey'].to_list()
            dft['nsn_survey_to_sub'] = dfb['nsn_survey'].to_list()
            dft['diff'] = dft['nsn']-dft['nsn_survey_to_sub']
            dft['diff'] = dft['diff'].clip(0)

            dfc['nsn_survey'] = dft[['nsn_survey', 'diff']].min(axis=1)

        if self.test_mode:
            ccols = ['field', 'nsn_z_0.0_0.1',
                     'nsn_z_0.0_0.1_low_sigma', 'nsn_survey']
            print('before')
            self.print_nsn_z(dfa)
            print('to substract')
            self.print_nsn_z(dfb)
            print('finally')
            self.print_nsn_z(dfc)

        dfc['survey'] = surveyName
        return dfc

    def print_nsn_z(self, dd):

        for ccol in dd.columns:
            print(ccol, dd[ccol].values)

    def sn_sample(self, data, nsn_z, fieldType='WFD'):
        """
        Method to grab the sn sample

        Parameters
        ----------
        data : pandas df
            Data to process.
        nsn_exp: int
           Number of expected SN
        nsn : int
            number of sn to get.
        frac_sigmaC : float
            frac of SN with sigmaC<0.04 to get.
        field: str
            field name
        nsn_lowz : int, optional
           Number of low-z SN. The default is 0.
        zlow : float, optional
           Redshift defining the low-z sample. The default is 0.1.

        Returns
        -------
        res : pandas df
            Sampled data.

        """

        if self.plot_test:
            if fieldType == 'WFD':
                print('data in this footprint:', len(data)/10.)
                self.plot_Moll(data, rot=(180., 0., 0.))
            if fieldType == 'DDF':
                self.plot_Moll(data, nside=128)

        res = pd.DataFrame()
        nsn_survey = nsn_z['nsn_survey'].astype(int).values[0]

        if nsn_survey >= len(data):
            res = data
        else:
            # grab the random sample
            if fieldType != 'WFD':
                res = data.sample(n=nsn_survey)
            else:
                res = self.sn_sample_z(data, nsn_z)

        return res

    def sn_sample_z(self, data, nsn_z):
        """
        Method to build SNe Ia sample vz z

        Parameters
        ----------
        data : pandas df
            Data to process.
        nsn_z : pandas df
            nsn vs z.

        Returns
        -------
        res : pandas df
            SN sample.

        """

        # first: sample_lowsigma_mu
        idx = data['sigma_mu'] <= self.sigmaInt
        sel_data = pd.DataFrame(data[idx])
        sample_lowsigma = self.sample_max_lowz_new(
            sel_data, nsn_z, suffix='_low_sigma', suffix_search=True)

        del sel_data

        # nsn_z_low = self.estimate_nsn_z_allfields_sigma_mu(
        #    sample_lowsigma)

        nsn_z_low = self.estimate_nsn_z_sigma_mu(sample_lowsigma)
        # print(nsn_z_low)
        # correct nsn_z to grab high sigma_mu sample

        nsn_corr = self.correct_nsn_survey(
            nsn_z, nsn_z_low, nsn_survey_corr=True)

        # print(nsn_corr)

        if self.test_mode:
            print('get high sample')

        sample_highsigma = self.sample_max_lowz_new(
            data[~idx], nsn_corr, suffix='_low_sigma', suffix_search=False)

        res = pd.concat((sample_lowsigma, sample_highsigma))

        return res

    def apply_footprint(self, data, footprint):
        """
        Function to superimpose footprint on data

        Parameters
        ----------
        data : pandas df
            Data to process.
        footprint : str
            Footprint name.

        Returns
        -------
        data : pandas df
            Data with footprint.

        """

        idx = self.footprints['footprint'] == footprint
        sel_foot = self.footprints[idx]
        if len(sel_foot) > 0:
            list_pix = sel_foot['healpixID'].to_list()
            idxb = data['healpixID'].isin(list_pix)
            data = data[idxb]
        else:
            print('warning: footprint not found', footprint)

        return data

    def plot_Moll(self, data, nside=64, rot=(0., 0., 0.)):
        """
        Method to plot data in (RA,Dec) (Mollweide view)

        Parameters
        ----------
        data : pandas df
            Data to plot.
        nside : int, optional
            nside healpix parameter. The default is 64.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        from sn_tools.sn_visu import get_map, plot_pixels
        map_pixel = get_map(nside)

        ll = data['healpixID'].to_list()

        idx = map_pixel['healpixID'].isin(ll)

        map_pixel.loc[~idx, 'weight'] = -1

        plot_pixels(map_pixel, rot=rot)

        plt.show()

    def get_sigma_mu_fraction_in_data(self, data):
        """
        Method to estimate the fraction of SNe Ia with sigmaC < max_sigmaC

        Parameters
        ----------
        data : pandas df
            Data to process.

        Returns
        -------
        float
            SNe fraction with sigma_mu<=0.12 in data.

        """

        idx = data['sigma_mu'] <= self.max_sigma_mu

        frac = len(data[idx])/len(data)

        return np.round(frac, 3)

    def estimate_nsn_z_survey(self, data):

        nsn = self.estimate_nsn_z_sigma_mu(data)

        """
        surveyName = data['survey'].unique()[0]
        nsn['survey'] = surveyName
        nsn = self.get_nsn_from_survey(nsn)
        """
        return nsn

    def estimate_nsn_z_sigma_mu(self, data, varx='z'):
        """
        Method to estimate nsn per z bin

        Parameters
        ----------
        data : pandas df
            Data to process.
        varx : str, optional
            variable of interest. The default is 'z'.

        Returns
        -------
        pandas df
            nsn per z bins.

        """

        dfa = self.estimate_nsn_z_allfields(data)
        dfb = self.estimate_nsn_z_allfields(data, sigma_mu=self.sigmaInt)

        df = dfa.merge(dfb, left_on=['field', self.timescale, 'nsn', 'nsn_survey'],
                       right_on=['field', self.timescale, 'nsn', 'nsn_survey'],
                       suffixes=['', '_low_sigma'])

        """
        if 'survey' not in df.columns:
            surveyName = data['survey'].unique()[0]
            df['survey'] = surveyName
        """
        return df

    def estimate_nsn_z_allfields_sigma_mu(self, data_all, varx='z_fit'):
        """
        Method to estimate nsn per z bin

        Parameters
        ----------
        data : pandas df
            Data to process.
        varx : str, optional
            variable of interest. The default is 'z'.

        Returns
        -------
        pandas df
            nsn per z bins.

        """

        fields = data_all['field'].unique()

        df = pd.DataFrame()
        for field in fields:
            idx = data_all['field'] == field
            sel_data = data_all[idx]

            dfa = self.estimate_nsn_z_allfields(sel_data)

            dfb = self.estimate_nsn_z_allfields(
                sel_data, sigma_mu=self.sigmaInt)

            dfc = dfa.merge(dfb, left_on=['field', 'season', 'nsn', 'nsn_survey'],
                            right_on=['field', 'season', 'nsn', 'nsn_survey'],
                            suffixes=['', '_low_sigma'])

            df = pd.concat((df, dfc))

        return df

    def estimate_nsn_z_allfields(self, data_all, varx='z_fit', sigma_mu=1.e6):
        """
        Method to estimate nsn per z bin

        Parameters
        ----------
        data : pandas df
            Data to process.
        varx : str, optional
            variable of interest. The default is 'z'.

        Returns
        -------
        pandas df
            nsn per z bins.

        """

        idx = data_all['sigma_mu'] <= sigma_mu
        data = data_all[idx]

        df = data.groupby(['field']).apply(
            lambda x: self.estimate_nsn_z(x)).reset_index()
        df[self.timescale] = int(data[self.timescale].median())
        df['nsn'] = len(data_all)
        df['nsn_survey'] = len(data_all)

        return df

    def estimate_nsn_z(self, data, varx='z_fit'):
        """
        Method to estimate nsn per z bin

        Parameters
        ----------
        data : pandas df
            Data to process.
        varx : str, optional
            variable of interest. The default is 'z'.

        Returns
        -------
        pandas df
            nsn per z bins.

        """

        zlim = np.arange(0.0, 1.2, 0.1)
        group = data.groupby(pd.cut(data[varx], zlim))

        nsn = group.size().to_list()

        bin_center = (zlim[:-1] + zlim[1:])/2
        delta_bin = np.mean(np.diff(bin_center))/2

        zmax = np.round(bin_center+delta_bin, 1)
        zmin = np.round(bin_center-delta_bin, 1)
        zmin = list(map(str, zmin))
        zmax = list(map(str, zmax))
        nsnstr = ['nsn_z']*len(zmin)
        r = [nsnstr, zmin, zmax]

        cols = list(map("_".join, zip(*r)))
        nsnb = np.array(nsn)
        nsnb = nsnb.reshape((1, len(nsn))).tolist()
        dfb = pd.DataFrame(nsnb, columns=cols)
        nsn_tot = np.sum(nsn)
        dfb['nsn'] = nsn_tot
        dfb['nsn_survey'] = nsn_tot

        return dfb

    def sample_max_lowz(self, data, nsn, nsn_lowz, frac_WFD_low_sigmaC):
        """
        Method to choose a sample maximizing lowz sn

        Parameters
        ----------
        data : pandas df
            Data to process.
        nsn : int
            total number of SNe Ia.
        nsn_lowz : int
            Number of low-z SNe Ia.
        frac_WFD_low_sigma: float
          fraction of SNe Ia with sigmaC < 0.4

        Returns
        -------
        res : pandas df
            Result.

        """

        zlim = np.arange(0.0, 1.1, 0.1)
        r = []
        nsn_tot = 0
        restot = pd.DataFrame()
        for i in range(len(zlim)-1):
            zmin = zlim[i]
            zmax = zlim[i+1]
            nname = 'nsn_z_{}_{}'.format(np.round(zmin, 1), np.round(zmax, 1))
            idx = data['z_fit'] >= zmin
            idx &= data['z_fit'] < zmax

            nsn_exp = (nsn_lowz[nname].values[0])*frac_WFD_low_sigmaC
            nsn_exp = int(nsn_exp)
            nsn_diff = nsn-nsn_tot
            if nsn_diff >= 0:
                sel_data = data[idx]
                nsn_rand = np.min([nsn_exp, nsn_diff])
                ndata = len(sel_data)
                nsn_rand = np.min([nsn_rand, ndata])
                res = sel_data.sample(nsn_rand)
                restot = pd.concat((restot, res))
                nsn_tot += nsn_rand

        """
        idx = data['z'] <= zlow
        if nsn_lowz > 0:
            res = data[idx].sample(nsn_lowz)
        else:
            res = pd.DataFrame()

        nsn_rem = nsn-nsn_lowz
        resb = data[~idx].sample(nsn_rem)
        res = pd.concat((res, resb))
        """

        return restot

    def sample_max_lowz_new(self, data, nsn_z, suffix='_low_sigma',
                            suffix_search=False):

        idxa = nsn_z.columns.str.contains('nsn')
        idxb = nsn_z.columns.str.contains(suffix)

        idx = idxa & idxb
        if suffix_search == False:
            idx = idxa & ~idxb

        ccols = nsn_z.columns[idx].to_list()

        nsn_survey = nsn_z['nsn_survey'].mean()
        nsn_ = nsn_survey

        if nsn_ == 0:
            return pd.DataFrame()

        sn = pd.DataFrame()
        ia, ib = -2, -1
        if suffix_search:
            ia -= 2
            ib -= 2

        for col in ccols:
            spl = col.split('_')
            zmin = float(spl[ia])
            zmax = float(spl[ib])

            nsn_exp = nsn_z[col].mean()
            if nsn_exp < 1:
                continue

            nsn_sample = int(np.min([nsn_exp, nsn_]))

            if self.test_mode:
                print('allo', np.unique(data[['field']]),
                      data[self.timescale].unique(), len(data))

            idxd = data['z_fit'] >= zmin
            idxd &= data['z_fit'] < zmax
            sel_data = pd.DataFrame(data[idxd])

            sn_ = sel_data.sample(nsn_sample)
            sn = pd.concat((sn, sn_))

            nsn_ -= nsn_sample
            del sel_data
            if nsn_ == 0:
                break

        return sn

    def get_info(self, survey, field):
        """
        Method to grab survey info

        Parameters
        ----------
        survey : pandas df
            Survey infos.
        field : str
            Field to consider.

        Returns
        -------
        zmax : float
            Max redshift for the field.
        sigmaC : float
            sigma color max value.
        season_min : int
            Min season of observation.
        season_max : int
            Max season of observation.

        """

        idx = survey['field'] == field
        sel = survey[idx]
        zmax = sel['host_effi'].values[0]
        season_min = sel['season_min'].values[0]
        season_max = sel['season_max'].values[0]

        return zmax, season_min, season_max

    def effi_zhost(self, data, host_effi_key):
        """
        Method to apply zhost efficiencies on SN distribution

        Parameters
        ----------
        data : pandas df
            Data to process.
        host_effi_key: str
          key dict for host_effi

        Returns
        -------
        data_new : pandas df
            data with host effi applied.

        """
        if not self.host_effi:
            return data

        bin_width = 0.1
        zmin = 0.0
        zmax = 1.2
        bins = np.arange(zmin, zmax, bin_width)
        bins_center = (bins[:-1] + bins[1:])/2
        group = data.groupby(pd.cut(data['z_fit'], bins))
        df = pd.DataFrame(group.size().to_list(), columns=['nsn'])

        df['effi'] = self.host_effi[host_effi_key](bins_center)
        df['nsn_effi'] = (df['nsn']*df['effi']).apply(np.ceil).astype(int)
        df['zmin'] = bins_center-bin_width/2.
        df['zmax'] = bins_center+bin_width/2.
        df['z'] = bins_center

        """
        if self.test_mode:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(df['z'], df['effi'])
            plt.show()
        """
        data_new = pd.DataFrame()
        for io, row in df.iterrows():
            zmin = row['zmin']
            zmax = row['zmax']
            nsn = int(row['nsn_effi'])
            if nsn <= 0:
                continue
            idx = data['z_fit'] >= zmin
            idx &= data['z_fit'] < zmax
            sel = data[idx]
            if not sel.empty:
                dd = sel.sample(nsn)
                data_new = pd.concat((data_new, dd))

        return data_new

    def plot_sample_zhost(self, data, data_new, field):
        """
        Method to plot (check) nsn vs z for two sets of data

        Parameters
        ----------
        data : pandas df
            First set of data.
        data_new : pandas df
            second set of data.
        field : str
            Field name.

        Returns
        -------
        None.

        """

        import matplotlib.pyplot as plt
        from sn_analysis.sn_calc_plot import bin_it

        bins = np.arange(0.0, 1.2, 0.1)
        group = data.groupby(pd.cut(data['z'], bins))
        group_new = data_new.groupby(pd.cut(data_new['z'], bins))
        print('before', group.size())
        print('after', group_new.size())

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(field)

        df = bin_it(data, bins=bins)
        dfb = bin_it(data_new, bins=bins)
        ax.plot(df['z'], df['NSN'], marker='o', mfc='None', color='k')
        ax.plot(dfb['z'], dfb['NSN'], marker='*', color='r')

        ax.set_xlabel('z')
        ax.set_ylabel('N$_{SN}$')
        ax.grid()

        print('hohoho', len(data),
              data['field'].unique(), data.columns)

        figb, axb = plt.subplots()
        figb.suptitle(field)
        axb.plot(data['pixRA'], data['pixDec'], 'k.')

        plt.show()


def analyze_data(data, add_str=''):
    """
    Function to analyze data

    Parameters
    ----------
    data : pandas df
        data to process.
    add_str: str, opt
        to add a str to the columns. The default is ''

    Returns
    -------
    outdict : dict
        output dict.

    """

    res = data.groupby(['field', 'season']).apply(
        lambda x: pd.DataFrame({'NSN': [len(x)]})).reset_index()

    resb = res.groupby(['field']).apply(
        lambda x: pd.DataFrame({'NSN': [x['NSN'].sum()]})).reset_index()

    outdict = {}
    nsn_tot = 0
    for i, row in resb.iterrows():
        field = row['field']
        nsn = row['NSN']
        outdict['{}{}'.format(field, add_str)] = nsn
        nsn_tot += nsn

    outdict['all_Fields{}'.format(add_str)] = nsn_tot
    return outdict


def analyze_data_sample(data, add_str='',
                        fields=['COSMOS', 'XMM-LSS', 'ELAISS1',
                                'CDFS', 'EDFSa', 'EDFSb']):
    """
    Function to analyze data

    Parameters
    ----------
    data : pandas df
        data to process.
    add_str: str, opt
        to add a str to the columns. The default is ''

    Returns
    -------
    outdict : dict
        output dict.

    """

    ztypes = ['photz', 'spectroz']

    dd = {}

    # get the total number z>0.7, 0.8

    for vv in [0.7, 0.8]:
        idx = data['z_fit'] >= vv
        sel = data[idx]
        dd['nsn_z_{}{}'.format(np.round(vv, 1), add_str)] = len(sel)

    for field in fields:
        idx = data['field'] == field
        sel_data = data[idx]
        dd['{}{}'.format(field, add_str)] = len(sel_data)
        if len(sel_data) > 0:
            for zt in ztypes:
                idxb = sel_data['zType'] == zt
                selb = sel_data[idxb]
                dd['{}_{}{}'.format(field, zt, add_str)] = len(selb)

        else:
            for zt in ztypes:
                dd['{}_{}{}'.format(field, zt, add_str)] = 0

    fieldTypes = ['DDF', 'WFD']

    for fieldType in fieldTypes:
        idx = data['fieldType'] == fieldType
        sel_data = data[idx]
        dd['{}{}'.format(fieldType, add_str)] = len(sel_data)
        if len(sel_data) > 0:
            for zt in ztypes:
                idxb = sel_data['zType'] == zt
                selb = sel_data[idxb]
                dd['{}_{}{}'.format(fieldType, zt, add_str)] = len(selb)
        else:
            for zt in ztypes:
                dd['{}_{}{}'.format(fieldType, zt, add_str)] = 0

    dd['all_Fields{}'.format(add_str)] = len(data)

    surveys = data['survey'].unique()

    for survey in surveys:
        idx = data['survey'] == survey
        sel = data[idx]
        dd['{}{}'.format(survey, add_str)] = len(sel)

    return dd
    """
    res = data.groupby(['field', 'season']).apply(
        lambda x: pd.DataFrame({'NSN': [len(x)]})).reset_index()

    resb = res.groupby(['field']).apply(
        lambda x: pd.DataFrame({'NSN': [x['NSN'].sum()]})).reset_index()

    outdict = {}
    nsn_tot = 0
    for i, row in resb.iterrows():
        field = row['field']
        nsn = row['NSN']
        outdict['{}{}'.format(field, add_str)] = nsn
        nsn_tot += nsn

    outdict['all_Fields{}'.format(add_str)] = nsn_tot
    return outdict
    """
