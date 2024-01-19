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


class HD_random:
    def __init__(self,
                 vardf=['z', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',
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
        for key, vals in self.fitconfig.items():

            fitparNames = list(vals.keys())
            fitparams = list(vals.values())
            myfit = MyFit(dataValues, self.dataNames,
                          fitparNames=fitparNames, prior=self.prior,
                          par_protect_fit=self.par_protect_fit)
            # get sigmaInt

            sigmaInt = myfit.get_sigmaInt()

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


class Random_survey:
    def __init__(self, dataDir_DD, dbName_DD,
                 dataDir_WFD, dbName_WFD, sellist, seasons,
                 survey=pd.DataFrame([('COSMOS', 1.1, 1.e8, 1, 10)],
                                     columns=['field', 'zmax', 'sigmaC',
                                              'season_min', 'season_max']),
                 sigmaInt=0.12, host_effi={},
                 frac_WFD_low_sigmaC=0.8, max_sigmaC=0.04,
                 test_mode=0, lowz_optimize=0.1, timescale='year', nrandom=50):
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
        frac_WFD_low_sigmaC : float, optional
             fraction of WFD SNe Ia with low sigmaC. The default is 0.8.
        max_sigmaC : float, optional
             Max sigmaC value defining the low sigmaC sample.
             The default is 0.04.
        test_mode: int, optional
            to run the program in test mode. The default is 0.
        lowz_optimize: float, opt.
           z-value where the number of SN should be maximized.
        timescale : str, optional
          Time scale to estimate the cosmology. The default is 'year'.

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
        self.frac_WFD_low_sigmaC = frac_WFD_low_sigmaC
        self.max_sigmaC = max_sigmaC
        self.test_mode = test_mode
        self.lowz_optimize = lowz_optimize
        self.timescale = timescale
        self.nrandom = nrandom

        # load data per season
        self.data = self.build_random_samples()

    def build_random_samples(self):
        """
        Method to build a random sample SN of DDF+WFD

        Returns
        -------
        sn_sample : pandas df
            The sn random samples (nrandom realization of the survey).

        """

        nsn = pd.DataFrame()
        sn_sample = pd.DataFrame()

        zlim = np.arange(0.0, 1.1, 0.1)
        rname = []
        for i in range(len(zlim)-1):
            zmin = zlim[i]
            zmax = zlim[i+1]
            nname = 'nsn_z_{}_{}'.format(np.round(zmin, 1), np.round(zmax, 1))
            rname.append(nname)

        for seas in self.seasons:
            print('Building samples', self.timescale, seas)
            ddf = self.load_data(self.dataDir_DD, self.dbName_DD,
                                 'DDF_spectroz', 'DDF', [seas])
            nsn_ddf = self.load_nsn_summary(
                self.dataDir_DD, self.dbName_DD, 'DDF_spectroz')
            wfd_spectroz = self.load_data(self.dataDir_WFD, self.dbName_WFD,
                                          'WFD_spectroz', 'WFD', [seas])

            nsn_wfd_spectroz = self.load_nsn_summary(
                self.dataDir_WFD, self.dbName_WFD, 'WFD_spectroz')

            wfd_photz = self.load_data(self.dataDir_WFD, self.dbName_WFD,
                                       'WFD_photz', 'WFD', [seas])

            nsn_wfd_photz = self.load_nsn_summary(
                self.dataDir_WFD, self.dbName_WFD, 'WFD_photz')

            wfd_spectroz['field'] = 'WFD_spectroz'
            nsn_wfd_spectroz['field'] = 'WFD_spectroz'
            wfd_photz['field'] = 'WFD_photz'
            nsn_wfd_photz['field'] = 'WFD_photz'

            print('hello', nsn_wfd_photz)
            print('hello', nsn_wfd_spectroz)

            nsn = pd.concat((nsn_ddf, nsn_wfd_spectroz))
            nsn = nsn.fillna(0)

            nsn['nsn'] = nsn['nsn'].astype(int)
            nsn[rname] = nsn[rname].astype(int)

            sn_data = pd.concat((ddf, wfd_spectroz))

            for i in range(self.nrandom):
                sn_samp = self.random_sample(nsn, sn_data, self.survey, [seas])

                sn_samp = self.correct_mu(sn_samp)

                """
                cols = ['Cov_t0t0', 'Cov_t0x0', 'Cov_t0x1', 
                        'Cov_t0color', 'Cov_x0x0',
                        'Cov_x0x1', 'Cov_x0color', 'Cov_x1x1', 
                        'Cov_x1color', 'Cov_colorcolor']
                print(sn_samp[cols])
                """
                sn_samp[self.timescale] = seas
                sn_samp['survey_real'] = i+1

            # self.plot_mu(sn_samp, 'mu_SN')

                sn_sample = pd.concat((sn_sample, sn_samp))

        # self.plot_nsn_z(sn_sample)

        return sn_sample

    def get_nsn_from_survey(self, nsn, survey) -> pd.DataFrame:

        fields = nsn['field'].unique()

        for field in fields:
            idx = survey['field'] == field
            sel_survey = survey[idx]
            seas_min = sel_survey['season_min']
            seas_max = sel_survey['season_max']

            idxb = nsn['field'] == field
            idxb &= nsn[self.timescale] >= seas_min
            idxb &= nsn[self.timescale] <= seas_max
            sel_nsn = nsn[idxb]

    def build_random_sample(self):
        """
        Method to build a random sample SN of DDF+WFD

        Returns
        -------
        sn_sample : pandas df
            The sn random sample.

        """

        nsn = pd.DataFrame()
        sn_sample = pd.DataFrame()
        for seas in self.seasons:
            ddf = self.load_data(self.dataDir_DD, self.dbName_DD,
                                 'DDF_spectroz', 'DDF', [seas])
            nsn_ddf = self.load_nsn_summary(
                self.dataDir_DD, self.dbName_DD, 'DDF_spectroz')
            wfd = self.load_data(self.dataDir_WFD, self.dbName_WFD,
                                 'WFD_spectroz', 'WFD', [seas])

            nsn_wfd = self.load_nsn_summary(
                self.dataDir_WFD, self.dbName_WFD, 'WFD_spectroz')

            nsn = pd.concat((nsn_ddf, nsn_wfd))
            nsn = nsn.fillna(0)

            nsn['nsn'] = nsn['nsn'].astype(int)
            nsn['nsn_z_0.1'] = nsn['nsn_z_0.1'].astype(int)
            nsn['nsn_z_0.2'] = nsn['nsn_z_0.2'].astype(int)

            sn_data = pd.concat((ddf, wfd))

            sn_samp = self.random_sample(nsn, sn_data, self.survey, [seas])

            sn_samp = self.correct_mu(sn_samp)

            # self.plot_mu(sn_samp, 'mu_SN')

            sn_sample = pd.concat((sn_sample, sn_samp))

        # self.plot_nsn_z(sn_sample)

        return sn_sample

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
            files += glob.glob(search_path)

            for fi in files:
                da = pd.read_hdf(fi)
                df = pd.concat((df, da))

        return df

    def load_nsn_summary(self, dataDir, dbName, runType):
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

        return res

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
        var_mu = data['Cov_mbmb']\
            + (alpha**2)*data['Cov_x1x1']\
            + (beta**2)*data['Cov_colorcolor']\
            + 2*alpha*data['Cov_x1mb']\
            - 2*beta*data['Cov_colormb']\
            - 2*alpha*beta*data['Cov_x1color']
        bins = data['z'].to_list()
        dist_mu = cosmo.distmod(bins).value

        sigmu = np.sqrt(var_mu).to_list()
        sigma_mu = [np.sqrt(sigmu[i]**2+self.sigmaInt**2)
                    for i in range(len(sigmu))]

        mu = [gauss(dist_mu[i], sigma_mu[i]) for i in range(len(dist_mu))]
        data['mu_SN'] = mu
        # data['sigma_mu_SN'] = sigmu
        data['sigma_mu'] = sigmu

        return data

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
        df = bin_it_mean(data, xvar='z', yvar=yvar)
        print(df)
        ax.errorbar(df['z'], df[yvar], yerr=df['{}_std'.format(yvar)],
                    marker='o', color='k', mfc='None', ms=5)

        from astropy.cosmology import w0waCDM
        cosmo = w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa)

        bins = np.arange(df['z'].min(), 1.1, 0.02)
        f = cosmo.distmod(bins).value

        ax.plot(bins, f, 'bs', ms=5)

        df['diff'] = df[yvar]-f
        figb, axb = plt.subplots()

        axb.plot(df['z'], df['diff'], 'ko')

        idx = df['z'] >= 0.5
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
            ax.plot(selbin['z'], selbin['NSN'])

        plt.show()

    def random_sample(self, nsn_field_season, sn_data, survey, seasons):
        """
        Function to extract a random sample of SN

        Parameters
        ----------
        nsn_field_season : pandas df
            reference data with eg NSN.
        sn_data : pandas df
            original df where data are extracted from.
        survey : pandas df
            list of field+zmac+sigmaC+season_min+season_max.

        Returns
        -------
        df_res : pandas df
            Resulting random sample.

        """

        df_res = pd.DataFrame()
        df_orig = pd.DataFrame()

        print('survey', survey)
        print('nsn_field_season', nsn_field_season)

        for season in seasons:
            # grab the fields according to the survey
            idx = survey['season_min'] <= season
            idx &= season <= survey['season_max']
            fields = survey[idx]['field'].unique()

            # loop on fields
            for field in fields:
                # grab the max allowed number of SN per season
                idf = survey['field'] == field
                nsn_max_season = int(survey[idf]['nsn_max_season'].mean())

                # grab the number of sn
                ida = nsn_field_season['field'] == field
                print('io', field, nsn_field_season[ida])
                ida &= nsn_field_season[self.timescale] == season
                nsn_exp = int(nsn_field_season[ida]['nsn'].mean())
                nsn = np.min([nsn_exp, nsn_max_season])

                """
                nsn_z_opti = 0
                if self.lowz_optimize > 0:
                    vv = 'nsn_z_{}'.format(np.round(self.lowz_optimize, 1))
                    nsn_z_opti = int(nsn_field_season[ida][vv].mean())
                """
                # get survey info
                host_effi_key, season_min,\
                    season_max = self.get_info(survey, field)

                # get data
                idb = sn_data['field'] == field
                idb &= sn_data['season'] == season
                sel_sn = sn_data[idb]

                # grab sn sample
                nsn_z_opti = nsn_field_season[ida]
                res = self.sn_sample(
                    sel_sn, nsn_exp, nsn, field,
                    nsn_z_opti, zlow=self.lowz_optimize)

                # select data according to the survey parameters
                # idb = res['z'] <= zmax
                idb = res[self.timescale] >= season_min
                idb = res[self.timescale] <= season_max

                sela = res[idb]

                if self.test_mode:
                    df_orig = pd.concat((df_orig, sela))

                # correct for zhost efficiency

                res_host = self.effi_zhost(
                    sela, host_effi_key)

                # if self.test_mode:
                #    self.plot_sample_zhost(sela, res_host, field)

                df_res = pd.concat((df_res, res_host))

        if self.test_mode:
            ll_ud = ['COSMOS', 'XMM-LSS']
            idxa = df_orig['field'].isin(ll_ud)
            idxb = df_res['field'].isin(ll_ud)
            idxba = df_orig['field'].isin(['WFD'])
            idxbb = df_res['field'].isin(['WFD'])
            idxca = df_orig['field'].isin(['WFD']+ll_ud)
            idxcb = df_res['field'].isin(['WFD']+ll_ud)

            self.plot_sample_zhost(df_orig[idxa], df_res[idxb], 'UD')
            self.plot_sample_zhost(df_orig[~idxa], df_res[~idxb], 'WFD+DD')
            self.plot_sample_zhost(df_orig[idxba], df_res[idxbb], 'WFD')
            self.plot_sample_zhost(df_orig[~idxca], df_res[~idxcb], 'DD')

        return df_res

    def sn_sample(self, data, nsn_exp, nsn, field, nsn_lowz=0, zlow=0.1):
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

        res = pd.DataFrame()
        if nsn >= len(data):
            res = data
        else:
            # grab the random sample
            if field != 'WFD':
                res = data.sample(n=nsn)
            else:
                # sample build out of two: sigma_C<=0.04 and sigma_C>=0.04
                if nsn > nsn_exp:
                    nsn = nsn_exp

                frac = self.get_sigmaC_fraction_in_data(data)
                nsn_exp_low_sigmaC = int(nsn_exp*frac)
                # nsn_exp_high_sigmaC = int(nsn_exp*(1-frac))
                nsn_wanted_low_sigmaC = nsn*self.frac_WFD_low_sigmaC
                # nsn_wanted_high_sigmaC = nsn-nsn_wanted_low_sigmaC

                nsn_frac = np.min([nsn_exp_low_sigmaC, nsn_wanted_low_sigmaC])
                nsn = int(nsn)
                nsn_frac = int(nsn_frac)

                idx = data['sigmaC'] <= self.max_sigmaC
                resa = self.sample_max_lowz(
                    data[idx], nsn_frac, nsn_lowz, self.frac_WFD_low_sigmaC)
                resb = self.sample_max_lowz(
                    data[~idx], nsn-nsn_frac, nsn_lowz, 1.-self.frac_WFD_low_sigmaC)
                res = pd.concat((resa, resb))

        if self.test_mode:
            print(field, len(res))
            if field == 'WFD':
                idx = res['z'] <= 0.1
                idb = data['z'] <= 0.1
                idc = res['sigmaC'] <= self.max_sigmaC
                print('hello', len(data), len(data[idb]), len(
                    res[idx]), len(res[idc]))

        return res

    def get_sigmaC_fraction_in_data(self, data):
        """
        Method to estimate the fraction of SNe Ia with sigmaC < max_sigmaC

        Parameters
        ----------
        data : pandas df
            Data to process.

        Returns
        -------
        float
            SNe fraction with sigmaC<=0.04 in data.

        """

        idx = data['sigmaC'] <= self.max_sigmaC

        frac = len(data[idx])/len(data)

        return np.round(frac, 3)

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
            idx = data['z'] >= zmin
            idx &= data['z'] < zmax

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

    def sample_max_lowz_deprecated(self, data, nsn, nsn_lowz, zlow):
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
        zlow : float
            Max redshift for low-z SNe Ia.

        Returns
        -------
        res : pandas df
            Result.

        """

        idx = data['z'] <= zlow
        if nsn_lowz > 0:
            res = data[idx].sample(nsn_lowz)
        else:
            res = pd.DataFrame()

        nsn_rem = nsn-nsn_lowz
        resb = data[~idx].sample(nsn_rem)
        res = pd.concat((res, resb))

        return res

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
        group = data.groupby(pd.cut(data['z'], bins))
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
            idx = data['z'] >= zmin
            idx &= data['z'] < zmax
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

        """
        figb, axb = plt.subplots()
        figb.suptitle(field)
        dfc = df.merge(dfb, left_on=['z'], right_on=['z'])
        dfc['ratio'] = dfc['NSN_y']/dfc['NSN_x']
        axb.plot(dfc['z'], dfc['ratio'])
        """
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
