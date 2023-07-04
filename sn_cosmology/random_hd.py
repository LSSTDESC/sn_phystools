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
                        'mu_SN', 'sigma_mu_SN'],
                 dataNames=['z', 'x1', 'color', 'mb', 'Cov_x1x1',
                            'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
                            'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu',
                            'mu_SN', 'sigma_mu_SN'],
                 fitconfig={},
                 par_protect_fit=['Om0'],
                 prior=pd.DataFrame({'varname': ['Om0'], 'refvalue': [0.3], 'sigma': [0.0073]})):
        """


        Parameters
        ----------
        vardf : TYPE, optional
            DESCRIPTION. The default is ['z', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',                        'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',                        'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu',                        'mu_SN', 'sigma_mu_SN'].
        dataNames : TYPE, optional
            DESCRIPTION. The default is ['z', 'x1', 'color', 'mb', 'Cov_x1x1',                            'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',                            'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu',                            'mu_SN', 'sigma_mu_SN'].
        fitconfig : TYPE, optional
            DESCRIPTION. The default is {}.
        par_protect_fit : TYPE, optional
            DESCRIPTION. The default is ['Om0'].
        prior : TYPE, optional
            DESCRIPTION. The default is pd.DataFrame({'varname': ['Om0'],'refvalue':[0.3], 'sigma':[0.0073]}).

        Returns
        -------
        None.

        """

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
            pd.DataFrame({'varname': ['Om0'],'refvalue':[0.3], 'sigma':[0.0073]}).

        Returns
        -------
        None.

        """

        self.vardf = vardf
        self.dataNames = dataNames
        self.fitconfig = fitconfig
        self.par_protect_fit = par_protect_fit
        self.prior = prior

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
            print(dict_fit)
            # fisher estimation
            # fisher_cov = myfit.covariance_fisher(fitparams)
            # print('Fisher', fisher_cov)
            print('')
            dict_fits[key] = dict_fit

        return dict_fits


class Random_survey:
    def __init__(self, dataDir_DD, dbName_DD,
                 dataDir_WFD, dbName_WFD, sellist, seasons,
                 survey=pd.DataFrame([('COSMOS', 1.1, 1.e8, 1, 10)],
                                     columns=['field', 'zmax', 'sigmaC',
                                              'season_min', 'season_max']),
                 sigmaInt=0.12):
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

        # load data per season
        self.data = self.build_random_sample()

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
            nsn['nsn'] = nsn['nsn'].astype(int)
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
            search_path = '{}/SN_{}_{}_{}.hdf5'.format(
                search_dir, fieldType, dbName, seas)
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

        theName = '{}/nsn_{}.hdf5'.format(theDir, dbName)

        res = pd.read_hdf(theName)

        return res

    def correct_mu(self, data, H0=70, Om0=0.3, Ode0=0.7,
                   w0=-1., wa=0.0):
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

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """

        from random import gauss
        from astropy.cosmology import w0waCDM
        cosmo = w0waCDM(H0=H0, Om0=Om0, Ode0=Ode0, w0=w0, wa=wa)

        sigmu = data['sigma_mu'].to_list()
        bins = data['z'].to_list()
        dist_mu = cosmo.distmod(bins).value
        sigma_mu = [np.sqrt(sigmu[i]**2+self.sigmaInt**2)
                    for i in range(len(sigmu))]
        mu = [gauss(dist_mu[i], sigma_mu[i]) for i in range(len(dist_mu))]
        data['mu_SN'] = mu
        data['sigma_mu_SN'] = sigma_mu

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
                ida &= nsn_field_season['season'] == season
                nsn = int(nsn_field_season[ida]['nsn'].mean())

                nsn = np.min([nsn, nsn_max_season])
                # get data
                idb = sn_data['field'] == field
                idb &= sn_data['season'] == season
                sel_sn = sn_data[idb]

                # grab the random sample
                res = sel_sn.sample(n=nsn)

                # filter here
                idx = survey['field'] == field
                zmax = survey[idx]['zmax'].values[0]
                sigmaC = survey[idx]['sigmaC'].values[0]
                season_min = survey[idx]['season_min'].values[0]
                season_max = survey[idx]['season_max'].values[0]

                idb = res['z'] <= zmax
                idb &= res['sigmaC'] <= sigmaC
                idb &= res['season'] >= season_min
                idb &= res['season'] <= season_max

                df_res = pd.concat((df_res, res[idb]))

        return df_res


def analyze_data(data):
    """
    Function to analyze data

    Parameters
    ----------
    data : pandas df
        data to process.

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
        outdict[field] = nsn
        nsn_tot += nsn

    outdict['all_Fields'] = nsn_tot
    return outdict
