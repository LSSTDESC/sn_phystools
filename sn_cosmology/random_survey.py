#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 16:01:07 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_tools.sn_io import checkDir
import numpy as np
import pandas as pd
from sn_cosmology.cosmo_tools import load_footprints, load_host_effi
from sn_cosmology.cosmo_tools import load_data_season, random_LSST
from sn_cosmology.cosmo_tools import clean_survey, analyze_survey
from sn_cosmology.cosmo_tools import get_seasons, dump_survey_season
from sn_cosmology.random_hd import Random_survey
from sn_tools.sn_utils import multiproc


class Gen_Surveys:
    def __init__(self, param,
                 vardf=['z_fit', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',
                        'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
                        'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu',
                        'field', 'healpixID', 'year', 'Cov_t0t0', 'x0_fit',
                        'Cov_x0x0', 'Cov_x0x1', 'Cov_x0color', 'x0', 'x1',
                        'color', 'SNID', 'season_length', 'survey_area']):
        """
        class to generate random SN surveys (with spectroscopic scenarios)

        Parameters
        ----------
        param : dict
            parameters.
        vardf : list(str), optional
            Columns of the output df. 
            The default is ['z_fit', 'x1_fit', 'color_fit', 'mbfit', 
                            'Cov_x1x1','Cov_x1color', 'Cov_colorcolor', 
                            'Cov_mbmb','Cov_x1mb', 'Cov_colormb', 
                            'mu', 'sigma_mu','field', 'healpixID', 'year', 
                            'Cov_t0t0', 'x0_fit','Cov_x0x0', 'Cov_x0x1', 
                            'Cov_x0color', 'x0', 'x1','color', 'SNID', 
                            'season_length', 'survey_area'].

        Returns
        -------
        None.

        """

        # params
        self.param = param

        # info survey
        self.load_info_survey()

        # load host effi and footprint

        self.load_host_foot()

        # simu infos
        self.load_simu_infos()

        # init random_survey

        self.rand_survey = Random_survey(self.survey,
                                         self.footprints, param['timescale'],
                                         param['sigmaInt'], self.host_effi,
                                         H0=param['H0'],
                                         Om0=param['Om0'],
                                         Ode0=param['Ode0'],
                                         w0=param['w0'],
                                         wa=param['wa'],
                                         alpha=param['alpha'],
                                         beta=param['beta'],
                                         low_z_optimize=param['low_z_optimize'],
                                         plot_test=param['plot_test'],
                                         test_mode=param['test_mode'])
        # df variables
        self.vardf = vardf

    def load_info_survey(self):
        """
        Survey infos (seasons, survey, norm factors)

        Returns
        -------
        None.

        """

        self.seasons = get_seasons(self.param['seasons'])

        self.outDir = '{}/{}_{}'.format(self.param['surveyDir'],
                                        self.param['dbName_DD'],
                                        self.param['dbName_WFD'])
        checkDir(self.outDir)

        print('seasons', self.seasons)

        self.survey = pd.read_csv(self.param['surveyFile'], comment='#')

        print('Survey considered', self.survey['survey'].unique())

        # normalisation factors
        self.simu_norm_factor = pd.read_csv(
            self.param['simu_norm_factor'], comment='#')

    def load_host_foot(self):
        """
        Loading host effieiencies and footprints

        Returns
        -------
        None.

        """

        # load host_effi
        self.host_effi = load_host_effi(
            self.param['hosteffiDir'], self.survey['host_effi'].unique())

        # load footprints
        self.footprints = load_footprints(self.param['footprintDir'])

    def load_simu_infos(self):
        """
        Loading simulation infos

        Returns
        -------
        None.

        """

        # Load the data

        dataDir = {}

        dataDir['DDF'] = self.param['dataDir_DD']
        dataDir['WFD'] = self.param['dataDir_WFD']

        dbName = {}
        dbName['DDF'] = self.param['dbName_DD']
        dbName['WFD'] = self.param['dbName_WFD']

        fieldTypes = np.unique(
            self.survey[['zType', 'fieldType']].to_records(index=False))

        # sort to have spectroz first
        fieldTypes = sorted(fieldTypes.tolist())[::-1]

        self.dataDir = dataDir
        self.dbName = dbName
        self.fieldTypes = fieldTypes

    def __call__(self):
        """
        Main function: generate surveys for all the requested seasons

        Returns
        -------
        None.

        """

        for seas in self.seasons:
            # res = self.build_sample(sn_simu_season[seas], seas)
            # get data for this season
            sn_simu_seas = load_data_season(
                self.fieldTypes, self.dataDir, self.dbName, seas,
                select_WFD=self.param['select_WFD'],
                select_DDF=self.param['select_DDF'],
                timescale=self.param['timescale'], vardf=self.vardf)

            """
            for i in range(self.param['n_random_survey']):
                self.survey_season(sn_simu_seas, seas, i+1)
            """
            pp = {}
            pp['sn_simu_seas'] = sn_simu_seas
            pp['seas'] = seas
            nreals = self.param['n_random_survey']
            nproc = self.param['nproc']
            if nreals > 1:
                reals = range(1, nreals+1)
                if nreals < 8:
                    nproc = nreals
                multiproc(reals, pp, self.survey_realisations, nproc)
            else:
                self.survey_realisations([nreals], pp)

    def survey_realisations(self, n_real, pp, j=0, output_q=None):
        """
        MAke survey realisations

        Parameters
        ----------
        n_real : list(int)
            realisation number.
        pp : dict
            parameters.
        j : int, optional
            tag for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            Required in the multiprocessing mode. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        sn_simu_seas = pp['sn_simu_seas']
        seas = pp['seas']

        for i in n_real:
            self.survey_season(sn_simu_seas, seas, i)

        if output_q is not None:
            return output_q.put({j: 0})
        else:
            return 0

    def survey_season(self, sn_simu_seas, seas, nreal):
        """
        Realisation of a seasonal survey

        Parameters
        ----------
        sn_simu_seas : dict
            Simulated SN.
        seas : int
            season of interest.
        nreal : int
            realisation number.

        Returns
        -------
        None.

        """

        # make a realization of this survey
        rand_LSST = random_LSST(
            sn_simu_seas, self.simu_norm_factor, test_mode=self.param['test_mode'])

        full_survey = self.make_survey(rand_LSST)

        if self.param['analyze_survey']:
            analyze_survey(full_survey)

        # make a random survey for the season
        sn_sample, res_foot = self.rand_survey(rand_LSST, seas)

        # concat this
        # sn_sample = pd.concat((sn_sample, res))

        # clean the survey to remove duplicate
        sn_sample = clean_survey(sn_sample)

        if self.param['analyze_survey']:
            analyze_survey(sn_sample)

        # add realization number
        sn_sample['nreal'] = nreal

        # dump the sample
        # year = sn_sample[self.param['timescale']].mean()
        # year_max = sn_sample[pp['timescale']].max()

        dump_survey_season(sn_sample, seas, nreal, self.outDir)
        if self.param['save_full_survey']:
            dump_survey_season(full_survey, seas, nreal,
                               self.outDir,
                               add_str='_nospectroz')

    def make_survey(self, sdict):
        """
        To transform a dict of df to a unique df

        Parameters
        ----------
        sdict : dict
            dict of DataFrames.

        Returns
        -------
        res : pandas DataFrame
            Result.

        """

        res = pd.DataFrame()

        for key, vals in sdict.items():
            res = pd.concat((res, vals))

        return res
