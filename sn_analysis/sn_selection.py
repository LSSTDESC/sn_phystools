#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:45:41 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import operator
from sn_analysis.sn_tools import load_complete_dbSimu, complete_df
import glob
from sn_tools.sn_io import checkDir
import numpy as np
import pandas as pd


def selection_criteria():

    dict_sel = {}

    dict_sel['nosel'] = [('daymax', operator.ge, 0, 1)]
    # dict_sel['nosel_z0.7'] = [('z', operator.ge, 0.7)]

    sdict = {}
    sdict['phases'] = [('n_epochs_phase_minus_10', operator.ge, 1, 1),
                       # ('n_epochs_bef', operator.ge, 2),
                       ('n_epochs_phase_plus_20', operator.ge, 1, 2)]

    sdict['G10'] = [('n_epochs_m10_p35', operator.ge, 4, 3),
                    ('n_epochs_m10_p5', operator.ge, 1, 4),
                    ('n_epochs_p5_p20', operator.ge, 1, 5),
                    ('n_bands_m8_p10', operator.ge, 2, 6),
                    ('fitstatus', operator.eq, 'fitok', 7)]

    sdict['sigmaC'] = [('sigmaC', operator.le, 0.04, 8)]
    # sdict['z0.7'] = [('z', operator.ge, 0.7)]
    sdict['JLA'] = [('sigmat0', operator.le, 2., 8),
                    ('sigmax1', operator.le, 1, 9),
                    # ('sigmaC', operator.le, 0.04, 10)
                    ]

    dict_sel['G10_sigmaC'] = sdict['phases'] + sdict['G10']+sdict['sigmaC']

    # dict_sel['G10_sigmaC_z0.7'] = dict_sel['G10_sigmaC'] + sdict['z0.7']

    dict_sel['G10_JLA'] = sdict['phases']+sdict['G10']+sdict['JLA']

    dict_sel['G10_JLA_sigmaC'] = sdict['phases'] + \
        sdict['G10']+sdict['JLA']+sdict['sigmaC']

    # dict_sel['G10_JLA_z0.7'] = dict_sel['G10_JLA']+sdict['z0.7']

    sdict['sn_tight'] = [('n_epochs_bef', operator.ge, 5),
                         ('n_epochs_aft', operator.ge, 10),
                         ('n_epochs_phase_minus_10', operator.ge, 5),
                         ('n_epochs_phase_plus_20', operator.ge, 2),
                         ('sigmaC', operator.le, 0.04)]

    dict_sel['no_sel'] = [('z', operator.le, 3.)]

    dict_sel['cosmo_wfd'] = sdict['phases'] + \
        sdict['G10']+sdict['JLA']+sdict['sn_tight']

    return dict_sel


def select(res, list_sel):
    """
    Function to select a pandas df

    Parameters
    ----------
    res : pandas df
        data to select.

    Returns
    -------
    pandas df
        selected df.

    """
    idx = True
    for vals in list_sel:
        idx &= vals[1](res[vals[0]], vals[2])

    return res[idx]


class Select_filt:
    def __init__(self, dataDir, dbName, sellist, seasons,
                 zType='spectroz',  # nsn_factor=1,
                 listFields='COSMOS', fieldType='DDF',
                 outDir='Test', nproc=8,
                 timescale='year',
                 dataType='pandasDataFrame',
                 ebvofMW=100):
        """
        class to load and select SN - output results: one file per season/year

        Parameters
        ----------
        dataDir : str
            Data directory.
        dbName : str
            Dbname to process.
        sellist : dict
            Selection criteria.
        seasons : list(int)
            seasons to process.
        zType : str, optional
            host z-type (spectroz/photz). The default is 'spectroz'.
        listFields : list(str), optional
            List of fields to process. The default is 'COSMOS'.
        fieldType : str, optional
            Type of field (DDF/WFD). The default is 'DDF'.
        outDir : str, optional
            Output dir. The default is 'Test'.
        nproc : int, optional
            Number of proc. The default is 8.
        timescale : str, optional
            Timescale (year/season). The default is 'year'.
        dataType : str, optional
            Data type. The default is 'pandasDataFrame'.
        ebvofMW : float, optional
            E(B-V) selection criteria. The default is 100.

        Returns
        -------
        None.

        """

        self.dataDir = dataDir
        self.dbName = dbName
        self.sellist = sellist
        self.seasons = seasons
        self.zType = zType
        self.listFields = listFields
        self.fieldType = fieldType
        self.outDir = outDir
        self.nproc = nproc
        self.timescale = timescale
        self.dataType = dataType
        self.ebvofMW = ebvofMW

        self.outDir_full = self.init_dir()
        self.clean()

        self.process()

    def init_dir(self):
        """
        Method to create output dir

        Returns
        -------
        outDir_full : TYPE
            DESCRIPTION.

        """

        self.runType = '{}_{}'.format(self.fieldType, self.zType)
        outDir_full = '{}/{}/{}'.format(self.outDir, self.dbName, self.runType)
        checkDir(outDir_full)

        return outDir_full

    def clean(self):
        """
        Method to clean output dir

        Returns
        -------
        None.

        """

        import os

        dirContents = os.listdir(self.outDir_full)
        print('checking the folder', self.outDir_full)
        if not dirContents:
            print('Folder is Empty - processing')
        else:
            print('Folder is Not Empty - cleaning')
            cmd = 'rm {}/*'.format(self.outDir_full)
            os.system(cmd)
        """
        for seas in self.seasons:
            outName = self.get_name(seas)

            if os.path.isfile(outName):
                os.remove(outName)
        """

    def process(self):
        """
        Method to process data

        Returns
        -------
        None.

        """

        if self.fieldType == 'DDF':
            self.process_DDF()

        if self.fieldType == 'WFD':
            self.process_WFD()

    def process_DDF(self):
        """
        Method to process DDFs

        Returns
        -------
        None.

        """

        for seas in self.seasons:

            # print('processing season ', seas)
            # load DDFs
            data = load_complete_dbSimu(
                self.dataDir, self.dbName, self.runType,
                listDDF=self.listFields, seasons=str(seas),
                nproc=self.nproc, dataType=self.dataType)
            # print('loaded', seas, len(data))

            if data.empty:
                continue

            # E(B-V) cut
            idx = data['ebvofMW'] <= self.ebvofMW
            data = data[idx]

            # apply selection on Data
            sel_data = select(data, self.sellist)

            # get year
            sel_data = self.get_year(sel_data)

            # save the data
            self.save_data(sel_data, seas)

            # this is to get stat
            """
                stat, rname = get_stat(
                    sel_data, nsn_factor, timescale=timescale)
                stat[rname] = stat[rname].astype(int)
                stat_tot = pd.concat((stat_tot, stat))
                """

        # this is to get stat
        """
        if timescale == 'year':
            vv = ['nsn']+rname
            stat_tot = stat_tot.groupby(['field', timescale])[
                vv].sum().reset_index()

        stat_tot[timescale] = stat_tot[timescale].astype(int)
        stat_tot['nsn'] = stat_tot['nsn'].astype(int)
        outName_stat = '{}/nsn_{}_{}.hdf5'.format(
            outDir_full, dbName, timescale)
        store = pd.HDFStore(outName_stat, 'w')
        store.put('SN', stat_tot)

        # stat_tot.to_hdf(outName_stat, key='SN')
        """

    def process_WFD(self):
        """
        Method to process WFD

        Returns
        -------
        None.

        """

        deltaRA = 10.

        RAs = np.arange(0., 360.+deltaRA, deltaRA)

        RA_loop = []
        for RA in RAs[:-1]:
            RAmin = np.round(RA, 1)
            RAmax = RAmin+deltaRA
            RAmax = np.round(RAmax, 1)
            RA_loop.append((RAmin, RAmax))

        from sn_tools.sn_utils import multiproc
        params = {}
        multiproc(RA_loop, params, self.load_process, self.nproc)

    def load_process(self, toproc, params, j=0, output_q=None):
        """
        Method to load and process using multiprocessing

        Parameters
        ----------
        toproc : list((float, float))
            List of (RAmin, RAmax) to process.
        params : dict
            Parameters.
        j : int, optional
            internal int for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            Multiprocessing queue where to dump results. The default is None.

        Returns
        -------
        int
            Output data.

        """

        dfb = pd.DataFrame()
        for vv in toproc:
            self.load_process_RAs(vv[0], vv[1])
            # dfb = pd.concat((dfb, dd))
            """
            if len(dd) > 0:
                self.save_data_wfd(dd, vv[0], vv[1])
            """
        """
        if len(dfb) == 0:
            if output_q is not None:
                return output_q.put({j: 0})
            else:
                return 0

        for seas in self.seasons:
            idx = dfb['season'] == seas
            selb = dfb[idx]
            self.save_data(selb, seas)
        """
        if output_q is not None:
            return output_q.put({j: 0})
        else:
            return 0

    def save_data_wfd(self, sel_data, RAmin, RAmax):
        """
        Method to save WFD data

        Parameters
        ----------
        sel_data : pandas df
            Data to save.

        Returns
        -------
        None.

        """
        vals = ['n_epochs_bef', 'n_epochs_aft',
                'n_epochs_phase_minus_10', 'n_epochs_phase_plus_20',
                'n_epochs_m10_p35', 'n_epochs_m10_p5', 'n_epochs_p5_p20',
                'n_bands_m8_p10', 'Nfilt_10', 'Nfilt_15',
                'Nfilt_20', 'ndof', 'remove_sat', 'status']

        years = sel_data[self.timescale].unique()
        for vv in years:
            idx = sel_data[self.timescale] == vv
            selb = pd.DataFrame(sel_data[idx])

            for vvb in vals:
                selb[vvb] = selb[vvb].astype(int)

            selb.to_hdf(self.get_name_wfd(vv, RAmin, RAmax),
                        key='SN', append=True)

            del selb

    def load_process_RAs(self, RAmin, RAmax):
        """
        Method to load and process files


        Parameters
        ----------
        RAmin : float
            Min RA.
        RAmax : float
            Max RA.

        Returns
        -------
        None.

        """

        fullpath = '{}/{}/{}/*{}_{}*.hdf5'.format(self.dataDir, self.dbName,
                                                  self.runType, RAmin, RAmax)

        fis = glob.glob(fullpath)

        if len(fis) == 0:
            return pd.DataFrame()

        # load the data
        # data = pd.DataFrame()
        for fi in fis:
            print(fi)
            data = pd.read_hdf(fi)
            # estimate sigma_mu...
            if len(data) > 0:
                data = complete_df(data, alpha=0.13, beta=3.1)
            data['field'] = 'WFD'
            # data = pd.concat((data, dd))

            # E(B-V) cut
            idx = data['ebvofMW'] <= self.ebvofMW
            data = data[idx]
            # get year
            data = self.get_year(data)

            # apply selection on Data
            sel_data = select(data, self.sellist)

            if len(sel_data) > 0:
                self.save_data_wfd(sel_data, RAmin, RAmax)

        # return sel_data
        """
        for seas in self.seasons:
            idx = sel_data['season'] == seas
            selb = sel_data[idx]
            self.save_data(selb, seas)
        """

    def get_year(self, data):
        """
        Method to estimate the year
        and to add it as a df col

        Parameters
        ----------
        data : pandas df
            Data to process.

        Returns
        -------
        sel_data : pandas df
            original df plus year col.

        """

        sel_data = pd.DataFrame(data)
        sel_data = sel_data[sel_data.columns.drop(
            list(sel_data.filter(regex='mask')))]
        if 'selected' in sel_data.columns:
            sel_data = sel_data.drop(columns=['selected'])

        sel_data['year'] = 1
        if 'mjd_max' in sel_data.columns:
            tt = sel_data['mjd_max']-sel_data['lsst_start']
            # sel_data['year'] = sel_data['daymax']+60*(1.+sel_data['z'])
            # sel_data['year'] -= sel_data['lsst_start']
            tt /= 365.
            sel_data['year'] = np.ceil(tt)
            # print(sel_data['year'], sel_data['lsst_start'])
        sel_data['year'] = sel_data['year'].astype(int)
        # print(sel_data['year'])
        sel_data['chisq'] = sel_data['chisq'].astype(float)
        sel_data['sigmat0'] = np.sqrt(sel_data['Cov_t0t0'])
        sel_data['sigmax1'] = np.sqrt(sel_data['Cov_x1x1'])

        return sel_data

    def save_data(self, sel_data, seas):
        """
        Method to dump data on disk

        Parameters
        ----------
        sel_data : pandas df
            Data to dump.
        seas : int
            season.

        Returns
        -------
        None.

        """
        vals = ['n_epochs_bef', 'n_epochs_aft',
                'n_epochs_phase_minus_10', 'n_epochs_phase_plus_20',
                'n_epochs_m10_p35', 'n_epochs_m10_p5', 'n_epochs_p5_p20',
                'n_bands_m8_p10', 'Nfilt_10', 'Nfilt_15',
                'Nfilt_20', 'ndof', 'remove_sat', 'status']

        # save output data in pandas df
        if self.timescale == 'season':
            # store[seas].put('SN', sel_data)

            """
            outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
                self.outDir_full, self.fieldType, 
                self.dbName, self.timescale, seas)
            """
            sel_data.to_hdf(self.get_name(seas), key='SN')

        else:
            years = sel_data[self.timescale].unique()
            for vv in years:
                if vv == 0:
                    continue
                idx = sel_data[self.timescale] == vv
                selb = pd.DataFrame(sel_data[idx])
                """
                outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
                    self.outDir_full, self.fieldType, self.dbName, self.timescale, vv)
                """
                for pp in vals:
                    selb[pp] = selb[pp].astype(int)

                # selb.info(verbose=True)
                selb.to_hdf(self.get_name(vv), key='SN', append=True)
                del selb
                # store[vv].put('SN', selb)

    def get_name(self, seas):
        """
        Method to get the name of output files

        Parameters
        ----------
        seas : int
            season/year.

        Returns
        -------
        outName : str
            Output name.

        """

        outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
            self.outDir_full, self.fieldType,
            self.dbName, self.timescale, seas)

        return outName

    def get_name_wfd(self, seas, RAmin, RAmax):
        """
        Method to get the name of output files

        Parameters
        ----------
        seas : int
            season/year.

        Returns
        -------
        outName : str
            Output name.

        """

        outName = '{}/SN_{}_{}_{}_{}_{}_{}.hdf5'.format(
            self.outDir_full, self.fieldType,
            self.dbName, RAmin, RAmax, self.timescale, int(seas))

        return outName
