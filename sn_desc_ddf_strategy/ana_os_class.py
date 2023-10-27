#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:58:22 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
import pandas as pd
from . import plt
from .ana_os_tools import doInt, coadd_night, translate, nmax, gime_combi


class Plot_cadence:
    def __init__(self, dbDir, dbName, df_config_scen, outDir, corresp_dd_names):
        """
        Class to make cadence plots

        Parameters
        ----------
        dbDir : str
            data dir.
        dbName : str
            OS to consider.
        df_config_scen : pandas df
            survey config.
        outDir : str
            output dir for plots.
        corresp_dd_names: dict
            to have chosen field names for the display.

        Returns
        -------
        None.

        """
        self.outDir = outDir
        self.corresp_dd_names = corresp_dd_names

        idx = df_config_scen['scen'] == dbName

        dname = df_config_scen[idx][['field', 'fieldType']]
        dname = dname.rename(columns={'field': 'note'})

        data = np.load('{}/{}.npy'.format(dbDir, dbName), allow_pickle=True)

        # get numexposures for a single visit

        visitExposureTime_single = data['visitExposureTime'].mean()

        # coadd this
        from sn_tools.sn_stacker import CoaddStacker
        stacker = CoaddStacker()
        data_stacked = stacker._run(data)

        # estimate seasons
        from sn_tools.sn_obs import season
        data_season = season(data_stacked)

        df = pd.DataFrame.from_records(data_season)

        df = df.merge(dname, left_on=['note'], right_on=['note'])

        dfb = df.groupby(['note', 'fieldType', 'night']).apply(
            lambda x: coadd_night(x)).reset_index()

        dfb = doInt(dfb, ['season'])

        dfc = dfb.groupby(['note', 'fieldType', 'season']).apply(
            lambda x: self.coadd_season(x)).reset_index()

        dfc = doInt(dfc, ['season', 'cadence'])
        print(dfc)

        dfd = dfc.groupby(['fieldType', 'season', 'filter_alloc', 'visits_band',
                           'cadence', 'Tmin', 'Tmax']).apply(lambda x: self.coadd_field(x)).reset_index()

        dfd = doInt(dfd, ['season', 'cadence'])

        print(dfd[['fieldType', 'season', 'filter_alloc', 'visits_band',
                  'cadence', 'note', 'Tmin', 'Tmax']])

        dfe = dfd.groupby(['fieldType', 'note', 'filter_alloc', 'visits_band',
                           'cadence']).apply(lambda x: self.coadd_final(x)).reset_index()
        dfe = doInt(dfe, ['cadence'])

        print(dfe[['fieldType', 'seas_min', 'seas_max', 'filter_alloc', 'visits_band',
                  'cadence', 'note']])

        self.plot_resu(dfb, dfe, dbName.split('.npy')[0])

    def coadd_season(self, grp):
        """
        Method to perform coaddition per season

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        pandas df
            putput data.

        """
        """
        print('allo', np.unique(
            grp[['filter_alloc', 'visits_band']]), len(grp))
        filter_alloc = grp['filter_alloc'].to_list()
        visits_band = grp['visits_band'].to_list()
        """

        filter_alloc = []
        visits_band = []

        falloc_all = grp['filter_alloc'].unique()
        for ff in np.sort(falloc_all):
            filter_alloc.append(ff)
            idx = grp['filter_alloc'] == ff
            visits_band.append(grp[idx]['visits_band'].unique()[0])

        visits_band = map(str, visits_band)
        dictout = {}

        dictout['filter_alloc'] = ['_or_'.join(filter_alloc)]
        dictout['visits_band'] = ['_or_'.join(visits_band)]

        vv = 'observationStartMJD'
        grp = grp.sort_values(by=[vv])

        cadence = np.mean(grp[vv].diff())

        dictout['cadence'] = [cadence]
        Tmin = grp[vv].min()
        Tmax = grp[vv].max()

        dictout['Tmin'] = [np.round(Tmin, 3)]
        dictout['Tmax'] = [np.round(Tmax, 3)]

        return pd.DataFrame.from_dict(dictout)

    def coadd_field(self, grp):
        """
        Method to perform field coadds

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        pandas df
            Coadded data.

        """

        fields = list(grp['note'].unique())
        dictout = {}
        dictout['note'] = [','.join(fields)]

        return pd.DataFrame.from_dict(dictout)

    def coadd_final(self, grp):
        """
        Method to perform the 'final' coaddition

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        pandas df
            Coadded data.

        """

        seas_min = grp['season'].min()
        seas_max = grp['season'].max()
        Tmin = grp['Tmin'].min()
        Tmax = grp['Tmax'].max()

        seas_min -= 1

        dictout = {}
        dictout['seas_min'] = [seas_min]
        dictout['seas_max'] = [seas_max]
        dictout['Tmin'] = [Tmin]
        dictout['Tmax'] = [Tmax]

        return pd.DataFrame.from_dict(dictout)

    def plot_resu(self, df_all, df_coadd, dbName):
        """
        Method to plot the cadence results

        Parameters
        ----------
        df_all : pandas df
            Data to plot.
        df_coadd : pandas df
            Coadded data to plot.
        dbName : str
            OS name.

        Returns
        -------
        None.

        """

        configs = df_coadd['note'].unique()
        vala = 'observationStartMJD'
        valb = 'numExposures'
        valb = 'Nvisits'
        valc = 'MJD_season'
        fig, ax = plt.subplots(nrows=len(configs), figsize=(15, 8))
        dbNameb = '_'.join(dbName.split('_')[:-1])
        fig.suptitle(dbNameb, fontweight='bold')
        fig.subplots_adjust(wspace=0., hspace=0., bottom=0.15)
        configs = np.sort(configs)
        import re
        fsize = 15
        ccol = 'm'
        for i, conf in enumerate(configs):

            confb = re.sub('DD:', '', conf)
            confb = ','.join([self.corresp_dd_names[vv]
                             for vv in conf.split(',')])
            idx = df_coadd['note'] == conf
            sel = df_coadd[idx]

            nn = conf.split(',')[0]

            idxb = df_all['note'] == nn
            sel_all = df_all[idxb]
            sel_all = translate(sel_all)
            # print('hh', sel_all.columns)
            idm = sel_all['moonPhase'] <= 20.
            ax[i].plot(sel_all[idm][valc], sel_all[idm]
                       [valb], 'r.', mfc='None', ms=4)
            idm = sel_all['moonPhase'] > 20.
            ax[i].plot(sel_all[idm][valc], sel_all[idm]
                       [valb], 'b+', mfc='None', ms=4)

            ymin, ymax = ax[i].get_ylim()
            rymax = []
            for ib, row in sel.iterrows():
                seas_min = row['seas_min']
                seas_max = row['seas_max']
                filter_alloc = row['filter_alloc'].split('_or_')
                visits_band = row['visits_band'].split('_or_')
                cadence = row['cadence']
                # print(filter_alloc)
                # print(visits_band)
                # print(cadence)
                ymax = nmax(visits_band)
                rymax.append(ymax)
                ax[i].plot([seas_min]*2, [ymin, ymax],
                           linestyle='dashed', color='k')
                ax[i].plot([seas_max]*2, [ymin, ymax],
                           linestyle='dashed', color='k')
                faloc, nvis, combi1, combi2 = gime_combi(
                    filter_alloc, visits_band)
                seas_mean = 0.5*(seas_min+seas_max)
                yyy = 0.5*(ymax-ymin)+ymin
                k = 0.08
                k = 0.055

                # print('alors', nn, cadence, seas_min, seas_mean)
                if seas_min == 0:
                    k = 0.005

                    ax[i].text(k*seas_mean, 0.8,
                               combi1, color=ccol,
                               fontsize=fsize, transform=ax[i].transAxes)

                    ax[i].text(k*seas_mean, 0.65,
                               combi2, color=ccol,
                               fontsize=fsize, transform=ax[i].transAxes)

                    ax[i].text(k*seas_mean, 0.50,
                               'cad = {}'.format(cadence), color=ccol,
                               fontsize=fsize, transform=ax[i].transAxes)
                else:
                    if seas_mean <= 3:
                        ax[i].text(0.1*(seas_mean-0.5), 0.60,
                                   combi1, color=ccol,
                                   fontsize=fsize, transform=ax[i].transAxes)

                        ax[i].text(0.1*(seas_mean-0.5), 0.45,
                                   combi2, color=ccol,
                                   fontsize=fsize, transform=ax[i].transAxes)

                        ax[i].text(0.1*(seas_mean-0.5), 0.30,
                                   'cad = {}'.format(cadence), color=ccol,
                                   fontsize=fsize, transform=ax[i].transAxes)
                    else:
                        tp = '{}/{} - cad = {}'.format(
                            combi1, combi2, cadence)
                        """
                        ax[i].text(seas_mean/10., 0.60,
                                   combi1+'/'+combi2, color='b',
                                   fontsize=12, transform=ax[i].transAxes)

                        ax[i].text(seas_mean/10., 0.40,
                                   'cadence = {}'.format(cadence), color='b',
                                   fontsize=12, transform=ax[i].transAxes)
                        """
                        yyy = 0.40
                        if dbNameb == 'DDF_SCOC_pII' and nn == 'DD:COSMOS':
                            yyy = 0.60
                        if dbNameb == 'DDF_DESC_0.75_co':
                            yyy = 0.60
                        if dbNameb == 'DDF_DESC_0.70_co':
                            yyy = 0.60
                        ax[i].text(0.8*seas_mean/10., yyy,
                                   tp, color=ccol,
                                   fontsize=fsize, transform=ax[i].transAxes)
            ll = range(1, 11, 1)
            ax[i].set_xticks(list(ll))
            if i == len(configs)-1:
                for it in ll:
                    ax[i].text(0.1*(it-0.5), -0.15, '{}'.format(it),
                               transform=ax[i].transAxes)
                # ax[i].set_xlabel('Season', labelpad=15)
                ax[i].text(0.5, -0.30, 'Season',
                           transform=ax[i].transAxes, ha='center')

            ax[i].set_ylabel('N$_{visits}$')
            ax[i].set_xlim([0, 10])
            ax[i].text(0.97, 0.85, confb, color='dimgrey',
                       fontsize=15, transform=ax[i].transAxes, ha='right')
            ax[i].grid()
            ax[i].tick_params(axis='x', colors='white')

        # fig.tight_layout(pad=0)
        if self.outDir != '':
            outName = '{}/cadence_{}.png'.format(self.outDir, dbName)
            plt.savefig(outName)
            plt.close(fig)


def plot_cadence_deprecated(dbDir, dbName, df_config_scen, outDir):
    """
    Function to make cadence plots

    Parameters
    ----------
    dbDir : str
        data dir.
    dbName : str
        OS to consider.
    df_config_scen : pandas df
        survey config.
    outDir : str
        output dir for plots.

    Returns
    -------
    None.

    """

    idx = df_config_scen['scen'] == dbName

    dname = df_config_scen[idx][['field', 'fieldType']]
    dname = dname.rename(columns={'field': 'note'})

    data = np.load('{}/{}.npy'.format(dbDir, dbName))
    df = pd.DataFrame.from_records(data)

    df = df.merge(dname, left_on=['note'], right_on=['note'])

    dfb = df.groupby(['note', 'fieldType', 'night']).apply(
        lambda x: coadd_night(x)).reset_index()

    dfb = doInt(dfb, ['season'])
    print(dfb)

    dfc = dfb.groupby(['note', 'fieldType', 'season']).apply(
        lambda x: coadd_season(x)).reset_index()

    dfc = doInt(dfc, ['season', 'cadence'])
    print(dfc)

    dfd = dfc.groupby(['fieldType', 'season', 'filter_alloc', 'visits_band',
                       'cadence', 'Tmin', 'Tmax']).apply(lambda x: coadd_field(x)).reset_index()

    dfd = doInt(dfd, ['season', 'cadence'])

    print(dfd[['fieldType', 'season', 'filter_alloc', 'visits_band',
              'cadence', 'note', 'Tmin', 'Tmax']])

    dfe = dfd.groupby(['fieldType', 'note', 'filter_alloc', 'visits_band',
                       'cadence']).apply(lambda x: coadd_final(x)).reset_index()
    dfe = doInt(dfe, ['cadence'])

    print(dfe[['fieldType', 'seas_min', 'seas_max', 'filter_alloc', 'visits_band',
              'cadence', 'note']])

    plot_resu(dfb, dfe, dbName.split('.npy')[0], outDir)
