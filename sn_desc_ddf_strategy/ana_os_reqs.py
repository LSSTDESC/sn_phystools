#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:10:41 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
import pandas as pd
from . import plt, filtercolors
from .ana_OS_tools import translate, coadd_night, m5_coadd_grp


class Anaplot_OS:
    def __init__(self, dbDir, config_scen, Nvisits_LSST, budget, outDir='',
                 pz_requirement='input/DESC_cohesive_strategy/pz_requirements.csv',
                 filter_alloc_req='input/DESC_cohesive_strategy/filter_allocation.csv',
                 Nvisits_WL=8000,
                 fields=['DD:COSMOS', 'DD:XMM_LSS', 'DD:ELAISS1',
                         'DD:ECDFS', 'DD:EDFS_a', 'DD:EDFS_b']):
        """
        Class to plot OS parameters+compariosn wrt reqs

        Parameters
        ----------
        dbDir : str
            Data dir.
        config_scen : pandas df
            list of db+color+linestyle.
        Nvisits_LSST : int
            Total number of LSST visits.
        budget : float
            DD budget.
        outDir : str, optional
            output directory for the plot. The default is ''.
        pz_req_file : str, optional
            PZ requirements file. The default is 'input/DESC_cohesive_strategy/pz_requirements.csv'.
        filter_alloc_req : str, optional
            filter alloc for WL. The default is 'input/DESC_cohesive_strategy/filter_allocation.csv'.
        Nvisits_WL : int, optional
            WL required number of visits per season. The default is 8000.

        Returns
        -------
        None.

        """

        self.dbDir = dbDir
        self.config_scen = config_scen
        self.Nvisits_LSST = Nvisits_LSST
        self.budget = budget
        self.outDir = outDir
        self.pz_requirement = pz_requirement
        self.filter_alloc_req = filter_alloc_req
        self.Nvisits_WL = Nvisits_WL
        self.fields = fields

        self.data = self.load_data()

    def load_data(self):
        """
        Method to load the data

        Returns
        -------
        df : pandas df
            Data to plot.

        """

        df = pd.DataFrame()

        for i, row in self.config_scen.iterrows():
            dbName = row['dbName']
            data = np.load('{}/{}.npy'.format(self.dbDir,
                           dbName), allow_pickle=True)
            if 'season' not in data.dtype.names:
                data = self.get_season(data)
            data = pd.DataFrame.from_records(data)
            idx = data['note'].isin(self.fields)
            datac = data[idx]
            if not row['coadded']:
                datac = self.coadd(data[idx])
            # datac = data[idx]
            datac['dbName'] = dbName
            # datac = data[idx]
            df = pd.concat((df, datac))

        return df

    def get_season(self, data):

        if 'season' not in data.dtype.names:
            data_n = None
            from sn_tools.sn_obs import season
            for field in self.fields:
                idx = data['note'] == field
                data_seas = season(data[idx])
                if data_n is None:
                    data_n = data_seas
                else:
                    data_n = np.concatenate((data_n, data_seas))
            return data_n
        else:
            return data

    def coadd(self, data):

        df = pd.DataFrame()
        from sn_tools.sn_stacker import CoaddStacker
        stacker = CoaddStacker()
        for field in self.fields:
            idx = data['note'] == field
            obs = pd.DataFrame(stacker._run(data[idx].to_records(index=False)))
            obs['note'] = field
            df = pd.concat((df, obs))

        return df

    def plot_cadence_mean(self):

        col = 'observationStartMJD'
        cols = ['fiveSigmaDepth', 'fiveSigmaDepth_median', 'Nvisits']
        colsm = ['dbName', 'season', 'note']

        ls = dict(
            zip(self.fields, ['solid', 'dashed', 'dotted', 'solid', 'dashed', 'dotted']))
        marker = dict(zip(self.fields, ['.', '*', 's', '^', 'o', 'h']))

        for i, row in self.config_scen.iterrows():
            dbName = row['dbName']
            idx = self.data['dbName'] == dbName
            data = self.data[idx]

            data['Nvisits'] = data['visitExposureTime']/30.
            data['fiveSigmaDepth_median'] = data['fiveSigmaDepth'] - \
                1.25*np.log10(data['Nvisits'])

            dff = data.groupby(['dbName', 'season', 'note', 'night', 'filter'])[
                cols].median().reset_index()
            dfa = data.groupby(['dbName', 'season', 'note', 'night'])[
                col].median().reset_index()

            df = dfa.groupby(['dbName', 'season', 'note']).apply(
                lambda x: self.obs_cadence(x)).reset_index()

            dfb = data.groupby(['dbName', 'season', 'note']).apply(
                lambda x: self.obs_cadence_band(x)).reset_index()

            df = df.drop(columns=['level_3'])
            dfb = dfb.drop(columns=['level_3'])
            df = df.merge(dfb, left_on=colsm,
                          right_on=colsm, suffixes=('', ''))

            for field in df['note'].unique():

                idx = df['note'] == field
                sel = df[idx]
                self.plot_two(sel, dbName, field)

            tt = data.groupby(['dbName', 'note', 'season', 'filter'])[
                cols].median().reset_index()
            print(tt)
            tt.to_csv('test_{}.csv'.format(dbName))
            for field in dff['note'].unique():
                idx = dff['note'] == field
                sel = dff[idx]
                self.plot_m5(sel, dbName, field)

        print('hello', df[['note', 'season', 'cad_mean', 'cad_rms']])
        print('hello', df)

    def plot_m5(self, data, dbName, field):

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
        bands = 'ugrizy'
        ppos = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

        pos = dict(zip(bands, ppos))

        for b in bands:
            idx = data['filter'] == b
            sela = data[idx]
            seasons = sela['season'].unique()
            for seas in seasons:
                idxb = sela['season'] == seas
                selb = sela[idxb]
                ax[pos[b]].hist(selb['fiveSigmaDepth_median'], histtype='step')

    def plot_two(self, sel, dbName, field):

        bands = 'ugrizy'
        marker = dict(zip(bands, ['o', '*', 's', 'h', '^', 'v']))
        fig, ax = plt.subplots(nrows=2, figsize=(12, 8))
        fig.subplots_adjust(hspace=0.)
        fig.suptitle('{} - {}'.format(dbName, field))
        ax[0].errorbar(sel['season'], sel['cad_mean'],
                       yerr=sel['cad_rms'], color='k', marker='.')

        for b in bands:
            ax[1].errorbar(sel['season'], sel['cad_mean_{}'.format(b)],
                           yerr=sel['cad_rms_{}'.format(b)],
                           color=filtercolors[b], marker=marker[b],
                           mfc='None', label=b)

        ax[0].set_ylabel('cadence [night]')
        ax[1].set_ylabel('cadence [night]')
        ax[1].set_xlabel('season')
        ax[0].grid()
        ax[1].grid()
        ax[1].legend(loc='lower center', bbox_to_anchor=(1.05, 0.5),
                     ncol=1, fontsize=12, frameon=False)

        ax[0].set_xlim([0.95, 10.05])
        ax[1].set_xlim([0.95, 10.05])
        ax[0].set_xticklabels([])

    def obs_cadence(self, grp):

        dict_out = self.cadence(grp)

        """
        # cadence per band
        bands = grp['filter'].unique()
        for b in bands:
            idx = grp['filter'] == b
            sel = grp[idx]
            dd = self.cadence(sel, suffix='_{}'.format(b))
            dict_out.update(dd)
        """
        res = pd.DataFrame.from_dict(dict_out)

        return res

    def obs_cadence_band(self, grp):

        dict_out = {}

        # cadence per band
        bands = grp['filter'].unique()
        for b in bands:
            idx = grp['filter'] == b
            sel = grp[idx]
            dd = self.cadence(sel, suffix='_{}'.format(b))
            dict_out.update(dd)

        res = pd.DataFrame.from_dict(dict_out)

        return res

    def cadence(self, grp, col='observationStartMJD', suffix=''):

        grp = grp.sort_values(by=[col])
        # mean cadence (global)
        diff = grp[col].diff()

        cad_mean = diff.mean()
        cad_rms = diff.std()

        dict_out = {}
        dict_out['cad_mean{}'.format(suffix)] = [cad_mean]
        dict_out['cad_rms{}'.format(suffix)] = [cad_rms]

        return dict_out

    def plot_budget(self):
        """
        Method to plot the budget vs season

        Returns
        -------
        None.

        """

        vala = 'observationStartMJD'
        valb = 'numExposures'
        valc = 'MJD_season'
        norm = self.Nvisits_LSST

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.subplots_adjust(right=0.8)

        bud_max = 0.
        for i, row in self.config_scen.iterrows():
            dbName = row['dbName']
            idx = self.data['dbName'] == dbName
            data = self.data[idx]
            # sum numexposures by night
            dt = data.groupby(['night']).apply(
                lambda x: coadd_night(x)).reset_index()
            print(dt['season'].unique())
            dt = dt.sort_values(by=['night'])
            dt[valb] /= norm
            dt[valb] *= 100.
            # re-estimate seasons
            # dt = dt.drop(columns=['season'])
            # dt = season(dt.to_records(index=False), mjdCol=vala)
            selt = pd.DataFrame.from_records(dt)

            selt = translate(selt)
            selt = selt.sort_values(by=[valc])
            tp = np.cumsum(selt[valb])
            ax.plot(selt[valc], tp, label=dbName,
                    color=row['color'], linestyle=row['ls'])

            bud_max = np.max([bud_max, np.max(tp)])

        ll = range(1, 11, 1)
        ax.set_xticks(list(ll))
        # ax.set_xticks([])
        for it in ll:
            ax.text(0.1*(it-0.5), -0.05, '{}'.format(it),
                    transform=ax.transAxes)
        # ax[i].set_xlabel('Season', labelpad=15)
        ax.text(0.5, -0.10, 'Season',
                transform=ax.transAxes, ha='center')
        # plt.setp(ax.get_xticklabels(),
        #         ha="right", va="top", fontsize=12)
        ax.legend()
        ax.grid()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 1.05*bud_max])
        ax.plot([0, 10], [100.*self.budget]*2,
                linestyle='dashed', color='k', lw=2)
        ax.legend(loc='upper center',
                  bbox_to_anchor=(1.15, 0.7),
                  ncol=1, fontsize=12, frameon=False)

        ax.set_ylabel('DDF Budget [%]')
        plt.xticks(color='w')
        plt.tight_layout()

    def plot_m5_PZ(self):
        """
        Method to plot delta_m5 = m5_data_m5_req(PZ)

        Returns
        -------
        None.

        """

        # loading pz requirements
        pz_req = pd.read_csv(self.pz_requirement, comment='#')

        bands = 'ugrizy'

        restot = pd.DataFrame()
        # dbNames = ['DDF_Univ_SN', 'DDF_Univ_WZ']
        for dbName in self.config_scen['dbName'].to_list():
            idx = self.data['dbName'] == dbName
            data = self.data[idx]

            # for each field: estimate m5 coadd per band

            fields = data['note'].unique()

            res = data.groupby('note').apply(
                lambda x: m5_coadd_grp(x)).reset_index()
            res['name'] = dbName
            res = res.rename(columns={'filter': 'band'})
            res = res.merge(pz_req, left_on=['band'], right_on=[
                'band'], suffixes=('', '_ref'))
            res['diff_m5_y1'] = res['m5_y1']-res['m5_y1_ref']
            res['diff_m5_y2_y10'] = res['m5_y2_y10']-res['m5_y2_y10_ref']
            restot = pd.concat((restot, res))

        vv = ['note', 'm5_y1', 'm5_y2_y10', 'm5_y1_ref', 'm5_y2_y10_ref',
              'diff_m5_y1', 'diff_m5_y2_y10']
        print(restot[vv])

        self.plot_diff_m5_indiv(restot)
        self.plot_diff_m5_indiv(restot, vary='diff_m5_y1',
                                ylabel='$\Delta m_5=m_5^{DD}-m_5^{PZ}$',
                                title='y1')

    def plot_diff_m5(self, data, varx='name', vary='diff_m5_y2_y10'):
        """
        Method to plot diff_m5

        Parameters
        ----------
        data : pandas df
            Data to plot.
        varx : str, optional
            x-axis variable. The default is 'name'.
        vary : str, optional
            y-axis variable. The default is 'diff_m5_y2_y10'.

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(12, 8))
        fig.subplots_adjust(wspace=0., hspace=0.)
        list_fields = ['DD:COSMOS', 'DD:XMM_LSS', 'DD:ELAISS1',
                       'DD:ECDFS', 'DD:EDFS_a', 'DD:EDFS_b']
        list_posfields = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

        ipos = dict(zip(list_fields, list_posfields))

        fields = data['note'].unique()

        for field in fields:

            i = ipos[field][0]
            j = ipos[field][1]

            idx = data['note'] == field

            sela = data[idx]

            bands = sela['band'].unique()

            for b in bands:
                idxa = sela['band'] == b
                selb = sela[idxa]
                color = filtercolors[b]
                ax[i, j].plot(selb[varx], selb[vary],
                              '{}o'.format(color), mfc='None')

            ax[i, j].tick_params(axis='x', labelrotation=20.,
                                 labelsize=10, labelleft=1)
            if i < 2:
                ax[i, j].set_xticks([])
            if j == 1:
                ax[i, j].set_yticks([])

    def plot_diff_m5_indiv(self, data, varx='name', vary='diff_m5_y2_y10',
                           ylabel='$\Delta m_5=m_5^{DD}-m_5^{PZ}$',
                           title='y2 to y10', ybar=0.):
        """
        Method to plot dif_m5 = m5_data-m5_req(PZ)

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        varx : TYPE, optional
            DESCRIPTION. The default is 'name'.
        vary : TYPE, optional
            DESCRIPTION. The default is 'diff_m5_y2_y10'.
        ylabel : TYPE, optional
            DESCRIPTION. The default is '$\Delta m_5=m_5^{DD}-m_5^{PZ}$'.
        title : TYPE, optional
            DESCRIPTION. The default is 'y2 to y10'.
        ybar : TYPE, optional
            DESCRIPTION. The default is 0..

        Returns
        -------
        None.

        """

        fields = data['note'].unique()

        for field in fields:

            fig, ax = plt.subplots(figsize=(12, 8))
            fig.suptitle('{} - {}'.format(field, title), fontweight='bold')
            fig.subplots_adjust(bottom=0.2)
            idx = data['note'] == field

            sela = data[idx]

            bands = sela['band'].unique()

            for b in bands:
                idxa = sela['band'] == b
                selb = sela[idxa]
                color = filtercolors[b]
                ax.plot(selb[varx], selb[vary], marker='o',
                        color=color, mfc='None', markeredgewidth=2, ms=12)

            # ax.tick_params(axis='x', labelrotation=20.,
            #               labelsize=10, labelright=True)
            # ax.set_xticklabels(ax.get_xticklabels(),
            #                   rotation=20, ha="right")

            ax.set_ylabel(ylabel)
            ax.grid()
            xmin, xmax = ax.get_xlim()

            ax.plot([xmin, xmax], [ybar]*2,
                    linestyle='dashed', lw=3, color='k')
            plt.setp(ax.get_xticklabels(), rotation=45,
                     ha="right", va="top", fontsize=12)
            # plt.tight_layout()
            xdep = 0.92
            ydep = 0.6
            shift = 0.05
            for ik, b in enumerate('ugrizy'):
                ax.plot([xdep, xdep+0.02], [ydep-ik*shift]*2, linestyle='solid',
                        color=filtercolors[b], transform=fig.transFigure, clip_on=False)
                ax.text(xdep+0.030, ydep-ik*shift, b, horizontalalignment='center',
                        verticalalignment='center', transform=fig.transFigure)
            ax.set_xlim([xmin, xmax])
            if self.outDir != '':
                outName = '{}/{}_{}.png'.format(self.outDir,
                                                vary, field.split(':')[-1])
                plt.savefig(outName)
                plt.close(fig)

    def plot_Nvisits_WL(self):
        """
        Method to plot the ratio of visits visits_data/visits_req_WL

        Returns
        -------
        None.

        """

        # loading filter_alloc reqs.
        Nv_WL = pd.read_csv(self.filter_alloc_req, comment='#')

        Nv_WL['Nv_WL'] = Nv_WL['frac_band']*self.Nvisits_WL
        Nv_WL['Nv_WL'] = Nv_WL['Nv_WL'].astype(int)
        print('reference', Nv_WL)
        # get the corresponding number of visits

        bands = 'ugrizy'

        restot = pd.DataFrame()
        # dbNames = ['DDF_Univ_SN', 'DDF_Univ_WZ']
        for dbName in self.config_scen['dbName'].to_list():

            idx = self.data['dbName'] == dbName
            data = self.data[idx]

            # get the total number of visits per band and per field
            df = data[['note', 'filter', 'numExposures']]
            print(df)
            sumdf = df.groupby(['note', 'filter']).apply(
                lambda x: pd.DataFrame({'Nv_WL': [x['numExposures'].sum()]})).reset_index()
            sumdf = sumdf.rename(columns={'filter': 'band'})
            sumdf = sumdf.merge(Nv_WL, left_on=['band'], right_on=[
                'band'], suffixes=['', '_ref'])
            sumdf['name'] = dbName
            sumdf['ratio_Nv_WL'] = sumdf['Nv_WL']/sumdf['Nv_WL_ref']
            restot = pd.concat((restot, sumdf))

        self.plot_diff_m5_indiv(restot, varx='name', vary='ratio_Nv_WL',
                                # ylabel='$\frac{N_visits}{N_{visits}^{WL}$',
                                ylabel=r'$\frac{N_{visits}^{DD}}{N_{visits}^{WL}}$',
                                title='WL reqs', ybar=1)
