import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from dataclasses import dataclass
from . import plt, filtercolors


@dataclass
class DDF:
    Nf: float  # number of fields
    Ns: float  # number of seasons/field
    Nv: float  # number of visits/season
    """
    cad: float  # cadence of observation
    sl: float  # season length
    zlim: float  # redshift completeness
    """


class FiveSigmaDepth_Nvisits:
    def __init__(self, requirements='input/DESC_cohesive_strategy/pz_requirements.csv',
                 Nvisits_WL_season=800,
                 frac_band=dict(
                     zip('ugrizy', [0.06, 0.09, 0.23, 0.23, 0.19, 0.20])),
                 m5_single=dict(
                     zip('ugrizy', [23.65, 24.38, 23.99, 23.55, 22.92, 22.16])),
                 Ns_y2_y10=9):
        """
        class to estimate Nvisits from m5 and m5 from Nvisits

        Parameters
        ----------
        requirements : str, optional
            csv file of requirements. The default is 'pz_requirements.csv'.
        Nvisits_WL_season: int, optional.
            Number of visits per season required by WL.
        frac_band: dict, opt
            filter allocation. The default is
            dict(zip('ugrizy', [0.06, 0.09, 0.23, 0.23, 0.19, 0.20]))
        m5_single: dict, opt
            m5 single visit. The default is
            dict(zip('ugrizy', [23.65, 24.38, 23.99, 23.55, 22.92, 22.16]
        Ns_y2_y10: int, opt.
            number of season to reach to y2_y10 req. The default is 9.

        Returns
        -------
        None.

        """
        # filter allocation
        self.frac_band = frac_band

        # Nseasons to reach y2_y10 req
        self.Ns_y2_y10 = Ns_y2_y10

        # load requirements from csv file

        self.m5_req = self.load_req(requirements)

        print('requirements', self.m5_req)

        # m5 single
        m5_b = {}
        for key, vals in m5_single.items():
            m5_b[key] = [vals]

        # transform dict in pandas df
        self.msingle = pd.DataFrame(m5_single.keys(), columns=['band'])
        self.msingle['m5_med_single'] = m5_single.values()

        msingle_calc, summary = self.get_Nvisits(
            self.msingle, self.m5_req)

        # get Nvisits requirements from WL
        nvisits_WL = self.filter_allocation(Nvisits_WL_season)

        self.msingle_calc, self.summary = self.merge_reqs(
            msingle_calc, nvisits_WL)

    def load_req(self, requirements):
        """
        Method to load the requirements file

        Parameters
        ----------
        requirements : str
            requirement file name.

        Returns
        -------
        df_pz : pandas df
            array with requirement parameters

        """

        df_pz = pd.read_csv(requirements, comment='#')

        ll = df_pz['m5_y2_y10'].to_list()
        delta_mag = 0.05
        ll = list(map(lambda x: x - delta_mag, ll))
        df_pz['m5_y2_y10_m'] = ll

        ll = list(map(lambda x: x + 2*delta_mag, ll))
        df_pz['m5_y2_y10_p'] = ll

        return df_pz

    def filter_allocation(self, Nvisits=800):
        """
        Method to estimate filter allocation and estimate
       the number of visits per band corresponding to Nvisits

        Parameters
        ----------
        Nvisits : int, optional
            number of reference visits. The default is 800.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """

        r = []
        for b in 'ugrizy':
            b_alloc = self.frac_band[b]
            nv = np.round(b_alloc*Nvisits, 0)
            r.append((b, np.round(100.*b_alloc), nv))
            print(b, np.round(100.*b_alloc, 1), int(nv))

        df = pd.DataFrame(
            r, columns=['band', 'filter_allocation', 'Nvisits_WL_season'])

        return df

    def merge_reqs(self, msingle_calc, nvisits_WL):
        """
        Method to merge reqs in terms of visits

        Parameters
        ----------
        msingle_calc : pandas df
            requirements from PZ.
        nvisits_WL : pandas df
            requirements from WL.

        Returns
        -------
        None.

        """

        var = ['band', 'm5_med_single', 'Nvisits_y1', 'Nvisits_y2_y10_p',
               'Nvisits_y2_y10_m', 'Nvisits_y2_y10', 'nseason_y2_y10',
               'm5_y1', 'm5_y2_y10']

        msingle_PZ = msingle_calc[var]

        msingle_all = msingle_PZ.merge(
            nvisits_WL, left_on=['band'], right_on=['band'])

        msingle_all['Nvisits_WL_y1'] = msingle_all['Nvisits_WL_season']
        msingle_all['Nvisits_WL_y2_y10'] = msingle_all['Nvisits_WL_season'] * \
            msingle_all['nseason_y2_y10']

        for tt in ['y1', 'y2_y10']:
            msingle_all['Nvisits_WL_PZ_{}'.format(tt)] = np.max(
                msingle_all[['Nvisits_{}'.format(tt),
                             'Nvisits_WL_{}'.format(tt)]], axis=1)

        """
        vva = ['Nvisits_WL_y1', 'Nvisits_WL_y2_y10',
               'Nvisits_y1', 'Nvisits_y2_y10',
               'Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10']
        print('merged', msingle_all[vva])
        """

        varb = ['band', 'Nvisits_y1',
                'Nvisits_y2_y10', 'Nvisits_y2_y10_p', 'Nvisits_y2_y10_m',
                'Nvisits_WL_y1', 'Nvisits_WL_y2_y10',
                'Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10']

        # estimate m5 from WL_PZ reqs
        for tt in ['y1', 'y2_y10']:
            msingle_all['m5_WL_PZ_{}'.format(tt)] = \
                msingle_all['m5_med_single']\
                + 1.25*np.log10(msingle_all['Nvisits_WL_PZ_{}'.format(tt)])

        # m5 +- 0.05
        tt = 'y2_y10'
        delta_m5 = 0.05
        dm5 = dict(zip(['p', 'm'], [+1, -1]))

        for key, vals in dm5.items():
            msingle_all['m5_WL_PZ_{}_{}'.format(
                tt, key)] = msingle_all['m5_WL_PZ_{}'.format(tt)]+vals*delta_m5

        # Nvisits corresponding to m5+-0.05
        for key, vals in dm5.items():
            msingle_all['Nvisits_WL_PZ_{}_{}'.format(tt, key)] =\
                10**(0.8*(msingle_all['m5_WL_PZ_{}_{}'.format(
                    tt, key)]-msingle_all['m5_med_single']))

        # restimate m5_y1 and m5_y2_y10

        for tt in ['y1', 'y2_y10']:
            msingle_calc['m5_{}'.format(
                tt)] = msingle_all['m5_WL_PZ_{}'.format(tt)]

        # print('icib', msingle_all[['Nvisits_WL_PZ_y2_y10',
        #      'Nvisits_WL_PZ_y2_y10_p', 'Nvisits_WL_PZ_y2_y10_m']])

        vv = ['Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10',
              'Nvisits_WL_PZ_y2_y10_p', 'Nvisits_WL_PZ_y2_y10_m', 'Nvisits_y1',
              'Nvisits_y2_y10', 'Nvisits_y2_y10_p', 'Nvisits_y2_y10_m']

        summary = msingle_all[vv].sum()

        return msingle_all, summary

    def get_Nvisits(self, msingle, df_pz):
        """
        Method to estimate the number of visits depending on m5

        Parameters
        ----------
        msingle : pandas df
            array with m5 single exp. values.
        df_pz : pandas df
            array with config (target) m5 values

        Returns
        -------
        msingle : pandas df
            array with m5 single exp. values+ target
        summary : pandas df
            array with sum of visits (over field and band)

        """

        msingle = msingle.merge(df_pz, left_on=['band'], right_on=['band'])

        llv = []

        ccols = df_pz.columns.to_list()
        ccols.remove('band')
        ccols = list(map(lambda it: it.split('m5_')[1], ccols))
        nseas = dict(zip(ccols, [1]+[self.Ns_y2_y10]*3))

        for vv in ccols:
            diff = msingle['m5_{}'.format(vv)]-msingle['m5_med_single']
            Nv = 'Nvisits_{}'.format(vv)
            msingle[Nv] = 10**(0.8 * diff)
            msingle['nseason_{}'.format(vv)] = nseas[vv]
            llv.append(Nv)
        if 'field' in msingle.columns:
            summary = msingle.groupby(['field'])[llv].sum().reset_index()
        else:
            summary = msingle[llv].sum()

        return msingle, summary

    def get_Nvisits_from_frac(self, Nvisits,
                              col='Nvisits_WL_PZ_y2_y10'):
        """
        Method to estimate the number of visits per band from a ref

        Parameters
        ----------
        Nvisits : int
            number of visits (total).
        col : str, optional
            ref col to estimate filter allocation.
            The default is 'Nvisits_y2_y10'.

        Returns
        -------
        df : pandas df
            array with the number of visits per band.

        """

        ntot = self.msingle_calc[col].sum()
        r = []

        for b in 'ugrizy':
            idx = self.msingle_calc['band'] == b
            frac = self.msingle_calc[idx][col].values/ntot
            r.append((b, frac[0]*Nvisits))

        df = pd.DataFrame(r, columns=['band', 'Nvisits'])

        return df

    def m5_from_Nvisits(self, Nvisits):
        """


        Parameters
        ----------
        Nvisits : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """

        df = self.get_Nvisits_from_frac(Nvisits, col='Nvisits_WL_PZ_y2_y10')
        df = df.merge(self.msingle, left_on=['band'], right_on=['band'])
        df = df.merge(self.m5_req, left_on=['band'], right_on=['band'])
        # df['m5'] = df['m5_med_single']+1.25*np.log10(df['Nvisits'])
        # df['delta_m5'] = df['m5']-df['m5_y2_y10']

        return df

    def m5_band_from_Nvisits(self, m5_resu_orig, m5single, sl_DD=180., cad_DD=4,
                             frac_moon=0.20, swap_filter_moon='y'):
        """
        Method to estimate m5 per band from the total number of visits

        Parameters
        ----------
        m5_resu_orig : pandas df
            m5 values.
        sl_DD : float, optional
            season length. The default is 180..
        cad_DD : float, optional
            cadence of observation. The default is 4.
        frac_moon : float, optional
            fraction of visits with no Moon. The default is 0.20.
        swap_filter_moon: str, optional.
            filter to swap with u for low moon phase nights. The default is z.

        Returns
        -------
        m5_resu : pandas df
            m5 results.
        tt : pandas df
            m5 results (y1).

        """

        m5_resu = pd.DataFrame(m5_resu_orig)

        # m5_resu['Nvisits'] = m5_resu['Nvisits'].astype(int)

        m5_resu = m5_resu[['name', 'band', 'Nvisits', 'Nseasons']]
        m5_resu = self.visits_night_from_frac(m5_resu, sl=sl_DD,
                                              cad=cad_DD, col='Nvisits',
                                              colb='m5_y2_y10',
                                              seasoncol='Nseasons',
                                              frac_moon=frac_moon,
                                              swap_filter_moon=swap_filter_moon)

        topp = m5_resu[['name', 'band', 'Nvisits',
                        'Nvisits_night']]

        topp.to_csv('DD_res2.csv', index=False)

        # now estimate the number of visits per night for y1

        print('there man', m5single.columns)

        tt = m5single[['band', 'Nvisits_y1', 'm5_y1', 'm5_med_single']]
        tt = m5single[['band', 'Nvisits_WL_PZ_y1',
                       'm5_WL_PZ_y1', 'm5_med_single']]
        tt = pd.DataFrame(tt)
        tt['Nseasons'] = 1

        # tt['Nvisits_y1'] = tt['Nvisits_y1'].astype(int)

        """
        tt['Nvisits_y1_night'] = tt.apply(
            lambda x: x['Nvisits_y1']/(fracs[x['band']]*Nnights_DD_season), axis=1)

        tt['Nvisits_y1'] = tt['Nvisits_y1'].astype(int)
        tt['Nvisits_y1_night'] = tt['Nvisits_y1_night'].astype(int)
        """
        tt = self.visits_night_from_frac(tt, sl=sl_DD,
                                         cad=cad_DD, col='Nvisits_WL_PZ_y1',
                                         colb='m5_WL_PZ_y1',
                                         seasoncol='Nseasons',
                                         frac_moon=frac_moon,
                                         swap_filter_moon=swap_filter_moon)

        return m5_resu, tt

    def visits_night_from_frac(self, tab, sl, cad, col='Nvisits',
                               colb='m5_y2_y10', seasoncol='Nseasons',
                               frac_moon=0.20, swap_filter_moon='z'):
        """
        Method to estimate the number of visits per night according to moon frac

        Parameters
        ----------
        tab : pandas df
            data to process.
        sl : float
            season length.
        cad : float
            cadence of observation.
        col : str, optional
            visit col to process. The default is 'Nvisits'.
        colb : str, optional
            m5 col to process. The default is 'm5_y2_y10'.
        frac_moon : float, optional
            frac of nights with no moon. The default is 0.20.
        swap_filter_moon: str, optional.
            filter to swap with u for low moon phase nights. The default is z.

        Returns
        -------
        tab : pandas df
            array with visits and m5.

        """

        print('there we go', tab, swap_filter_moon)

        nights_season = int(sl/cad)
        bands = 'ugrizy'

        fracs = {}
        for b in bands:
            fracs[b] = 1
            if b == 'u':
                fracs[b] = frac_moon
            if b == swap_filter_moon:
                fracs[b] = 1.-frac_moon
        # frac_night = [frac_moon, 1., 1., 1., 1.-frac_moon, 1.]
        # fracs = dict(zip(bands, frac_night))
        tab = pd.DataFrame(tab)
        tab['{}_night'.format(col)] = tab.apply(
            lambda x: x[col]/x[seasoncol] /
            (fracs[x['band']]*nights_season),
            axis=1)

        # tab['{}_night'.format(col)] = tab['{}_night'.format(col)].astype(int)
        tab['cad'] = cad
        tab['sl'] = sl
        tab['nights_season'] = nights_season

        frac_df = pd.DataFrame(list(bands), columns=['band'])
        frac_night = []
        for b in bands:
            frac_night.append(fracs[b])

        frac_df['frac_night'] = frac_night
        tab.loc[:, 'Nseasons'] = tab[seasoncol]
        tab = tab.merge(frac_df, left_on=['band'], right_on=['band'])

        tab = tab.round({'{}'.format(col): 0, '{}_night'.format(col): 0})
        tab['{}_recalc'.format(col)] = tab['{}_night'.format(
            col)]*tab['nights_season']*tab['frac_night']*tab['Nseasons']

        """
        tab['m5_recalc'] = 1.25 * \
            np.log10(tab['{}_recalc'.format(col)])+tab['m5_med_single']
        tab['diff_m5'] = tab['m5_recalc']-tab[colb]
        """

        return tab


class DB_Infos:
    def __init__(self, dbDir='../DB_Files',
                 dbName='draft_connected_v2.99_10yrs.npy'):
        """
        Class to grab infos (m5 single exp values, filter allocation in the WFD)

        Parameters
        ----------
        dbDir : TYPE, optional
            DESCRIPTION. The default is '../DB_Files'.
        dbName : TYPE, optional
            DESCRIPTION. The default is 'draft_connected_v2.99_10yrs.npy'.

        Returns
        -------
        None.

        """

        # load data - DDF
        self.data = self.load_DDF(dbDir, dbName)

        # m5 single exp - median
        m5_single = pd.DataFrame.from_records(
            self.get_median_m5())

        m5_single_dict = {}
        for i, row in m5_single.iterrows():
            m5_single_dict[row['band']] = row['m5_med_single']

        self.m5_single = m5_single_dict

        # get frac events Moon-on
        nref = len(self.data)
        idx = self.data['moonPhase'] <= 20.
        frac_moon = len(self.data[idx])/len(self.data)

        self.frac_moon = np.round(frac_moon, 2)

        fa = self.filter_allocation(dbDir, dbName)

        fa_dict = {}
        for i, row in fa.iterrows():
            fa_dict[row['band']] = row['filter_allocation']

        self.filter_alloc = fa_dict

    def load_DDF(self, dbDir, dbName, DDList=['COSMOS', 'ECDFS',
                                              'EDFS_a', 'EDFS_b',
                                              'ELAISS1', 'XMM_LSS']):
        """
        Method to load DDFs

        Parameters
        ----------
        dbDir : str
            location dir of the database.
        dbName : str
            db name (OS) to load.
        DDList : list(str), optional
            list of DDFs to consider. The default is ['COSMOS', 'ECDFS',
                                                      'EDFS_a', 'EDFS_b',
                                                      'ELAISS1', 'XMM_LSS'].

        Returns
        -------
        data : array
            DDF observations.

        """

        fullPath = '{}/{}'.format(dbDir, dbName)
        tt = np.load(fullPath)

        data = None
        for field in DDList:
            idx = tt['note'] == 'DD:{}'.format(field)
            if data is None:
                data = tt[idx]
            else:
                data = np.concatenate((data, tt[idx]))

        return data

    def filter_allocation(self, dbDir, dbName):
        """
        Method to estimate filter allocation and estimate
       the number of visits per band corresponding to Nvisits

        Parameters
        ----------
        dbDir : str
            dbDir.
        dbName : str
            db name.
        Nvisits : int, optional
            number of reference visits. The default is 800.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """

        fullPath = '{}/{}'.format(dbDir, dbName)
        obs = np.load(fullPath)
        ntot = len(obs)

        r = []
        for b in 'ugrizy':
            idx = obs['band'] == b
            sel = obs[idx]
            b_alloc = len(sel)/ntot
            r.append((b, b_alloc))

        df = pd.DataFrame(
            r, columns=['band', 'filter_allocation'])

        return df

    def get_median_m5(self):
        """
        Method to get the median m5 per band (all fields)

        Parameters
        ----------
        None

        Returns
        -------
        msingle : array
            median m5 values (per band).

        """

        r = []

        for b in 'ugrizy':
            idxb = self.data['band'] == b
            idxb &= self.data['airmass'] <= 1.5
            selb = self.data[idxb]
            r.append((b, np.median(selb['fiveSigmaDepth'])))

        msingle = np.rec.fromrecords(r, names=['band', 'm5_med_single'])

        return msingle


class FiveSigmaDepth_Nvisits_fromdb_deprecated:
    def __init__(self, dbDir='../DB_Files',
                 dbName='draft_connected_v2.99_10yrs.npy',
                 requirements='input/DESC_cohesive_strategy/pz_requirements.csv',
                 Nvisits_WL_season=800):
        """
        class to estimate Nvisits from m5 and m5 from Nvisits

        Parameters
        ----------
        dbDir : str, optional
            location dir of the db to load. The default is '../DB_Files'.
        dbName : str, optional
            db Name to load. The default is 'draft_connected_v2.99_10yrs.npy'.
        requirements : str, optional
            csv file of requirements. The default is 'pz_requirements.csv'.
        Nvisits_WL_season: int, optional.
            Number of visits per season required by WL.

        Returns
        -------
        None.

        """

        # load data - DDF
        self.data = self.load_DDF(dbDir, dbName)

        # get frac events Moon-on
        nref = len(self.data)
        idx = self.data['moonPhase'] <= 20.
        frac_moon = len(self.data[idx])/len(self.data)

        self.frac_moon = np.round(frac_moon, 2)

        # load requirements from csv file

        self.m5_req = self.load_req(requirements)

        # get m5 single exp. median
        self.msingle = pd.DataFrame.from_records(self.get_median_m5())

        self.m5_single_field = pd.DataFrame.from_records(
            self.get_median_m5_field())

        print('m5 single field', self.m5_single_field)

        msingle_calc, summary = self.get_Nvisits(
            self.msingle, self.m5_req)

        # get Nvisits requirements from WL
        nvisits_WL = self.filter_allocation(
            dbDir, dbName, Nvisits_WL_season)

        print('nvisits from filter_alloc', nvisits_WL)

        self.msingle_calc, self.summary = self.merge_reqs(
            msingle_calc, nvisits_WL)

    def load_req(self, requirements):
        """
        Method to load the requirements file

        Parameters
        ----------
        requirements : str
            requirement file name.

        Returns
        -------
        df_pz : pandas df
            array with requirement parameters

        """

        df_pz = pd.read_csv(requirements, comment='#')

        ll = df_pz['m5_y2_y10'].to_list()
        delta_mag = 0.05
        ll = list(map(lambda x: x - delta_mag, ll))
        df_pz['m5_y2_y10_m'] = ll

        ll = list(map(lambda x: x + 2*delta_mag, ll))
        df_pz['m5_y2_y10_p'] = ll

        return df_pz

    def load_DDF(self, dbDir, dbName, DDList=['COSMOS', 'ECDFS',
                                              'EDFS_a', 'EDFS_b',
                                              'ELAISS1', 'XMM_LSS']):
        """
        Method to load DDFs

        Parameters
        ----------
        dbDir : str
            location dir of the database.
        dbName : str
            db name (OS) to load.
        DDList : list(str), optional
            list of DDFs to consider. The default is ['COSMOS', 'ECDFS',
                                                      'EDFS_a', 'EDFS_b',
                                                      'ELAISS1', 'XMM_LSS'].

        Returns
        -------
        data : array
            DDF observations.

        """

        fullPath = '{}/{}'.format(dbDir, dbName)
        tt = np.load(fullPath)

        data = None
        for field in DDList:
            idx = tt['note'] == 'DD:{}'.format(field)
            if data is None:
                data = tt[idx]
            else:
                data = np.concatenate((data, tt[idx]))

        return data

    def filter_allocation(self, dbDir, dbName, Nvisits=800):
        """
        Method to estimate filter allocation and estimate
       the number of visits per band corresponding to Nvisits

        Parameters
        ----------
        dbDir : str
            dbDir.
        dbName : str
            db name.
        Nvisits : int, optional
            number of reference visits. The default is 800.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """

        fullPath = '{}/{}'.format(dbDir, dbName)
        obs = np.load(fullPath)
        ntot = len(obs)

        r = []
        for b in 'ugrizy':
            idx = obs['band'] == b
            sel = obs[idx]
            b_alloc = len(sel)/ntot
            nv = np.round(b_alloc*Nvisits, 0)
            r.append((b, np.round(100.*b_alloc), nv))
            print(b, np.round(100.*b_alloc, 1), int(nv))

        df = pd.DataFrame(
            r, columns=['band', 'filter_allocation', 'Nvisits_WL_season'])

        return df

    def merge_reqs(self, msingle_calc, nvisits_WL):
        """
        Method to merge reqs in terms of visits

        Parameters
        ----------
        msingle_calc : pandas df
            requirements from PZ.
        nvisits_WL : pandas df
            requirements from WL.

        Returns
        -------
        None.

        """

        var = ['band', 'm5_med_single', 'Nvisits_y1', 'Nvisits_y2_y10_p',
               'Nvisits_y2_y10_m', 'Nvisits_y2_y10', 'nseason_y2_y10',
               'm5_y1', 'm5_y2_y10']

        msingle_PZ = msingle_calc[var]

        msingle_all = msingle_PZ.merge(
            nvisits_WL, left_on=['band'], right_on=['band'])

        msingle_all['Nvisits_WL_y1'] = msingle_all['Nvisits_WL_season']
        msingle_all['Nvisits_WL_y2_y10'] = msingle_all['Nvisits_WL_season'] * \
            msingle_all['nseason_y2_y10']

        for tt in ['y1', 'y2_y10']:
            msingle_all['Nvisits_WL_PZ_{}'.format(tt)] = np.max(
                msingle_all[['Nvisits_{}'.format(tt),
                             'Nvisits_WL_{}'.format(tt)]], axis=1)

        varb = ['band', 'Nvisits_y1',
                'Nvisits_y2_y10', 'Nvisits_y2_y10_p', 'Nvisits_y2_y10_m',
                'Nvisits_WL_y1', 'Nvisits_WL_y2_y10',
                'Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10']
        print('alors2', msingle_all[varb])

        # estimate m5 from WL_PZ reqs
        for tt in ['y1', 'y2_y10']:
            msingle_all['m5_WL_PZ_{}'.format(tt)] = \
                msingle_all['m5_med_single']\
                + 1.25*np.log10(msingle_all['Nvisits_WL_PZ_{}'.format(tt)])

        # m5 +- 0.05
        tt = 'y2_y10'
        delta_m5 = 0.05
        dm5 = dict(zip(['p', 'm'], [+1, -1]))

        for key, vals in dm5.items():
            msingle_all['m5_WL_PZ_{}_{}'.format(
                tt, key)] = msingle_all['m5_WL_PZ_{}'.format(tt)]+vals*delta_m5

        # Nvisits corresponding to m5+-0.05
        for key, vals in dm5.items():
            msingle_all['Nvisits_WL_PZ_{}_{}'.format(tt, key)] =\
                10**(0.8*(msingle_all['m5_WL_PZ_{}_{}'.format(
                    tt, key)]-msingle_all['m5_med_single']))

        # restimate m5_y1 and m5_y2_y10

        for tt in ['y1', 'y2_y10']:
            msingle_calc['m5_{}'.format(
                tt)] = msingle_all['m5_WL_PZ_{}'.format(tt)]

        # print('icib', msingle_all[['Nvisits_WL_PZ_y2_y10',
        #      'Nvisits_WL_PZ_y2_y10_p', 'Nvisits_WL_PZ_y2_y10_m']])

        vv = ['Nvisits_WL_PZ_y1', 'Nvisits_WL_PZ_y2_y10',
              'Nvisits_WL_PZ_y2_y10_p', 'Nvisits_WL_PZ_y2_y10_m', 'Nvisits_y1',
              'Nvisits_y2_y10', 'Nvisits_y2_y10_p', 'Nvisits_y2_y10_m']

        summary = msingle_all[vv].sum()

        return msingle_all, summary

    def get_median_m5_field(self):
        """
        Method to get m5 per band and per DD field

        Parameters
        ----------
        None

        Returns
        -------
        msingle : array
            median m5 per band and per field.

        """

        r = []
        for field in np.unique(self.data['note']):
            idxa = self.data['note'] == field
            sela = self.data[idxa]
            for b in 'ugrizy':
                idxb = sela['band'] == b
                idxb &= sela['airmass'] <= 1.5
                selb = sela[idxb]
                print(b, np.median(selb['fiveSigmaDepth']))
                r.append(
                    (b, np.median(selb['fiveSigmaDepth']),
                     field.split(':')[-1]))

        msingle = np.rec.fromrecords(
            r, names=['band', 'm5_med_single', 'field'])

        return msingle

    def get_median_m5(self):
        """
        Method to get the median m5 per band (all fields)

        Parameters
        ----------
        None

        Returns
        -------
        msingle : array
            median m5 values (per band).

        """

        r = []

        for b in 'ugrizy':
            idxb = self.data['band'] == b
            idxb &= self.data['airmass'] <= 1.5
            selb = self.data[idxb]
            r.append((b, np.median(selb['fiveSigmaDepth'])))

        msingle = np.rec.fromrecords(r, names=['band', 'm5_med_single'])

        return msingle

    def get_Nvisits(self, msingle, df_pz):
        """
        Method to estimate the number of visits depending on m5

        Parameters
        ----------
        msingle : pandas df
            array with m5 single exp. values.
        df_pz : pandas df
            array with config (target) m5 values

        Returns
        -------
        msingle : pandas df
            array with m5 single exp. values+ target
        summary : pandas df
            array with sum of visits (over field and band)

        """

        msingle = msingle.merge(df_pz, left_on=['band'], right_on=['band'])

        llv = []

        ccols = df_pz.columns.to_list()
        ccols.remove('band')
        ccols = list(map(lambda it: it.split('m5_')[1], ccols))
        nseas = dict(zip(ccols, [1, 9, 9, 9]))

        for vv in ccols:
            diff = msingle['m5_{}'.format(vv)]-msingle['m5_med_single']
            Nv = 'Nvisits_{}'.format(vv)
            msingle[Nv] = 10**(0.8 * diff)
            msingle['nseason_{}'.format(vv)] = nseas[vv]
            llv.append(Nv)
        if 'field' in msingle.columns:
            summary = msingle.groupby(['field'])[llv].sum().reset_index()
        else:
            summary = msingle[llv].sum()

        return msingle, summary

    def get_Nvisits_from_frac(self, Nvisits,
                              col='Nvisits_WL_PZ_y2_y10'):
        """
        Method to estimate the number of visits per band from a ref

        Parameters
        ----------
        Nvisits : int
            number of visits (total).
        col : str, optional
            ref col to estimate filter allocation.
            The default is 'Nvisits_y2_y10'.

        Returns
        -------
        df : pandas df
            array with the number of visits per band.

        """

        ntot = self.msingle_calc[col].sum()
        r = []

        for b in 'ugrizy':
            idx = self.msingle_calc['band'] == b
            frac = self.msingle_calc[idx][col].values/ntot
            r.append((b, frac[0]*Nvisits))

        df = pd.DataFrame(r, columns=['band', 'Nvisits'])

        return df

    def m5_from_Nvisits(self, Nvisits):
        """


        Parameters
        ----------
        Nvisits : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """

        df = self.get_Nvisits_from_frac(Nvisits, col='Nvisits_WL_PZ_y2_y10')
        df = df.merge(self.msingle, left_on=['band'], right_on=['band'])
        df = df.merge(self.m5_req, left_on=['band'], right_on=['band'])
        # df['m5'] = df['m5_med_single']+1.25*np.log10(df['Nvisits'])
        # df['delta_m5'] = df['m5']-df['m5_y2_y10']

        return df

    def m5_band_from_Nvisits(self, m5_resu_orig, m5single, sl_DD=180., cad_DD=4,
                             frac_moon=0.20, swap_filter_moon='z'):
        """
        Method to estimate m5 per band from the total number of visits

        Parameters
        ----------
        m5_resu_orig : pandas df
            m5 values.
        sl_DD : float, optional
            season length. The default is 180..
        cad_DD : float, optional
            cadence of observation. The default is 4.
        frac_moon : float, optional
            fraction of visits with no Moon. The default is 0.20.
        swap_filter_moon: str, optional.
            filter to swap with u for low moon phase nights. The default is z.

        Returns
        -------
        m5_resu : pandas df
            m5 results.
        tt : pandas df
            m5 results (y1).

        """

        m5_resu = pd.DataFrame(m5_resu_orig)

        # m5_resu['Nvisits'] = m5_resu['Nvisits'].astype(int)

        m5_resu = m5_resu[['name', 'band', 'Nvisits', 'Nseasons']]
        m5_resu = self.visits_night_from_frac(m5_resu, sl=sl_DD,
                                              cad=cad_DD, col='Nvisits',
                                              colb='m5_y2_y10',
                                              seasoncol='Nseasons',
                                              frac_moon=frac_moon,
                                              swap_filter_moon=swap_filter_moon)

        topp = m5_resu[['name', 'band', 'Nvisits',
                        'Nvisits_night']]

        topp.to_csv('DD_res2.csv', index=False)

        # now estimate the number of visits per night for y1

        print('there man', m5single.columns)

        tt = m5single[['band', 'Nvisits_y1', 'm5_y1', 'm5_med_single']]
        tt = m5single[['band', 'Nvisits_WL_PZ_y1',
                       'm5_WL_PZ_y1', 'm5_med_single']]
        tt['Nseasons'] = 1

        # tt['Nvisits_y1'] = tt['Nvisits_y1'].astype(int)

        """
        tt['Nvisits_y1_night'] = tt.apply(
            lambda x: x['Nvisits_y1']/(fracs[x['band']]*Nnights_DD_season), axis=1)

        tt['Nvisits_y1'] = tt['Nvisits_y1'].astype(int)
        tt['Nvisits_y1_night'] = tt['Nvisits_y1_night'].astype(int)
        """
        tt = self.visits_night_from_frac(tt, sl=sl_DD,
                                         cad=cad_DD, col='Nvisits_WL_PZ_y1',
                                         colb='m5_WL_PZ_y1',
                                         seasoncol='Nseasons',
                                         frac_moon=frac_moon,
                                         swap_filter_moon=swap_filter_moon)

        return m5_resu, tt

    def visits_night_from_frac(self, tab, sl, cad, col='Nvisits',
                               colb='m5_y2_y10', seasoncol='Nseasons',
                               frac_moon=0.20, swap_filter_moon='z'):
        """
        Method to estimate the number of visits per night according to moon frac

        Parameters
        ----------
        tab : pandas df
            data to process.
        sl : float
            season length.
        cad : float
            cadence of observation.
        col : str, optional
            visit col to process. The default is 'Nvisits'.
        colb : str, optional
            m5 col to process. The default is 'm5_y2_y10'.
        frac_moon : float, optional
            frac of nights with no moon. The default is 0.20.
        swap_filter_moon: str, optional.
            filter to swap with u for low moon phase nights. The default is z.

        Returns
        -------
        tab : pandas df
            array with visits and m5.

        """

        print('there we go', tab, swap_filter_moon)

        nights_season = int(sl/cad)
        bands = 'ugrizy'

        fracs = {}
        for b in bands:
            fracs[b] = 1
            if b == 'u':
                fracs[b] = frac_moon
            if b == swap_filter_moon:
                fracs[b] = 1.-frac_moon
        # frac_night = [frac_moon, 1., 1., 1., 1.-frac_moon, 1.]
        # fracs = dict(zip(bands, frac_night))
        tab['{}_night'.format(col)] = tab.apply(
            lambda x: x[col]/x[seasoncol] /
            (fracs[x['band']]*nights_season),
            axis=1)

        # tab['{}_night'.format(col)] = tab['{}_night'.format(col)].astype(int)
        tab['cad'] = cad
        tab['sl'] = sl
        tab['nights_season'] = nights_season

        frac_df = pd.DataFrame(list(bands), columns=['band'])
        frac_night = []
        for b in bands:
            frac_night.append(fracs[b])

        frac_df['frac_night'] = frac_night
        tab['Nseasons'] = tab[seasoncol]
        tab = tab.merge(frac_df, left_on=['band'], right_on=['band'])

        tab = tab.round({'{}'.format(col): 0, '{}_night'.format(col): 0})
        tab['{}_recalc'.format(col)] = tab['{}_night'.format(
            col)]*tab['nights_season']*tab['frac_night']*tab['Nseasons']

        """
        tab['m5_recalc'] = 1.25 * \
            np.log10(tab['{}_recalc'.format(col)])+tab['m5_med_single']
        tab['diff_m5'] = tab['m5_recalc']-tab[colb]
        """

        return tab


class DD_Scenario:
    def __init__(self, Nv_LSST=2100000,  # total number of visits
                 budget_DD=0.07,  # DD budget
                 NDDF=5,  # 5 DDFs
                 Nseason=10,  # number of season of observation
                 sl_UD=180.,  # season length UD fields
                 cad_UD=2.,  # cadence of observation UD fields
                 cad_DD=3.,  # cadence of observation DD fields
                 sl_DD=180.,  # season length DD fields
                 Nf_DD_y1=3,  # number of DDF year 1
                 Nv_DD_y1=998,  # number of DD visits year 1
                 Ns_DD=9,
                 # nvisits vs zcomp
                 nvisits_zcomp_file='input/DESC_cohesive_strategy/Nvisits_zcomp_paper.csv',
                 m5_single_zcomp_file='input/DESC_cohesive_strategy/m5_single_zcomp_paper.csv',
                 m5_single_OS=pd.DataFrame(),
                 Nf_combi=[(1, 3), (2, 2), (2, 3), (2, 4)],  # UD combi
                 zcomp=[0.66, 0.80, 0.75, 0.70],  # correesponding zcomp
                 scen_names=['DDF_SCOC', 'DDF_DESC_0.80',
                             'DDF_DESC_0.75',
                             'DDF_DESC_0.70'],  # scenario name
                 frac_moon=0.20,
                 obs_UD_DD=1,
                 Nv_DD_max=4000):
        """
        class to estimate DD scenarios

        Parameters
        ----------
        Nv_LSST : float, optional
            Total number of LSST visits (10 yrs). The default is 2100000.
        budget_DD : float, optional
            DD budget. The default is 0.07.
        NDDF : int, optional
            Total number of DDFs. The default is 5.
        Nseason : int, optional
            Total number of seasons. The default is 10.
        sl_UD : float, optional
            season length for UD fields. The default is 180..
        cad_UD : float, optional
            cadence of observation for UD fields. The default is 2..
        cad_DD : float, optional
            cadence of observation for Deep Fields. The default is 4..
        sl_DD : float, optional
            season length for Deep Fields. The default is 180..
        Nf_DD_y1 : int, optional
            Number of Deep fields in y1. The default is 3.
        Nv_DD_y1 : int, optional
            Number of DD visits in y1/field. The default is 998.
        Ns_DD: int, optional
            number of season of observations of the DD fields. The default is 9.
        nvisits_zcomp_file : csv file, optional
            Nvisits<-> zcomplete (SNe Ia). The default is '
            input/DESC_cohesive_strategy/Nvisits_zcomp_paper.csv'.
        m5_single_zcomp_file : csv file, optional
            m5 single visit used to estimate Nvisits <-> zcomplete (SNe Ia).
            The default is '
            input/DESC_cohesive_strategy/m5_single_zcomp_paper.csv'.
        m5_single_OS: pandas df, optional
            array of m5 single visite value to correct nvisits_zcomp_file.
        Nf_combi : list(pairs), optional
            UD config (Nfields, Nseasons). The default is
            [(1, 3), (2, 2), (2, 3), (2, 4)].
        zcomp : list(float), optional
            zcomplete for UD fields. The default is [0.66, 0.80, 0.75, 0.70].
        scen_names : list(str), optional
            scenario names. The default is ['DDF_SCOC','DDF_DESC_0.80',
                                         'DDF_DESC_0.75','DDF_DESC_0.70'].
        frac_moon : float, optional
            fraction of nights with no moon. The default is 0.20.
        obs_UD_DD : int, optional
            To observe UD fields as DD. The default is 1.
        Nv_DD_max : int, optional
           Max number of Nvisits per DD field/season. The default is 4000.
        Returns
        -------
        None.

        """

        self.Nv_LSST = Nv_LSST
        self.budget_DD = budget_DD
        self.NDDF = NDDF
        self.Nseason = Nseason
        self.sl_UD = sl_UD
        self.cad_UD = cad_UD
        self.cad_DD = cad_DD
        self.sl_DD = sl_DD
        self.Ns_DD = Ns_DD
        self.Nf_DD_y1 = Nf_DD_y1
        self.Nv_DD_y1 = Nv_DD_y1
        self.Nf_combi = Nf_combi
        self.zcomp = zcomp
        self.scen_names = scen_names
        self.obs_UD_DD = obs_UD_DD
        self.Nv_DD_max = Nv_DD_max

        # load zlim vs nvisits
        dfa = pd.read_csv(nvisits_zcomp_file, comment='#')

        # load single m5 visits used to estimate nvisits_zcomp_file
        m5_single_zcomp = pd.read_csv(m5_single_zcomp_file, comment='#')

        dfb = self.correct_SNR(dfa, m5_single_zcomp, m5_single_OS)
        # dfb = pd.DataFrame(dfa)
        print(dfb)

        # interpolators
        self.zlim_nvisits = interp1d(dfb['nvisits'], dfb['zcomp'],
                                     bounds_error=False, fill_value=0.)
        self.nvisits_zlim = interp1d(dfb['zcomp'], dfb['nvisits'],
                                     bounds_error=False, fill_value=0.)

        # interpolators per band
        self.nvisits_zlim_band = {}

        for b in 'grizy':
            self.nvisits_zlim_band[b] = interp1d(dfb['zcomp'], dfb[b],
                                                 bounds_error=False,
                                                 fill_value=0.)

    def correct_SNR(self, dfa, m5_single_zcomp, m5_single_OS):
        """
        Method to correct for the m5 single variations
        between current m5 single (from a simulation) and m5 single used to
        estimate the number of visits vs zcompleteness.

        Parameters
        ----------
        dfa : pandas df
            data to process.
        m5_single_zcomp : pandas df
            m5 single exp. values to estimate Nvisits <-> zcomplete.
        m5_single_OS : pandas df
            current m5 single exp..

        Returns
        -------
        None.

        """

        """
        print(dfa)
        print(m5_single_zcomp)
        print(m5_single_OS)
        print(test)
        """

        m5_single = m5_single_zcomp.merge(
            m5_single_OS, left_on=['band'], right_on=['band'])
        m5_single['delta_m5'] = m5_single['m5_med_single'] - \
            m5_single['m5_single']

        # print(m5_single)

        dfb = pd.DataFrame(dfa)
        dfb['nvisits'] = 0
        for io, row in m5_single.iterrows():
            b = row['band']
            if b == 'u':
                continue
            k = 10**(-0.8*row['delta_m5'])
            print(b, k)
            if b == 'g' or b == 'r':
                dfb['nvisits'] += dfb['{}'.format(b)]
                continue
            vval = '{}'.format(b)
            dfb[vval] = k*dfb[vval]
            dfb[vval] = dfb[vval].astype(int)
            dfb['nvisits'] += dfb[vval]

        return dfb

    def get_Nv_DD(self, Nf_UD, Ns_UD, Nv_UD, Nf_DD, Ns_DD, Nv_DD, k):
        """
        Function to estimate the number of DD visits per season

        Parameters
        ----------
        Nf_UD : int
            nb UD fields.
        Ns_UD : int
            nb season per UD field.
        Nv_UD : int
            nb visits per season and per UD field.
        Nf_DD : int
            nb DD fields.
        Ns_DD : int
            nb season per DD field.
        Nv_DD : int
            nb visits per season and per DD field.
        k : float
            equal to Nf_UD/Nf_DD.

        Returns
        -------
        Nv_DD : float
            nb visits per DD field and per season.

        """

        # UD = DDF(Nf_UD, Ns_UD, Nv_UD, cad_UD, sl_UD, -1)
        # DD = DDF(NDD-Nf_UD, Ns_DD, -1, cad_DD, sl_DD, -1)
        UD = DDF(Nf_UD, Ns_UD, Nv_UD)
        DD = DDF(self.NDDF-Nf_UD, Ns_DD, Nv_DD)

        Nv_DD = self.budget_DD*self.Nv_LSST
        Nv_DD -= self.Nf_DD_y1*self.Nv_DD_y1
        Nv_DD -= UD.Nf*UD.Ns*UD.Nv
        Nv_DD /= (self.NDDF-UD.Nf)*DD.Ns+k*UD.Nf*UD.Ns

        return Nv_DD

    def get_combis(self):
        """
        Method to get nvisits depend on combination (Nf_UD,Ns_UD)

        Returns
        -------
        restot : array
            the result

        """

        r = []
        dftot = pd.DataFrame()
        for combi in self.Nf_combi:
            Nf_UD = combi[0]
            Ns_UD = combi[1]
            print('hello combi', Nf_UD, Ns_UD)
            """
            Ns_DD = (self.NDDF*self.Nseason-self.Nf_DD_y1-Nf_UD*Ns_UD)
            Ns_DD /= (self.NDDF-Nf_UD)

            for k in np.arange(1., 60., 1.):
                res = self.get_Nv_DD(Nf_UD, Ns_UD, -1,
                                     self.NDDF-Nf_UD, Ns_DD, -1, k)
                print(k, res, k*res, self.cad_DD, self.sl_DD,
                      self.cad_UD, self.sl_UD, Nf_UD, Ns_UD)
                r.append((k, res, k*res, res*self.cad_DD/self.sl_DD,
                          k*res*self.cad_UD/self.sl_UD, Nf_UD, Ns_UD,
                          self.zlim_nvisits(res*self.cad_DD/self.sl_DD)))
            """
            Nv_DD = np.arange(100., self.Nv_DD_max, 100)
            df = pd.DataFrame(Nv_DD, columns=['Nv_DD'])
            Nv_UD_season = self.get_Nv_UD_season(
                self.budget_DD, self.Nv_LSST, self.NDDF,
                Nf_UD, Ns_UD, Nv_DD, self.Ns_DD, self.Nv_DD_y1)
            df['Nv_UD_night'] = Nv_UD_season*self.cad_UD/self.sl_UD
            print(Nv_UD_season)
            df['Nf_UD'] = Nf_UD
            df['Ns_UD'] = Ns_UD
            df['Nv_UD'] = Nv_UD_season
            df['Nv_DD_night'] = df['Nv_DD']*self.cad_DD/self.sl_DD
            df['zcomp'] = self.zlim_nvisits(df['Nv_UD_night'])

            df['Nf_UD'] = df['Nf_UD'].astype(int)
            df['Ns_UD'] = df['Ns_UD'].astype(int)
            dftot = pd.concat((dftot, df))
        """
        restot = np.rec.fromrecords(r, names=[
            'k', 'Nv_DD', 'Nv_UD', 'Nv_DD_night',
            'Nv_UD_night', 'Nf_UD', 'Ns_UD', 'zcomp'])
        """
        # select combis with Nv_UD_night > 0
        idx = dftot['Nv_UD_night'] > 0
        dftot = dftot[idx]

        restot = dftot.to_records(index=False)
        return restot

    def get_Nv_UD_season(self, budget_DD, Nv_LSST, NDDF,
                         Nf_UD, Ns_UD, Nv_DD, Ns_DD, Nv_DD_y1):

        res = budget_DD*Nv_LSST
        res -= (NDDF-Nf_UD)*Nv_DD*Ns_DD
        Ns_UD_DD = Ns_DD-Ns_UD
        if Ns_UD_DD < 0:
            Ns_UD_DD = 0.
        if self.obs_UD_DD == 0:
            Ns_UD_DD = 0.0
        res -= Nf_UD*Nv_DD*Ns_UD_DD
        res -= NDDF*Nv_DD_y1

        res /= (Ns_UD*Nf_UD)

        return res

    def get_scenario(self):
        """
        Method to get the DD scenatios for the UD fields

        Returns
        -------
        scenario : dict
            scenario parameters.

        """

        scenario = {}

        for i in range(len(self.Nf_combi)):
            nv_UD = self.cad_UD*self.nvisits_zlim(self.zcomp[i])
            nv_UD = int(np.round(nv_UD, 0))
            name = self.scen_names[i]
            scenario[self.Nf_combi[i]] = [nv_UD, name, self.zcomp[i]]

        return scenario

    def get_zcomp_req(self):
        """
        Method to grab zcomp reauirements

        Returns
        -------
        zcomp_req : dict
            requirement values.

        """

        zc = '$z_{complete}^{UDF}$'

        zcomp_req = {}
        for z in self.zcomp:
            nv = self.cad_UD*self.nvisits_zlim(z)
            key = '{}={:0.2f}'.format(zc, np.round(z, 2))
            zcomp_req[key] = int(np.round(nv, 2))

        return zcomp_req

    def get_zcomp_req_err(self, zcomp=[0.80, 0.75, 0.70], delta_z=0.01):
        """
        Method to grab zcomp requirements error


        Parameters
        ----------
        zcomp : list(float), optional
            zcomp values. The default is [0.80, 0.75, 0.70].
        delta_z: float, optional
            delta_z value for err req. The default is 0.01

        Returns
        -------
        zcomp_req_err : dict
            req error values.

        """

        zcomp_req_err = {}

        for z in zcomp:
            zmin = z-delta_z
            zmax = z+delta_z
            zcomp_req_err['z_{}'.format(z)] = (
                self.cad_UD*self.nvisits_zlim(zmin),
                self.cad_UD*self.nvisits_zlim(zmax))

        return zcomp_req_err

    def plot(self, restot, varx='Nv_DD_night',
             legx='N$_{visits}^{DD}/obs. night}$',
             vary='Nv_UD_night',
             legy='N$_{visits}^{UD}/obs. night}$', scenario={}, figtitle='',
             zcomp_req={}, pz_wl_req={},
             pz_wl_req_err={}, zcomp_req_err={}, deep_universal={}, scoc_pII={}):
        """
        Method to plot the results

        Parameters
        ----------
        restot : array
            results (with nvisits).
        varx : str, optional
            x-axis variable. The default is 'Nv_DD_night'.
        legx : str, optional
            x-axis label. The default is 'N$_{visits}^{DD}/obs. night}$'.
        vary : str, optional
            y-axis variable. The default is 'Nv_UD_night'.
        legy : str, optional
            y-axis legend. The default is 'N$_{visits}^{UD}/obs. night}$'.
        scenario : dict, optional
            scenarios for DDF survey. The default is {}.
        figtitle : str, optional
            figure title. The default is ''.
        zcomp_req : dict, optional
            zcomp requirements. The default is {}.
        pz_wl_req : dict, optional
            pz_wl requirements. The default is {}.
        pz_wl_req_err : dict, optional
            pz_wl req errors. The default is {}.
        zcomp_req_err : dict, optional
            zcomp requirement errors. The default is {}.
        deep_universal: dict, optional
            deep universal scenario result. The default is {}.

        Returns
        -------
        res : array
            DDF scenario (taking requirements into account).

        """

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.suptitle(figtitle)
        fig.subplots_adjust(right=0.8)
        # ls = dict(zip([1, 2, 3], ['solid', 'dotted', 'dashed']))
        # mark = dict(zip([2, 3, 4,5,6], ['s', 'o', '^']))
        ls = ['solid', 'dotted', 'dashed']
        mark = ['s', 'o', '^']
        vx = -1
        vy = -1
        for_res = []
        for it, (Nf_UD, Ns_UD) in enumerate(np.unique(restot[['Nf_UD', 'Ns_UD']])):
            idx = restot['Nf_UD'] == Nf_UD
            idx &= restot['Ns_UD'] == Ns_UD
            sel = restot[idx]
            # lstyle = ls[Nf_UD]
            # mmark = mark[Ns_UD]
            lstyle = ls[it]
            mmark = mark[it]

            label = '$(N_f^{UDF},N_{s}^{UDF})$'
            lab = '{} = ({},{})'.format(label, Nf_UD, Ns_UD)
            ax.plot(sel[varx], sel[vary], label=lab, marker=mmark,
                    linestyle=lstyle, mfc='None', ms=7, color='k')
            if scenario:
                tag = scenario[(Nf_UD, Ns_UD)]
                nv_UD = tag[0]
                name = tag[1]
                zcomp = tag[2]
                interp = interp1d(sel[vary], sel[varx],
                                  bounds_error=False, fill_value=0.)
                nv_DD = interp(nv_UD)
                ax.plot([nv_DD], [nv_UD], marker='s', ms=20,
                        color='b', mfc='None', markeredgewidth=2)
                nameb = '{}_SN'.format(name)
                for_res.append(
                    (nameb, zcomp, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD),
                     self.cad_UD, self.sl_UD, 0))

                ax.text(nv_DD-500, nv_UD-20, nameb, color='b', fontsize=12)

                if pz_wl_req and Nf_UD >= 2:
                    nv_DD_n = pz_wl_req['WL_PZ_y2_y10'][1]
                    interpb = interp1d(
                        sel[varx], sel[vary], bounds_error=False)
                    nv_UD_n = interpb(nv_DD_n)
                    ax.plot([nv_DD_n], [nv_UD_n], marker='*', ms=20,
                            color='b', mfc='None', markeredgewidth=2)
                    namec = '{}_WZ'.format(name)
                    ax.text(nv_DD_n+50, nv_UD_n, namec,
                            color='b', fontsize=12)
                    for_res.append(
                        (namec, zcomp, Nf_UD, Ns_UD, int(nv_UD_n), int(nv_DD_n),
                         self.cad_UD, self.sl_UD, 1))
                    nv_UD = np.mean([nv_UD, nv_UD_n])
                    nv_DD = np.mean([nv_DD, nv_DD_n])

                # print('scenario', name, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD))
                namea = '{}_co'.format(name)
                for_res.append(
                    (namea, zcomp, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD),
                     self.cad_UD, self.sl_UD, 2))

                ax.plot([nv_DD], [nv_UD], marker='o', ms=15,
                        color='b', mfc='None', markeredgewidth=3.)
                ax.plot([nv_DD], [nv_UD], marker='.', ms=5,
                        color='b', mfc='None', markeredgewidth=3.)
                # ax.text(1.05*nv_DD, 1.05*nv_UD, name, color='b', fontsize=12)
                if vx < 0:
                    dd = 0.75

                    vx = np.abs(nv_DD-dd*nv_DD)
                if vy < 0:
                    vy = np.abs(nv_UD-0.95*nv_UD)
                ax.text(nv_DD-vx, nv_UD-vy, namea, color='b', fontsize=12)
                # print(name, int(nv_DD), int(nv_UD), vx, vy)

        # xmin = np.max([np.min(restot[varx]), 500])
        xmin = np.min(restot[varx])
        xmax = np.max(restot[varx])
        ymin = np.min(restot[vary])
        ymax = np.max(restot[vary])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        if pz_wl_req_err:
            for key, vals in pz_wl_req_err.items():
                xminb = vals[0]
                xmaxb = vals[1]
                # poly = [(xmin,yminb),(xmax,yminb),(xmax,ymaxb),(xmin,ymaxb)]
                ax.fill_between([xminb, xmaxb], ymin, ymax,
                                color='yellow', alpha=0.2)
        if zcomp_req_err:
            for key, vals in zcomp_req_err.items():
                yminb = vals[0]
                ymaxb = vals[1]
                # poly = [(xmin,yminb),(xmax,yminb),(xmax,ymaxb),(xmin,ymaxb)]
                ax.fill_between([xmin, xmax], yminb, ymaxb,
                                color='yellow', alpha=0.2)

        ax.set_xlabel(r'{}'.format(legx))
        ax.set_ylabel(r'{}'.format(legy))

        coltext = 'r'
        if zcomp_req:
            xmin, xmax = ax.get_xlim()
            k = 0
            for key, vals in zcomp_req.items():
                ymin = vals
                ymax = vals
                if k == 0:
                    tt = 1.01*ymin
                    k = tt-ymin
                ax.plot([xmin, xmax], [ymin, ymax], ls='dotted', color=coltext)
                ax.text(0.85*xmax, ymin+k, key, fontsize=12, color=coltext)

        if pz_wl_req:
            ymin, ymax = ax.get_ylim()
            k = 0
            for key, vals in pz_wl_req.items():
                x = vals[0]
                xmin = vals[1]
                xmax = xmin
                if k == 0:
                    tt = 1.01*xmin
                    k = tt-xmin
                ax.plot([xmin, xmax], [ymin, ymax], ls='dotted', color=coltext)
                ax.text(xmin+k, x, key, fontsize=12, rotation=270,
                        color=coltext, va='top')
        if deep_universal:
            for key, vals in deep_universal.items():
                x = vals[0]
                y = vals[1]
                ax.plot([x]*2, [ymin, ymax], color='g', ls='dotted')
                ax.text(1.005*x, y, key, fontsize=12, rotation=270,
                        color='g', va='top')
                Nv_UD_night = x*self.cad_DD/self.sl_DD
                for_res.append(('DDF_Univ_WZ', 0.5, 0, 1,
                                int(Nv_UD_night), int(x), self.cad_DD, self.sl_DD, 3))
                Nv_UD_night = x*self.cad_DD/self.sl_DD
                for_res.append(('DDF_Univ_SN', 0.5, 5, 9,
                                int(Nv_UD_night), int(x), self.cad_DD, self.sl_DD, 3))
        if scoc_pII:
            for key, vals in scoc_pII.items():
                x = vals[0]
                y = vals[1]
                ax.plot(x, y, color='g', marker='*')
                ax.text(1.01*x, y, key, fontsize=12,
                        color='g', va='top')
                for_res.append(('DDF_SCOC_pII', 0.5, 1, 3,
                                int(y), int(x), self.cad_DD, self.sl_DD, 3))

        ax.legend(bbox_to_anchor=(1.3, 0.55),
                  ncol=1, frameon=False, fontsize=13)
        ax.grid()

        res = None
        if for_res:
            res = np.rec.fromrecords(for_res, names=['name', 'zcomp', 'Nf_UD',
                                                     'Ns_UD',
                                                     'nvisits_UD_night',
                                                     'nvisits_DD_season',
                                                     'cad', 'sl', 'scen_type'])
        return res

    def finish(self, res):
        """
        Method to re-estimate the number of visits to avoid rounding pb.

        Parameters
        ----------
        res : pandas df
            data to process.

        Returns
        -------
        df_res : pandas df
            final data.

        """

        df_res = pd.DataFrame.from_records(res)
        df_res['zcomp_new'] = self.zlim_nvisits(
            df_res['nvisits_UD_night']/self.cad_UD)
        df_res['delta_z'] = df_res['zcomp']-df_res['zcomp_new']

        bands = 'grizy'

        for b in bands:
            nv = self.nvisits_zlim_band[b](df_res['zcomp_new'])
            df_res[b] = nv*self.cad_UD
            df_res[b] = df_res[b].astype(int)

        df_res['nvisits_UD_night_recalc'] = df_res[list(bands)].sum(axis=1)

        # print('recalc', df_res[list(bands)].sum(axis=1))
        # if mismatch between Nvisits and Nvisits_recalc-> diff on z-band
        df_res['z'] += df_res['nvisits_UD_night'] - \
            df_res['nvisits_UD_night_recalc']
        df_res['nvisits_UD_night_recalc'] = df_res[list(bands)].sum(axis=1)

        idx = df_res['name'] != 'DDF_SCOC'
        df_res = df_res[idx]
        df_res = df_res.round({'delta_z': 2})

        # print(df_res.columns)
        nights_UD = self.sl_UD/self.cad_UD
        n_UD = df_res['Nf_UD']*df_res['Ns_UD'] * \
            df_res['nvisits_UD_night']*nights_UD
        # print('alors', df_res['Nf_UD'].values, df_res['Ns_UD'].values,
        #      df_res['nvisits_UD_night'].values, nights_UD)
        df_res['nvisits_UD_season'] = df_res['nvisits_UD_night']*nights_UD
        for b in bands:
            df_res['{}_season'.format(b)] = df_res['{}'.format(b)]*nights_UD

        n_DD = self.NDDF*self.Nseason
        n_DD -= self.Nf_DD_y1
        n_DD -= df_res['Nf_UD']*df_res['Ns_UD']
        n_DD *= df_res['nvisits_DD_season']
        # print('hello', n_UD.values, n_DD.values, self.Nf_DD_y1*self.Nv_DD_y1)
        df_res['Nvisits'] = n_UD+n_DD+self.Nf_DD_y1*self.Nv_DD_y1
        df_res['budget'] = df_res['Nvisits']/self.Nv_LSST

        return df_res

    def plot_budget_time_deprecated(self, df):
        """
        Method to plot the budget vs time

        Parameters
        ----------
        sel : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        years = range(0, 11)
        r = []
        for i, row in df.iterrows():
            Nf_UD = row['Nf_UD']
            Ns_UD = row['Ns_UD']
            budget = 0.
            for year in years:
                to = year <= Ns_UD+1
                nn = to*1
                n_UD = Nf_UD*nn*row['nvisits_UD_night']*self.sl_UD/self.cad_UD
                n_DD = (self.NDDF-Nf_UD*nn)*row['nvisits_DD_season']
                if year == 1:
                    n_DD = self.Nf_DD_y1*self.Nv_DD_y1
                    n_UD = 0
                if year > 0:
                    budget_y = (n_UD+n_DD)/self.Nv_LSST
                    budget += budget_y
                    bb = 100*budget
                    r.append((row['name'], year, budget, bb, budget_y))
                    print('helllo', year, n_DD, n_UD)
                else:
                    r.append((row['name'], year, 0.0, 0.0, 0.0))

        res = np.rec.fromrecords(
            r, names=['name', 'year', 'budget', 'budget_per', 'budget_yearly'])

        fig, ax = plt.subplots(figsize=(14, 8))

        names = np.unique(res['name'])
        ls = dict(zip([0, 1, 2], ['solid', 'dotted', 'dashed']))
        colors = dict(zip([0, 1, 2], ['blue', 'k', 'red']))
        for io, nn in enumerate(names):
            idx = res['name'] == nn
            sel = res[idx]
            ax.plot(sel['year'], sel['budget_per'], linestyle=ls[io],
                    color=colors[io], label=nn)

        ax.grid()
        ax.set_xlabel('Year')
        ax.set_ylabel('DDF budget [%]')
        ax.legend(frameon=False)
        ax.set_xlim(0, 10)
        ax.set_ylim(0., 7.)


def check_m5_deprecated(df_res, m5class):

    r = []
    for i, row in df_res.iterrows():
        for b in bands:
            r.append((b, row['{}_season'.format(b)], row['name']))

    df_new = pd.DataFrame(r, columns=['band', 'Nvisits', 'name'])
    print(df_new)

    df_new = df_new.merge(m5class.msingle, left_on='band', right_on='band')

    yrs = 'y2_y10'
    df_new['m5_UD_{}'.format(yrs)] = df_new['m5_med_single'] + \
        1.25*np.log10(df_new['Nvisits'])
    df_new = df_new.merge(m5class.req, left_on='band', right_on='band')
    df_new['diff_m5_{}'.format(yrs)] = df_new['m5_UD_{}'.format(
        yrs)]-df_new['m5_{}'.format(yrs)]+np.log10(9)

    print(df_new[['name', 'band', 'Nvisits', 'm5_med_single',
          'm5_{}'.format(yrs), 'diff_m5_{}'.format(yrs)]])

    df_new[['name', 'band', 'Nvisits', 'diff_m5_{}'.format(yrs)]].to_csv(
        'ddf_res3.csv', index=False)


def get_band_season_deprecated(grp):

    res = {}

    for b in 'ugrizy':
        idx = grp['band'] == b
        selit = grp[idx]
        res[b] = selit['nvisits_band_season'].mean()

    return res


def get_budget_deprecated(grp, NDD, m5_resu, m5_nvisits_y1, Nv_LSST):

    Nf_UD = grp['Nf_UD'].mean()
    Ns_UD = grp['Ns_UD'].mean()
    name = grp.name

    grp['nvisits_band_season'] = grp['n_night_season']*grp['nvisits_night']
    n_UD_season = grp['nvisits_band_season'].sum()
    n_UD_band_season = get_band_season(grp)

    # get m5 results

    idx = m5_resu['name'] == name
    sel_m5 = m5_resu[idx]
    sel_m5['nvisits_band_season'] = sel_m5['Nvisits_night'] * \
        sel_m5['frac_night']*sel_m5['nights_season']

    n_DD_y2_y10 = sel_m5['nvisits_band_season'].sum()
    n_DD_band_y2_y10_season = get_band_season(sel_m5)

    # get second m5 year 1 req
    m5_nvisits_y1['nvisits_band_season'] = m5_nvisits_y1['Nvisits_y1_recalc']
    n_DD_y1 = m5_nvisits_y1['Nvisits_y1_recalc'].sum()
    n_DD_band_y1_season = get_band_season(m5_nvisits_y1)

    print('here', n_UD_season, n_DD_y2_y10, n_DD_y1)
    print('hhb', n_DD_band_y2_y10_season, n_DD_band_y1_season)

    years = range(0, 11)
    budget = 0.
    r = []
    bands = 'ugrizy'
    for year in years:
        to = year <= Ns_UD+1
        nn = to*1
        n_UD = Nf_UD*nn*n_UD_season
        n_DD = n_DD_y2_y10*(5-Nf_UD)
        nvisits_band = dict(zip(bands, [0.]*len(bands)))

        if year == 1:
            n_DD = n_DD_y1
            n_UD = 0
            for b in 'ugrizy':
                nvisits_band[b] = n_DD_band_y1_season[b]
        else:
            for b in 'ugrizy':
                nvisits_band[b] = Nf_UD*nn*n_UD_band_season[b]
                nvisits_band[b] += (NDD-Nf_UD*nn)*n_DD_band_y2_y10_season[b]
        if year > 0:
            budget_y = (n_UD+n_DD)/Nv_LSST
            budget += budget_y
            bb = 100*budget
            add = [nvisits_band[b] for b in bands]
            r.append([year, budget, bb, budget_y, Nf_UD, Ns_UD]+add)
            print('helllo', year, n_DD, n_UD, budget, budget_y, Nf_UD, Ns_UD)
        else:
            r.append([year, 0.0, 0.0, 0.0, 0.0, 0., 0., 0., 0., 0., 0., 0.0])

    res = np.rec.fromrecords(
        r, names=['year', 'budget', 'budget_per', 'budget_yearly',
                  'Nf_UD', 'Ns_UD']+list(bands))

    return pd.DataFrame.from_records(res)


def complete_df(dfa, fieldType='DD', Nfields=5, year=1):
    """
    Function to add infos in dfa

    Parameters
    ----------
    dfa : pandas df
        data to complete.
    fieldType : str, optional
        DDf field type (UD or DD). The default is 'DD'.
    Nfields : int, optional
        number of fields of this type. The default is 5.
    year : float, optional
        year of obs. The default is 1.

    Returns
    -------
    df : pandas df
        final df.

    """

    df = pd.DataFrame(dfa)
    df['fieldType'] = fieldType
    df['Nfields'] = Nfields
    df['year'] = year

    return df


def get_final_scenario(grp, NDD, m5_resu, m5_nvisits_y1):
    """
    Function to get the final scenario

    Parameters
    ----------
    grp : pandas group
        Data to process.
    NDD : int
        total number of ddf.
    m5_resu : pandas df
        link m5 req / nvisits - y2 to y10
    m5_nvisits_y1 : pandas df
        link m5 req / nvisits - y1

    Returns
    -------
    df_res : pandas df
        final scenario.

    """

    Nf_UD = grp['Nf_UD'].mean()
    Ns_UD = grp['Ns_UD'].mean()
    name = grp.name

    grp['nvisits_band_season'] = grp['frac_night'] * \
        grp['n_night_season']*grp['nvisits_night']

    print(grp, grp.columns)
    cols = ['zcomp', 'n_night_season', 'band',
            'nvisits_night', 'nvisits_band_season', 'cad', 'sl', 'frac_night']
    config_UD = grp[cols]

    """
    n_UD_season = grp['nvisits_band_season'].sum()
    n_UD_band_season = get_band_season(grp)
    """
    # get m5 results

    idx = m5_resu['name'] == name
    sel_m5 = m5_resu[idx]
    sel_m5['nvisits_band_season'] = sel_m5['Nvisits_night'] * \
        sel_m5['frac_night']*sel_m5['nights_season']

    print(sel_m5.columns)

    sel_m5 = sel_m5.rename(
        columns={'nights_season': 'n_night_season',
                 'Nvisits_night': 'nvisits_night'})
    print(sel_m5.columns)
    sel_m5['zcomp'] = 0.0
    config_DD_y2_y10 = sel_m5[cols]

    """
    print(config_DD_y2_y10)

    n_DD_y2_y10 = sel_m5['nvisits_band_season'].sum()
    n_DD_band_y2_y10_season = get_band_season(sel_m5)
    """

    # get second m5 year 1 req

    m5_nvisits_y1 = m5_nvisits_y1.rename(
        columns={'nights_season': 'n_night_season',
                 'Nvisits_WL_PZ_y1_night': 'nvisits_night'})
    m5_nvisits_y1['nvisits_band_season'] = m5_nvisits_y1['Nvisits_WL_PZ_y1_recalc']
    m5_nvisits_y1['zcomp'] = 0.
    m5_nvisits_y1['name'] = name

    """
    n_DD_y1 = m5_nvisits_y1['Nvisits_y1_recalc'].sum()
    n_DD_band_y1_season = get_band_season(m5_nvisits_y1)
    """
    config_DD_y1 = m5_nvisits_y1[cols]

    """
    print(config_DD_y1)

    print('here', n_UD_season, n_DD_y2_y10, n_DD_y1)
    print('hhb', n_DD_band_y2_y10_season, n_DD_band_y1_season)
    """

    years = range(1, 11)
    # budget = 0.
    # r = []
    # bands = 'ugrizy'

    df_res = pd.DataFrame()

    for year in years:
        to = year <= Ns_UD+1
        nn = to*1
        # n_UD = Nf_UD*nn*n_UD_season
        # n_DD = n_DD_y2_y10*(5-Nf_UD)
        # nvisits_band = dict(zip(bands, [0.]*len(bands)))

        if year == 1:
            # n_DD = n_DD_y1
            df = complete_df(config_DD_y1, 'DD', NDD-Nf_UD, year)
            df_res = pd.concat((df_res, df))
            df = complete_df(config_DD_y1, 'UD', Nf_UD, year)
            df_res = pd.concat((df_res, df))
        else:

            Nf_UD_seas = Nf_UD*nn
            if Nf_UD_seas > 0:
                df = complete_df(config_UD, 'UD', Nf_UD_seas, year)
                df = pd.DataFrame(config_UD)
                df_res = pd.concat((df_res, df))
            else:
                df = complete_df(config_DD_y2_y10, 'UD', Nf_UD, year)
                df_res = pd.concat((df_res, df))

            df = complete_df(config_DD_y2_y10, 'DD', NDD-Nf_UD, year)
            df_res = pd.concat((df_res, df))

    return df_res


def analyze_scenario_deprecated(df, m5_nvisits):

    resa = df.groupby(['name', 'fieldType', 'Nfields', 'year'])[
        'nvisits_band_season'].sum().reset_index()

    print(resa)


class Delta_m5:
    def __init__(self, df, m5_nvisits):
        """
        class to estimate and plot delta_m5 wrt reqs.

        Parameters
        ----------
        df : pandas df
            Data to process.
        m5_nvisits : pandas df
            m5 vs nvisits vals.

        Returns
        -------
        None.

        """

        idx = df['year'] == 1
        res_y1 = df[idx]
        res_y2_y10 = df[~idx]

        res_y1 = res_y1.groupby(['name', 'fieldType', 'Nfields', 'band'])[
            'nvisits_band_season'].sum().reset_index()

        m5_y1 = m5_nvisits[['band', 'm5_med_single', 'm5_y1']]

        res_y1 = res_y1.merge(m5_y1, left_on=['band'], right_on=['band'])
        res_y1['diff_m5'] = res_y1['m5_med_single']+1.25 * \
            np.log10(res_y1['nvisits_band_season'])-res_y1['m5_y1']

        res_y2_y10 = res_y2_y10.groupby(['name', 'fieldType', 'Nfields', 'band'])[
            'nvisits_band_season'].sum().reset_index()
        # print('hello', res_y2_y10)
        # print(test)
        m5_y2_y10 = m5_nvisits[['band', 'm5_med_single', 'm5_y2_y10']]

        res_y2_y10 = res_y2_y10.merge(
            m5_y2_y10, left_on=['band'], right_on=['band'])

        res_y2_y10['diff_m5'] = res_y2_y10['m5_med_single']+1.25 * \
            np.log10(res_y2_y10['nvisits_band_season'])-res_y2_y10['m5_y2_y10']

        print(res_y1)
        self.plot_deltam5(res_y1, figtitle='WL+PZ requirements y1')
        print(res_y2_y10)

        self.plot_deltam5(res_y2_y10)

    def plot_deltam5(self, df, xvar='name', xlabel='', yvar='diff_m5',
                     ylabel='$\Delta m_5=m_5^{DD}-m_5^{PZ}$',
                     figtitle='WL+PZ requirements y2_y10'):
        """
        Method to plot delta_m5 vs scenario

        Parameters
        ----------
        df : pandas df
            data to plot.
        xvar : str, optional
            x-axis val. The default is 'name'.
        xlabel : str, optional
            x-axis label. The default is ''.
        yvar : str, optional
            y-axis var. The default is 'diff_m5'.
        ylabel : str, optional
            y-axis label. The default is '$\Delta m_5=m_5^{DD}-m_5^{PZ}$'.
        figtitle : str, optional
            fig title. The default is 'WL+PZ requirements y2_y10'.

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(figtitle)
        fig.subplots_adjust(bottom=0.2)
        bands = df['band'].unique()
        ls = dict(zip(['UD', 'DD'], ['solid', 'dashed']))
        marker = dict(zip(['UD', 'DD'], ['o', 's']))

        df['scen_type'] = df['name'].str.split('_').str.get(-1)
        df = df.sort_values(by='name')

        for b in bands:
            idx = df['band'] == b
            sela = df[idx]
            sela = sela.sort_values(by='name')
            fieldTypes = sela['fieldType'].unique()
            for ft in fieldTypes:
                idxa = sela['fieldType'] == ft
                selb = sela[idxa]
                selb = selb.sort_values(by='name')
                """
                ax.plot(selb[xvar], selb[yvar], linestyle=ls[ft],
                        color=filtercolors[b])
                """

                ax.plot(selb[xvar], selb[yvar],  # linestyle='None',
                        color=filtercolors[b], marker=marker[ft],
                        mfc='None', ms=12, markeredgewidth=2)

        ax.text(0.3, 0.9, 'UD Field', horizontalalignment='center',
                verticalalignment='center', transform=fig.transFigure)
        ax.plot([0.23], [0.9], marker='o', linestyle='None', mfc='None', ms=12,
                color='k', transform=fig.transFigure, clip_on=False, markeredgewidth=2)

        ax.text(0.7, 0.9, 'DD Field', horizontalalignment='center',
                verticalalignment='center', transform=fig.transFigure)
        ax.plot([0.63], [0.9], marker='s', linestyle='None', mfc='None', ms=12,
                color='k', transform=fig.transFigure, clip_on=False, markeredgewidth=2)

        xdep = 0.92
        ydep = 0.6
        shift = 0.05
        for ik, b in enumerate('ugrizy'):
            ax.plot([xdep, xdep+0.02], [ydep-ik*shift]*2, linestyle='solid',
                    color=filtercolors[b], transform=fig.transFigure, clip_on=False)
            ax.text(xdep+0.030, ydep-ik*shift, b, horizontalalignment='center',
                    verticalalignment='center', transform=fig.transFigure)

        ax.grid()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis='x', labelrotation=290, labelright=True,
                       labelsize=12)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")


class Budget_time:
    def __init__(self, df, Nv_LSST, budget_DD):
        """
        class to estimate and plot the budget vs time

        Parameters
        ----------
        df : pandas df
            data to process.
        Nv_LSST : int
            total number of LSST visits (10 years).
        budget_DD: float
            DDF allocated budget

        Returns
        -------
        None.

        """

        self.Nv_LSST = Nv_LSST
        print('there aaaaaaaaaaa',
              df[['band', 'nvisits_band_season']])
        budget = df.groupby('name').apply(
            lambda x: self.budget_scenario(x)).reset_index()

        print(budget)

        budget['budget_per'] = 100.*budget['budget']
        budget = budget.sort_values(by=['year'])
        self.plot_budget_time(budget, budget_DD)

    def budget_scenario(self, grp):
        """
        Method to estimate the budget vs time per scenario

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        nv : pandas df
            budget vs time.

        """

        grp['nvisits_field'] = grp['nvisits_band_season']*grp['Nfields']

        print('there mmmmm',
              grp[['band', 'nvisits_field', 'nvisits_band_season']])
        nv = grp.groupby(
            ['year'])['nvisits_field'].sum().reset_index()

        nv['budget'] = nv['nvisits_field'].cumsum()/self.Nv_LSST

        # add year 0 with 0 budget
        y0 = pd.DataFrame([(0, 0.0)], columns=['year', 'budget'])
        nv = pd.concat((nv, y0))

        return nv

    def plot_budget_time(self, res, budget_DD):
        """
        Method to plot the budget vs time

        Parameters
        ----------
        res: pandas df
            Data to plot.
        budget_DD: float
            Reference DD budget

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.subplots_adjust(right=0.85)
        names = np.unique(res['name'])
        keys = ['0.70', '0.75', '0.80', 'pII', 'Univ_WZ', 'Univ_SN']
        lss = ['solid', 'dotted', 'dashed', 'solid', 'solid', 'solid']
        ls = dict(zip(keys, lss))
        colors = dict(
            zip(range(12), ['blue', 'k', 'red']*3+['m', 'green', 'orange']))
        res = res.sort_values(by=['name'])
        for io, nn in enumerate(names):
            lst = 'None'
            for key, vals in ls.items():
                if key in nn:
                    lst = vals
            idx = res['name'] == nn
            sel = res[idx]
            sel = sel.sort_values(by=['year'])
            # print('hhh', nn, lst, colors[io])
            ax.plot(sel['year'], sel['budget_per'], linestyle=lst,
                    color=colors[io], label=nn)

        ax.grid()
        ax.set_xlabel('Season', fontweight='bold')
        ax.set_ylabel('DDF budget [%]', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 0.7),
                  ncol=1, fontsize=12, frameon=False)
        ax.set_xlim(0, 10)
        budget_max = np.max(sel['budget_per'])
        ax.set_ylim(0., 1.01*budget_max)

        ax.plot([0, 10], [100.*budget_DD]*2, linestyle='dashed', color='k')


class Scenario_time:
    def __init__(self, df, swap_filter_moon='z'):
        """
        class to plot scenario (ie nvisits/band/obs. night per year)

        Parameters
        ----------
        df : pandas df
            data to plot.
        swap_filter_moon: str, opt.
           filter to swap wih u - low moon phases. The default is z.

        Returns
        -------
        None.

        """

        self.swap_filter_moon = swap_filter_moon

        res = df.groupby(['name']).apply(
            lambda x: self.scenario_fields(x)).reset_index()

        res['scen_type'] = res['name'].str.split('_').str.get(-1)

        # plot it please
        for scen_type in res['scen_type'].unique():
            idx = res['scen_type'] == scen_type
            sel = res[idx]
            self.plot_scenario(sel)
            self.plot_scenario(sel, visitName='visits_night')

    def scenario_fields(self, df):
        """
        method to estimate scenario per fieldtype (UD or DD)

        Parameters
        ----------
        df : pandas df
            Data to process.

        Returns
        -------
        rr : pandas df
            scenario per fieldtype

        """

        rr = df.groupby(['fieldType', 'Nfields']).apply(
            lambda x: self.scenario_years(x))

        return rr

    def scenario_years(self, grp):
        """
        Method to estimlate scenario per year

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        rr : pandas df
            scenario per year

        """

        rr = grp.groupby('year').apply(
            lambda x: self.visits_str(x))

        return rr

    def visits_str(self, grp):
        """
        Method to estimate nvisits/band/obs night as a str

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        resfi: pandas df
            filter allocation and total number of visits per obs. night.

        """

        ff = dict(zip(grp['band'].to_list(), grp['nvisits_night'].to_list()))
        # Nfields = int(grp['Nfields'].mean())

        bands = 'ugrizy'

        import copy
        ff_moon = copy.deepcopy(ff)
        ff_moon[self.swap_filter_moon] = 0
        nv_moon = '/'.join(['{}'.format(int(ff_moon[key])) for key in bands])
        nv_moon_night = np.sum([int(ff_moon[key]) for key in bands])

        ff_nomoon = copy.deepcopy(ff)
        ff_nomoon['u'] = 0
        nv_nomoon = '/'.join(['{}'.format(int(ff_nomoon[key]))
                              for key in bands])
        nv_nomoon_night = np.sum([int(ff_nomoon[key]) for key in bands])

        # res = '{}||{}'.format(nv_moon, nv_nomoon)
        resb = '{}/{}'.format(nv_moon_night, nv_nomoon_night)

        nv = '/'.join(['{}'.format(int(ff[key])) for key in bands])
        # nv_night = np.sum([int(ff[key]) for key in bands])

        res = '{}'.format(nv)
        # resb = '{}'.format(nv_night)

        resfi = pd.DataFrame({'visits': [res], 'visits_night': [resb]})

        return resfi

    def plot_scenario(self, df, visitName='visits'):
        """
        method to plot scenario

        Parameters
        ----------
        df : pandas df
            data to plot.

        Returns
        -------
        None.

        """

        names = df['name'].unique()

        ls = dict(zip(['UD', 'DD'], ['solid', 'dashed']))
        corresp = dict(zip(['UD', 'DD'], ['Ultra-Deep Field', 'Deep Field']))

        fig, axa = plt.subplots(nrows=3, figsize=(12, 10))
        fig.subplots_adjust(hspace=0)
        for io, name in enumerate(names):
            ax = axa[io]
            idx = df['name'] == name
            sel = df[idx]

            dict_pos = {}
            fieldTypes = sorted(sel['fieldType'].unique())[::-1]
            for ftype in fieldTypes:
                idxa = sel['fieldType'] == ftype
                sela = sel[idxa]

                ya = sela['Nfields'].mean()
                # complete line up to end of y10
                ax.plot([10, 11], [ya]*2, linestyle=ls[ftype], color='k', lw=3)
                ax.plot(sela['year'], sela['Nfields'],
                        linestyle=ls[ftype], color='k', lw=3)

                tt = sela.groupby(visitName).apply(lambda x:
                                                   pd.DataFrame({'yearmin':
                                                                 [x['year'].min()],
                                                                 'yearmax':
                                                                 [x['year'].max()]})).reset_index()

                y = [ya-0.1, ya+0.1]
                k = 1
                shift = 0
                if ftype == 'DD':
                    # k = -1
                    shift = 1

                for iu, row in tt.iterrows():
                    ccol = 'g'
                    yrmin = row['yearmin']
                    yrmax = row['yearmax']
                    xa = [yrmin]*2
                    xb = [yrmax+1]*2
                    xmean = np.mean(xa+xb)
                    xmeanarr = xmean

                    if xmean >= 2 and xmean <= 4:
                        ccol = 'r'

                    if xmean < 2:
                        ccol = 'b'

                    if yrmax == yrmin:
                        xmean *= 0.7
                    if xmean == 3:
                        xmean += 0.2

                    """
                    if yrmax == yrmin:
                        xmean = yrmin
                    """
                    if io != 0:
                        ax.plot(xa, y, linestyle=ls[ftype], color='k', lw=3)
                    else:
                        if iu == 1:
                            ax.plot(xa, y, linestyle=ls[ftype],
                                    color='k', lw=3, label=corresp[ftype])

                    ax.plot(xb, y, linestyle=ls[ftype], color='k', lw=3)

                    """
                    ax.arrow(yrmin, ya, yrmax+1-yrmin, 0.0, width=0.05,
                             length_includes_head=True, fc='yellow', head_length=0.1, ec='k', alpha=0.20)

                    ax.arrow(yrmin+1, ya, -1, 0.0, width=0.05,
                             length_includes_head=True, fc='yellow', head_length=0.1, ec='k', alpha=0.20)
                    """

                    """
                    ax.arrow(yrmin, ya, yrmax+1-yrmin, 0.0, width=0.03,
                             length_includes_head=True, fc='k', head_length=0.2, ec='k')

                    ax.arrow(yrmin+1, ya, -1, 0.0, width=0.03,
                             length_includes_head=True, fc='k', head_length=0.2, ec='k', ls=ls[ftype])
                    """
                    """
                    arrow = mpatches.FancyArrowPatch(
                        (yrmin, ya), (yrmax+1, ya), mutation_scale=100,
                        arrowstyle='<->,head_length=0.4,head_width=0.4,tail_width=0.4', lw=8, color='yellow')
                    """
                    # ax.add_patch(arrow)

                    txt = row[visitName]
                    props = dict(boxstyle='round',
                                 facecolor='wheat', alpha=0.5)
                    # ax.text(xmean, ya+0.5, txt, bbox=props)
                    if txt not in dict_pos.keys():
                        dict_pos[txt] = [xmean, xmeanarr, ya]

                    xmean = dict_pos[txt][0]
                    xmeanarr = dict_pos[txt][1]
                    ya = dict_pos[txt][2]

                    arrowprops = dict(arrowstyle="->")
                    arrowprops = dict(
                        facecolor=ccol, arrowstyle="simple", edgecolor=ccol)
                    ax.annotate(txt, xy=(xmeanarr, ya+shift),
                                xytext=(xmean, ya+0.5),
                                arrowprops=arrowprops, color=ccol)
                    """
                    ax.annotate(txt, xy=(xmean, ya+1), xytext=(xmean, ya+1.5),
                                    arrowprops=dict(arrowstyle="->"))
                    """

            # break
            axb = ax.twinx()
            axb.set_ylabel(name, rotation=270., fontweight='bold',
                           color='orange', labelpad=35, fontsize=15)
            axb.set_yticks([])
            ax.grid()
            if io == 2:
                ax.set_xlabel('year', fontweight='bold')
            if io == 1:
                ax.set_ylabel('$N^{DDF}$')
            ax.set_xlim([1, 11])
            if io == 0:
                ax.legend(ncol=2, bbox_to_anchor=(0.1, 0.95), frameon=False)


def reshuffle(df_res, m5_resu, sl_UD, cad_UD, frac_moon, swap_filter_moon):
    """
    Functio to write the input df in a more usable way

    Parameters
    ----------
    df_res : pandas df
        data to process
    frac_moon : float
        frac of nights with moon (moon phase <= xx%)

    Returns
    -------
    df_resb : pandas df
        reshuffled df

    """
    r = []
    bbval = ['name', 'zcomp', 'Nf_UD', 'Ns_UD', 'nvisits_UD_night',
             'nvisits_DD_season', 'cad', 'sl', 'zcomp_new', 'Nvisits',
             'delta_z',
             'nvisits_UD_night_recalc', 'nvisits_UD_season', 'n_night_season']

    # get u-visits from z-visits
    # df_res['u_season'] = frac_moon*df_res['z_season']
    # df_res['z_season'] = (1.-frac_moon)*df_res['z_season']
    df_res = pd.DataFrame(df_res)
    df_res['n_night_season'] = df_res['sl']/df_res['cad']
    # df_res['u_recalc'] = df_res['u_season']/frac_moon/df_res['n_night_season']
    # df_res['z_recalc'] = df_res['z_season'] / \
    #    (1.-frac_moon)/df_res['n_night_season']
    for io, row in df_res.iterrows():
        ra = []
        for vv in bbval:
            ra.append(row[vv])
        for b in 'grizy':
            r.append(ra + [b, row[b]])
        # add a uband
        name = row['name']
        ipo = m5_resu['name'] == name
        ipo &= m5_resu['band'] == 'u'
        sel = m5_resu[ipo]
        nu_night = sel['Nvisits']/sel['Nseasons']
        n_nights = sl_UD/cad_UD
        nu_night /= (frac_moon*n_nights)
        r.append(ra+['u', np.round(nu_night[0], 0)])

    df_resb = pd.DataFrame(r, columns=bbval+['band', 'nvisits_night'])
    bands = 'ugrizy'
    fracs = pd.DataFrame(list(bands), columns=['band'])
    ro = []
    for b in bands:
        ffrac_b = 1
        if b == 'u':
            ffrac_b = frac_moon
        if b == swap_filter_moon:
            ffrac_b = 1.-frac_moon
        ro.append(ffrac_b)
    fracs['frac_night'] = ro

    df_resb = df_resb.merge(fracs, left_on=['band'], right_on=['band'])

    return df_resb


def nvisits_from_m5(res, m5class, Nseasons=9):
    """
    Function to get nvisits from m5 values

    Parameters
    ----------
    res : pandas df
        data to process.
    m5class : pandas df
        m5 single exp values.
    Nseasons : int, optional
        number of seasons. The default is 9.

    Returns
    -------
    m5_resu : pandas df
        m5 values corresponding to Nvisits.

    """

    m5_resu = pd.DataFrame()

    for vv in res:
        Nvisits = vv['nvisits_DD_season']*Nseasons
        res = m5class.m5_from_Nvisits(Nvisits=Nvisits)
        res['name'] = vv['name']
        res['Nseasons'] = Nseasons
        m5_resu = pd.concat((m5_resu, res))

    return m5_resu


class Delta_nvisits:
    def __init__(self, dfres, m5_nvisits):
        """
        class to estimate and plot Delta Nvisits per band (scenario vs req)

        Parameters
        ----------
        dfres : pandas df
            DD scenarios.
        m5_nvisits : pandas df
            M5 and Nvisits req to reach these m5.

        Returns
        -------
        None.

        """

        idx = dfres['year'] == 1
        df_y1 = dfres[idx][['name', 'fieldType',
                            'band', 'nvisits_band_season']]
        m5_nvisits_y1 = m5_nvisits[['band', 'Nvisits_WL_PZ_y1']]

        df_y1 = df_y1.merge(m5_nvisits_y1, left_on=['band'], right_on=['band'])
        df_y1['ratio_nvisits'] = df_y1['nvisits_band_season'] / \
            df_y1['Nvisits_WL_PZ_y1']

        df_y2_y10 = dfres[~idx]
        m5_nvisits_y2_y10 = m5_nvisits[['band', 'Nvisits_WL_PZ_y2_y10']]
        df_y2_y10 = df_y2_y10.groupby(['name', 'band', 'fieldType'])[
            'nvisits_band_season'].sum().reset_index()
        df_y2_y10 = df_y2_y10.merge(m5_nvisits_y2_y10, left_on=['band'],
                                    right_on=['band'])

        df_y2_y10['ratio_nvisits'] = df_y2_y10['nvisits_band_season'] / \
            df_y2_y10['Nvisits_WL_PZ_y2_y10']

        self.plot_delta_nvisits(df_y1, figtitle='WL+PZ requirements y1')
        self.plot_delta_nvisits(df_y2_y10)

    def plot_delta_nvisits(self, df, xvar='name', xlabel='',
                           yvar='ratio_nvisits',
                           ylabel=r'$\frac{N_{visits}^{DD}}{N_{visits}^{WL+PZ}}$',
                           figtitle='WL+PZ requirements y2_y10'):
        """
        Method to plot ratio of nvisits vs scenario

        Parameters
        ----------
        df : pandas df
            data to plot.
        xvar : str, optional
            x-axis val. The default is 'name'.
        xlabel : str, optional
            x-axis label. The default is ''.
        yvar : str, optional
            y-axis var. The default is 'diff_m5'.
        ylabel : str, optional
            y-axis label. The default is '$\Delta m_5=m_5^{DD}-m_5^{PZ}$'.
        figtitle : str, optional
            fig title. The default is 'WL+PZ requirements y2_y10'.

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(figtitle)
        fig.subplots_adjust(bottom=0.2)
        bands = df['band'].unique()
        ls = dict(zip(['UD', 'DD'], ['solid', 'dashed']))
        marker = dict(zip(['UD', 'DD'], ['o', 's']))

        df['scen_type'] = df['name'].str.split('_').str.get(-1)
        df = df.sort_values(by='scen_type')

        for b in bands:
            idx = df['band'] == b
            sela = df[idx]
            fieldTypes = sela['fieldType'].unique()
            for ft in fieldTypes:
                idxa = sela['fieldType'] == ft
                selb = sela[idxa]
                """
                ax.plot(selb[xvar], selb[yvar], linestyle=ls[ft],
                        color=filtercolors[b])
                """
                ax.plot(selb[xvar], selb[yvar], linestyle='None',
                        color=filtercolors[b], marker=marker[ft],
                        mfc='None', ms=12, markeredgewidth=2)

        ax.grid()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis='x', labelrotation=290, labelright=True,
                       labelsize=12)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")


def moon_recovery(df, swap_filter_moon='z'):
    """
    Function to compensate for moon swapping filter

    Parameters
    ----------
    df : pandas df
        data to process.
    swap_filter_moon : str, optional
        Filter removed during low moon phases. The default is 'z'.

    Returns
    -------
    dfb : pandas df
        Resulting data.

    """

    idx = df['band'] == swap_filter_moon
    sel = df[idx]
    dfb = df[~idx]

    # print('hhh', sel[['frac_night', 'fieldType', 'band', 'year']])
    # correct for the number of visits on the swap filter
    sel['nvisits_night'] /= sel['frac_night']

    sel['nvisits_night'] = sel['nvisits_night'].astype(int)
    sel.loc[sel['nvisits_night'] == 1, 'nvisits_night'] = 2

    dfb = pd.concat((dfb, sel))

    return dfb


def reverse_df(df):
    """
    Function to modify df struct

    Parameters
    ----------
    df : pandas df
        data to process.

    Returns
    -------
    pandas df
        restructured df.

    """

    outDict = {}
    for b in 'ugrizy':
        idx = df['band'] == b
        sel = df[idx]
        outDict[b] = sel['nvisits_night'].to_list()

    return pd.DataFrame.from_dict(outDict)


def uniformize(dfres, name='DDF_Univ_SN', Nv_LSST=2.1e6, budget=0.07):
    """
    Function to make a uniform survey from UD fields (converted to DD at the end)
    and correct for u-band to
    fit in the budget

    Parameters
    ----------
    dfres : pandas df
        Data to process.
    name : str, optional
        survey to modify. The default is 'DDF_Univ_SN'.
    Nv_LSST : float, optional
        Total number of LSST visits. The default is 2.1e6.
    budget : float, optional
        DD budget. The default is 0.07.

    Returns
    -------
    df_calc : pandas df
        Processed data.

    """

    idx = dfres['name'] == name
    df_other = dfres[~idx]

    idx &= dfres['fieldType'] == 'UD'
    seldf = dfres[idx]

    df_UD = pd.DataFrame(seldf)

    idxa = df_UD['year'] > 1

    # estimate number of visits - y1
    df_UD_y1 = pd.DataFrame(df_UD[~idxa])
    df_UD_y1['nvisits_season'] = df_UD_y1['nvisits_band_season'] * \
        df_UD_y1['Nfields']

    N_y1 = df_UD_y1['nvisits_season'].sum()

    # estimate number of visits y2 to y10 - no u-band
    df_UD_y2_y10 = pd.DataFrame(df_UD[idxa])
    idxb = df_UD_y2_y10['band'] != 'u'

    df_UD_y2_y10_no_u = pd.DataFrame(df_UD_y2_y10[idxb])
    df_UD_y2_y10_no_u['nvisits_season'] = df_UD_y2_y10_no_u['nvisits_band_season'] * \
        df_UD_y2_y10_no_u['Nfields']

    N_y2_y10_no_u = df_UD_y2_y10_no_u['nvisits_season'].sum()

    # estimate remaining number of u-visits to fit the budget
    Nvisits_u = budget*Nv_LSST-N_y1-N_y2_y10_no_u

    idxf = df_UD['year'] > 1
    idxf &= df_UD['band'] == 'u'
    Nfields = df_UD[idxf]['Nfields'].mean()
    df_UD.loc[idxf, 'nvisits_band_season'] = Nvisits_u/9./Nfields
    df_UD.loc[idxf, 'nvisits_season'] = df_UD[idxf]['nvisits_band_season']*Nfields

    df_UD.loc[idxf, 'nvisits_night'] = df_UD[idxf]['nvisits_band_season'] / \
        df_UD[idxf]['n_night_season']/df_UD[idxf]['frac_night']

    # int it
    for vv in ['nvisits_night', 'nvisits_band_season']:
        #df_UD.loc[idx, vv] *= cad_UD*sl_UD/(cad_DD*sl_DD)
        df_UD[vv] = df_UD[vv].astype(int)

    df_UD['fieldType'] = 'DD'
    df_calc = pd.concat((df_UD, df_other))

    # estimate budget here
    """
    df_calc = pd.DataFrame(df_UD)

    df_calc['nvisits_season'] = df_calc['nvisits_night'] * \
        df_calc['n_night_season']*df_calc['Nfields']*df_calc['frac_night']

    rr_ud = df_calc.groupby(['year'])['nvisits_season'].sum().reset_index()

    print('finally', np.sum(df_calc['nvisits_season'])/Nv_LSST)
    """
    return df_calc
