#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:19:36 2026

@author: philippe.gris@clermont.in2p3.fr
"""
import sncosmo
from sn_telmodel.sn_throughputs import get_telescope
import numpy as np
from astropy.table import Table,vstack
import pandas as pd
from sn_tools.sn_io import check_get_file
from sn_tools.sn_cosmo_model import cosmo_wrapper
from sn_telmodel.sn_transtools import zp_from_config

class SNflux:
    def __init__(self,x1,color,daymax,z,ebvofMW,
                 model='salt3',
                 version='2.0',
                 absmag=-19.0906,
                 magsys='vega',band='bessellB',
                 tel_dir = 'throughputs',
                 throughputsDir = 'baseline',
                 atmosDir = 'atmos',
                 tag_tel = '1.9',
                 airmass=1.2,
                 aerosol=0.05,
                 pwv=5.0,
                 ozone=300,
                 x0_file='x0_norm_-19.0906_salt3.npy',
                 x0_dir='reference_files',
                 web_path='https://me.lsst.eu/gris/DESC_SN_pipeline',
                 cosmo_params=dict(zip(['de_params','de_class','de_model',
                                        'de_eos','H0','Om0','Ode0','class_loc'],
                                       [dict(zip(['w0','wa'],[-1,0.])),
                                        'w0waCDM','CPL','w0+wa*z/(1+z)',
                                        70.,0.3,0.7,'astropy.cosmology'])),
                 phases_sed=[-10,0.,20]):
        """
        class to estimate SN Ia  flux vs time

        Parameters
        ----------
        x1 : float
            SN stretch parameter.
        color : float
            SN color parameter.
        x0 : float
            SN x0 parameter.
        daymax :  float
            SN daymax (T0) parameter.
        z : float
            SN redshift parameter.
        model : str, optional
            sncosmo model. The default is 'salt3'.
        version : float, optional
            sncosmo model version. The default is 2.0.
        absmag : float, optional
            SN abs mag. The default is -19.0906.
        magsys : str, optional
            mag system. The default is 'vega'.
        band : str, optional
            SN band. The default is 'bessellB'.
        tel_dir : str, optional
            telescope directory. The default is 'throughputs'.
        throughputsDir : str, optional
            throughput directory. The default is 'baseline'.
        atmosDir : str, optional
            atmosphere directory. The default is 'atmos'.
        tag_tel : str, optional
            telescope tag version. The default is '1.9'.
        airmass : float, optional
            airmass default value. The default is 1.2.
        aerosol : float, optional
            aerosol default value. The default is 0.0.
        pwv : float, optional
            pwv default value. The default is 4.0.
        ozone : float, optional
            ozone default value. The default is 400.

        Returns
        -------
        None.

        """
        
        
        # SN parameters
        self.x1 = x1
        self.color = color
        self.ebvofMW= ebvofMW
        self.daymax = daymax
        self.z = z
        self.model = model
        self.version = version
        self.absmag = absmag
        self.band = band
        self.magsys = magsys
        
        
        #telescope parameters
        self.tel_dir = tel_dir
        self.throughputsDir = throughputsDir
        self.atmosDir = atmosDir
        self.tag_tel = tag_tel
        
        #atmos parameters
        self.airmass = airmass
        self.pwv = pwv
        self.aerosol = aerosol
        self.ozone = ozone
        
        #grab x0 norm file and grab values
        check_get_file(web_path,x0_dir, x0_file)
        self.x0_grid = np.load('{}/{}'.format(x0_dir,x0_file))
        
        #instanciate cosmology
        self.cosmology=cosmo_wrapper(cosmo_params)
        
        
        #instances of SN and telescope
        
        self.sn = self.get_sn()
        self.telescope = self.get_telescope()
        
        #getting the zeropoints
        config = {}
        config['sigma'] = {}
        for vv in ['airmass','pwv','ozone','aerosol']:
            config[vv] = eval('{}'.format(vv))
            config['sigma']['{}'.format(vv)] = 0
        config['ntrial'] = {}
        config['ntrial']['zp']=1   
        config['atmosDir'] = atmosDir
        config['name'] = 'LSST'
        config['telescope'] = {}
        config['telescope']['dir'] = tel_dir
        config['telescope']['tag'] = tag_tel
        config['throughputDir'] = throughputsDir
        
        zp_airmass = zp_from_config(config)
        
        zp = {}
        for b in 'ugrizy':
            val = zp_airmass['zp'][b](airmass).tolist()
            zp[b] =float(np.round(val,3))
        
        self.zp = zp
       
        tmin = daymax-20*(1+z)
        tmax = daymax+45*(1+z)
        tstep = 0.5
        self.mjds = np.arange(tmin,tmax,tstep).tolist()
        
        mjds_from_ph = []
        if phases_sed:
            for ph in phases_sed:
                mjd_ = ph*(1.+self.z)+daymax
                mjds_from_ph.append(mjd_)
            self.mjds_sed=mjds_from_ph
        else:
            self.mjds_sed=self.mjds
            
        
    def get_sn(self):
        """
        Method for SN Ia instance in sncosmo

        Returns
        -------
        sn : sncosmo.Model
            SN Ia instance.

        """
        
        source = sncosmo.get_source(self.model, self.version)
        
        if self.model == 'salt3':
            wave_min = 1500.
            wave_max = 24990.
            source._wave[0] = wave_min  # used to be 1700
            source._wave[-1] = wave_max
           
        wmin = wave_min*(1+self.z)
        wmax = np.min([wave_max*(1+self.z),33333])
        
        self.wave = np.arange(wmin, wmax, 1.)
        #self.wave *= (1.+self.z)
        
        dustmap = sncosmo.OD94Dust()
        sn = sncosmo.Model(source=source,
                           effects=[dustmap, dustmap],
                           effect_names=['host', 'mw'],
                           effect_frames=['rest', 'obs'])
        sn.set(z=self.z)
        sn.set(t0=self.daymax)
        sn.set(x1=self.x1)
        sn.set(c=self.color)
        x0 = self.get_x0()
        sn.set(x0=x0)
        sn.set(mwebv=self.ebvofMW)
        
        return sn
    
    def get_x0(self,alpha=0.13,beta=3.1):
       """

       Method to estimate x0 from (alpha, beta,sigmaint)

       Returns
       -------
       X0 : TYPE
           DESCRIPTION.

       """

       from scipy.interpolate import griddata

       x0_grid = griddata((self.x0_grid['x1'], self.x0_grid['color']),
                          self.x0_grid['x0_norm'], 
                          (self.x1, self.color),
                          method='nearest')
       
       lumidist= self.cosmology.luminosity_distance(self.z).value*1.e3  # in kpc
       x0 = x0_grid / lumidist ** 2

       x0 *= np.power(10., 0.4*(alpha *self.x1 - beta *self.color))
       
       return x0
   
    def get_telescope(self):
        """
        Method for a telescope instance

        Returns
        -------
        telescope : Throughput class
            Total Throughput.

        """
        
        telb = '{}_{}'.format(self.tel_dir, self.tag_tel)
        through_dir = '{}/{}'.format(telb, self.throughputsDir)
        atmos_dir = '{}/{}'.format(telb, self.atmosDir)
        telescope = get_telescope(tel_dir=telb,
                                  through_dir=through_dir,
                                  atmos_dir=atmos_dir,
                                  tag=self.tag_tel, 
                                  airmass=self.airmass,
                                  aerosol=self.aerosol, 
                                  pwv=self.pwv, 
                                  ozone=self.ozone)
         
        return telescope
        
        
    def get_flux(self,lc_data=None):
        """
        Method to get sn_fluxes from LC (time+filters)

        Parameters
        ----------
        lc_data : astropy table
            flux vs time (from obs).

        Returns
        -------
        None.

        """
        
        #complete lc
        
        lc = self.complete_lc(lc_data)
        
        # register bands
        ## necessity to drop duplicates!!!!!!!
        ccols = ['band_cosmo','filter','airmass','pwv','aerosol','ozone']
        lc_nodup = lc[ccols].drop_duplicates()
        
        self.register_bands(lc_nodup)
        
        #grab the fluxes
        bands = lc['filter'].unique()
        
        lc_df = pd.DataFrame()
        
        for b in bands:
            idx = lc['filter'] == b
            lcb = pd.DataFrame(lc[idx])
            
            flux = self.sn.bandflux(lcb['band_cosmo'], lcb['time'], 
                                    zpsys=lcb['zpsys'],zp=lcb['zp'])
            
            df_ = pd.DataFrame(flux.tolist(),columns=['flux'])
            df_['filter'] = 'LSST:'+b
            df_['filter_notel'] = b
            df_['time'] = lcb['time'].to_list()
            df_['zp'] = lcb['zp'].to_list()
            df_['zpsys'] = lcb['zpsys'].to_list()
            df_['mag'] = -2.5*np.log(df_['flux'])+df_['zp']
            lc_df = pd.concat((lc_df,df_))
        
        #self.plot_flux(lc_df,lc_data)
        
        lc_df['phase'] = (lc_df['time']-self.daymax)/(1.+self.z)
        return lc_df
        
    def complete_lc(self,lc_data=None):
        """
        Method to complete lc_data with obs

        Parameters
        ----------
        lc_data : astropy table, optional.
            Data to process. The default is None.

        Returns
        -------
        lc_tot : pandas df
            output lc.

        """

        lc = Table()  
        
        if lc_data is not None:
            #print(lc_data[['pwv','aerosol','ozone']])
        
            ccols = ['time','band_cosmo','filter',
                     'airmass','pwv','aerosol','ozone',
                     'zpsys','zp']
        
            lc = lc_data[ccols]
        
        lc_full = self.get_full_lc()
        
        if lc_data is not None:
            lc_tot = pd.concat((lc.to_pandas(),lc_full))
        else:
            lc_tot =pd.DataFrame(lc_full)
        
        
        return lc_tot
        
    def get_full_lc(self):
        """
        Method to estimate the full LC

        Returns
        -------
        df_lc : pandas df
            Full LC.

        """
        
        ccols = ['time','band_cosmo','filter',
                 'airmass','pwv','aerosol','ozone',
                 'zpsys','zp']
        
        filters = 'grizy'
        
        airmassb = np.round(self.airmass,2)
        pwvb = np.round(self.pwv,2)
        aerosolb = np.round(self.aerosol,2)
        ozoneb = np.round(self.ozone,2)
        
        b_filt = {}
        for fi in filters:
            bcols = self.telescope.site_name+'::'
            bcols += fi+'_'
            bcols += '{}'.format(airmassb)+'_' 
            bcols +='{}'.format(pwvb)+'_'      
            bcols += '{}'.format(ozoneb)+'_' 
            bcols +='{}'.format(aerosolb)
            b_filt[fi] = bcols
            
        r = []
        
        df_lc = pd.DataFrame()
        for key,vals in b_filt.items():
            
            df = pd.DataFrame(self.mjds, columns=['time'])
            df['band_cosmo'] = vals
            df['filter'] = key
            df['airmass'] = airmassb
            df['pwv'] = pwvb
            df['ozone'] = ozoneb
            df['aerosol'] = aerosolb
            df['zpsys'] = 'ab'
            df['zp'] = self.zp[key]
            df_lc = pd.concat((df_lc,df))
          
        return df_lc
        
        
    def register_bands(self,data):
        """
        Method to register bands in sncosmo

        Parameters
        ----------
        data : pandas df
            Data to register.

        Returns
        -------
        None.

        """
        from sn_tools.sn_utils import register_bands_sncosmo
        
        for i, row in data.iterrows():
            bandname = row['band_cosmo']
            band = row['filter']
            airmass = row['airmass']
            pwv = row['pwv']
            ozone = row['ozone']
            aerosol = row['aerosol']
            register_bands_sncosmo(sncosmo, self.telescope,
                                  bandname, band,
                                  airmass, pwv, ozone, aerosol)
        
    def plot_flux(self, lc_flux,lc_data=None):
        """
        Method to plot fluxes

        Parameters
        ----------
        lc_flux : pandas df
            The full light curve.
        lc_data : astropy table, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if lc_data is not None:
            bands = np.unique(lc_data['filter'])
        else:
            bands = lc_flux['filter'].unique()
    
        import matplotlib.pyplot as plt
        
        for b in bands:
            fig, ax = plt.subplots()
            idx = lc_flux['filter'] == b
            idx &= lc_flux['flux'] > 0.
            sel = lc_flux[idx]
            sel = sel.sort_values(by=['time'])
            ax.plot(sel['time'],sel['flux'])
            if lc_data is not None:
                idx = lc_data['filter'] == b
                sel_data = lc_data[idx]
                ax.errorbar(sel_data['time'],sel_data['flux'],
                            yerr=sel_data['fluxerr'],
                            marker='o',color='r',linestyle='None')
        plt.show()
        
    def sn_sed_mjd(self, mjd):
         """
         Method to generate SED flux
    
         Parameters
         ----------
         mjd : float
             MJD for the flux generation.
    
         Returns
         -------
         sed : astropy table
             generated fluxes.
    
         """
    
         fluxes = 10.*self.sn.flux(mjd, self.wave)
         sed = Table([fluxes], names=['flux'])
         sed['wavelength'] = self.wave
         sed['fluxerr'] = 0.0
    
         idx = sed['wavelength'] >= 4500.
         idx &= sed['wavelength'] <= 20000.
         return Table(sed[idx])        
        
    def get_sed(self):
        """
        Method to estimate SN Ia SEDs for self.mjds 

        Returns
        -------
        sed : list(astropy tables)
            List of SEDs.

        """
       
        sed = []
        for io, mjd in enumerate(self.mjds_sed):
            phase = (mjd-self.daymax)/(1+self.z)
            sedm = self.sn_sed_mjd(mjd)
            metadata = {}
            metadata['phase'] = float(np.round(phase,2))
            metadata['mjd'] = mjd
            tab = Table(sedm)
            tab.meta = metadata
            sed.append(tab)
          
        return sed
         
         
         
        