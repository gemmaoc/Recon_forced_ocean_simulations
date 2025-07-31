#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 16:39:11 2022

@author: gemma
"""

import numpy as np
from Functions import load_1d_data, load_3d_data


#%%

# Parent directories containing data
global recon_dir
home_dir = '/Users/gemma/Documents/Research/LMR_analysis/1940_ASE_Analysis/'
recon_dir = home_dir+'Data/Reconstruction/' #Contains all O'Connor et al. recons and Dalaiden et al. recon
model_dir = home_dir+'Data/Model/' #Contains Pacemaker ensemble, CESM PreIndustrial Ctrl simulation, LENS historical ensemble
verif_dir = home_dir+'Data/Verification/' #contains ERA5, ERSST

# Reconstruction names (corresponding to filenames)
# To do: change the names of these to match published names
run72 = 'iCESM_LME_GKO1_all_lin_1mc_1900_2005_GISBrom_1880_2019_1deg_res_full_ens'
run73 = 'HadCM3_all_lin_1mc_1900_2005_GISBrom_1880_2019_1deg_res_full_ens'
run78 = 'PACE_super_GKO1_all_linPSM_1mc_1900_2005_GISBrom_1880_2019_1_deg_res_full_ens'
run82 = 'LENS_super_GKO1_all_linPSM_1mc_1800_2005_GISBrom_1880_2019'
dal_run = 'Dalaiden_2021_*vname*_ano_annual_recon-antarctic_1800-2000'

# Single proxy runs
run26 = 'icesm_lme_GKO1_coral_only_lin_1mc_1900_2005' 
run51 = 'iCESM_LME_GKO1_ice_only_1mc_1900_2005_psm_calib_GISBrom_1880_2019'
run52 = 'PACE_super_GKO1_ice_only_1mc_1900_2005_psm_calib_GISBrom_1880_2019'
run31 = 'pace_super_GKO1_coral_only_lin_1mc_1900_2005' 

# User: Edit path to match recon filenames. There must be one file for each variable
# Instead of writing each variable in the path, use '*vname*' and the function will replace the vname
cesm_recon_path = recon_dir + run72 + '_*vname*.nc'
pace_recon_path = recon_dir + run78 + '_*vname*.nc'
cesm_coral_recon_path = recon_dir + run26 + '_*vname*.nc'
cesm_ice_recon_path = recon_dir + run51+ '_*vname*.nc'
pace_coral_recon_path = recon_dir + run31 + '_*vname*.nc'
pace_ice_recon_path = recon_dir + run52 + '_*vname*.nc'
# These recons are for making Figure A1, not used in the main analyses
hadcm_recon_path = recon_dir + run73 + '_*vname*.nc'
lens_recon_path = recon_dir + run82 + '_*vname*.nc'
dal_recon_path = recon_dir + dal_run+ '.nc'

# paths for full reconstruction ensmeble
cesm_recon_ens_path = recon_dir + 'Full Ensemble/' + run72 + '_*vname*_full.nc'
pace_recon_ens_path = recon_dir + 'Full Ensemble/' + run78 + '_*vname*_full.nc'

#%% Reconstruction class

#Create a class with dataset meta so that changes propogate to all scripts

class Reconstruction:
    
    
    def __init__(self, name, path, psl_color, u10_color):
        
        self.name = name
        self.path = path
        # Set colors for ASE shelf break time series
        self.psl_color = psl_color
        self.u10_color = u10_color
        
    # Add methods for reconstructions that have geostrophic winds and full ensembles
    def add_g_u10_color(self,g_u10_color):
        self.g_u10_color = g_u10_color
        
    def add_ens_path(self,ens_path):
        self.ens_path = ens_path
    
    def get_geo_wind_ens(self,recon_u10_full_ens,time_per,region):
    
        # generate geostrophic wind full ensemble using variance from u10 ens
        # add spread from u10 full ensemble to ensemble mean g_u10
        recon_g_u10_em = load_1d_data(self.path.replace('*vname*','g_u10'),\
                                      'g_u10', region, time_per = time_per)
        recon_u10_em = load_1d_data(self.path.replace('*vname*','u10'),\
                                    'u10', region, time_per = time_per)
        # Var is same for each year so get ens from first year
        yr1_full_ens = recon_u10_full_ens.isel(time = 0)
        # Get differences between em and each member.shape 100
        spacings = yr1_full_ens - recon_u10_em.isel(time = 0)
        
        # Add spacings to each year in geostrophic wind ensemble mean
        n_yrs, n_ens = recon_u10_full_ens.shape
        # Replace copy of u10 full ens with geostrophic ensemble 
        recon_g_u10_full_ens = recon_u10_full_ens.copy() #shape (n_yrs, n_ens)
        
        
        for i in range(n_ens):
            
            ens_i_data = []
            ens_spacing_value = spacings[i]
            
            for j in range(n_yrs):
                
                ens_i_data.append(recon_g_u10_em[j] + ens_spacing_value)
                
            recon_g_u10_full_ens[:,i] = ens_i_data
        
        return recon_g_u10_full_ens
    
    def calc_scram_ens(self,full_ens_data,n_draws):
        """
        Calculates n_ens timeseries by randomly drawing from recon ensemble members.
    
        Parameters
        ----------
        full_ens_data : 2d xr data array
            recon full ensemble data (probably shape 86, 100)
        n_draws : int
            how many timeseries of scrambled ensemble members to make
    
        Returns
        -------
        scram_ens : 2d numpy array
            shape n_draws, n_years
    
        """
        
        n_yrs = full_ens_data.shape[0]
        scram_ens = np.zeros((n_draws,n_yrs))
        #iterate through number of random timeseries you want
        for i in range(n_draws):
            # print('working on random draw',i)
            rand_timeseries_i = []
            #get random draw for each year in timeseries
            for year_idx in range(n_yrs):
                rand_idx = int(np.random.uniform(low=0,high=99))#inclusive
                rand_val = full_ens_data[year_idx,rand_idx]
                rand_timeseries_i.append(rand_val)
            rand_timeseries_i = np.array(rand_timeseries_i) #has shape n_years,n_lats,n_lons
            scram_ens[i,:] = rand_timeseries_i
        
        return scram_ens
        
        
# Primary recons used in analysis
cesm_recon = Reconstruction('Natural-prior recon', cesm_recon_path, '#253494', '#993404')
cesm_recon.add_g_u10_color('#cc4c02')
cesm_recon.add_ens_path(cesm_recon_ens_path)

pace_recon = Reconstruction('Anthro-prior recon', pace_recon_path, '#41b6c4', '#ec7014')
pace_recon.add_g_u10_color('#fe9929')
pace_recon.add_ens_path(pace_recon_ens_path)

# Single proxy recons generated in O'Connor et al. 1940 study
cesm_coral_recon = Reconstruction('Natural-prior coral only recon',\
                                  cesm_coral_recon_path,'#ce1256',None)
cesm_ice_recon = Reconstruction('Natural-prior ice only recon',cesm_ice_recon_path,None,None)

pace_coral_recon = Reconstruction('Anthro-prior coral only recon',\
                                  pace_coral_recon_path,'indianred',None)
pace_ice_recon = Reconstruction('Anthro-prior ice only recon',pace_ice_recon_path,None,None)

# Comparison reconstructions from O'Connor et al. 2021 and Dalaiden et al. 2021
hadcm_recon = Reconstruction('OConnor 2021 HadCM recon',hadcm_recon_path,None,None)
lens_recon = Reconstruction('OConnor 2021 LENS recon',lens_recon_path,None,None)
dal_recon = Reconstruction('Dalaiaden 2021 CESM LM recon',dal_recon_path,None,None)


#%% Model class

class Model:
    
    def __init__(self, name, em_path, ens_path, members):
        
        self.name = name
        self.em_path = em_path
        self.ens_path = ens_path
        self.members = members
        
    def set_model_time(self,time):
        self.time = time
        
    def concat_ens_1d(self, vname, region):
        """
        Concatenate the members in an ensemble to produce one long 1d array
        I.e., to load the PI control simulation which is saved as multiple files

        Parameters
        ----------
        vname : str
            'psl' or 'u10' or whatever vars the model has
        region : str
            must be a key in region_dict

        Returns
        -------
        ens_data_concat : 1d np array
            array of concatenated ensemble members in specified region

        """
        
        ens_data_list = []
        for member in self.members:
            
            mem_path = self.ens_path.replace('*vname*', vname)
            mem_path = mem_path.replace('*num*', member)
            mem_data = load_1d_data(mem_path, vname, region,np_array = True)
            ens_data_list.append(mem_data)
                
        #make model data one big nparray
        ens_data_concat = np.concatenate(list(ens_data_list))
        
        return ens_data_concat
        
        
    def remove_model_em_ref(self, member, vname, time_per, anom_ref, \
                            region = None, np_array = False):
        """
        Put a single ensemble member in anomaly space relative to the 
        model ensemble mean during the given anomaly ref period. 
        I.e., put a member of the Pacific Pacemaker ensemble in anomaly
        space relative to the Pacific Pacemaker ensemble mean so that
        all members are relative to the same reference. 

        Parameters
        ----------
        member : an instance of the Model class
            i.e, the Pacific Pacemaker ensemble (pac_pace_model)
        vname : str
            variable name in the file
        time_per : list of two int years
            time period of ensemble member data desired (inclusive)
        anom_ref : list of two int years
            the period over which to average the ensemble mean data
            the ensemble member data will be relative to the values during this period
        region : str, optional
            Whether to select a region over which to do the calculation. 
            The default is None.
        np_array : bool, optional
            If False (default), returns the data as an xr data array
            If true, converts to a np array

        Returns
        -------
        member_anom : xr data array (unless np_array = True)
            3D data array of ensemble member data in anomalies 

        """
    
        # Get ensemble mean data during anom ref period
        model_em_ref_data = load_3d_data(self.em_path.replace('*vname*',vname),\
                               vname, anom_ref, region=region)
        model_em_ref_mean = model_em_ref_data.mean(dim='time')
    
        # Get ensemble member data
        member_path = self.ens_path.replace('*num*',member)
        member_data = load_3d_data(member_path.replace('*vname*',vname),vname,\
                                    time_per,region=region)
            
        # Remove ensemble mean ref from ensemble member data
        member_anom = member_data - model_em_ref_mean
        
        return member_anom
    
    def get_ens_anom_nat(self, vname, region, anom_ref):
        """
        For a model ensemble dataset which has each member saved as a separate file, 
        remove the anomaly reference period in each member for the specified ref period, 
        and remove the LENS Historical EM to isolate the internal component.
        
        It assumes that the model has the same number of years as LENS historical EM, 
        so alter code if that is not true.
        
        This method is called for calculating the probability of the 1940 event, 
        in processing the LENS historical ensemble data. 

        Parameters
        ----------
        vname : str
            psl or u10
        region : str
            must be a key in region_dict
        anom_ref : list of 2 int years
            period over which to remove the mean values to make anomalies

        Returns
        -------
        mod_anom_nat : np array
            shape n_ens, n_yrs
            contains ensemble data in anomalies with anthro component removed

        """
        
        mod_path = self.ens_path.replace('*vname*',vname)
        members = self.members
        n_mems = len(members)
        n_yrs = len(self.time)
        mod_anom_nat = np.zeros((n_mems,n_yrs))
        anom_start, anom_stop = anom_ref
        lens_hist_em_data_anom = load_1d_data(lens_hist.em_path.replace('*vname*',vname),\
                                              vname, region, anom_ref = anom_ref)
        
        for mem_i in range(n_mems):
            
            mem_path = mod_path.replace('*num*',members[mem_i])
            mem_data = load_1d_data(mem_path, vname, region)
            
            #put in anomaly space
            mem_anom_ref = mem_data.sel(time = slice(anom_start,anom_stop))
            mem_data_anom = mem_data - mem_anom_ref.mean()
            
            #remove LENS EM
            mem_data_anom_nat = mem_data_anom - lens_hist_em_data_anom
            
            #populate
            mod_anom_nat[mem_i,:] = mem_data_anom_nat
            
        return mod_anom_nat
        
pac_pace_model = Model('PACE', \
                 '../Model/PAC_PACE/annual_*vname*_PAC_PACE_ens_mean_1920_2005.nc',\
                 '../Model/PAC_PACE/annual_*vname*_PAC_PACE_ens_*num*_1920_2005.nc',\
                 ['01','02','03','04','05','06','07','08','09','10',\
                 '11','12','13','14','15','16','17','18','19','20'])

lens_hist = Model('LENS historical', \
                  model_dir + 'LENS_hist/annual_*vname*_LENS_ens_mean_1920_2005.nc',\
                  model_dir+'LENS_hist/annual_*vname*_LENS_ens_*num*_1920_2005.nc',\
                  ['001','002','003','004','005','006','007','008','009','010',\
                   '011','012','013','014','015','016','017','018','019','020',\
                   '021','022','023','024','025','026','027','028','029','030',\
                   '031','032','033','034','035','101','102','103','104','105'])
lens_hist.set_model_time(np.linspace(1920,2005,86,dtype=int))

# LENS PI is one long simulation, but saved as 100-year files (and last one is 101 years)
# Concatatenate these files to get the simulation (there is no ensemble mean)
pi_years1 = np.linspace(400,2000,(2000-400)//100+1)
ens_mems_1 = [str(int(year)) for year in pi_years1]
pi_years_2 = np.linspace(499,2099,(2099-499)//100+1)
ens_mems_2 = [str(int(year)) for year in pi_years_2]
pi_members = [e1+'_'+e2 for e1,e2 in zip(ens_mems_1,ens_mems_2)]+['2100_2200']

pi_ctrl = Model('PI Control',None,\
                model_dir + 'LENS_PI_ctrl/annual_*vname*_LENS_preindustrial_*num*.nc',\
                pi_members)
pi_time = np.linspace(400,2200,2200-400+1)
pi_ctrl.set_model_time(pi_time)
    

