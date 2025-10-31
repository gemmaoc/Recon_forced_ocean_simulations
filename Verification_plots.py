#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:10:02 2020

Creates a figure showing timeseries of a user-defined variable averaged over a user-defined region 
in proxy reconstructions and ERA5. Shifts anom ref period in recons to 1979-2005. 
Includes corr and CE in legend. 


@author: gemma
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import signal
import importlib
import Functions
importlib.reload(Functions)
from Functions import load_1d_data, calc_1d_corr, calc_1d_ce


#%% User defined params
anom_ref = 1979,2005
# region = 'ase_domain'
region = 'WAIS'
# region = 'WAIS_ice_cores'
recon_start,recon_stop = 1800,2005
recons = ['cesm2_pace', 'cesm2_lens', 'cesm1_pace','cesm1_lens', 'cesm1-lme']
vname = 'tas' #u10, v10, tas, psl, pr, dlw, dsw, spfh2m
parent_dir = '/Users/gemma/Documents/Data/'


#%% load data 

# get recon data
recon_dir = parent_dir + 'Proxy_reconstructions/'
recon_path_dict = {'cesm2_pace':'CESM2_PAC_PACE_recon_1800_2005/CESM2_PAC_PACE_recon_1800_2005_',
                   'cesm2_lens':'CESM2_LENS_recon_1800_2005/CESM2_LENS_recon_1800_2005_',
                   'cesm1_lens':'CESM1_LENS_recon_1800_2005/CESM1_LENS_recon_1800_2005_',
                   'cesm1-lme':'iCESM1_LME_recon_1800_2005/iCESM1_LME_recon_1800_2005_',
                   'cesm1_pace':'CESM1_PAC_PACE_recon_1800_2005/CESM1_PAC_PACE_recon_1800_2005_'}

time_per = recon_start,recon_stop
recon_data = []
for recon in recons:
    recon_1d, recon_units = load_1d_data(recon_dir+recon_path_dict[recon]+vname+'.nc', vname, region, 
                            time_per = time_per, anom_ref = anom_ref)
    recon_data.append(recon_1d)
print('Recon units:', recon_units)
recon_time = recon_1d.time


#%%get ERA5 
era_vname_dict = {'u10':'u10','v10':'v10','tas':'t2m','sst':'sst','psl':'msl',
                  'dsw':'ssrd', 'dlw':'strd', 'pr':'tp', 'spfh2m':'q2m'}
verif_dir = parent_dir + 'Reanalysis/ERA5/'
era_1d, era_units = load_1d_data(verif_dir+'era5_annual_1979_2024.nc', era_vname_dict[vname], \
                                region, anom_ref = anom_ref)
print('ERA5 units:', era_units)
if vname == 'pr':
    era_1d = era_1d / (365.25 * 24 * 3600)  # convert m/year to m/s
    era_1d = era_1d * 1000  # convert m/s to kg/m2/s (density of water = 1000 kg/m3)
    era_units = 'kg m$^{-2}$ s$^{-1}$'
era_time = era_1d.time

if vname == 'dlw' or vname == 'dsw':
    # for radiation, multiply by 12 because you have a double division problem in your conversion from monthly to annual'
    era_1d = era_1d * 12
    

#%% Calc stats with ERA5 over period of overlap
print(vname)

recon_stats = []
era_np = np.array(era_1d.sel(time=slice(anom_ref[0], anom_ref[1])))


for i in range(len(recons)):
    recon_i = recon_data[i]
    recon_tseries = recon_i.sel(time=slice(anom_ref[0], anom_ref[1]))
    recon_np = np.array(recon_tseries)

    corr,p_val = calc_1d_corr(recon_np, era_np,
                              return_format = 'float')
    ce = calc_1d_ce(recon_np, era_np, return_format= 'float')

    stat_str = recons[i]+' recon (r = {:.2f}, p = {:.3f}, CE =  {:.2f})'.format(corr, p_val, ce)
    recon_stats.append(stat_str)

    # print std dev over overlap period
    recon_std = np.std(recon_np)
    era_std = np.std(era_np)
    print('{} std dev: recon = {:.4f}, ERA5 = {:.4f}'.format(recons[i], recon_std, era_std), recon_units)


#%% Make Figure

# generate colors from  cmap
colors = plt.cm.viridis_r(np.linspace(0.05, 0.85, len(recons)))

fig = plt.figure(figsize=(8,4))
plt.plot(era_time, era_1d, label='ERA5', color='black', linewidth=2)
for i in range(len(recons)):
    plt.plot(recon_time, recon_data[i], label=recon_stats[i], linewidth=1.5, color=colors[i])
if vname in ['psl','u10','v10']:
    plt.legend(loc='lower left', fontsize=7)
else:
    plt.legend(loc='upper left', fontsize=7)
plt.xlabel('Year', fontsize=14)
plt.ylabel(vname + ' ('+ recon_units + ')', fontsize=14)
plt.title('Time Series of {} over {}'.format(vname, region), fontsize=16)
plt.grid(True)

# %%
