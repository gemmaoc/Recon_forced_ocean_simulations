#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:10:02 2020

Plota 3-panel figure 1:
Timeseries of SLP and U averaged over ASE shelf break region from from 1900 to 2005
with 2 recons and ERA5. Shifts anom ref period in recons to 1979-2005. 
Includes corr, sig, and CE in legend. 

Also calculates the magnitude of the 1940 event, classified in 10 different ways
And whether the 1940 event is unique in the 20th century 
(and whether the trend influences its uniqueness)

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
region = 'ase_domain'
recon_start,recon_stop = 1800,2005
recons = ['cesm2_pace']
vname = 'pr' #u10, v10, tas, psl, pr, dlw, dsw, spfh2m

#%% load data 

# get recon data
recon_dir = '../Data/Reconstructions/'
recon_path_dict = {'cesm2_pace':'PAC_PACE2_super_GKO1_all_bilinPSM_1mc_1800_2005_GISBrom_mitgcm_vars_'}

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
                  'dsw':'ssrd', 'dlw':'strd', 'pr':'tp'}
verif_dir = '../Data/Reanalysis/ERA5/'
era_1d, era_units = load_1d_data(verif_dir+'era5_annual_1979_2024.nc', era_vname_dict[vname], \
                                region, anom_ref = anom_ref)
print('ERA5 units:', era_units)
if vname == 'pr':
    # convert from m to kg/m2/s
    # 1 mm/day = 1e-3 m/day = 1e-3/86400 m/s = 1.1574e-8 m/s
    # ERA5 'tp' is in meters (total precipitation), annual mean, so convert m/year to kg/m2/s
    # 1 m/year = 1 m / (365.25*24*3600) s = 1 / 31_557_600 m/s
    era_1d = era_1d / (365.25 * 24 * 3600)  # convert m/year to m/s
    era_1d = era_1d * 1000  # convert m/s to kg/m2/s (density of water = 1000 kg/m3)
    era_units = 'kg m$^{-2}$ s$^{-1}$'
era_time = era_1d.time

    

#%% Calc stats with ERA5 over period of overlap

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


#%% Make Figure

fig = plt.figure(figsize=(8,4))
plt.plot(era_time, era_1d, label='ERA5', color='black', linewidth=2)
for i in range(len(recons)):
    plt.plot(recon_time, recon_data[i], label=recon_stats[i], linewidth=2)
plt.legend(loc='lower left', fontsize=12)
plt.xlabel('Year', fontsize=14)
plt.ylabel(vname + ' ()'+ recon_units + ')', fontsize=14)
plt.title('Time Series of {} over {}'.format(vname, region), fontsize=16)
plt.grid(True)

# %%
