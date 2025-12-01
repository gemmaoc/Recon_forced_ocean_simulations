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
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches


#%% User defined params

# Set climate variable
vname = 'tas' #u10, v10, tas, psl, pr, dlw, dsw, spfh2m

# Time period settings
recon_start,recon_stop = 1800,2005
anom_ref = 1979,2005

# regions are defined in Functions.py
# region = 'ase_domain'
region = 'WAIS'
# region = 'WAIS_ice_cores'

# which recons to plot
# recons = ['cesm2_pace', 'cesm2_lens', 'cesm1_pace','cesm1_lens', 'cesm1-lme']
recons = ['cesm2_pace', 'cesm2_pace_no_ice', 'cesm2_pace_ice_only']

# Set directory location of proxy and reanalysis data. It looks for subdirectories "Proxy_reconstructions" and "Reanalysis/ERA5"
parent_dir = '/Users/gemma/Documents/Data/'


#%% load data 

# get recon data
recon_dir = parent_dir + 'Proxy_reconstructions/'
recon_path_dict = {'cesm2_pace':'CESM2_PAC_PACE_recon_1800_2005/CESM2_PAC_PACE_recon_1800_2005_',
                   'cesm2_lens':'CESM2_LENS_recon_1800_2005/CESM2_LENS_recon_1800_2005_',
                   'cesm1_lens':'CESM1_LENS_recon_1800_2005/CESM1_LENS_recon_1800_2005_',
                   'cesm1-lme':'iCESM1_LME_recon_1800_2005/iCESM1_LME_recon_1800_2005_',
                   'cesm1_pace':'CESM1_PAC_PACE_recon_1800_2005/CESM1_PAC_PACE_recon_1800_2005_',

                   'cesm2_pace_no_ice':'CESM2_PAC_PACE_recon_no_ice_1800_2005/CESM2_PAC_PACE_recon_no_ice_1800_2005_',
                   'cesm2_lens_no_ice':'CESM2_LENS_recon_no_ice_1800_2005/CESM2_LENS_recon_no_ice_1800_2005_',
                   'cesm1_lens_no_ice':'CESM1_LENS_recon_no_ice_1800_2005/CESM1_LENS_recon_no_ice_1800_2005_',
                   'cesm1-lme_no_ice':'iCESM1_LME_recon_no_ice_1800_2005/iCESM1_LME_recon_no_ice_1800_2005_',
                   'cesm1_pace_no_ice':'CESM1_PAC_PACE_recon_no_ice_1800_2005/CESM1_PAC_PACE_recon_no_ice_1800_2005_',

                   'cesm2_pace_ice_only':'CESM2_PAC_PACE_recon_ice_only_1800_2005/CESM2_PAC_PACE_recon_ice_only_1800_2005_',
                   'cesm2_pace_ice_only':'CESM2_PAC_PACE_recon_ice_only_1800_2005/CESM2_PAC_PACE_recon_ice_only_1800_2005_',
                   'cesm1_lens_ice_only':'CESM1_LENS_recon_ice_only_1800_2005/CESM1_LENS_recon_ice_only_1800_2005_',
                   'cesm1-lme_ice_only':'iCESM1_LME_recon_ice_only_1800_2005/iCESM1_LME_recon_ice_only_1800_2005_',
                   'cesm1_pace_ice_only':'CESM1_PAC_PACE_recon_ice_only_1800_2005/CESM1_PAC_PACE_recon_ice_only_1800_2005_',
                   }

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

# %% make a map showing the region 

def normalize_lon_deg_e(lon_deg_e):
    """Convert longitude in degrees East [0..360) to [-180..180) used by cartopy."""
    if lon_deg_e >= 180:
        return lon_deg_e - 360
    return lon_deg_e


def plot_antarctica_region_box(lon_e_min, lon_e_max, lat_min, lat_max, show=True):
    """
    Plot Antarctica and draw a red box over the specified region.
    
    Parameters:
    -----------
    lon_e_min, lon_e_max : float
        Longitude bounds in degrees East
    lat_min, lat_max : float  
        Latitude bounds in degrees
    show : bool
        Whether to display the plot
    """
    # convert longitudes to -180..180
    lon_min = normalize_lon_deg_e(lon_e_min)
    lon_max = normalize_lon_deg_e(lon_e_max)

    # Use a South Polar Stereographic projection
    proj = ccrs.SouthPolarStereo()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Set extent of Antarctica visible
    lon_pad = 5
    lat_north = -65
    ax.set_extent([lon_min - lon_pad, lon_max + lon_pad*2, lat_min - 8, lat_north], crs=ccrs.PlateCarree())

    # Add land and coastline
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f0f0f0')
    ax.coastlines(resolution='50m')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # Draw the red arc box over region
    n_points = 100
    lons = np.linspace(lon_min, lon_max, n_points)
    bottom_lons = lons
    bottom_lats = np.full_like(lons, lat_min)
    top_lons = lons
    top_lats = np.full_like(lons, lat_max)
    left_lats = np.linspace(lat_min, lat_max, n_points)
    left_lons = np.full_like(left_lats, lon_min)
    right_lats = np.linspace(lat_min, lat_max, n_points)
    right_lons = np.full_like(right_lats, lon_max)
    
    # Plot each edge
    ax.plot(bottom_lons, bottom_lats, 'r-', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    ax.plot(top_lons, top_lats, 'r-', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    ax.plot(left_lons, left_lats, 'r-', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    ax.plot(right_lons, right_lats, 'r-', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)

    # Add a small marker at center and annotate
    center_lon = (lon_e_min + lon_e_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0
    center_lon_norm = normalize_lon_deg_e(center_lon)
    ax.text(center_lon_norm, center_lat - 1.0, f"{lon_e_min}°E–{lon_e_max}°E\n{lat_min}°–{lat_max}°", color='red',
            horizontalalignment='center', transform=ccrs.PlateCarree(), fontsize=9)

    # Title
    ax.set_title(f'{region} region', fontsize=16)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# Create the map
lat1, lat2, lon1, lon2 = Functions.region_dict[region]
plot_antarctica_region_box(lon1, lon2, lat1, lat2)



# %%
