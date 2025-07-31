#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 17:55:37 2022

Contains all major functions used in recon-forced analysis. 

@author: gemma
"""

import xarray as xr
import numpy as np
from scipy import stats, signal


#%%

global region_dict
region_dict = {'ASE SB':[-72,-70,245,258],'Nino3.4':[-5,5,190,240],\
               'South Pacific':[-80,-15,100,350],'ase_domain':[-76,-70,245,260]}


#%% LOADING 1D TIME SERIES DATA

def load_1d_data(path, vname, region, time_per = None, anom_ref=None):
    
    """
    Load a gridded dataset from a netCDF file and return data for specified region, variable, 
    and time period. 
    
    If anom_ref is specified, the time_per must include the anom_ref.
    
    Works for 3d datasets with times, lats, and lons or 4d datasets with members
    If only one member, returns 3d array
    If it is an ensemble, returns 4d array
    
    The times in the dataset need to be int or float years (not strings) and labeled time. 
    Designed for loading data from reconstructions, ERA5, ERSSTv5, etc.

    Parameters
    ----------
    path : str
        contains path to netCDF dataset
    vname : str
        variable name as shown in dataset
    region : str
        name of region to average over.
        Options are from directory region_dict
    time_per : list of 2 ints, optional
        start year, stop year of data to select (inclusive)
    anom_ref : list of 2 ints, optional
        list of 2 int years (inclusive). The default is None.
        If defined, data are shifted to specified anom ref period.
        If data are not anomalies, they will be converted to anomalies.

    Returns
    -------
    data_reg: 1d xr data array of floats
        If data array, contains coordinate time
    units: str

    """
    
    # Get location and timing info
    lat1,lat2,lon1,lon2 = region_dict[region]
    
    # Load dataset
    ds = xr.open_dataset(path)
    if time_per:
        start,stop = time_per
        ds = ds.sel(time = slice(start,stop))
    # Slice region based on ordering of lats and name of lat/lon
    try:
        if ds.lat[0] < ds.lat[1]:
            ds = ds.sel(lat = slice(lat1,lat2),\
                        lon=slice(lon1,lon2))
        else:
            ds = ds.sel(lat = slice(lat2,lat1),\
                        lon=slice(lon1,lon2))
        data = ds.get(vname)
        # Remove if it has an unecessary member dimension (keep if more than 1 member)
        if len(data.shape) == 4:
            if 1 in data.shape:
                data = np.squeeze(data) 
        #avg over region
        data_reg = data.mean(dim = 'lat')
        data_reg = data_reg.mean(dim = 'lon')
        
    except:
        print('Exception passed trying to get data called',vname)
        if ds.latitude[0] < ds.latitude[1]:
            ds = ds.sel(latitude = slice(lat1,lat2),\
                        longitude=slice(lon1,lon2))
        else:
            ds = ds.sel(latitude = slice(lat2,lat1),\
                        longitude=slice(lon1,lon2))
        data = ds.get(vname)
        # Remove if it has an unecessary member dimension (keep if more than 1 member)
        if len(data.shape) == 4:
            if 1 in data.shape:
                data = np.squeeze(data) 
        #avg over region
        data_reg = data.mean(dim = 'latitude')
        data_reg = data_reg.mean(dim = 'longitude')
        
    time = ds.time
    
    #put in anomaly space relative to specified period
    if anom_ref:
        
        anom_start,anom_stop = anom_ref
        #calculate mean over new anom ref period and shift data by that amount
        offset = np.mean(data_reg.sel(time = slice(anom_start,anom_stop)))
        data_reg = data_reg - offset

    # Get units from the variable in the dataset
    try:
        units = ds[vname].attrs.get('units', '')
    except Exception:
        units = ''

    return data_reg, units
    


def load_3d_data(path, vname, time_per, anom_ref = None, region = None, \
                 np_array = False):
    """
    Load map of gridded data for specified time period.
    Only handles data from a single member or ensemble mean.
    Option to change anom ref period (this takes a minute).
    
    Note that for changing the anom ref, it assumes the shape is (time, lat, lon)
    Need to update code to accommodate datasets with other shapes.
    

    Parameters
    ----------
    path : str
        path containing .nc file
    vname : str
        name of variable corresponding to .nc file naming
    time_per : list of 2 ints
        start, stop of timeseries (incl.)
    anom_ref : list of 2 ints, optional
        time period over which to remove the mean. The default is None.
    region : str
        Option to specify region. must be a key in the region_dict above.
        Useful for moving the anom ref period so it does it only in the needed region.
    np_array : bool
        if False (default), returns them as xr data arrays
        if True, returns data and times as np arrays

    Returns
    -------
    data: 3d array (xr data array unless specified as np array)
        3d array of data specified with shape n_years, n_lats,nlons
    """

    start,stop = time_per
    # put in anomaly space relative to specified period (at every loc)
    if anom_ref:
        
        
        print('Moving anomaly reference period to', anom_ref, '...')
        ds = xr.open_dataset(path)
        anom_start,anom_stop = anom_ref
        
        try:
            
            # Select region of data if specified
            if region:
                lat1,lat2,lon1,lon2 = region_dict[region]
                ds = ds.sel(lat = slice(lat1,lat2),lon = slice(lon1,lon2))
            #need all years to ensure we include the full anom ref period
            data_all_yrs = ds.get(vname)
            data_all_yrs = np.squeeze(data_all_yrs)
            #copy data with specified years to replace with anom values
            data_anom = data_all_yrs.copy() 
            data_anom = data_anom.sel(time = slice(start,stop))
            nlat = len(ds.lat)
            nlon = len(ds.lon)
            
            for lat_i in range(nlat):
                
                print(f'for lat {lat_i} of {nlat}...')
                
                for lon_j in range(nlon):
                
                    # Calculate mean during ref period in this location
                    data_loc = data_all_yrs.isel(lat = lat_i,lon = lon_j)
                    ref_avg_loc = np.mean(data_loc.sel(time = slice(anom_start, anom_stop)))
                    #Subtract reference mean in this loc from the years specified
                    data_anom_loc = data_loc.sel(time=slice(start,stop)) - ref_avg_loc 
                    data_anom[:,lat_i,lon_j] = data_anom_loc #assumes it follows this order
        except: 
            
            # Select region of data if specified
            if region:
                lat1,lat2,lon1,lon2 = region_dict[region]
                ds = ds.sel(latitude = slice(lat1,lat2),longitude = slice(lon1,lon2))
            #need all years to ensure we include the full anom ref period
            data_all_yrs = ds.get(vname)
            data_all_yrs = np.squeeze(data_all_yrs)
            #copy data with specified years to replace with anom values
            data_anom = data_all_yrs.copy() 
            data_anom = data_anom.sel(time = slice(start,stop))
            nlat = len(ds.latitude)
            nlon = len(ds.longitude)
            
            for lat_i in range(nlat):
                
                print(f'for lat {lat_i+1} of {nlat}...')
                
                for lon_j in range(nlon):
                
                    # Calculate mean during ref period in this location
                    data_loc = data_all_yrs.isel(latitude = lat_i,longitude = lon_j)
                    ref_avg_loc = np.mean(data_loc.sel(time = slice(anom_start, anom_stop)))
                    #Subtract reference mean in this loc from the years specified
                    data_anom_loc = data_loc.sel(time=slice(start,stop)) - ref_avg_loc 
                    data_anom[:,lat_i,lon_j] = data_anom_loc
        
        # select data for given years
        data = data_anom
        
    else:
        ds = xr.open_dataset(path)
        start,stop = time_per
        ds = ds.sel(time=slice(start,stop))
        if region:
                lat1,lat2,lon1,lon2 = region_dict[region]
                ds = ds.sel(lat = slice(lat1,lat2),lon = slice(lon1,lon2))
        data = ds.get(vname)
        data = np.squeeze(data) 
        
    if np_array:
        data = np.array(data)
        
    # put in hPa if SLP
    if vname == 'psl':
        try:
            if data.units == 'Pa' or data.units == 'Pascals':
                data = data/100
        # If units not specified, assume in Pascals
        except:
            data = data/100

    
    return data



def load_1d_data_raw_times(path,var,time_per,region,anom_ref=None):
    
    """
    Load Dalaiden recon timeseries for specified variable in a specified region
    over specified period. Returns as numpy array.
    Separate function to handle different time format.
    Returns SLP in hPa.
    Anom ref period 1941-1990 unless specified.

    Parameters
    ----------
    path : str
     path to file (ending in .nc)
    var : str
        Variable name corresponding to var name in .nc file
        Ex: '850hpa-Uwind','SLP','SAT'
    time_per: list of 2 integers
        Start year, stop year of data to collect (inclusive)
    region : str
        Region over which to average
        Accepted: 'ASE' (see dictionary for locations)
    anom_ref: list with 2 ints, optional
        List with start,stop year for anomaly reference period
        Only needed if want to change ref period from 1941,1990

    Returns
    -------
    time: 1d np array
        Contains times associated with returned data
    data_reg: 1d np array
        1d array of data specified

    """
    
    
    #get timing and loc info
    start,stop = time_per
    lat1,lat2,lon1,lon2 = region_dict[region]
    
    ds = xr.open_dataset(path,decode_times=False)
    ds = ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    data = ds.get(var)
    
    #get time slice via indexing bc error with calendar
    start_idx = start - 1800
    stop_idx = stop - 1800 + 1
    data = data[start_idx:stop_idx] #shape n_years,n_lats,n_lons
    
    #average over region
    data_reg = np.mean(data,axis = 1)
    data_reg = np.mean(data_reg,axis = 1)
    data_reg = np.array(data_reg)
    
    #calculate anomalies relative to specified period
    if anom_ref:
        anom_start,anom_stop = anom_ref
        #calculate mean of data over new anom ref period and shift data by that much
        os = np.mean(data_reg[anom_start-start:anom_stop-start+1])
        data_reg = data_reg - os
    
    if var == 'SLP':
        data_reg = data_reg/100
    
    time = np.linspace(start,stop,stop-start+1)
    
    return time, data_reg



#STATS CALCULATIONS-----------------------------------------------------------

# Calculate correlations between 2 timeseries
def calc_1d_corr(x,y,return_format = 'string'):
    """
    Calculates correlations between 2 numpy arrays x and y. 
    Accounts for atuocorrelation in assessment of significance. 
    Tests for significance with 95% confidence.
    Returns strings. Designed for print statements and text labels in plots. 

    Parameters
    ----------
    x : 1d numpy array
    y : 1d numpy array of same length as x
    return_format : str, optional
        default returns string of corr to 2 decimal places, and '*' for sig p-val
        alternative is a 'float' of corr and p-val

    Returns
    -------
    corr_str : str
        Pearson r correlation with 2 decimal points
    sig : str
        Returns '*' if significance with 95% confidence (p-val<0.05).
        Else ''

    """
    
    #calculate sample correlation
    corr = stats.pearsonr(x,y)[0]
    if return_format == 'string':
        corr_formatted = "{:.2f}".format(corr)
    else:
        corr_formatted = corr
    
    #calculate number of independent obs n_eff
    autocorr_x = abs(stats.pearsonr(x[0:-1],x[1:])[0])
    autocorr_y = abs(stats.pearsonr(y[0:-1],y[1:])[0])
    n = len(x)
    n_eff = n * (1-autocorr_x * autocorr_y)/(1+autocorr_x * autocorr_y)
    
    #do t-test
    dof = n_eff -2
    t_stat = corr * np.sqrt(dof/(1-corr**2))
    p = stats.t.sf(np.abs(t_stat), dof)*2 
    #p_string = "{:.3f}".format(p)
    if p < 0.05:
        sig = '*'
    else:
        sig = ''
    if return_format == 'float':
        sig = p
    
    return corr_formatted, sig

# Calculate coefficient of efficiency between 2 timeseries
def calc_1d_ce(x,v,return_format = 'string'):
    """
    Calculates coefficient of efficiency between a test 1d array and a 
    verification 1d array.

    Parameters
    ----------
    x : 1d numpy array
        Test dataset (i.e., reconstruction)
    v : 1d numpy array of same length as x
        Verification dataset (i.e., ERA5)
    return_format: string, optional
        default is string, other option is 'float'

    Returns
    -------
    ce_str : str or float, depending on return_format
        CE value (if string, with 2 decimal places)

    """
    #x = recon, v = verif
    
    error = v - x
    
    # CE following equation 7 in Hakem et al., 2016
    numer = np.sum(np.power(error,2))
    denom = np.sum(np.power((v-np.mean(v)),2))
    CE    = 1 - (numer/denom)
    
    if return_format == 'string':
        ce_formatted = '{:.2f}'.format(CE)
    elif return_format == 'float':
        ce_formatted = CE
    
    return ce_formatted


def calc_trends_simple(time,data):
    """
    calculatees trends with linear regress
    uses all data points for degrees of freeedom for determining significance
    signficance is simply the p value in the stats linregress package
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(time, data)
    slope_str = '{:.2f}'.format(slope*100)+'/cent'
    p_val_str = '{:.3f}'.format(p_value)
    
    return slope_str,p_val_str


def trend_sig_dof(x,y):
    """
    x = independent var, i.e. time
    y = dependent var, i.e. climate data
    trend calc is same as simple func
    significanec accounts for autocorrelation in samplesize
    uses t-statistic to calculate 95% CI on estimate of slope
    See page 52 of Hartmann Regression Notes (552_Notes_3)


    """
    
    regress = stats.linregress(x,y)
    trend = regress[0]
    intcpt = regress[1]
    fit = intcpt + trend * x
    resids = y - fit
    
    #calc autocorr of resids at lag 1
    r_lag1 = stats.pearsonr(resids[0:-1],resids[1:])[0]
    
    #calculate effective n using bretherton et al., 1999
    n = len(y)
    n_eff = n * ((1-abs(r_lag1)) / (1+abs(r_lag1)))
    
    #calculate t stats and bounds #1-0.025 is 2 tail, 1-0.05 is 1 tail
    t_crit = stats.t.ppf(1-0.025, n_eff-2)
    sd_resid = np.sum(resids**2) / (n_eff-2) #sd=std ev
    sd_trend = sd_resid / (np.sqrt(n_eff) * np.std(x))
    
    conf_int_val = t_crit * sd_trend
    
    return [trend - conf_int_val, trend+conf_int_val], trend, intcpt

# %%
