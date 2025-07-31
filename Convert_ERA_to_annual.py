#%%
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

#%% 

# Load both datasets
file_dir = '../Data/Reanalysis/ERA5/'
ds1 = xr.open_dataset(file_dir+"era5_monthly_d2m_temp_wind_psl_1979_2024.nc") 
ds2 = xr.open_dataset(file_dir+"era5_monthly_ssrd_strd_tp_1979_2024.nc")


#%% Combine the datasets into one
ds_combined = xr.merge([ds1, ds2])

#%% Group by year and convert to annual values
seconds_in_year = 365 * 24 * 60 * 60  # 31,536,000
annual_groups = ds_combined.groupby("valid_time.year")

# Initialize a dict to hold annual DataArrays
annual_vars = {}
for var in ds_combined.data_vars:
    if var in ["ssrd", "strd"]:
        # Sum over the year, then convert to W/m^2
        annual_sum = annual_groups[var].sum(dim="valid_time", keep_attrs=True)
        annual_vars[var] = annual_sum / seconds_in_year
        annual_vars[var].attrs["units"] = "W m-2"
        annual_vars[var].attrs["note"] = "Converted from J/m^2 by sum over year and divide by seconds in year"
    else:
        # Use mean for other variables
        annual_vars[var] = annual_groups[var].mean(dim="valid_time", keep_attrs=True)

# Combine into a new Dataset
import pandas as pd
annual_years = annual_vars[list(annual_vars.keys())[0]].year.values
annual_time = pd.to_datetime([f"{int(y)}-07-01" for y in annual_years])
ds_annual = xr.Dataset({k: v for k, v in annual_vars.items()})
ds_annual = ds_annual.assign_coords(time=("year", annual_time))
ds_annual = ds_annual.swap_dims({"year": "time"})

#%% Remove empty number dim
if 'number' in ds_annual:
    ds_annual = ds_annual.drop_vars('number')

#%% Save to NetCDF
new_fname = file_dir+"era5_annual_1979_2024.nc"
ds_annual.to_netcdf(new_fname)
print('Saved annual data as ', new_fname)

#%% Rename 'year' dimension to 'time'
ds = xr.open_dataset(new_fname)
ds = ds.rename({'year': 'time'})
ds.to_netcdf(file_dir+"era5_annual_1979_2024_year.nc")
print('Renamed year to time in ', new_fname)

#%% test that annual data looks right
ds = xr.open_dataset(new_fname)
test_var = ds.get('u10')
test_var_year = test_var.sel(year = 2015)
test_var_mean = test_var.mean(dim='year')
test_var_anom = test_var_year - test_var_mean
test_var_anom.plot()

# %%
