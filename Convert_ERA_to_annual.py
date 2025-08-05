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

#%% Helper function to convert dew point (K) and pressure (Pa) to specific humidity (kg/kg)
def dewpoint_to_specific_humidity(d2m, ps):
    # Constants
    Rd = 287.05  # J/(kg·K)
    Rv = 461.5   # J/(kg·K)
    epsilon = Rd / Rv  # ~0.622
    # Saturation vapor pressure over liquid water (Pa)
    es = 6.112 * np.exp((17.67 * (d2m - 273.15)) / (d2m - 29.65)) * 100  # convert hPa to Pa
    q = epsilon * es / (ps - (1 - epsilon) * es)
    return q

#%% Group by year and convert to annual values
seconds_in_year = 365 * 24 * 60 * 60  # 31,536,000

# Initialize a dict to hold annual DataArrays
annual_vars = {}
for var in ds_combined.data_vars:
    print(var)
    if var in ["ssrd", "strd"]:
        # for radiation, sum over the year, then convert from J/m^2 to W/m^2
        annual_sum = ds_combined[var].groupby("valid_time.year").sum(dim="valid_time", keep_attrs=True)
        annual_vars[var] = annual_sum / seconds_in_year
        annual_vars[var].attrs["units"] = "W m-2"
        annual_vars[var].attrs["note"] = "Converted from J/m^2 by sum over year and divide by seconds in year"
    elif var == "d2m":
        # Will handle d2m separately for specific humidity
        continue
    else:
        # Use mean for other variables
        annual_vars[var] = ds_combined[var].groupby("valid_time.year").mean(dim="valid_time", keep_attrs=True)

# Calculate annual mean specific humidity at 2m
d2m = ds_combined["d2m"]  # [K]
ps = ds_combined["msl"]  # [Pa]
q2m = dewpoint_to_specific_humidity(d2m, ps)
q2m_annual = q2m.groupby("valid_time.year").mean(dim="valid_time", keep_attrs=True)
q2m_annual.name = "q2m"
q2m_annual.attrs["units"] = "kg kg-1"
q2m_annual.attrs["long_name"] = "Specific humidity at 2m"
q2m_annual.attrs["note"] = "Converted from dew point temperature (d2m) and pressure (msl)"
annual_vars["q2m"] = q2m_annual

#%% Combine into a new Dataset
ds_annual = xr.Dataset({k: v for k, v in annual_vars.items()})
ds_annual = ds_annual.rename({'year': 'time'})

# Remove empty number dim
if 'number' in ds_annual:
    ds_annual = ds_annual.drop_vars('number')

#%% Save to NetCDF
new_fname = file_dir+"era5_annual_1979_2024.nc"
ds_annual.to_netcdf(new_fname)
print('Saved annual data as ', new_fname)


#%% test that annual data looks right
ds = xr.open_dataset(new_fname)
test_var = ds.get('tp')
test_var_year = test_var.sel(time = 2015)
test_var_mean = test_var.mean(dim='time')
test_var_anom = test_var_year - test_var_mean
test_var_anom.plot()

# %%
