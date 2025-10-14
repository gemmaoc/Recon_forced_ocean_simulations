#%%
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

#%% 

# Load both datasets
file_dir = '/Users/gemma/Documents/Data/Reanalysis/ERA5/'
ds1 = xr.open_dataset(file_dir+"era5_monthly_d2m_temp_wind_psl_1979_2024.nc") 
ds2 = xr.open_dataset(file_dir+"era5_monthly_ssrd_strd_tp_1979_2024.nc")

# Replace ds2 timestamps with ds1 timestamps (ignore hours)
ds2 = ds2.assign_coords(valid_time=ds1.valid_time)

#%% Combine the datasets into one
ds_combined = xr.merge([ds1, ds2])

#%% convert dew point (K) and pressure (Pa) to specific humidity (kg/kg)

def specific_humidity_monthly(p_Pa, Td_K):
    # convert celsius to kelvin
    Td_C = Td_K - 273.15

    # saturation vapor pressure at dew point (Pa)
    e = 611.2 * np.exp((17.67 * Td_C) / (Td_C + 243.5)) 
    
    # mixing ratio (kg/kg)
    w = (0.622 * e) / (p_Pa - e)
    
    # specific humidity (kg/kg)
    q = w / (1.0 + w)

    return q

#%% Group by year and convert to annual values (takes a while)
seconds_per_year = 365.25 * 24 * 60 * 60
annual_vars = {}
for var in ["ssrd", "strd"]:#ds_combined.data_vars:
    print(var)
    if var in ["ssrd", "strd"]:
        # for radiation, convert each monthly total (J/m^2) to mean power (W/m^2) by dividing by seconds in month,
        # then average these monthly means over the year
        # Calculate seconds per month for each timestamp
        annual_sum = ds_combined[var].groupby("valid_time.year").sum(dim="valid_time", keep_attrs=True)
        annual_mean = annual_sum / seconds_per_year 
        annual_vars[var] = annual_mean
        annual_vars[var].attrs["units"] = "W m-2"
        annual_vars[var].attrs["note"] = "Converted from J/m^2 by dividing by seconds in month (float), then averaged over year"
    elif var == "d2m":
        # Will handle d2m separately for specific humidity
        continue
    else:
        # Use mean for other variables
        annual_vars[var] = ds_combined[var].groupby("valid_time.year").mean(dim="valid_time", keep_attrs=True)

# Calculate annual mean specific humidity at 2m
d2m_K = ds_combined["d2m"]  # [K]
ps_Pa = ds_combined["msl"]  # [Pa]
q2m_monthly = specific_humidity_monthly(ps_Pa, d2m_K)
q2m_annual = q2m_monthly.groupby("valid_time.year").mean(dim="valid_time", keep_attrs=True)
q2m_annual.name = "q2m"
q2m_annual.attrs["units"] = "kg kg-1"
q2m_annual.attrs["long_name"] = "Specific humidity at 2m"
q2m_annual.attrs["note"] = "Converted from dew point temperature (d2m) and pressure (msl)"
annual_vars["q2m"] = q2m_annual



#%% test plot
q2m_annual.isel(year=1).plot()
plt.title('Annual mean specific humidity at 2m -- test year')
plt.show()

#%% test ASE timeseries: longwave radiation
lat1,lat2,lon1,lon2 = -76,-70,245,260
var_annual = annual_vars['strd']
var_ase = var_annual.sel(latitude=slice(lat2,lat1), longitude=slice(lon1,lon2))
var_ase = var_ase.mean(dim=['latitude','longitude'])
var_ase_anom = var_ase - var_ase.mean(dim='year')
var_ase_anom.plot()
plt.title('annual mean var for ASE domain')
plt.show()


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
test_var = ds.get('strd')
lat1,lat2,lon1,lon2 = -76,-70,245,260
var_ase = test_var.sel(latitude=slice(lat2,lat1), longitude=slice(lon1,lon2))
var_ase = var_ase.mean(dim=['latitude','longitude'])
var_ase_anom = var_ase - var_ase.mean(dim='time')
var_ase_anom.plot()
plt.title('annual mean anomaly for test var over ASE domain')
plt.show()

# %%
