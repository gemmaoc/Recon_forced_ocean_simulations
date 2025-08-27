#%%
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#%%

# Load the sparse dataframe pickle file
df = pd.read_pickle('../Data/Proxies/LMRdb_vGKO1_Metadata.df.pckl')

# Check columns for proxy type and lat/lon
# Common column names: 'archive_type', 'lat', 'lon'
proxy_type_col = 'Archive type' 
lat_col = 'Lat (N)'
lon_col = 'Lon (E)'

# Group by proxy type
grouped = df.groupby(proxy_type_col)

#%%
# Set up the map
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Define custom colors for specific proxy types
custom_colors = {
    'Ice Cores': 'tab:cyan',
    'Tree Rings': 'tab:green',
    'Corals and Sclerosponges': 'tab:orange',
    'Bivalve': 'tab:red',
    'Lake Cores': 'tab:purple',
    'Speleothems': 'tab:pink',
    'Marine Cores': 'tab:blue',
}

# Color map for other proxy types
colors = plt.cm.tab10.colors
proxy_types = list(grouped.groups.keys())
color_map = {}
color_idx = 0
for ptype in proxy_types:
    if ptype in custom_colors:
        color_map[ptype] = custom_colors[ptype]
    else:
        color_map[ptype] = colors[color_idx % len(colors)]
        color_idx += 1

# Plot each proxy type
for i, (ptype, group) in enumerate(grouped):
    ax.scatter(group[lon_col], group[lat_col], 
               label=ptype, 
               color=color_map[ptype], 
               s=30, alpha=0.7, edgecolor='k')

plt.legend(title='Proxy Type', loc='lower left', fontsize='small')
plt.title('Global Locations of Proxy Records by Type')
plt.tight_layout()
plt.savefig('../Plots/Proxy_Locations_by_Type.png', dpi=300, bbox_inches='tight')
print('saved fig')
plt.show()
# %%
