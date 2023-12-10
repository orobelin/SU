import pandas as pd
import xarray as xr

data = pd.read_table('SG_velo2001_ESPG3006_table_Adjoint_std.asc', sep=' ', header=None, names=('x','y','Vx','Vy','Stdx','Stdy'))
data = data[data.Vx>0]

data.to_csv('SG_velo2001_ESPG3006_table_Adjoint_std_filter_0.asc', sep=' ', header=None, index=False)


data = xr.open_dataset('SG_Velo2001_ESPG_3006.nc')
Vel = data.velocity

mask = Vel.where(Vel>0)
mask.to_netcdf('SG_velo2001_ESPG3006_table_filter_0.nc')
