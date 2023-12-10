std = '1e-2'
#ncwa -a Time -O Grid_Std_1e-2.nc Grid.nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#data = xr.open_dataset('Grid_Std_' + std + '.nc')
data = xr.open_dataset('Grid.nc')
data = data.where(data.thickness > 1)
Us = np.sqrt(data['velocity 2']**2+data['velocity 1']**2)
Uv = data['velocity 3']

Us_max = Us.max()
Uv_max = Uv.max()

C = data['bottom copt'].mean()

with open("Summary.txt", "w") as file:
    file.write("Us max = %s\nUv max = %s\nCopt mean = %s\n" %(Us_max.values, Uv_max.values, C.values))

list = data['bottom copt'].values.flatten()
flierprops = dict(marker='o', markerfacecolor='green', markersize=5, linestyle='none')
plt.figure(figsize=(15,10))
plt.boxplot(list[~np.isnan(list)], whis=[5,95], patch_artist = True, vert=False, flierprops=flierprops)
plt.xlabel('Copt values')
plt.title('Boxplot of Copt values (whis=1.5IQR)')
plt.savefig('Copt_repart.png')

plt.figure()
plt.imshow(data['bottom copt'])
plt.title('Copt repartition')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig('Copt_map.png')
