import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data = xr.open_dataset('wdirs/Transient/Grid.nc')
year_list = data.Time.values

for i in range (4):
	year = year_list[i]
	low = data.thickness[i].where(data['top icymask'][i] == 1.0)
	high = data.thickness[i + 1].where(data['top icymask'][i + 1] == 1.0)

	#Thickness change in m
	dH = high.mean() - low.mean()
	#Area change in km²
	low_area = (low*10*10 - low).sum()
	high_area = (high*10*10 - high).sum()
	dS = (high_area - low_area) * 10**(-6)
	#Volume change in 10⁶ m³
	dV = (((high - low)*10*10).sum()) * 10**(-6)
	#Length change in m
	low = data.x.where(data['top icymask'][i] == 1.0).max()
	high = data.x.where(data['top icymask'][i + 1] == 1.0).max()
	dL = high - low

	with open("Summary.txt", "a") as file:
    		file.write("Year = %s\nVolume change = %s\nSurface change = %s\nLength change = %s\n\n\n" %(year, dV.values, dS.values, dL.values))



