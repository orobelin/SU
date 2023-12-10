C=0.8

import numpy as np
import math
import xarray as xr
import matplotlib.pyplot as plt

#open files
#velocity_obs = xr.open_dataset('../../../data/SG_Velo2001_ESPG_3006.nc')
velocity_obs = xr.open_dataset('SG_Velo2001_ESPG_3006.nc')
#mask = xr.open_dataset('../../../data/Surface_2015.nc')
mask = xr.open_dataset('Surface_2015.nc')
data = xr.open_dataset('Grid_C_%s.nc'%C)

#Max horizontal and vertical velocitites
u_s = data.Band1
#u_s = ((data.xvelsurf**2+data.yvelsurf**2)**(0.5))
#w_s = data.zvelsurf
u_s_max = np.max(u_s.values)
#w_s_maw = np.max(w_s.max.values)


#Difference between obs and model
uobs = velocity_obs.velocity
X = velocity_obs.x
Y = velocity_obs.y

u_s = u_s.interp(x=(X),y=(Y),method="nearest")
diff = uobs - u_s
square = diff**2
MSE = square.mean()
RMSE = MSE**(0.5)


#Compute r²
PCC = xr.corr(u_s, uobs)
R2 = PCC**2

#plot
X=mask.x
Y=mask.y
#Linear regression
plt.figure()
plt.xlabel('u_obs (m/y)')
plt.ylabel('u_s (m/y)')
plt.scatter(uobs,u_s)
plt.plot(uobs,uobs)
plt.text(10, 20, 'r²= %s' % (str.format("{0:.3f}", R2)),fontsize=10,bbox=dict(facecolor='red', alpha=0.5))
#plt.show()
plt.savefig('Regression.png')


#Plot the colormap of thickness difference
	#mask to include the map of Störglacen
#h = np.ma.array(data.thick,mask=data.thick!=0)

plt.figure()
plt.imshow(mask.Band1, extent=(X[0],X[-1],Y[0],Y[-1]),origin='lower',cmap='cool')

plt.title('Difference between observation and model')
diff.plot(cmap='seismic')
plt.colorbar(label='velocity difference')
plt.text(X[1075], Y[-1], 'RMSE= %s' % (str.format("{0:.3f}", RMSE)),fontsize=10,bbox=dict(facecolor='red', alpha=0.5))
plt.xlim(X[0],X[-1])
plt.ylim(Y[0],Y[-1])
#plt.show()
plt.savefig('Colormap.png')


#Eventually boxplot/histogram



