import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib as mpl
from PIL import Image

y_change = pd.read_csv('Y_change.dat', header=0, sep=' ')
data = xr.open_dataset('Grid_optim.nc')
optim = xr.open_dataset('Grid_optim.nc')
reg1 = xr.open_dataset('Grid_reg_1.nc')
reg2 = xr.open_dataset('Grid_reg_2.nc')
y_change = pd.read_csv('Y_change.dat', header=0, sep=' ')
year_list = (1959, 1969, 1980, 1990, 1999, 2017)
area_list = (3.32, 3.26, 3.23, 3.22, 3.22 ,3.17)

rho = 800 / 1000

Hmax_2015 = 268.97
V_2015 = 291.867
Hmax_2021 = 265.63
V_2021 = 283.785
V_1959 = 341.158


#Plot surface change along the center line
line = pd.read_csv('center_line.csv')
x = line.X
y = line.Y
l = len(line)
length = (x - x.min()).values

list = (optim, reg1, reg2)
name = ('optim', 'R1', 'R2')

#profile
i=0
plt.figure()
for k in (list):
    dZs1 = (k.zs[1] - k.zs[21]).where(k['top icymask'][21] > 0.0)
    dZs2 = (k.zs[21] - k.zs[34]).where(k['top icymask'][34] > 0.0)
    
    interp = dZs1.interp(x=(x), y=(y), method="nearest")
    dZs1 = [interp.values[i][i] for i in range (l)]
    
    plt.scatter(length, dZs1, marker='.', label=name[i])
    i += 1
    
plt.title('Relative change in surface elevation 1961-1982')
plt.xlim(20, 2500)
plt.ylim(-30, 15)
plt.legend()
plt.grid(linestyle='--')
########################plt.show()

i=0
plt.figure()
for k in (list):
    dZs1 = (k.zs[1] - k.zs[21]).where(k['top icymask'][21] > 0.0)
    dZs2 = (k.zs[21] - k.zs[34]).where(k['top icymask'][34] > 0.0)
    
    interp = dZs2.interp(x=(x), y=(y), method="nearest")
    dZs2 = [interp.values[i][i] for i in range (l)]
    
    plt.scatter(length, dZs2, marker='.', label=name[i])
    i += 1
    
plt.title('Relative change in surface elevation 1982-1995')
plt.xlim(200, 2500)
plt.ylim(-30, 15)
plt.legend()
plt.grid(linestyle='--')
########################plt.show()

#colormap
vmin=-15
vmax=15
cmap='RdBu'
Tab = Image.open('table.png')
for k in (list):
    p=1
    fig, ax = plt.subplots(2, 3, figsize=(12,7))
    dZs1 = (k.zs[1] - k.zs[21]).where(k['top icymask'][21] > 0.0)
    dZs2 = (k.zs[21] - k.zs[34]).where(k['top icymask'][34] > 0.0)
    
    interp = dZs1.interp(x=(x), y=(y), method="nearest")
    dZs1 = [interp.values[i][i] for i in range (l)]
    
    interp = dZs2.interp(x=(x), y=(y), method="nearest")
    dZs2 = [interp.values[i][i] for i in range (l)]
    
    ax[0][0].scatter(length, dZs1, marker='.', label='1961-1982')
    ax[0][0].scatter(length, dZs2, marker='.', label='1982-1995')
    ax[0][0].set_title('Relative change in surface elevation')
    ax[0][0].set_xlim(200, 2500)
    ax[0][0].set_ylim(-30, 15)
    ax[0][0].set_ylabel('dH [m]', labelpad=-40)
    ax[0][0].set_xlabel('length [m]', labelpad=-30)
    ax[0][0].legend()
    ax[0][0].grid(linestyle='--')
    for i in range (5):
        year = year_list[i]
        year_range = year - 1959
        year_next = year_list[i+1]
        year_next_range = year_next - 1959
        dZs = (k.zs[year_next_range] - k.zs[year_range]).where(k['top icymask'][year_range] > 0.0)
        
        if p>2:
            j=1
            ax[j][p-3].imshow(dZs, vmin=vmin, vmax=vmax, cmap=cmap)
            ax[j][p-3].set_title('¨%s - %s'%(year, year_next), fontsize='xx-large')
            ax[j][p-3].set_xticklabels([])
            ax[j][p-3].set_yticklabels([])
            ax[j][p-3].text(275, 200, 'dH = %.2f'%(dZs.mean().values))
        else:
            j=0
            ax[j][p].imshow(dZs, vmin=vmin, vmax=vmax, cmap=cmap)
            ax[j][p].set_title('¨%s - %s'%(year, year_next), fontsize='xx-large')
            ax[j][p].set_xticklabels([])
            ax[j][p].set_yticklabels([])
            ax[j][p].text(275, 200, 'dH = %.2f'%(dZs.mean().values))
        p += 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax[1],orientation='horizontal', label='Diff Zs')
    
    #ax[0][0].imshow(Tab)
    #ax[0][0].axis('off')"

