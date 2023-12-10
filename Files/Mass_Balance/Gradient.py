import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#data from BolinC
MB = pd.DataFrame(columns=['year', 'ELA', 'db/dz(abla)_mean', 'R2', 'db/dz(abla)_diff', 'R2', 'db/dz(abla)_reg', 'R2', 'db/dz(accu)_mean', 'R2', 'db/dz(accu)_diff', 'R2', 'db/dz(accu)_reg', 'R2', 'Head elevation', 'db/dz(accu)', 'r²', 'db/dz(abla)', 'r²'])
folder_path = 'Annual_MB/'

#data from WGMS
WGMS = pd.read_csv('data/Annual_MB_WGMS.csv', header=0, sep=',', usecols=('YEAR','LOWER_BOUND','UPPER_BOUND',"WINTER_BALANCE","SUMMER_BALANCE",'AREA','ANNUAL_BALANCE'))
WGMS.ANNUAL_BALANCE /= 1000
WGMS.WINTER_BALANCE /= 1000
WGMS.SUMMER_BALANCE /= 1000
MEAN_WGMS = WGMS[WGMS.LOWER_BOUND == 9999]
WGMS = WGMS[WGMS.LOWER_BOUND != 9999]
WGMS['Zmean'] = (WGMS.LOWER_BOUND + WGMS.UPPER_BOUND) / 2
ELA_WGMS = pd.read_csv('data/mass_balance_overview.csv', header=0, sep=',', usecols=('WGMS_ID','YEAR','ELA'))
ELA_WGMS = ELA_WGMS[ELA_WGMS.WGMS_ID == 332]

#Ablation gradients already computed in BolinC
value = {'year':[2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011], 'grad':[0.48, 0.75, 0.71, 0.38, 0.42, 0.71, 0.44, 0.58]}
grad_bolin = pd.DataFrame(value)

#create a file to save dV and dH observed for the simulation post-process
Y_change = pd.DataFrame(columns=['year', 'dV_obs [10⁶ m³]', 'dV_simu [10⁶ m³]', 'S_obs [km²]', 'S_simu [km²]', 'Ba_obs [m.w.e]', 'Ba_simu [m.w.e]', 'diff_S [km²]'])

#Linear regression
Reg = pd.DataFrame(columns=['R2(abla)_1', 'R2(abla)_2', 'R2(abla)_3', 'R2(accu)_1', 'R2(accu)_2', 'R2(accu)_3'])
Reg_2 = pd.DataFrame(columns=['year', 'abla1', 'abla2', 'accu1', 'accu2'])

#Parameters for plot
rows = ['db/dz abla', 'db/dz accu']
columns = ['diff', 'mean']

for year in range(1959,2020):
	if year<2012 and year!= 1960 and year != 1963:
		file = folder_path + "%s" %year + "MBtable.dat"
		data = pd.read_csv(file,header=None,sep='\s+',skiprows=(0,1,2,3,4),usecols=(0,1,2,4,6,7,8))
		data.columns = ("LOWER_BOUND", "UPPER_BOUND", "SURFACE", "SUMMER_BALANCE", "WINTER_BALANCE", "ANNUAL_VOLUME","ANNUAL_BALANCE")
		dV = (data.ANNUAL_VOLUME.iloc[-1]) * 10**(-6)
		S = (data.SURFACE.iloc[-1]) * 10**(-6)
		dH = data.ANNUAL_BALANCE.iloc[-1]
		Y_change.loc[year] = (year, dV, 0, S, 0, dH, 0, 0)

		if year < 1981 or year == 1990 :
			ELA = int((pd.read_csv(file,header=None,sep='\s+',skiprows=(1),nrows=(1),usecols=(1,))).values)
			Head = data.UPPER_BOUND.iloc[-1]
			Bottom = data.LOWER_BOUND.iloc[0]
			Ba = data.ANNUAL_BALANCE.iloc[-1]
			Bw = data.WINTER_BALANCE.iloc[-1]
			Bs = data.SUMMER_BALANCE.iloc[-1]

			data = data[:-1]
			data['Zmean'] = (data.LOWER_BOUND + data.UPPER_BOUND) / 2
			
			#Reshape the dataframe to insure that values in ablation / accumulation area are under / above ELA with a negative / positive balance
			abla_data = data[(data.LOWER_BOUND < ELA) & (data.ANNUAL_BALANCE < 1.0)]
			accu_data = data[(data.LOWER_BOUND > ELA) & (data.ANNUAL_BALANCE > -1.0)]

			#Gradient db / dz in (m w.e) / (100 m) using the mean of the gradient computed between each band and ELA
			grad_abla_mean = round(((abla_data.ANNUAL_BALANCE / (abla_data.LOWER_BOUND - ELA)) * 100).mean(),2)
			grad_accu_mean = round(((accu_data.ANNUAL_BALANCE / (accu_data.UPPER_BOUND - ELA)) * 100).mean(),2)

			#Gradient db / dz in (m w.e) / (100 m) between the point at the upper/lower point of accu/abla area and ELA
			grad_abla_diff = round(((abla_data.ANNUAL_BALANCE.iloc[0]) / (abla_data.LOWER_BOUND.iloc[0] - ELA)) * 100,2)
			grad_accu_diff = round(((accu_data.ANNUAL_BALANCE.iloc[-1]) / (accu_data.UPPER_BOUND.iloc[-1] - ELA)) * 100,2)

		###############2 - Compute the gradients using data from WGMS from 1981 to 2019
	if year>1980:
		data = WGMS[WGMS.YEAR == year]
		ELA = int(ELA_WGMS.ELA[ELA_WGMS.YEAR == year].values)
		Head = data.UPPER_BOUND.iloc[-1]
		Bottom = data.LOWER_BOUND.iloc[0]
		Ba = (MEAN_WGMS.ANNUAL_BALANCE[MEAN_WGMS.YEAR == year]).values
		Bw = (MEAN_WGMS.WINTER_BALANCE[MEAN_WGMS.YEAR == year]).values
		Bs = (MEAN_WGMS.SUMMER_BALANCE[MEAN_WGMS.YEAR == year]).values

		#Reshape the dataframe to insure that values in ablation / accumulation area are under / above ELA with a negative / positive balance
		abla_data = data[(data.LOWER_BOUND < ELA) & (data.ANNUAL_BALANCE < 1000.0)]
		accu_data = data[(data.LOWER_BOUND > ELA) & (data.ANNUAL_BALANCE > -1000.0)]
		#Gradient db / dz in (m w.e) / (100 m) using the mean of the gradient computed between each band and ELA
		grad_abla_mean = round((((abla_data.ANNUAL_BALANCE) / (abla_data.LOWER_BOUND - ELA)) * 100).mean(),2)
		grad_accu_mean = round((((accu_data.ANNUAL_BALANCE) / (accu_data.UPPER_BOUND - ELA)) * 100).mean(),2)
			
		#Gradient db / dz in (m w.e) / (100 m) between the point at the upper/lower point of accu/abla area and ELA
		grad_abla_diff = round(((abla_data.ANNUAL_BALANCE.iloc[0]) / (abla_data.LOWER_BOUND.iloc[0] - ELA)) * 100,2)
		grad_accu_diff = round(((accu_data.ANNUAL_BALANCE.iloc[-1]) / (accu_data.UPPER_BOUND.iloc[-1] - ELA)) * 100,2)

	#linear regression with degree 1, 2 and 3
	#ablation
	R2 = np.zeros([2, 5])
	x = (abla_data.Zmean.values - ELA).reshape(-1,1)
	y = abla_data.ANNUAL_BALANCE.values
	model = LinearRegression(fit_intercept=False).fit(x,y)
	grad_abla_reg = float(model.coef_ * 100)
	R2[0][2] = float(model.score(x, y))

	x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
	model = LinearRegression(fit_intercept=False).fit(x_, y)
	grad_abla_reg2 = model.coef_
	R2[0][3] = float(model.score(x_, y))

	x_ = PolynomialFeatures(degree=3, include_bias=False).fit_transform(x)
	model = LinearRegression(fit_intercept=False).fit(x_, y)
	grad_abla_reg3 = model.coef_
	R2[0][4] = float(model.score(x_, y))

	#accumulation
	x = (accu_data.Zmean.values - ELA).reshape(-1,1)
	y = accu_data.ANNUAL_BALANCE.values
	model = LinearRegression(fit_intercept=False).fit(x,y)
	grad_accu_reg = float(model.coef_ * 100)
	R2[1][2] = float(model.score(x, y))

	x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
	model = LinearRegression(fit_intercept=False).fit(x_, y)
	grad_accu_reg2 = model.coef_
	R2[1][3] = float(model.score(x_, y))

	x_ = PolynomialFeatures(degree=3, include_bias=False).fit_transform(x)
	model = LinearRegression(fit_intercept=False).fit(x_, y)
	grad_accu_reg3 = model.coef_
	R2[1][4] = float(model.score(x_, y))

	#Compute r²
	grad = np.array([[grad_abla_mean, grad_abla_diff, grad_abla_reg], [grad_accu_mean, grad_accu_diff, grad_accu_reg]])
	Z = [abla_data.Zmean.to_numpy(), accu_data.Zmean.to_numpy()]
	yi = [abla_data.ANNUAL_BALANCE.to_numpy(), accu_data.ANNUAL_BALANCE.to_numpy()]
	dbdz = np.zeros(2)
	for i in range (2):
		for j in range (2):
			y = grad[i][j] * ((Z[j] - ELA) / 100)
			y_mean = y.mean()
			SCReg = ((y - y_mean)**2).sum()
			SCRes = ((yi[j] - y)**2).sum()
			SCT = SCReg + SCRes
			R2[i][j] = SCReg / SCT
			#pos = max(enumerate(R2[0]))
			dbdz[i] = grad[i][0]
#dbdz[i] = grad[i][pos[0]]
	R2 = np.around(R2,2)
	MB.loc[year] = [year, ELA, grad_abla_mean, R2[0][0], grad_abla_diff, R2[0][1], grad_abla_reg, R2[0][2], grad_accu_mean, R2[1][0], grad_accu_diff, R2[1][1], grad_accu_reg, R2[1][2], Head, dbdz[0], R2[0].max(), dbdz[1], R2[1].max()]
	Reg.loc[year] = [R2[0][2], R2[0][3], R2[0][4], R2[1][2], R2[1][3], R2[1][4]]
	Reg_2.loc[year] = [year, float(grad_abla_reg2[0]), float(grad_abla_reg2[1]), float(grad_accu_reg2[0]), float(grad_accu_reg2[1])]
		
	#Plot Balance in function of elevation
	plt.figure(figsize=(10,7))
	plt.scatter(data.ANNUAL_BALANCE, data.Zmean, s=0.75, label='Ba', color='blue')
	#plt.scatter(data.WINTER_BALANCE, data.Zmean, s=0.75, label='Bw', color='grey')
	#plt.scatter(data.SUMMER_BALANCE, data.Zmean, s=0.75, label='Bs', color='orange')
	z = np.linspace(1100, 1800, 700)

	#Plot and compute the r² for gradient already computed in bolinC
	if (year>2003) and (year<2012):
		grad = grad_bolin.grad[grad_bolin.year == year].values
		y = grad * ((Z[0] - ELA) / 100)
		y_mean = y.mean()
		SCReg = ((y - y_mean)**2).sum()
		SCRes = ((yi[0] - y)**2).sum()
		SCT = SCReg + SCRes
		R2_B = SCReg / SCT
		for i in range (len(z)):
			if z[i]<ELA:
				Mb[i] = grad * ((z[i]-ELA) / 100)
			else:
				Mb[i] = 0

		plt.plot(Mb, z, label='grad_Bolin (r² = %.2f)' %(R2_B), color='blue')

		#.1 Mean method
	Mb = np.zeros(len(z))
	for i in range (len(z)):
		if z[i]<ELA:
			Mb[i] = (grad_abla_mean) * ((z[i]-ELA) / 100)
		else:
			Mb[i] = (grad_accu_mean) * ((z[i]-ELA) / 100)
	plt.plot(Mb, z, label='grad_mean', color ='red')
		#.2 Diff method
	for i in range (len(z)):
		if z[i]<ELA:
			Mb[i] = (grad_abla_diff) * ((z[i]-ELA) / 100)
		else:
			Mb[i] = (grad_accu_diff) * ((z[i]-ELA) / 100)
	plt.plot(Mb, z, label='grad_diff', color='yellow')

		#.3 Linear regression method
	for i in range (len(z)):
		if z[i]<ELA:
			Mb[i] = (grad_abla_reg) * ((z[i]-ELA) / 100)
		else:
			Mb[i] = (grad_accu_reg) * ((z[i]-ELA) / 100)
	plt.plot(Mb, z, label='grad_reg\nr² abla = %.2f\nr² accu = %.2f' %(R2[0][2], R2[1][2]), color='black')

	plt.axvline(x=0, linewidth=.25, color='blue')
	plt.axhline(y=ELA, linewidth=.25, color='blue')
	plt.axhline(y=Head, linewidth=.75, color='black', linestyle='--',  dashes=(20, 80))
	plt.axhline(y=Bottom, linewidth=.75, color='black', linestyle='dashed',  dashes=(20, 80))
	plt.title('Ba (z)\nyear = %s\nBa = %.2f'%(year, Ba), fontsize = 'xx-large')
	plt.xlim(-5, 5)
	plt.xticks(fontsize='x-large')
	plt.xlabel('Annual Mass Balance (m.w.e)', fontsize = 'xx-large')
	plt.ylim(1100, 1800)
	plt.yticks(fontsize='x-large')
	plt.ylabel('Elevation (m)', fontsize = 'xx-large')
	plt.legend(fontsize = 'xx-large', loc='upper left')
	table = plt.table(cellText=([[grad_abla_diff,grad_abla_mean], [grad_accu_diff,grad_accu_mean]]), rowLabels=['db/dz abla', 'db/dz accu'],colLabels=['diff', 'mean'], loc='lower right', cellLoc='center', rowColours=['grey', 'grey'], colColours=['yellow', 'red'])
	table.scale(0.3, 3)
	table.auto_set_font_size(False)
	table.set_fontsize(15)
	plt.savefig('Plot/MB_%s' %year)
	plt.close()

	plt.figure(figsize=(10,7))
	plt.scatter(data.ANNUAL_BALANCE, data.Zmean, s=0.75, label='Ba', color='blue')
	#.1 deg=1	
	for i in range (len(z)):
		if z[i]<ELA:
			Mb[i] = (grad_abla_reg) * ((z[i]-ELA) / 100)
		else:
			Mb[i] = (grad_accu_reg) * ((z[i]-ELA) / 100)
	plt.plot(Mb, z, label='R1', color='yellow')
	#.2 deg=2
	for i in range (len(z)):
		if z[i]<ELA:
			Mb[i] = grad_abla_reg2[1] * (z[i]-ELA)**2 + grad_abla_reg2[0] * (z[i]-ELA)
		else:
			Mb[i] = grad_accu_reg2[1] * (z[i]-ELA)**2 + grad_accu_reg2[0] * (z[i]-ELA)
	plt.plot(Mb, z, label='R2', color='blue')
	#.2 deg=3
	for i in range (len(z)):
		if z[i]<ELA:
			Mb[i] = grad_abla_reg3[2] * (z[i]-ELA)**3 + grad_abla_reg3[1] * (z[i]-ELA)**2 + grad_abla_reg3[0] * (z[i]-ELA)
		else:
			Mb[i] = grad_accu_reg3[2] * (z[i]-ELA)**3 + grad_accu_reg3[1] * (z[i]-ELA)**2 + grad_accu_reg3[0] * (z[i]-ELA)
	plt.plot(Mb, z, label='R3', color='red')

	plt.axvline(x=0, linewidth=.25, color='blue')
	plt.axhline(y=ELA, linewidth=.25, color='blue')
	plt.axhline(y=Head, linewidth=.75, color='black', linestyle='--',  dashes=(20, 80))
	plt.axhline(y=Bottom, linewidth=.75, color='black', linestyle='dashed',  dashes=(20, 80))
	plt.title('Ba (z)\nyear = %s\nBa = %.2f'%(year, Ba), fontsize = 'xx-large')
	plt.xlim(-5, 5)
	plt.xticks(fontsize='x-large')
	plt.xlabel('Annual Mass Balance (m.w.e)', fontsize = 'xx-large')
	plt.ylim(1100, 1800)
	plt.yticks(fontsize='x-large')
	plt.ylabel('Elevation (m)', fontsize = 'xx-large')
	plt.legend(fontsize = 'xx-large', loc='upper left')
	table = plt.table(cellText=([[R2[0][2],R2[0][3],R2[0][4]], [R2[1][2],R2[1][3],R2[1][4]]]), rowLabels=['db/dz abla', 'db/dz accu'],colLabels=['R1', 'R2', 'R3'], loc='lower right', cellLoc='center', rowColours=['grey', 'grey'], colColours=['yellow', 'blue', 'red'])
	table.scale(0.3, 3)
	table.auto_set_font_size(False)
	table.set_fontsize(15)
	plt.savefig('Plot/Regression/Regression %s' %year)
	plt.close()
	

mean = MB.mean()

#replace missing values
MB['db/dz(abla)'].loc[1960] = mean['db/dz(abla)']
MB['db/dz(abla)'].loc[1963] = mean['db/dz(abla)']
MB['db/dz(accu)'].loc[1960] = mean['db/dz(accu)']
MB['db/dz(accu)'].loc[1963] = mean['db/dz(accu)']

MB.to_csv('OUTPUT/ELA.dat',sep=' ',index=False,header=False,columns=('year','ELA'))
#MB.to_csv('OUTPUT/abla.dat',sep=' ',index=False,header=False,columns=('year','db/dz(abla)'))
#MB.to_csv('OUTPUT/accu.dat',sep=' ',index=False,header=False,columns=('year','db/dz(accu)'))
MB.to_csv('OUTPUT/abla.dat',sep=' ',index=False,header=False,columns=('year','db/dz(abla)_reg'))
MB.to_csv('OUTPUT/accu.dat',sep=' ',index=False,header=False,columns=('year','db/dz(accu)_reg'))
MB.to_csv('OUTPUT/head.dat',sep=' ',index=False,header=False,columns=('year','Head elevation'))

#File for post-process
Y_change.to_csv('OUTPUT/Y_change.dat', sep = ' ', index=False)

#save the file with all the data
mean['year'] = 'resume'
MB.loc[year+1] = mean
MB = MB.round(3)
MB.to_csv('OUTPUT/SMB.csv',sep=',',index=False,header=True)
print(MB)

#File for linear regression
mean = Reg.mean()
Reg.loc[year+1] = mean
Reg = Reg.round(3)
Reg.to_csv('OUTPUT/Reg.csv',sep=',',index=True,header=True)
Reg_2.to_csv('OUTPUT/abla1.dat',sep=' ',index=False,header=False,columns=('year','abla1'))
Reg_2.to_csv('OUTPUT/abla2.dat',sep=' ',index=False,header=False,columns=('year','abla2'))
Reg_2.to_csv('OUTPUT/accu1.dat',sep=' ',index=False,header=False,columns=('year','accu1'))
Reg_2.to_csv('OUTPUT/accu2.dat',sep=' ',index=False,header=False,columns=('year','accu2'))
print(Reg)
