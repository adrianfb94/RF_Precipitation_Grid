from __future__ import print_function
import numpy as np
from netCDF4 import Dataset, num2date
import ast

import nctoolkit as cdo
import pandas as pd 
import glob


from scipy.io import savemat
import numpy.ma as ma

import calendar


def cant_hour_month(year, month):
    Scaling = calendar.monthrange(year=year, month=month)[1]*24
    return Scaling




def agrega_cero(x):
    if x==13:
        return '01'
    if len(str(int(x)))==1:
        return '0'+str(int(x))
    else:
        return str(int(x))

def save_data_model(filename, model, path_dataset):

    if model == 'era5':

        d = {}
        flag_lat_lon = False

        ncfile = cdo.open_data(filename)
        data = ncfile.to_xarray(cdo_times = True)

        LON = data.variables['longitude']
        LAT = data.variables['latitude']

        lon, lat = np.meshgrid(LON,LAT)

        if not flag_lat_lon:
            d['lat'] = lat
            d['lon'] = lon
            flag_lat_lon = True 


        dates = np.array(data.variables['time'])
        dates = pd.to_datetime(dates)
        dates = dates.values.astype('datetime64[s]').tolist()
        dates = np.array(dates)

        pre = np.array(data.variables['tp'])

        for i in range(len(dates)): 
            fecha = '_'.join([str(dates[i]).split(" ")[0],str(dates[i]).split(" ")[1]])
            new_fecha = '-'.join([fecha.split('-')[0],fecha.split('-')[1]])
            print(model,new_fecha)
            mask = np.isnan(pre[i,:])
            pre[i,:][mask] = -9999
            d[new_fecha] = pre[i,:]

        savemat('/'.join([path_dataset,model+'.mat']),d)

    elif model == 'era5_bc':

        d = {}
        flag_lat_lon = False


        ncfile = cdo.open_data(filename)
        data = ncfile.to_xarray(cdo_times = True)


        LON = data.variables['longitude']
        LAT = data.variables['latitude']

        lon, lat = np.meshgrid(LON,LAT)

        if not flag_lat_lon:
            d['lat'] = lat
            d['lon'] = lon
            flag_lat_lon = True 


        dates = np.array(data.variables['time'])
        dates = pd.to_datetime(dates)
        dates = dates.values.astype('datetime64[s]').tolist()
        dates = np.array(dates)

        pre = np.array(data.variables['tp'])

        for i in range(len(dates)): 
            fecha = '_'.join([str(dates[i]).split(" ")[0],str(dates[i]).split(" ")[1]])
            new_fecha = '-'.join([fecha.split('-')[0],fecha.split('-')[1]])
            print(model,new_fecha)
            mask = [pre[i,:]>9e30]
            pre[i,:][mask] = -9999
            d[new_fecha] = pre[i,:]
        savemat('/'.join([path_dataset,model+'.mat']),d)

    elif model == 'mswep_new':

        d = {}
        flag_lat_lon = False

        ncfile = cdo.open_data(filename)
        data = ncfile.to_xarray(cdo_times = True)
        LON = data.variables['lon']
        LAT = data.variables['lat']

        lon, lat = np.meshgrid(LON,LAT)

        if not flag_lat_lon:
            d['lat'] = lat
            d['lon'] = lon
            flag_lat_lon = True 


        dates = np.array(data.variables['time'])
        dates = pd.to_datetime(dates)
        dates = dates.values.astype('datetime64[s]').tolist()
        dates = np.array(dates)

        pre = np.array(data.variables['precip'])


        for i in range(len(dates)):  
            fecha = '_'.join([str(dates[i]).split(" ")[0],str(dates[i]).split(" ")[1]])
            new_fecha = '-'.join([fecha.split('-')[0],fecha.split('-')[1]])
            print(model,new_fecha)
            d[new_fecha] = pre[i,:]
        
        savemat('/'.join([path_dataset,model+'.mat']),d)




    elif model == 'chirps':

        d = {}
        flag_lat_lon = False


        data = Dataset(filename)

        LON = data.variables['longitude']
        LAT = data.variables['latitude']

        lon, lat = np.meshgrid(LON,LAT)

        if not flag_lat_lon:
            d['lat'] = lat
            d['lon'] = lon
            flag_lat_lon = True 


        time = data.variables['time']
        dates = num2date(time, units=time.units)


        pre = data.variables['precip']

        for i in range(len(dates)): 
            fecha = '_'.join([str(dates[i]).split(" ")[0],str(dates[i]).split(" ")[1]])
            new_fecha = '-'.join([fecha.split('-')[0],fecha.split('-')[1]])
            print(model,new_fecha)
            d[new_fecha] = pre[i,:]

        savemat('/'.join([path_dataset,model+'.mat']),d)


    elif model == 'cdt_m':

        d = {}
        flag_lat_lon = False


        data = Dataset(filename)


        LON = data.variables['Lon']
        LAT = data.variables['Lat']

        lon, lat = np.meshgrid(LON,LAT)

        if not flag_lat_lon:
            d['lat'] = lat
            d['lon'] = lon
            flag_lat_lon = True 

        pre = data.variables['precip']

        dates = data.variables['time']
        for i in range(len(dates)):  
            new_fecha = '-'.join([str(dates[i])[:4],str(dates[i])[4:6]])
            d[new_fecha] = pre[i,:]
            print(model,new_fecha)

        savemat('/'.join([path_dataset,model+'.mat']),d)

    elif model == 'gpcc':

        d = {}
        flag_lat_lon = False


        ncfile = cdo.open_data(filename)
        data = ncfile.to_xarray(cdo_times = True)

        LON = data.variables['lon']
        LAT = data.variables['lat']

        lon, lat = np.meshgrid(LON,LAT)

        if not flag_lat_lon:
            d['lat'] = lat
            d['lon'] = lon
            flag_lat_lon = True 


        dates = np.array(data.variables['time'])
        dates = pd.to_datetime(dates)
        dates = dates.values.astype('datetime64[s]').tolist()
        dates = np.array(dates)

        pre = np.array(data.variables['precip'])


        for i in range(len(dates)):
            fecha = '_'.join([str(dates[i]).split(" ")[0],str(dates[i]).split(" ")[1]])
            new_fecha = '-'.join([fecha.split('-')[0],fecha.split('-')[1]])
            print(model,new_fecha)
            d[new_fecha] = pre[i,:]
        
        savemat('/'.join([path_dataset,model+'.mat']),d)

    elif model == 'gpm':
        d = {}
        flag_lat_lon = False

        for infile in sorted(glob.glob(filename+'*.nc')):
            fecha = infile.split('/')[-1].split('_')[-1].split('.')[0]

            data = Dataset(infile)

            LON = data.variables['lons']
            LAT = data.variables['lats']

            lon, lat = np.meshgrid(LON,LAT)

            if not flag_lat_lon:
                d['lat'] = lat
                d['lon'] = lon
                flag_lat_lon = True 

            pre = data.variables['precip'][0,:]

            time = data.variables['times']
            dates = num2date(time, units=time.units)[0]

            new_fecha = str(dates)[:7]
            year = int(new_fecha.split('-')[0])
            month = int(new_fecha.split('-')[1])
            pre = pre*cant_hour_month(year,month)
            d[new_fecha] = pre
            print(model,new_fecha)

        savemat('/'.join([path_dataset,model+'.mat']),d)


path = '/media/adrianfb/Datos/Guyana-data'

path_dataset = '/'.join([path,'dataset'])

filename_era5 = '/'.join([path,'era5/ERA5_precip_monthly.nc'])

filename_era5_bc = '/'.join([path,'era5/era5_10km_corrected75'])

filename_mswep_new = '/'.join([path,'mswep/mswep_monthly-A1981_2020.nc'])

filename_chirps = '/'.join([path,'chirps/guyana_chirps-v2.0.monthly.nc'])

filename_cdt_m = '/'.join([path,'cdt/CDT-dekadal_monthly.nc'])

filename_gpcc = '/'.join([path,'GPCC/gpcc_guyana.1981.2019.nc'])

filename_gpm = '/'.join([path,'gpm/GPM_mensuales/'])


name_models = ['era5','era5_bc','cdt_m', 'chirps','mswep_new', 'gpcc', 'gpm']
folder_models = [filename_era5, filename_era5_bc, filename_cdt_m, filename_chirps, filename_mswep_new, filename_gpcc, filename_gpm]



for i in range(len(name_models)):
    save_data_model(folder_models[i],name_models[i],path_dataset)


