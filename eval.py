from __future__ import print_function
import numpy as np
import numpy.ma as ma 

import matplotlib.pyplot as plt 
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib as mpl

import os, glob 

import pandas as pd 
from netCDF4 import Dataset, num2date, date2num 
import datetime as dt
import time

import matplotlib.pyplot as plt 

from scipy.io import loadmat 
from scipy.interpolate import NearestNDInterpolator as interp_nearest 
from scipy.stats import pearsonr as pcor

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score, r2_score

import hydroeval as he

import joblib

import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
from math import ceil
import matplotlib.ticker as mticker


def remove_keys(dataset):
    for key in ['__header__', '__version__', '__globals__']:
        _ = dataset.pop(key)
    return dataset


def extract_data(path_data, model):

    name_model = model + '_interp_nearest'

    data = loadmat('/'.join([path_data,name_model+'.mat']))

    data = remove_keys(data)

    fechas = []
    for key in data.keys():
        fechas.append(key)

    lat = data['lat']
    lon = data['lon']


    fechas.remove('lat')
    fechas.remove('lon')

    fechas.sort()

    return data, fechas


def combine(models):
    comb = []
    for i in range(len(models)):
        for j in range(i,len(models)):
            if i!=j:
                for k in range(len(models)):
                    if i!=k and j!=k:
                        comb.append([models[i],models[j],models[k]])
    return list(comb)

def save_model(model, path, name):
    joblib.dump(model, '/'.join([path, name+'.pkl']))

def load_model(path, name):
    rf = joblib.load('/'.join([path,name+'.pkl']))
    return rf

import pybaobabdt
def run_train(f_metrics, x_models, y_model, X_train, y_train, X_test, y_test, name_model_rf):

    print('x1_{} x2_{} y_{}\n'.format(x_models[0],x_models[1],y_model).upper())
    print('Training X Shape:', X_train.shape)
    print('Training y Shape:', y_train.shape)
    print('Testing X Shape:', X_test.shape)
    print('Testing y Shape:', y_test.shape)

    modelo = RandomForestRegressor(
                n_estimators = 16,
                criterion    = 'mse',
                max_depth    = 20,
                max_features = 'auto',
                oob_score    = False,
                n_jobs       = -1,
                random_state = 123
             )

    print(modelo.fit(X_train, y_train.ravel()))
    score = modelo.score(X_train,y_train)
    print('score','%.4f'%(score))

    if not os.path.isfile(path_data+'/eval/scores/score_'+name_model_rf+'.npy'):
        np.save(path_data+'/eval/scores/score_'+name_model_rf+'.npy',-1e5)

    old_score = np.load(path_data+'/eval/scores/score_'+name_model_rf+'.npy')
    if score>old_score:
        save_model(model=modelo, path=path_data+'/eval/models', name='modelo_'+name_model_rf+'_entrenado')
        np.save(path_data+'/eval/scores/score_'+name_model_rf+'.npy',score)
        

        print('saved score {}'.format('%.4f'%(score)))

    else:
        print('Trainning did not improve over previous performance. Old score {}'.format('%.4f'%(old_score)))

    rf = modelo.predict(X = X_test)

    importances = list(modelo.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x_models, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    f_metrics.write(' '.join([10*'=','Trainning Random Forest Regressor',10*'=','\n']))

    print(10*'=','Trainning Random Forest Regressor',10*'=')
    print()
    print('explained_variance_score ({},{})       -> {}'.format('rf',y_model,round(explained_variance_score(rf,y_test),2)))
    print('explained_variance_score ({},{})   -> {}'.format(x_models[0],y_model,round(explained_variance_score(X_test[:,0].ravel(),y_test),2)))
    print('explained_variance_score ({},{})    -> {}'.format(x_models[1],y_model,round(explained_variance_score(X_test[:,1].ravel(),y_test),2)))
    print()
    print('  root_mean_sqared_error ({},{})       -> {}'.format('rf',y_model,round(mean_squared_error(rf,y_test,squared=False), 2)))
    print('  root_mean_sqared_error ({},{})   -> {}'.format(x_models[0],y_model,round(mean_squared_error(X_test[:,0].ravel(),y_test,squared=False), 2)))
    print('  root_mean_sqared_error ({},{})    -> {}'.format(x_models[1],y_model,round(mean_squared_error(X_test[:,1].ravel(),y_test,squared=False), 2)))
    print()
    print('     mean_absolute_error ({},{})       -> {}'.format('rf',y_model,round(mean_absolute_error(rf,y_test), 2)))
    print('     mean_absolute_error ({},{})   -> {}'.format(x_models[0],y_model,round(mean_absolute_error(X_test[:,0].ravel(),y_test), 2)))
    print('     mean_absolute_error ({},{})    -> {}'.format(x_models[1],y_model,round(mean_absolute_error(X_test[:,1].ravel(),y_test), 2)))
    print()
    print('                r2_score ({},{})       -> {}'.format('rf',y_model,round(r2_score(rf,y_test),2)))
    print('                r2_score ({},{})   -> {}'.format(x_models[0],y_model,round(r2_score(X_test[:,0].ravel(),y_test),2)))
    print('                r2_score ({},{})    -> {}'.format(x_models[1],y_model,round(r2_score(X_test[:,1].ravel(),y_test),2)))
    print()
    print('                    pcor ({},{})       -> {}'.format('rf',y_model,round(pcor(rf,y_test)[0][0],2)))
    print('                    pcor ({},{})   -> {}'.format(x_models[0],y_model,round(pcor(X_test[:,0].ravel(),y_test)[0][0],2)))
    print('                    pcor ({},{})    -> {}'.format(x_models[1],y_model,round(pcor(X_test[:,1].ravel(),y_test)[0][0],2)))
    print()
    print()
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    print()
    print()

    f_metrics.write(' '.join(['explained_variance_score ({},{})       -> {}'.format('rf',y_model,round(explained_variance_score(rf,y_test),2)),'\n']))
    f_metrics.write(' '.join(['explained_variance_score ({},{})   -> {}'.format(x_models[0],y_model,round(explained_variance_score(X_test[:,0].ravel(),y_test),2)),'\n']))
    f_metrics.write(' '.join(['explained_variance_score ({},{})    -> {}'.format(x_models[1],y_model,round(explained_variance_score(X_test[:,1].ravel(),y_test),2)),'\n']))
    f_metrics.write('\n')
    f_metrics.write(' '.join(['  root_mean_sqared_error ({},{})       -> {}'.format('rf',y_model,round(mean_squared_error(rf,y_test,squared=False), 2)),'\n']))
    f_metrics.write(' '.join(['  root_mean_sqared_error ({},{})   -> {}'.format(x_models[0],y_model,round(mean_squared_error(X_test[:,0].ravel(),y_test,squared=False), 2)),'\n']))
    f_metrics.write(' '.join(['  root_mean_sqared_error ({},{})    -> {}'.format(x_models[1],y_model,round(mean_squared_error(X_test[:,1].ravel(),y_test,squared=False), 2)),'\n']))
    f_metrics.write('\n')
    f_metrics.write(' '.join(['     mean_absolute_error ({},{})       -> {}'.format('rf',y_model,round(mean_absolute_error(rf,y_test), 2)),'\n']))
    f_metrics.write(' '.join(['     mean_absolute_error ({},{})   -> {}'.format(x_models[0],y_model,round(mean_absolute_error(X_test[:,0].ravel(),y_test), 2)),'\n']))
    f_metrics.write(' '.join(['     mean_absolute_error ({},{})    -> {}'.format(x_models[1],y_model,round(mean_absolute_error(X_test[:,1].ravel(),y_test), 2)),'\n']))
    f_metrics.write('\n')
    f_metrics.write(' '.join(['                r2_score ({},{})       -> {}'.format('rf',y_model,round(r2_score(rf,y_test),2)),'\n']))
    f_metrics.write(' '.join(['                r2_score ({},{})   -> {}'.format(x_models[0],y_model,round(r2_score(X_test[:,0].ravel(),y_test),2)),'\n']))
    f_metrics.write(' '.join(['                r2_score ({},{})    -> {}'.format(x_models[1],y_model,round(r2_score(X_test[:,1].ravel(),y_test),2)),'\n']))
    f_metrics.write('\n')
    f_metrics.write(' '.join(['                    pcor ({},{})       -> {}'.format('rf',y_model,round(pcor(rf,y_test)[0][0],2)),'\n']))
    f_metrics.write(' '.join(['                    pcor ({},{})   -> {}'.format(x_models[0],y_model,round(pcor(X_test[:,0].ravel(),y_test)[0][0],2)),'\n']))
    f_metrics.write(' '.join(['                    pcor ({},{})    -> {}'.format(x_models[1],y_model,round(pcor(X_test[:,1].ravel(),y_test)[0][0],2)),'\n']))
    f_metrics.write('\n')


    return modelo


def calc_prom_not_nan(model):
    medias = np.zeros(13)
    c_mes = np.zeros(13)
    for i in range(len(fechas_comunes)):
        mes = int(fechas_comunes[i].split('-')[-1])
        for j in range(model.shape[1]):
            medias[mes] += model[i,j]
            c_mes[mes] += 1

    medias_model = []
    for i in range(1,len(medias)):
        medias_model.append(medias[i]/c_mes[i])

    return medias_model


def plot_marcha_anual(var_x1, var_x2, var_y, var_rf):
    prom_x1 = calc_prom_not_nan(var_x1)
    prom_x2 = calc_prom_not_nan(var_x2)
    prom_y = calc_prom_not_nan(var_y)
    prom_rf = calc_prom_not_nan(var_rf)

    plt.figure(figsize=(14,8))
    plt.plot(np.arange(1,13),prom_x1,color='r',marker='d',label=('x1_'+x_models[0]).upper(),markersize=8,linewidth=3)
    plt.plot(np.arange(1,13),prom_x2,color='b',marker='v',label=('x2_'+x_models[1]).upper(),markersize=8,linewidth=3)
    plt.plot(np.arange(1,13),prom_y,color='g',marker='^',label=('y_'+y_model).upper(),markersize=8,linewidth=3)
    plt.plot(np.arange(1,13),prom_rf,color='k',marker='s',label='RF',markersize=8,linewidth=3)

    plt.xticks(np.arange(1,13))
    plt.ylim([0,400])
    plt.grid()
    plt.legend()
    plt.savefig(path_data+'/eval/plots_marcha/marcha_anual_'+name_model_rf+'.png')
    print('listo marcha_anual_'+name_model_rf+'.png')
    return

def pallet(levels):
    cdict = {'colors': [(255,255,255), (215, 225, 255), (181, 201, 255),
                        (142, 178, 255), (127, 150, 255), (99, 112, 248),
                        (0, 99, 255), (0, 150, 150), (0, 198, 51),
                        (99, 235, 0), (150, 255, 0), (198, 255, 51),
                        (255, 245, 0), (255, 188, 0), (255, 85, 0),
                        (215, 0, 0), (170, 0, 0)],
             # 'clevls': np.linspace(levels[0], levels[1], 50)}
             'clevls': np.arange(levels[0], levels[1], 10)}

    return cdict

def make_cmap(cdict):

        try:
            colors, clevs = cdict['colors'], cdict['clevls']
        except:
            print ('Not a valid cdict')
            return []

        ncmaps = [(float(c[0]) / 255, float(c[1]) / 255, float(c[2]) / 255) for c in colors]

        cmap = mpl.colors.ListedColormap(ncmaps, name='from_list')

        return clevs, cmap


def plot_all(lon,lat,y_true, y_pred, x1, x2, units,fecha,path_out):

    size = 10
    paso_h=1
    cbposition='vertical'
    levels=np.array([0,500])

    palet=pallet(levels)
    clevs, cmap = make_cmap(palet)

    fig,ax = plt.subplots(1,4,figsize=(20,9))

    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                        wspace=0.02, hspace=0.02)


    ax = plt.subplot(141,projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.set_extent([np.min(lon),np.max(lon),np.min(lat),np.max(lat)], crs=ccrs.PlateCarree())

    cf = ax.contourf(lon, lat, x1, levels=clevs,cmap=cmap,transform=ccrs.PlateCarree(),extend='max')
    cbar_ax = fig.add_axes([0.1, 0.15, 0.8, 0.02])
    cb=fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = True
    gl.left_labels = True
    gl.right_labels=False
    gl.xlines = True

    lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)

    gl.xlocator = mticker.FixedLocator(lons)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': size, 'color': 'black'}
    gl.ylabel_style = {'size': size,'color': 'black'}

    ax.set_title(x_models[0].upper(),size=14)


    ax = plt.subplot(142,projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.set_extent([np.min(lon),np.max(lon),np.min(lat),np.max(lat)], crs=ccrs.PlateCarree())

    cf = ax.contourf(lon, lat, x2, levels=clevs,cmap=cmap,transform=ccrs.PlateCarree(),extend='both')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = True
    gl.left_labels = False
    gl.right_labels=False
    gl.xlines = True

    lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)

    gl.xlocator = mticker.FixedLocator(lons)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': size, 'color': 'black'}
    gl.ylabel_style = {'size': size,'color': 'black'}

    ax.set_title(x_models[1].upper(),size=14)


    ax = plt.subplot(143,projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.set_extent([np.min(lon),np.max(lon),np.min(lat),np.max(lat)], crs=ccrs.PlateCarree())

    cf = ax.contourf(lon, lat, y_true, levels=clevs,cmap=cmap,transform=ccrs.PlateCarree(),extend='both')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = True
    gl.left_labels = False
    gl.right_labels=False
    gl.xlines = True

    lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)

    gl.xlocator = mticker.FixedLocator(lons)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': size, 'color': 'black'}
    gl.ylabel_style = {'size': size,'color': 'black'}

    ax.set_title(y_model.upper(),size=14)


    ax = plt.subplot(144,projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.set_extent([np.min(lon),np.max(lon),np.min(lat),np.max(lat)], crs=ccrs.PlateCarree())

    cf = ax.contourf(lon, lat, y_pred, levels=clevs,cmap=cmap,transform=ccrs.PlateCarree(),extend='both')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
    gl.top_labels = True
    gl.left_labels = False
    gl.right_labels=True
    gl.xlines = True

    lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)

    gl.xlocator = mticker.FixedLocator(lons)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': size, 'color': 'black'}
    gl.ylabel_style = {'size': size,'color': 'black'}

    ax.set_title('Random Forest',size=14)



    plt.suptitle(fecha, size=14)

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    plt.savefig('/'.join([path_out,fecha+'.png']))

    plt.delaxes(ax)
    plt.close()

    return


def plot_save_parallel_all(cant, plot=False, save=False):
    for i in cant:
        if plot:
            plot_all(lon,lat,y_true=m_y_val[i,:,:],y_pred=m_rf[i,:,:],x1=m_X_val[i,:,:,0],x2=m_X_val[i,:,:,1],units='mm/month',fecha=fechas_comunes[i],path_out=path_out_full)
            # print(fechas_comunes[i]+'.png')
        if save:
            np.save('/'.join([path_rf,fechas_comunes[i]]),np.array([m_X_val[i,:,:,0],m_X_val[i,:,:,1],m_y_val[i,:,:],m_rf[i,:,:]]))
            # print(fechas_comunes[i]+'.npy')


def save_nc_all_dates(name_model):
    ncfile = Dataset('/'.join([path_data,'eval','nc_models',name_model+'.nc']),mode='w',format='NETCDF4') 

    lon_dim = ncfile.createDimension('lon', 102)
    lat_dim = ncfile.createDimension('lat', 152)
    time_dim = ncfile.createDimension('time', None)

    lon = ncfile.createVariable('lon', np.dtype('double').char, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'


    lat = ncfile.createVariable('lat', np.dtype('double').char, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'

    lat[:] = Lats 
    lon[:] = Lons 

    time = ncfile.createVariable('time', np.dtype('double').char, ('time',))
    time.units = 'days since 1981-01-01'
    time.calendar = 'standard'
    time.long_name = 'time'

    x1 = ncfile.createVariable('x1_'+x_models[0],np.dtype('double').char,('time','lat','lon'), fill_value=-9999., zlib=True)
    x1.units = 'mm/month'
    x1.standard_name = 'precipitation_'+x_models[0].upper()+'_model'

    x2 = ncfile.createVariable('x2_'+x_models[1],np.dtype('double').char,('time','lat','lon'), fill_value=-9999., zlib=True)
    x2.units = 'mm/month'
    x2.standard_name = 'precipitation_'+x_models[1].upper()+'_model'

    y = ncfile.createVariable('y_'+y_model,np.dtype('double').char,('time','lat','lon'), fill_value=-9999., zlib=True)
    y.units = 'mm/month'
    y.standard_name = 'precipitation_'+y_model.upper()+'_model'

    rf = ncfile.createVariable('rf',np.dtype('double').char,('time','lat','lon'), fill_value=-9999., zlib=True)
    rf.units = 'mm/month'
    rf.standard_name = 'precipitation_Random_Forest_model'


    years = []; months = []
    for file in sorted(glob.iglob('/'.join([path_data,'eval','data_models',name_model,'*.npy']))):
        years.append(int(file.split('/')[-1].split('.')[0].split('-')[0]))
        months.append(int(file.split('/')[-1].split('.')[0].split('-')[1]))


    dates = []
    for a,m in zip(years,months):
        dates.append([dt.datetime(a,m,1,0)])

    times = date2num(dates, time.units)
    time[:] = times[:,0]

    data_arr_x1 = []; data_arr_x2 = []; data_arr_y = []; data_arr_rf = []
    for file in sorted(glob.iglob('/'.join([path_data,'eval','data_models',name_model,'*.npy']))):

        file_temp = np.load(file)

        precip_x1 = file_temp[0,:]
        data_arr_x1.append(precip_x1)

        precip_x2 = file_temp[1,:]
        data_arr_x2.append(precip_x2)

        precip_y = file_temp[2,:]
        data_arr_y.append(precip_y)

        precip_rf = file_temp[3,:]
        data_arr_rf.append(precip_rf)


    data_arr_x1 = np.array(data_arr_x1)
    data_arr_x2 = np.array(data_arr_x2)
    data_arr_y = np.array(data_arr_y)
    data_arr_rf = np.array(data_arr_rf)


    x1[:,:,:] = data_arr_x1[:,:,:]
    x2[:,:,:] = data_arr_x2[:,:,:]

    y[:,:,:] = data_arr_y[:,:,:]
    rf[:,:,:] = data_arr_rf[:,:,:]


    ncfile.close()
    print('Dataset {} is closed!'.format(name_model))



def nearest_neighbor(x, y, z, xi, yi):
    f_nearest = interp_nearest(list(zip(x,y)),z)
    return f_nearest(xi,yi)


path_data = '/media/adrianfb/Datos/Guyana-data/dataset'

if not os.path.exists('/'.join([path_data,'eval','data_models'])):
    os.makedirs('/'.join([path_data,'eval','data_models']))

if not os.path.exists('/'.join([path_data,'eval','metrics'])):
    os.makedirs('/'.join([path_data,'eval','metrics']))

if not os.path.exists('/'.join([path_data,'eval','models'])):
    os.makedirs('/'.join([path_data,'eval','models']))

if not os.path.exists('/'.join([path_data,'eval','nc_models'])):
    os.makedirs('/'.join([path_data,'eval','nc_models']))

if not os.path.exists('/'.join([path_data,'eval','plots_marcha'])):
    os.makedirs('/'.join([path_data,'eval','plots_marcha']))

if not os.path.exists('/'.join([path_data,'eval','plots_models'])):
    os.makedirs('/'.join([path_data,'eval','plots_models']))

if not os.path.exists('/'.join([path_data,'eval','scores'])):
    os.makedirs('/'.join([path_data,'eval','scores']))


Lons = np.load('/'.join([path_data,'lon_ref.npy']))
Lats = np.load('/'.join([path_data,'lat_ref.npy']))


lon, lat = np.meshgrid(Lons, Lats)

cdt, fechas_cdt = extract_data(path_data, 'cdt_m')
chirps, fechas_chirps = extract_data(path_data, 'chirps')
mswep, fechas_mswep = extract_data(path_data, 'mswep')
gpcc, fechas_gpcc = extract_data(path_data, 'gpcc')
era5, fechas_era5 = extract_data(path_data, 'era5')
era5bc, fechas_era5bc = extract_data(path_data, 'era5_bc')

gpm, fechas_gpm = extract_data(path_data, 'gpm')

with_gpm = False 

if with_gpm:
    fechas_comunes =  set(fechas_gpcc) & set(fechas_mswep) &  set(fechas_gpm)  
else:
    fechas_comunes =  set(fechas_gpcc) & set(fechas_mswep) & set(fechas_chirps) & set(fechas_cdt) & set(fechas_era5) & set(fechas_era5bc) 



fechas_comunes = list(fechas_comunes)
fechas_comunes.sort()

print(fechas_comunes[0],fechas_comunes[-1])

name_models1 = ['chirps','mswep','gpcc']
name_models2 = ['chirps','mswep','cdt']
name_models3 = ['gpcc','mswep','cdt']
name_models4 = ['chirps','era5','mswep']
name_models5 = ['chirps','era5bc','mswep']
name_models6 = ['era5','cdt','mswep']
name_models7 = ['era5','mswep','gpcc']

name_models8 = ['gpcc','mswep','gpm']



if with_gpm:
    lista_name_models = [name_models8]
else:
    lista_name_models = [name_models1, name_models2, name_models3, name_models4, name_models5, name_models6, name_models7]



comb_full = []
for models in lista_name_models:
    comb = combine(models)
    for c in comb:
        comb_full.append(c)




mask_shape = mswep[fechas_comunes[0]].shape
mask_value_nan = np.nan

mask1 = np.array([chirps[fechas_comunes[0]]==-9999])
mask2 = np.array([era5[fechas_comunes[0]]==-9999])
mask3 = np.array([era5bc[fechas_comunes[0]]==-9999])
mask4 = np.array([cdt[fechas_comunes[0]]==-99])

mask = []
for m1,m2,m3,m4 in zip(mask1.reshape(-1,),mask2.reshape(-1,),mask3.reshape(-1,),mask4.reshape(-1,)):
    if ~m1 and ~m2 and ~m3 and ~m4:
        mask.append(False)
    else:
        mask.append(True)


mask_full = np.array(mask).reshape(mask_shape)

mask = list(mask_full)

cant = len(fechas_comunes)


if with_gpm:
    d_not_nan = []
    for i in range(cant):


        mswep_masked = ma.masked_array(mswep[fechas_comunes[i]],mask=mask)
        mask_mswep = ma.getmask(mswep_masked)
        mswep_masked[mask_mswep] = mask_value_nan
        mswep_masked[~mask_mswep] = mswep[fechas_comunes[i]][~mask_mswep]

        gpcc_masked = ma.masked_array(gpcc[fechas_comunes[i]],mask=mask)
        mask_gpcc = ma.getmask(gpcc_masked)
        gpcc_masked[mask_gpcc] = mask_value_nan
        gpcc_masked[~mask_gpcc] = gpcc[fechas_comunes[i]][~mask_gpcc]

        gpm_masked = ma.masked_array(gpm[fechas_comunes[i]],mask=mask)
        mask_gpm = ma.getmask(gpm_masked)
        gpm_masked[mask_gpm] = mask_value_nan
        gpm_masked[~mask_gpm] = gpm[fechas_comunes[i]][~mask_gpm]



        d_not_nan.append(np.array([mswep_masked[~np.isnan(mswep_masked)],
                                gpcc_masked[~np.isnan(gpcc_masked)],
                                gpm_masked[~np.isnan(gpm_masked)],
                                ]))

else:
    d_not_nan = []
    for i in range(cant):

        cdt_masked = ma.masked_array(cdt[fechas_comunes[i]],mask=mask)
        mask_cdt = ma.getmask(cdt_masked)
        cdt_masked[mask_cdt] = mask_value_nan
        cdt_masked[~mask_cdt] = cdt[fechas_comunes[i]][~mask_cdt]


        chirps_masked = ma.masked_array(chirps[fechas_comunes[i]],mask=mask)
        mask_chirps = ma.getmask(chirps_masked)
        chirps_masked[mask_chirps] = mask_value_nan
        chirps_masked[~mask_chirps] = chirps[fechas_comunes[i]][~mask_chirps]


        era5_masked = ma.masked_array(era5[fechas_comunes[i]],mask=mask)
        mask_era5 = ma.getmask(era5_masked)
        era5_masked[mask_era5] = mask_value_nan
        era5_masked[~mask_era5] = era5[fechas_comunes[i]][~mask_era5]


        era5bc_masked = ma.masked_array(era5bc[fechas_comunes[i]],mask=mask)
        mask_era5bc = ma.getmask(era5bc_masked)
        era5bc_masked[mask_era5bc] = mask_value_nan
        era5bc_masked[~mask_era5bc] = era5bc[fechas_comunes[i]][~mask_era5bc]


        mswep_masked = ma.masked_array(mswep[fechas_comunes[i]],mask=mask)
        mask_mswep = ma.getmask(mswep_masked)
        mswep_masked[mask_mswep] = mask_value_nan
        mswep_masked[~mask_mswep] = mswep[fechas_comunes[i]][~mask_mswep]


        gpcc_masked = ma.masked_array(gpcc[fechas_comunes[i]],mask=mask)
        mask_gpcc = ma.getmask(gpcc_masked)
        gpcc_masked[mask_gpcc] = mask_value_nan
        gpcc_masked[~mask_gpcc] = gpcc[fechas_comunes[i]][~mask_gpcc]


        d_not_nan.append(np.array([cdt_masked[~np.isnan(cdt_masked)],
                                    chirps_masked[~np.isnan(chirps_masked)],
                                    era5_masked[~np.isnan(era5_masked)],
                                    era5bc_masked[~np.isnan(era5bc_masked)],
                                    mswep_masked[~np.isnan(mswep_masked)],
                                    gpcc_masked[~np.isnan(gpcc_masked)],
                                    ]))




d_not_nan = np.array(d_not_nan)
d_not_nan = np.swapaxes(d_not_nan,1,2)

d_not_nan = d_not_nan.reshape(d_not_nan.shape[0]*d_not_nan.shape[1],d_not_nan.shape[2])

indexes = np.arange(d_not_nan.shape[0])
np.random.shuffle(indexes)

train_index = indexes[: int(0.9 * d_not_nan.shape[0])]
val_index = indexes[int(0.9 * d_not_nan.shape[0]) :]

train_dataset = d_not_nan[train_index]
val_dataset = d_not_nan[val_index]

if with_gpm:
    columnas = ['mswep','gpcc','gpm']
else:
    columnas = ['cdt','chirps','era5','era5bc','mswep','gpcc']

df_train = pd.DataFrame(train_dataset, columns=columnas)
df_test = pd.DataFrame(val_dataset, columns=columnas)
df_val = pd.DataFrame(d_not_nan, columns=columnas)


for c in comb_full[0:1]:
    name_model_rf = '_'.join([c[0],c[1],c[2],'rf'])
    x_models, y_model = name_model_rf.split('_')[0:2],name_model_rf.split('_')[-2]
    X_train, y_train = np.array([df_train[x_models[0]],df_train[x_models[1]]]).swapaxes(0,1), np.array([df_train[y_model]]).reshape(-1,1)
    X_test, y_test = np.array([df_test[x_models[0]],df_test[x_models[1]]]).swapaxes(0,1), np.array([df_test[y_model]]).reshape(-1,1)
    # name_model_rf = '_'.join([x_models[0],x_models[1],y_model,'rf'])
    print('name_model_rf ',name_model_rf)
    f_metrics = open(path_data+'/eval/metrics/metrics_'+name_model_rf+'.txt','w')

    if not os.path.isfile(path_data+'/eval/models/modelo_'+name_model_rf+'_entrenado.pkl'):
        modelo = run_train(f_metrics,x_models, y_model, X_train, y_train, X_test, y_test, name_model_rf)
    else:
        modelo = load_model(path=path_data+'/eval/models', name='modelo_'+name_model_rf+'_entrenado')


    print(10*'=','Validation Random Forest Regressor',10*'=')
    f_metrics.write(' '.join([10*'=','Validation Random Forest Regressor',10*'=','\n']))

    X_val, y_val = np.array([df_val[x_models[0]],df_val[x_models[1]]]).swapaxes(0,1), np.array([df_val[y_model]]).reshape(-1,1)
    rf = modelo.predict(X = X_val)

    importances = list(modelo.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x_models, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    print()
    print('explained_variance_score ({},{})       -> {}'.format('rf',y_model,round(explained_variance_score(rf,y_val),2)))
    print('explained_variance_score ({},{})   -> {}'.format(x_models[0],y_model,round(explained_variance_score(X_val[:,0].ravel(),y_val),2)))
    print('explained_variance_score ({},{})    -> {}'.format(x_models[1],y_model,round(explained_variance_score(X_val[:,1].ravel(),y_val),2)))
    print()
    print('  root_mean_sqared_error ({},{})       -> {}'.format('rf',y_model,round(mean_squared_error(rf,y_val,squared=False), 2)))
    print('  root_mean_sqared_error ({},{})   -> {}'.format(x_models[0],y_model,round(mean_squared_error(X_val[:,0].ravel(),y_val,squared=False), 2)))
    print('  root_mean_sqared_error ({},{})    -> {}'.format(x_models[1],y_model,round(mean_squared_error(X_val[:,1].ravel(),y_val,squared=False), 2)))
    print()
    print('     mean_absolute_error ({},{})       -> {}'.format('rf',y_model,round(mean_absolute_error(rf,y_val), 2)))
    print('     mean_absolute_error ({},{})   -> {}'.format(x_models[0],y_model,round(mean_absolute_error(X_val[:,0].ravel(),y_val), 2)))
    print('     mean_absolute_error ({},{})    -> {}'.format(x_models[1],y_model,round(mean_absolute_error(X_val[:,1].ravel(),y_val), 2)))
    print()
    print('                r2_score ({},{})       -> {}'.format('rf',y_model,round(r2_score(rf,y_val),2)))
    print('                r2_score ({},{})   -> {}'.format(x_models[0],y_model,round(r2_score(X_val[:,0].ravel(),y_val),2)))
    print('                r2_score ({},{})    -> {}'.format(x_models[1],y_model,round(r2_score(X_val[:,1].ravel(),y_val),2)))
    print()
    print('                    pcor ({},{})       -> {}'.format('rf',y_model,round(pcor(rf,y_val)[0][0],2)))
    print('                    pcor ({},{})   -> {}'.format(x_models[0],y_model,round(pcor(X_val[:,0].ravel(),y_val)[0][0],2)))
    print('                    pcor ({},{})    -> {}'.format(x_models[1],y_model,round(pcor(X_val[:,1].ravel(),y_val)[0][0],2)))
    print()
    print()
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    print()

    f_metrics.write(' '.join(['explained_variance_score ({},{})       -> {}'.format('rf',y_model,round(explained_variance_score(rf,y_val),2)),'\n']))
    f_metrics.write(' '.join(['explained_variance_score ({},{})   -> {}'.format(x_models[0],y_model,round(explained_variance_score(X_val[:,0].ravel(),y_val),2)),'\n']))
    f_metrics.write(' '.join(['explained_variance_score ({},{})    -> {}'.format(x_models[1],y_model,round(explained_variance_score(X_val[:,1].ravel(),y_val),2)),'\n']))
    f_metrics.write('\n')
    f_metrics.write(' '.join(['  root_mean_sqared_error ({},{})       -> {}'.format('rf',y_model,round(mean_squared_error(rf,y_val,squared=False), 2)),'\n']))
    f_metrics.write(' '.join(['  root_mean_sqared_error ({},{})   -> {}'.format(x_models[0],y_model,round(mean_squared_error(X_val[:,0].ravel(),y_val,squared=False), 2)),'\n']))
    f_metrics.write(' '.join(['  root_mean_sqared_error ({},{})    -> {}'.format(x_models[1],y_model,round(mean_squared_error(X_val[:,1].ravel(),y_val,squared=False), 2)),'\n']))
    f_metrics.write('\n')
    f_metrics.write(' '.join(['     mean_absolute_error ({},{})       -> {}'.format('rf',y_model,round(mean_absolute_error(rf,y_val), 2)),'\n']))
    f_metrics.write(' '.join(['     mean_absolute_error ({},{})   -> {}'.format(x_models[0],y_model,round(mean_absolute_error(X_val[:,0].ravel(),y_val), 2)),'\n']))
    f_metrics.write(' '.join(['     mean_absolute_error ({},{})    -> {}'.format(x_models[1],y_model,round(mean_absolute_error(X_val[:,1].ravel(),y_val), 2)),'\n']))
    f_metrics.write('\n')
    f_metrics.write(' '.join(['                r2_score ({},{})       -> {}'.format('rf',y_model,round(r2_score(rf,y_val),2)),'\n']))
    f_metrics.write(' '.join(['                r2_score ({},{})   -> {}'.format(x_models[0],y_model,round(r2_score(X_val[:,0].ravel(),y_val),2)),'\n']))
    f_metrics.write(' '.join(['                r2_score ({},{})    -> {}'.format(x_models[1],y_model,round(r2_score(X_val[:,1].ravel(),y_val),2)),'\n']))
    f_metrics.write('\n')
    f_metrics.write(' '.join(['                    pcor ({},{})       -> {}'.format('rf',y_model,round(pcor(rf,y_val)[0][0],2)),'\n']))
    f_metrics.write(' '.join(['                    pcor ({},{})   -> {}'.format(x_models[0],y_model,round(pcor(X_val[:,0].ravel(),y_val)[0][0],2)),'\n']))
    f_metrics.write(' '.join(['                    pcor ({},{})    -> {}'.format(x_models[1],y_model,round(pcor(X_val[:,1].ravel(),y_val)[0][0],2)),'\n']))
    f_metrics.write('\n')

    for pair in feature_importances:
        f_metrics.write('Variable: {:20} Importance: {}'.format(*pair)+'\n')


    print()
    f_metrics.write('\n')

    print(30*'=',x_models[0]+'-'+y_model,30*'=')
    f_metrics.write(' '.join([30*'=',x_models[0]+'-'+y_model,30*'=','\n']))

    simulations = X_val[:,0]
    evaluations = y_val

    nse = he.evaluator(he.nse, simulations, evaluations)

    kge, r, alpha, beta = he.evaluator(he.kge, simulations, evaluations)
    kge1, r1, alpha1, beta1 = he.evaluator(he.kgeprime, simulations, evaluations)
    kge2, r2, alpha2, beta2 = he.evaluator(he.kgenp, simulations, evaluations)


    rmse = he.evaluator(he.rmse, simulations, evaluations)
    mare = he.evaluator(he.mare, simulations, evaluations)
    pbias = he.evaluator(he.pbias, simulations, evaluations)
    cc,_ = pcor(simulations, evaluations)


    print('     nse: {}'.format('%.4f'%(nse[0])))
    print('     kge: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge[0]),'%.4f'%(r[0]),'%.4f'%(alpha[0]),'%.4f'%(beta[0])))
    print('kgeprime: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge1[0]),'%.4f'%(r1[0]),'%.4f'%(alpha1[0]),'%.4f'%(beta1[0])))
    print('   kgenp: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge2[0]),'%.4f'%(r2[0]),'%.4f'%(alpha2[0]),'%.4f'%(beta2[0])))
    print('    rmse: {}'.format('%.4f'%(rmse)))
    print('    mare: {}'.format('%.4f'%(mare)))
    print('   pbias: {}'.format('%.4f'%(pbias)))
    print('    pcor: {}'.format('%.4f'%(cc)))


    f_metrics.write('     nse: {}'.format('%.4f'%(nse[0]))+'\n')
    f_metrics.write('     kge: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge[0]),'%.4f'%(r[0]),'%.4f'%(alpha[0]),'%.4f'%(beta[0]))+'\n')
    f_metrics.write('kgeprime: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge1[0]),'%.4f'%(r1[0]),'%.4f'%(alpha1[0]),'%.4f'%(beta1[0]))+'\n')
    f_metrics.write('   kgenp: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge2[0]),'%.4f'%(r2[0]),'%.4f'%(alpha2[0]),'%.4f'%(beta2[0]))+'\n')
    f_metrics.write('    rmse: {}'.format('%.4f'%(rmse))+'\n')
    f_metrics.write('    mare: {}'.format('%.4f'%(mare))+'\n')
    f_metrics.write('   pbias: {}'.format('%.4f'%(pbias))+'\n')
    f_metrics.write('    pcor: {}'.format('%.4f'%(cc))+'\n')


    print()
    f_metrics.write('\n')

    print(30*'=',x_models[1]+'-'+y_model,30*'=')
    f_metrics.write(' '.join([30*'=',x_models[1]+'-'+y_model,30*'=','\n']))

    simulations = X_val[:,1]
    evaluations = y_val

    nse = he.evaluator(he.nse, simulations, evaluations)

    kge, r, alpha, beta = he.evaluator(he.kge, simulations, evaluations)
    kge1, r1, alpha1, beta1 = he.evaluator(he.kgeprime, simulations, evaluations)
    kge2, r2, alpha2, beta2 = he.evaluator(he.kgenp, simulations, evaluations)


    rmse = he.evaluator(he.rmse, simulations, evaluations)
    mare = he.evaluator(he.mare, simulations, evaluations)
    pbias = he.evaluator(he.pbias, simulations, evaluations)
    cc,_ = pcor(simulations, evaluations)


    print('     nse: {}'.format('%.4f'%(nse[0])))
    print('     kge: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge[0]),'%.4f'%(r[0]),'%.4f'%(alpha[0]),'%.4f'%(beta[0])))
    print('kgeprime: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge1[0]),'%.4f'%(r1[0]),'%.4f'%(alpha1[0]),'%.4f'%(beta1[0])))
    print('   kgenp: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge2[0]),'%.4f'%(r2[0]),'%.4f'%(alpha2[0]),'%.4f'%(beta2[0])))
    print('    rmse: {}'.format('%.4f'%(rmse)))
    print('    mare: {}'.format('%.4f'%(mare)))
    print('   pbias: {}'.format('%.4f'%(pbias)))
    print('    pcor: {}'.format('%.4f'%(cc)))


    f_metrics.write('     nse: {}'.format('%.4f'%(nse[0]))+'\n')
    f_metrics.write('     kge: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge[0]),'%.4f'%(r[0]),'%.4f'%(alpha[0]),'%.4f'%(beta[0]))+'\n')
    f_metrics.write('kgeprime: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge1[0]),'%.4f'%(r1[0]),'%.4f'%(alpha1[0]),'%.4f'%(beta1[0]))+'\n')
    f_metrics.write('   kgenp: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge2[0]),'%.4f'%(r2[0]),'%.4f'%(alpha2[0]),'%.4f'%(beta2[0]))+'\n')
    f_metrics.write('    rmse: {}'.format('%.4f'%(rmse))+'\n')
    f_metrics.write('    mare: {}'.format('%.4f'%(mare))+'\n')
    f_metrics.write('   pbias: {}'.format('%.4f'%(pbias))+'\n')
    f_metrics.write('    pcor: {}'.format('%.4f'%(cc))+'\n')


    print()
    f_metrics.write('\n')

    print(30*'=','rf'+'-'+y_model,30*'=')
    f_metrics.write(' '.join([30*'=','rf'+'-'+y_model,30*'=','\n']))

    simulations = rf
    evaluations = y_val

    nse = he.evaluator(he.nse, simulations, evaluations)

    kge, r, alpha, beta = he.evaluator(he.kge, simulations, evaluations)
    kge1, r1, alpha1, beta1 = he.evaluator(he.kgeprime, simulations, evaluations)
    kge2, r2, alpha2, beta2 = he.evaluator(he.kgenp, simulations, evaluations)


    rmse = he.evaluator(he.rmse, simulations, evaluations)
    mare = he.evaluator(he.mare, simulations, evaluations)
    pbias = he.evaluator(he.pbias, simulations, evaluations)
    cc,_ = pcor(simulations, evaluations)


    print('     nse: {}'.format('%.4f'%(nse[0])))
    print('     kge: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge[0]),'%.4f'%(r[0]),'%.4f'%(alpha[0]),'%.4f'%(beta[0])))
    print('kgeprime: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge1[0]),'%.4f'%(r1[0]),'%.4f'%(alpha1[0]),'%.4f'%(beta1[0])))
    print('   kgenp: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge2[0]),'%.4f'%(r2[0]),'%.4f'%(alpha2[0]),'%.4f'%(beta2[0])))
    print('    rmse: {}'.format('%.4f'%(rmse)))
    print('    mare: {}'.format('%.4f'%(mare)))
    print('   pbias: {}'.format('%.4f'%(pbias)))
    print('    pcor: {}'.format('%.4f'%(cc)))


    f_metrics.write('     nse: {}'.format('%.4f'%(nse[0]))+'\n')
    f_metrics.write('     kge: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge[0]),'%.4f'%(r[0]),'%.4f'%(alpha[0]),'%.4f'%(beta[0]))+'\n')
    f_metrics.write('kgeprime: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge1[0]),'%.4f'%(r1[0]),'%.4f'%(alpha1[0]),'%.4f'%(beta1[0]))+'\n')
    f_metrics.write('   kgenp: {}, r: {}, alpha: {}, beta: {}'.format('%.4f'%(kge2[0]),'%.4f'%(r2[0]),'%.4f'%(alpha2[0]),'%.4f'%(beta2[0]))+'\n')
    f_metrics.write('    rmse: {}'.format('%.4f'%(rmse))+'\n')
    f_metrics.write('    mare: {}'.format('%.4f'%(mare))+'\n')
    f_metrics.write('   pbias: {}'.format('%.4f'%(pbias))+'\n')
    f_metrics.write('    pcor: {}'.format('%.4f'%(cc))+'\n')

    f_metrics.close()

    X_val = X_val.reshape(cant,int(X_val.shape[0]/cant),X_val.shape[1])
    y_val = y_val.reshape(cant,int(y_val.shape[0]/cant))
    rf = rf.reshape(cant,int(rf.shape[0]/cant))

    print()
    datos_x1 = X_val[:,:,0]
    datos_x2 = X_val[:,:,1]
    datos_y = y_val
    datos_rf = rf

    plot_marcha_anual(datos_x1,datos_x2,datos_y,datos_rf)
    print()

    m_X_val = np.empty((cant,152,102,X_val.shape[-1]))
    m_y_val = np.empty((cant,152,102))
    m_rf = np.empty((cant,152,102))

    for i in range(cant):

        m_X_val[i,:,:,0][~mask_full] = X_val[i,:,0]
        m_X_val[i,:,:,0][mask_full] = -9999.0


        m_X_val[i,:,:,1][~mask_full] = X_val[i,:,1]
        m_X_val[i,:,:,1][mask_full] = -9999.0


        m_y_val[i,:][~mask_full] = y_val[i,:]
        m_y_val[i,:][mask_full] = -9999.0

        m_rf[i,:][~mask_full] = rf[i,:]
        m_rf[i,:][mask_full] = -9999.0 

    if not os.path.exists('/'.join([path_data,'eval','data_models',name_model_rf])):
        os.makedirs('/'.join([path_data,'eval','data_models',name_model_rf]))

    path_rf = '/'.join([path_data,'eval','data_models',name_model_rf])
    path_out_full = '/'.join([path_data,'eval','plots_models',name_model_rf])


    # Running parallel with frac*cpu_count   0<frac<1
    frac = 1.0
    n_process = None

    # create pool of workers:
    if np.isscalar(n_process):
        n_process = min(1, int(n_process))
    else:
        n_process = int(frac*cpu_count())

    print('Running with {:n} workers.'.format(n_process))
    print()

    pool = Pool(n_process)  

    # running in parallel
    function = partial(plot_save_parallel_all, plot=False, save=True)

    lista = []
    for i in range(cant):
        lista.append([i])


    pool.map(function,lista)

    pool.close()
    pool.join()


    print()
    print('Finished {}'.format(name_model_rf))
    print()

    print()
    save_nc_all_dates(name_model_rf)
    print()





