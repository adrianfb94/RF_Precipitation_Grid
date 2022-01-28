from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib as mpl
from scipy.interpolate import Rbf
from scipy.io import loadmat, savemat

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
from math import ceil
import matplotlib.ticker as mticker
import os, glob, math 

from multiprocessing import Pool, cpu_count
from functools import partial

from scipy.interpolate import NearestNDInterpolator as interp_nearest 


def agrega_cero(x):
    if len(str(int(x)))==1:
        return '0'+str(int(x))
    else:
        return str(int(x))


def f_mes(x):
    if x == 13:
        return 1
    else:
        return x

def mes_ant(x):
    if x==1:
        return agrega_cero(12)
    else:
        return(agrega_cero(x-1))

# Distance calculation, degree to km (Haversine method)
def harvesine(lon1, lat1, lon2, lat2):
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return(d)
# ------------------------------------------------------------
# Prediction
def idwr(x, y, z, xi, yi):
    lstxyzi = []
    flag_s = []
    c_flag_s = 0
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (harvesine(x[s], y[s], xi[p], yi[p]))
            if d == 0:
                flag_s.append(s)
            lstdist.append(d)
        if np.min(lstdist) > 0:        
            sumsup = list((1 / np.power(lstdist, 2)))
            suminf = np.sum(sumsup)
            sumsup = np.sum(np.array(sumsup) * np.array(z))
            u = sumsup / suminf
            xyzi = [xi[p], yi[p], u]
            lstxyzi.append(xyzi)
        else:
            xyzi = [xi[p], yi[p], z[flag_s[c_flag_s]]]
            lstxyzi.append(xyzi)
            c_flag_s += 1


    lstxyzi = np.asarray(lstxyzi)

    return lstxyzi[:,2]




def nearest_neighbor(x, y, z, xi, yi):
    f_nearest = interp_nearest(list(zip(x,y)),z)
    return f_nearest(xi,yi)


def remove_keys(dataset):
    for key in ['__header__', '__version__', '__globals__']:
        _ = dataset.pop(key)
    return dataset


def extract_data(path_data, model):


    data = loadmat('/'.join([path_data,model+'.mat']))

    data = remove_keys(data)
    if data['lon'][0,0]>0:
        data['lon'] = data['lon']-360

    fechas = []
    for key in data.keys():
        fechas.append(key)

    fechas.remove('lat')
    fechas.remove('lon')

    fechas.sort()

    return data, fechas


def reshape(lon, lat, rain):

    lon_ref = np.load('lon_ref.npy')
    lat_ref = np.load('lat_ref.npy')
    lon_ref, lat_ref = np.meshgrid(lon_ref, lat_ref)

    Lat_cond = (lat >= lat_ref.min()) & (lat <= lat_ref.max())
    Lon_cond = (lon >= lon_ref.min()) & (lon <= lon_ref.max())
            
    lat_lon = Lat_cond & Lon_cond

    num_lat = np.where(Lat_cond[:,0] == True)
    num_lon = np.where(Lon_cond[0,:] == True)
    

    dim_lat, dim_lon = len(num_lat[0]), len(num_lon[0])
   
    lat = lat[lat_lon].reshape((dim_lat, dim_lon))
    lon = lon[lat_lon].reshape((dim_lat, dim_lon))

    rain = rain[lat_lon].reshape((dim_lat, dim_lon))

    return lon, lat, rain


def pallet(levels):
        cdict = {'colors': [(255,255,255), (215, 225, 255), (181, 201, 255),
                            (142, 178, 255), (127, 150, 255), (99, 112, 248),
                            (0, 99, 255), (0, 150, 150), (0, 198, 51),
                            (99, 235, 0), (150, 255, 0), (198, 255, 51),
                            (255, 245, 0), (255, 188, 0), (255, 85, 0),
                            (215, 0, 0), (170, 0, 0)],
                 'clevls': np.linspace(levels[0], levels[1], 60)}
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



def plot_griddes_precip(lon,lat,data,x, y, z, units,fecha,path_out,model,method, scatter=False):

    levels=np.array([0,1000])

    palet=pallet(levels)
    clevs, cmap = make_cmap(palet)


    fig = plt.figure(figsize=(32, 18))
        
    ax = plt.subplot(111,projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.set_extent([np.min(lon),np.max(lon),np.min(lat),np.max(lat)], crs=ccrs.PlateCarree())


    paso_h=1
    cbposition='vertical'


    cf = ax.contourf(lon, lat, data, levels=clevs,cmap=cmap,transform=ccrs.PlateCarree(),extend='both')
    cb = fig.colorbar(cf, orientation=cbposition, aspect=30,shrink=0.6,pad=0.06)
    cb.set_label(units, size=17)
    cb.ax.tick_params(labelsize=17)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.2, linestyle='--')
    gl.xlines = True

    if scatter:
        plt.scatter(x, y, c=z, marker = 'o',cmap=cmap, linewidths=0.5, edgecolors = 'k', transform = ccrs.PlateCarree(), s=28,alpha=1)



    lons=np.arange(ceil(lon.min()),ceil(lon.max()),paso_h)
    gl.xlocator = mticker.FixedLocator(lons)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 17, 'color': 'black'}
    gl.ylabel_style = {'size': 17,'color': 'black'}

    plt.suptitle('_'.join([fecha,model,method]),size=17)

    if not os.path.exists('/'.join([path_out,'_'.join([model,'interp',method])])):
        os.makedirs('/'.join([path_out,'_'.join([model,'interp',method])]))

    path_save = '/'.join([path_out,'_'.join([model,'interp',method])])
    plt.savefig('/'.join([path_save,'.'.join([fecha,'png'])]))
    plt.close(fig)
    print(model,fecha+method+'.png')


    return 


def get_new_fechas(file, save_acumulates=False):

    station_data = loadmat(file)

    for key in ['__header__', '__version__', '__globals__']:
        _ = station_data.pop(key)

    keys = []
    for key in station_data.keys():
        keys.append(key)

    temp = station_data[keys[0]]

    fechas = []
    for i in range(temp.shape[0]):
        fechas.append(temp[i,3])

    fechas.sort()

    suma = 0
    d_fechas = {}
    d_estaciones = {}
    new_fechas = []
    d = {}
    l = []
    for i in range(len(keys)):
        mes_init = 1
        for j in range(len(fechas)):
            str_fecha = str(int(fechas[j]))
            anho = int(str_fecha[0:4])
            mes = int(str_fecha[4:6])
            dia = int(str_fecha[6:8])
            if mes == mes_init:
                if station_data[keys[i]][j,4]>0:
                    suma += station_data[keys[i]][j,4]



            if mes == f_mes(mes_init+1):
                if mes_init+1==13:
                    new_fecha = '-'.join([str(anho-1),mes_ant(mes)])
                else:
                    new_fecha = '-'.join([str(anho),mes_ant(mes_init+1)])

                if not new_fecha in new_fechas:
                    new_fechas.append(new_fecha)
                x = station_data[keys[i]][j,0] 
                y = station_data[keys[i]][j,1]
                z = suma
                label = keys[i]

                d_fechas[new_fecha] = [x,y,z,label]

                if station_data[keys[i]][j,4]>0:
                    suma = station_data[keys[i]][j,4]


                if mes_init != 12:
                    mes_init += 1
                else:
                    mes_init = 1

        d_estaciones[keys[i]] = d_fechas
        d_fechas = {}

    new_fechas.sort()

    return new_fechas, d_estaciones



def get_dict_stn(new_fechas, d_estaciones, save_acumulates=False):

    dict_stn = {}

    stn = d_estaciones.keys()
    for i in range(len(new_fechas)):
        f, x,y,z,label = [], [], [], [], []

        for s in stn:
            dates = d_estaciones[s].keys()
            for d in dates:
                if d == new_fechas[i]:
                    f.append(d)
                    x.append(d_estaciones[s][d][0])
                    y.append(d_estaciones[s][d][1])
                    z.append(d_estaciones[s][d][2])
                    label.append(d_estaciones[s][d][3])

        dict_stn[new_fechas[i]] = np.asarray([x,y,z]).T

    if save_acumulates:
        savemat('/'.join([path_dataset,'station_raw.mat']),dict_stn)


    return dict_stn


def main(fechas_comunes, plot=False, scatter=False):
    for i in range(len(models)):
        model, name_model = models[i], name_models[i]

        if name_model == 'station': 
            if not os.path.exists('/'.join([path_dataset,'path_temp',name_model,'idw'])):
                os.makedirs('/'.join([path_dataset,'path_temp',name_model,'idw']))
            else:
                pass

            for fecha in fechas_comunes:

                x,y,z = model[fecha][:,0], model[fecha][:,1], model[fecha][:,2]
                x_scatter, y_scatter, z_scatter = x, y, z

                xi, yi = np.load('lon_ref.npy'), np.load('lat_ref.npy')
                nx, ny = len(xi), len(yi)

                xi, yi = np.meshgrid(xi, yi)

                xi, yi = xi.flatten(), yi.flatten()

                grid_idw = idwr(x,y,z,xi,yi)
                grid_idw = grid_idw.reshape((ny, nx))

                xi, yi = xi.reshape((ny, nx)), yi.reshape((ny, nx))

                if plot:

                    plot_griddes_precip(xi,yi,grid_idw,x_scatter, y_scatter, z_scatter, units = "mm/month",fecha=fecha,
                                        path_out=path_out,model=name_model,method='idw',scatter=scatter)
                    

                np.save('/'.join([path_dataset,'path_temp',name_model,'idw',fecha]),[xi,yi,grid_idw])
                print(name_model,fecha+' idw.npy')

        else:
            if not os.path.exists('/'.join([path_dataset,'path_temp',name_model,'nearest'])):
                os.makedirs('/'.join([path_dataset,'path_temp',name_model,'nearest']))
            else:
                pass


            for fecha in fechas_comunes:

                x,y,z = model['lon'], model['lat'], model[fecha]
                x_scatter, y_scatter, z_scatter = reshape(x, y, z)

                x, y, z = reshape(x, y, z)

                x, y, z = x.flatten(), y.flatten(), z.flatten()

                xi, yi = np.load('lon_ref.npy'), np.load('lat_ref.npy')
                nx, ny = len(xi), len(yi)

                xi, yi = np.meshgrid(xi, yi)

                grid_nearest = nearest_neighbor(x,y,z,xi,yi)

                if plot:

                    plot_griddes_precip(xi,yi,grid_nearest,x_scatter, y_scatter, z_scatter, units = "mm/month",fecha=fecha,
                                        path_out=path_out,model=name_model,method='nearest_neighbor',scatter=scatter)
                    


                np.save('/'.join([path_dataset,'path_temp',name_model,'nearest',fecha]),[xi,yi,grid_nearest])
                print(name_model,fecha+' nearest.npy')



    return  


path_dataset = '/media/adrianfb/Datos/Guyana-data/dataset'
path_out = '/media/adrianfb/Datos/Guyana-data/dataset/out_interp'


era5, fechas_era5 = extract_data(path_dataset,'era5')
era5_bc, fechas_era5_bc = extract_data(path_dataset,'era5_bc')
cdt_m, fechas_cdt_m = extract_data(path_dataset, 'cdt_m')
chirps, fechas_chirps = extract_data(path_dataset, 'chirps')
mswep, fechas_mswep = extract_data(path_dataset, 'mswep_new')
gpcc, fechas_gpcc = extract_data(path_dataset, 'gpcc')

gpm, fechas_gpm = extract_data(path_dataset, 'gpm')


with_gpm = False

if with_gpm:
    fechas_comunes =  set(fechas_gpcc) & set(fechas_mswep) & set(fechas_gpm)  
else:
    fechas_comunes =  set(fechas_gpcc) & set(fechas_mswep) & set(fechas_chirps) & set(fechas_cdt_m) & set(fechas_era5) & set(fechas_era5_bc) 


fechas_comunes = list(fechas_comunes)

fechas_comunes.sort()


if with_gpm:
    models = [gpm]
    name_models = ['gpm']
else:
    models = [gpcc, mswep, chirps, cdt_m, era5, era5_bc]
    name_models = ['gpcc', 'mswep', 'chirps','cdt_m', 'era5', 'era5_bc']


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
function = partial(main,plot=False,scatter=False)

lista = []
for i in range(len(fechas_comunes)):
    lista.append([fechas_comunes[i]])


pool.map(function,lista)

pool.close()
pool.join()


print()
print('Finished')

print()
print('Creating the dict_data and .mat file')

for i in range(len(name_models)):
    if name_models[i] == 'station':
        method = 'idw'
    else:
        method = 'nearest'

    path_i = '/'.join([path_dataset,'path_temp',name_models[i],method])
    d = {}
    flag_lat_lon = False
    files = sorted(glob.iglob('/'.join([path_i,'*.npy'])))
    if len(files) != 0:
        for file in files:
            file_temp = np.load(file)
            xi = file_temp[0,:]
            yi = file_temp[1,:]
            grid = file_temp[2,:]
            fecha = file.split('/')[-1].split('.')[0]
            print('Reading ',name_models[i], fecha+'.npy')

            if not flag_lat_lon:
                d['lon'] = xi 
                d['lat'] = yi
                flag_lat_lon = True 

            d[fecha] = grid

        savemat('/'.join([path_dataset,'_'.join([name_models[i],'interp',method+'.mat'])]),d)
        print()
        print(name_models[i],method+'.mat')


        os.system('rm -r ' + path_i)
        print()
        print('Removed folder ',path_i)    







