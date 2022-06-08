import gemini3d.read as read
import importlib  
plotcurv = importlib.import_module("pygemini-scripts.gridoutput.plotcurv")
# from pygemini-scripts.gridoutput.plotcurv import plotcurv3D
from matplotlib.pyplot import show
from gemini3d.grid.gridmodeldata import model2magcoords,model2geogcoords, geog2dipole
from gemini3d.grid.convert import unitvecs_geographic, geog2geomag
import numpy as np
import matplotlib.pyplot as plt
import helpers
from lompe.secsy.secsy import utils as secsy
from scipy.linalg import lstsq
from lompe.ppigrf import igrf
import datetime as dt
import xarray as xr


# load some sample data (3D)
direc = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/gemini_output/"
cfg = read.config(direc)
xg = read.grid(direc)
dims = xg['lx']
# parm="ne"
# var = ["ne", "Ti", "Te", "v1", "v2", "v3", "J1", "J2", "J3", "Phi"]
var = ["v1", "v2", "v3", "Phi"]

parm = var[0]
times = cfg["time"][-1:]


##################3
#SECS description of electric field on top
##############

#Global variables
height = 500 #km. height of secs CS grid
RE = 6371.2 #Earth radius in km


#Define CS grid
grid, grid_ev = helpers.make_csgrid(xg, height=height, crop_factor=0.75, resolution_factor=0.5) #outer (secs), inner (evaluation) grid

#B vectors from model output, project on geo ENU frame
[egalt,eglon,eglat]=unitvecs_geographic(xg)    
Be = np.sum(xg["e1"]*eglon*xg['Bmag'][...,np.newaxis],3)
Bn = np.sum(xg["e1"]*eglat*xg['Bmag'][...,np.newaxis],3)
Bu = np.sum(xg["e1"]*egalt*xg['Bmag'][...,np.newaxis],3)


#Load the data
t = times[0]
dat = read.frame(direc, t, var=var)

#Convert velocity to grographic components, use ENU notation
vu, ve, vn = helpers.model_vec2geo_vec(xg, dat, param='v')


#Resample on spherical shell at specified height inside CS grid domain
galti, gloni, glati, vei = model2geogcoords(xg, ve, 1, dims[2], dims[1], wraplon=True, 
                altlims=(height*1e3-1,height*1e3+1), glatlims=(grid.lat.min(),grid.lat.max()),
                glonlims=(grid.lon.min(),grid.lon.max()))
galti, gloni, glati, vni = model2geogcoords(xg, vn, 1, dims[2], dims[1], wraplon=True, 
                altlims=(height*1e3-1,height*1e3+1), glatlims=(grid.lat.min(),grid.lat.max()),
                glonlims=(grid.lon.min(),grid.lon.max()))
galti, gloni, glati, vui = model2geogcoords(xg, vu, 1, dims[2], dims[1], wraplon=True, 
                altlims=(height*1e3-1,height*1e3+1), glatlims=(grid.lat.min(),grid.lat.max()),
                glonlims=(grid.lon.min(),grid.lon.max()))
galti, gloni, glati, Bei = model2geogcoords(xg, xr.DataArray(Be), 1, dims[2], dims[1],
            wraplon=True, altlims=(height*1e3-1,height*1e3+1),
            glatlims=(grid.lat.min(),grid.lat.max()), 
            glonlims=(grid.lon.min(),grid.lon.max()))
galti, gloni, glati, Bni = model2geogcoords(xg, xr.DataArray(Bn), 1, dims[2], dims[1],
            wraplon=True, altlims=(height*1e3-1,height*1e3+1),
            glatlims=(grid.lat.min(),grid.lat.max()), 
            glonlims=(grid.lon.min(),grid.lon.max()))
galti, gloni, glati, Bui = model2geogcoords(xg, xr.DataArray(Bu), 1, dims[2], dims[1],
            wraplon=True, altlims=(height*1e3-1,height*1e3+1),
            glatlims=(grid.lat.min(),grid.lat.max()), 
            glonlims=(grid.lon.min(),grid.lon.max()))

#Initiate SECS
glatmesh, glonmesh=np.meshgrid(glati,gloni, indexing='xy')
use = np.isfinite(vei.flatten()) & grid.ingrid(glonmesh.flatten(), glatmesh.flatten())
Ge_cf, Gn_cf = secsy.get_SECS_J_G_matrices(glatmesh.flatten()[use], glonmesh.flatten()[use], 
            grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
            RI=6371.2 * 1e3 + height * 1e3, current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
Ge_df, Gn_df = secsy.get_SECS_J_G_matrices(glatmesh.flatten()[use], glonmesh.flatten()[use], 
            grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
            RI=6371.2 * 1e3 + height * 1e3, current_type = 'divergence_free', singularity_limit=grid.Lres*0.5)

#Convert to E-field "measurement" by invoking corss product E = - v x B. ENU components are in geographical, spherical frame.
Ee_model = -(vni.flatten() * Bui.flatten() - vui.flatten() * Bni.flatten())
En_model = (vei.flatten() * Bui.flatten() - vui.flatten() * Bei.flatten())
Eu_model = -(vei.flatten() * Bni.flatten() - vni.flatten() * Bei.flatten())
Ee_model_mesh = -(vni * Bui - vui * Bni)
En_model_mesh = (vei * Bui - vui * Bei)
# Ee_model_meshs = np.concatenate((Ee_model_meshs,Ee_model_mesh[0,:,:,None]), axis=2)
# En_model_meshs = np.concatenate((En_model_meshs,En_model_mesh[0,:,:,None]), axis=2)
# Ve_model_mesh = vei
# Ve_model_meshs = np.concatenate((Ve_model_meshs, Ve_model_mesh[0,:,:,None]), axis=2)
d = np.hstack((Ee_model[use],En_model[use])) #Append to data vector

Gcf = np.vstack((Ge_cf,Gn_cf)) 
Gdf = np.vstack((Ge_df,Gn_df)) 

#Solve
m_cf = lstsq(Gcf, d)[0] 
m_df = lstsq(Gdf, d)[0] 


#Compare SECS representation of convection with input data
plt.subplots(1,1)
# plt.scatter(glonmesh.flatten()[use], glatmesh.flatten()[use], c = Ee_model[use], cmap='seismic', vmin=-0.04, vmax=0.04)
plt.pcolormesh(glonmesh, glatmesh, Ee_model_mesh[0,:,:], cmap='seismic', vmin=-0.04, vmax=0.04)
plt.colorbar()
# plt.quiver(glonmesh.flatten()[use][::19], glatmesh.flatten()[use][::19], Ee_model[use][::19], En_model[use][::19], color='green')
plt.quiver(glonmesh[::9,::9].flatten(), glatmesh[::9,::9].flatten(), Ee_model_mesh[0,:,:][::9,::9].flatten(), En_model_mesh[0,:,:][::9,::9].flatten(), color='green', scale=0.4, width=0.005)
# plt.scatter(glonmesh.flatten()[use], glatmesh.flatten()[use], c = vei[0,:,:].flatten()[use], cmap='seismic', vmin=-1000, vmax=1000)
# plt.pcolormesh(glonmesh, glatmesh, vni[0,:,:], vmin=-1000,vmax=1000, cmap='seismic')
plt.xlim(-4,37)
plt.ylim(65,77)
plt.xlabel('glon')
plt.ylabel('glat')
plt.title('Eastward electric field at 500 km in GEMINI')

#SECS prediction
# grid_ev = helpers.make_csgrid(xg, height=height, crop_factor=0.9)
Ge_, Gn_ = secsy.get_SECS_J_G_matrices(grid_ev.lat.flatten(), grid_ev.lon.flatten(), grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + height * 1e3, current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
G_pred = np.vstack((Ge_, Gn_))
Ecf = G_pred.dot(m_cf)
Ge_, Gn_ = secsy.get_SECS_J_G_matrices(grid_ev.lat.flatten(), grid_ev.lon.flatten(), grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + height * 1e3, current_type = 'divergence_free', singularity_limit=grid.Lres*0.5)
G_pred = np.vstack((Ge_, Gn_))
Edf = G_pred.dot(m_df)
# phi, theta = geog2geomag(grid_ev.lon.flatten(),grid_ev.lat.flatten()) #phi, theta are the magnetic centered dipole coords of the resampled locations
# Be, Bn, Bu = igrf(grid_ev.lon.flatten(), grid_ev.lat.flatten(), height, dtime) # in nT in ENU
Ecf_e = Ecf[0:Ecf.shape[0]//2] 
Ecf_n = Ecf[Ecf.shape[0]//2:]
Edf_e = Edf[0:Edf.shape[0]//2] 
Edf_n = Edf[Edf.shape[0]//2:]

# Ve = (En / (Bu.flatten()*1e-9))# - (Eu / (Bn.flatten()*1e-9))
# Vn = -Ee / (Bu.flatten()*1e-9)


######
#SECS decomposition plot
plt.subplot(1,3,1)
# plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Vn, cmap='seismic', vmin=-1000, vmax=1000)
# plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Ecf_e+Edf_e, cmap='seismic', vmin=-0.04, vmax=0.04)
plt.pcolormesh(grid_ev.lon, grid_ev.lat, (Ecf_e).reshape(grid_ev.lat.shape), cmap='seismic', vmin=-0.04, vmax=0.04)
# plt.quiver()
plt.colorbar()
nn = 3
EEe = Ecf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()# + Edf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
EEn = Ecf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()# + Edf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
plt.quiver(grid_ev.lon[::nn,::nn].flatten(), grid_ev.lat[::nn,::nn].flatten(), EEe, EEn, color='green', scale=0.4, width=0.005)
plt.xlim(-4,37)
plt.ylim(65,77)
plt.xlabel('glon')
plt.ylabel('glat')
plt.title('SECS: CF Eastward electric field at 500 km')

plt.subplot(1,3,2)
# plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Vn, cmap='seismic', vmin=-1000, vmax=1000)
# plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Ecf_e+Edf_e, cmap='seismic', vmin=-0.04, vmax=0.04)
plt.pcolormesh(grid_ev.lon, grid_ev.lat, (Edf_e).reshape(grid_ev.lat.shape), cmap='seismic', vmin=-0.04, vmax=0.04)
# plt.quiver()
plt.colorbar()
nn = 3
EEe = Edf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()# + Edf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
EEn = Edf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()# + Edf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
plt.quiver(grid_ev.lon[::nn,::nn].flatten(), grid_ev.lat[::nn,::nn].flatten(), EEe, EEn, color='green', scale=0.4, width=0.005)
plt.xlim(-4,37)
plt.ylim(65,77)
plt.xlabel('glon')
plt.ylabel('glat')
plt.title('SECS: DF Eastward electric field at 500 km')

plt.subplot(1,3,3)
# plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Vn, cmap='seismic', vmin=-1000, vmax=1000)
# plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Ecf_e+Edf_e, cmap='seismic', vmin=-0.04, vmax=0.04)
plt.pcolormesh(grid_ev.lon, grid_ev.lat, (Ecf_e+Edf_e).reshape(grid_ev.lat.shape), cmap='seismic', vmin=-0.04, vmax=0.04)
# plt.quiver()
plt.colorbar()
nn = 3
EEe = Ecf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten() + Edf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
EEn = Ecf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten() + Edf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
plt.quiver(grid_ev.lon[::nn,::nn].flatten(), grid_ev.lat[::nn,::nn].flatten(), EEe, EEn, color='green', scale=0.4, width=0.005)
plt.xlim(-4,37)
plt.ylim(65,77)
plt.xlabel('glon')
plt.ylabel('glat')
plt.title('SECS: CF+DF Eastward electric field at 500 km')



############
#Variability plots
plt.subplots(1,1)
sss = np.nanmean(Ve_model_meshs[:,:,1:],axis=2)
pcolormesh(glonmesh, glatmesh,sss, vmin=-1000,vmax=1000)
plt.colorbar()
plt.xlabel('glon')
plt.ylabel('glat')
plt.title('Eastward velocity (geo) mean [m/s]')


plt.subplots(1,1)
sss = np.nanstd(Ve_model_meshs[:,:,1:],axis=2)
pcolormesh(glonmesh, glatmesh,sss, vmin=0,vmax=12)
plt.colorbar()
plt.xlabel('glon')
plt.ylabel('glat')
plt.title('Eastward velocity (geo) std [m/s]')









#####################################################################
#Some sample plotting examples from Matt
################################



# these plotting functions will internally grid data
print("Plotting...")
plotcurv.plotcurv3D(xg, dat[parm], cfg, lalt=128, llon=128, llat=128, coord="geographic")


###############################################################################
# produce gridded dataset arrays from model output for user
###############################################################################
lalt=256; llon=128; llat=256;

# regrid data in geographic
print("Sampling in geographic coords...")
galti, gloni, glati, parmgi = model2geogcoords(xg, dat[parm], lalt, llon, llat, wraplon=True)

# regrid in geomagnetic
print("Sampling in geomagnetic coords...")
malti, mloni, mlati, parmmi = model2magcoords(xg, dat[parm], lalt, llon, llat)


# bring up plot
show(block=False)

###############################################################################
# read in a vector quantity, rotate into geographic components and then grid
###############################################################################
v1=dat["v1"]; v2=dat["v2"]; v3=dat["v3"];
[egalt,eglon,eglat]=unitvecs_geographic(xg)    
#^ returns a set of geographic unit vectors on xg; these are in ECEF geomag comps
#    like all other unit vectors in xg

# each of the components in models basis projected onto geographic unit vectors
vgalt=( np.sum(xg["e1"]*egalt,3)*dat["v1"] + np.sum(xg["e2"]*egalt,3)*dat["v2"] + 
    np.sum(xg["e3"]*egalt,3)*dat["v3"] )
vglat=( np.sum(xg["e1"]*eglat,3)*dat["v1"] + np.sum(xg["e2"]*eglat,3)*dat["v2"] +
    np.sum(xg["e3"]*eglat,3)*dat["v3"] )
vglon=( np.sum(xg["e1"]*eglon,3)*dat["v1"] + np.sum(xg["e2"]*eglon,3)*dat["v2"] + 
    np.sum(xg["e3"]*eglon,3)*dat["v3"] )

# must grid each (geographic) vector components separately
print("Sampling vector compotnents in geographic...")
galti, gloni, glati, vgalti = model2geogcoords(xg, vgalt, lalt, llon, llat, wraplon=True)
galti, gloni, glati, vglati = model2geogcoords(xg, vglat, lalt, llon, llat, wraplon=True)
galti, gloni, glati, vgloni = model2geogcoords(xg, vglon, lalt, llon, llat, wraplon=True)

# for comparison also grid the flows in the model coordinate system componennts
galti, gloni, glati, v1i = model2geogcoords(xg, dat["v1"], lalt, llon, llat, wraplon=True)
galti, gloni, glati, v2i = model2geogcoords(xg, dat["v2"], lalt, llon, llat, wraplon=True)
galti, gloni, glati, v3i = model2geogcoords(xg, dat["v3"], lalt, llon, llat, wraplon=True)

# quickly compare flows in model components vs. geographic as a meridional slice
plt.subplots(1,3)

plt.subplot(2,3,1)
plt.pcolormesh(glati,galti,v1i[:,64,:])
plt.xlabel("glat")
plt.ylabel("glon")
plt.title("$v_1$")
plt.colorbar()

plt.subplot(2,3,2)
plt.pcolormesh(glati,galti,v2i[:,64,:])
plt.xlabel("glat")
plt.ylabel("glon")
plt.colorbar()
plt.title("$v_2$")

plt.subplot(2,3,3)
plt.pcolormesh(glati,galti,v3i[:,64,:])
plt.xlabel("glat")
plt.ylabel("glon")
plt.colorbar()
plt.title("$v_3$")

plt.subplot(2,3,4)
plt.pcolormesh(glati,galti,vgalti[:,64,:])
plt.xlabel("glat")
plt.ylabel("glon")
plt.title("$v_r$")
plt.colorbar()

plt.subplot(2,3,5)
plt.pcolormesh(glati,galti,vglati[:,64,:])
plt.xlabel("glat")
plt.ylabel("glon")
plt.colorbar()
plt.title("$v_{mer}$")

plt.subplot(2,3,6)
plt.pcolormesh(glati,galti,vgloni[:,64,:])
plt.xlabel("glat")
plt.ylabel("glon")
plt.colorbar()
plt.title("$v_{zon}$")