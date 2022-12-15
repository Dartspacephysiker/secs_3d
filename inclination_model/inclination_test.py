#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:48:08 2022

@author: jone

Investigate the inclination effect in the synthetic dataset that is produced 
with large inclination.

"""

import numpy as np
import matplotlib.pyplot as plt
from secsy import cubedsphere
from secs_3d.gemini_tools import RE
import secs_3d.gemini_tools as gemini_tools
import secs_3d.secs3d as secs3d
from scipy.linalg import lstsq
from secs_3d import visualization
from pysymmetry.utils.spherical import sph_to_car, car_to_sph

    

#Load dataset
# position = (20, 70)
position = (0, 35)
data = np.load('inclination_dataset_sph_'+str(position[0])+'-'+str(position[1])+'.npy', allow_pickle=True).item()
d_grid = data['grid']
ext_factor = 0

#Set up new analysis grid within the domain of the dataset
# First, find the observations from bottom of synthetic dataset, that projects 
# radially onto the original CS grid at the top boundary (d_grid) that the dataset is 
# made from. This selection is used to make a new grid (grid) that should have 
# data all the way to the top.
height = np.min(data['alt'])
# ii, jj = d_grid.bin_index(data['lon'][data['alt']==height],data['lat'][data['alt']==height])
# # i_min = ii.min()
# i_min = min(i for i in ii if i >= 0)
# i_max = ii.max()
# # j_min = jj.min()
# j_min = min(i for i in jj if i >= 0)
# j_max = jj.max()
# xi_min = d_grid.xi_mesh[i_min,j_min]
# xi_max = d_grid.xi_mesh[i_max,j_max]
# eta_min = d_grid.eta_mesh[i_min,j_min]
# eta_max = d_grid.eta_mesh[i_max,j_max]
# crop = 1
# ires = 15
# jres = 2
# grid = cubedsphere.CSgrid(cubedsphere.CSprojection(d_grid.projection.position, 
#             d_grid.projection.orientation), 0, 0, 0, 0, R = (RE+height)*1e3, 
#             edges = (d_grid.xi_mesh[0,j_min:j_max+1][::crop] , 
#                      d_grid.eta_mesh[i_min:i_max+1,0][::crop]))
# grid = cubedsphere.CSgrid(cubedsphere.CSprojection(d_grid.projection.position, 
#             d_grid.projection.orientation), 0, 0, 0, 0, R = (RE+height)*1e3, 
#             edges = (np.linspace(d_grid.xi_mesh[0,j_min],d_grid.xi_mesh[0,j_max+1],jres), 
#             np.linspace(d_grid.eta_mesh[i_min,0], d_grid.eta_mesh[i_max+1,0], ires)))
# L, W, Lres, Wres = 160e3,540e3, 30e3, 30e3 # (160, 340 km, 20, 20 km), L will be in magnetic east-west direction
L, W, Lres, Wres = 160e3,340e3, 20e3, 20e3 # (160, 340 km, 20, 20 km), L will be in magnetic east-west direction
centerlat = data['lat'][-1,:,:].flatten().mean()-0.1#+0.55
centerlat = data['lat'][-1,:,:].flatten().mean()+0.55#0.55
centerlon = data['lon'][-1,:,:].flatten().mean()
grid = cubedsphere.CSgrid(cubedsphere.CSprojection(np.array([centerlon,centerlat]), 
            d_grid.projection.orientation), L, W, Lres, Wres, R = (RE+height)*1e3) 

#Plot grids (input data (blue) and evaluation, green) to see how they match
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(azim=-72, elev=-16)
visualization.spherical_grid(ax, data['lat'], data['lon'], data['alt'], color='blue')
visualization.field_aligned_grid(ax, grid, data['alts_grid'], color='green', fullbox=True)
for kk in range(data['lat'][0,-1,:].size)[::2]:
    visualization.plot_field_line(ax, data['lat'][0,-1,kk], data['lon'][0,-1,kk], 
                              data['alts_grid'])

##############
singularity_limit=d_grid.Lres
alts_grid = np.concatenate((np.arange(100,150,2),np.arange(150,190,4),np.arange(190, 
                                    np.max(data['alt'])+40,20)))
# alts_grid = data['alts_grid']
altres = np.diff(alts_grid)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))

#Grid dimensions
K = alts_grid.shape[0] #Number of vertival layers
I = grid.shape[0] #Number of cells in eta direction
J = grid.shape[1]  #Number of cells in xi direction 

#Prepare the synthetic dataset for inversion, based on jperp
jperp = True
crop = 3
if jperp:
    data['je'] = data['jperp_phi'][:,::crop,::crop]
    data['jn'] = -data['jperp_theta'][:,::crop,::crop]
    data['ju'] = data['jperp_r'][:,::crop,::crop]
    savestr = '_jperp'
else:
    data['je'] = data['jphi'][:,::crop,::crop]
    data['jn'] = -data['jtheta'][:,::crop,::crop]
    data['ju'] = data['jr'][:,::crop,::crop]
    savestr = '_fullj'
data['lat'] = data['lat'][:,::crop,::crop]
data['lon'] = data['lon'][:,::crop,::crop]
data['alt'] = data['alt'][:,::crop,::crop]
data['fac'] = data['fac'][:,::crop,::crop]
data['Be'] = data['B'][2,:,::crop,::crop]
data['Bn'] = -data['B'][1,:,::crop,::crop]
data['Bu'] = data['B'][0,:,::crop,::crop]
#Set jperp=False in below line since our dataset is already jperp
# jj, lat, lon, alt = gemini_tools.prepare_model_data(grid, data, alts_grid, 
                        # jperp=False, ext_factor=ext_factor)
# d = np.hstack((jj[0], jj[1], jj[2])) # (r, theta, phi components)
d = np.hstack((data['ju'].flatten(), -data['jn'].flatten(), data['je'].flatten())) # (r, theta, phi components)
G = secs3d.make_G(grid, alts_grid, data['lat'], data['lon'], data['alt'], 
                  ext_factor=ext_factor)
# br, btheta, bphi = secs3d.make_b_unitvectors(jj[3],jj[4],jj[5])
br, btheta, bphi = secs3d.make_b_unitvectors(data['Bu'].flatten(), 
                            -data['Bn'].flatten(),data['Be'].flatten())
B = secs3d.make_B(br, btheta, bphi)    
P = secs3d.make_P(br.size)
G = P.T.dot(B.dot(P.dot(G)))
GTG = G.T.dot(G)
R = np.diag(np.ones(GTG.shape[0])) * np.median(np.diag(GTG)) * 0.01
GTd = G.T.dot(d)

#Solve
m = lstsq(GTG + R, GTd, cond=0.)[0]
keep = {'data':data, 'm':m}
np.save('m'+savestr+'_'+str(position[0])+'-'+str(position[1])+'.npy', keep)

###########################################

#Evaluate model
# position = (20, 70)
position = (0, 35)
jperp = True
if jperp:
    savestr='_jperp'
else:
    savestr='_fullj'
m_ = np.load('m'+savestr+'_'+str(position[0])+'-'+str(position[1])+'.npy', allow_pickle=True).item()
m = m_['m']
try:
    data = m_['data'].item()
except:
    data = m_['data']
d = np.hstack((data['ju'].flatten(), -data['jn'].flatten(), data['je'].flatten())) # (r, theta, phi components)


#Make G to evaluate for full j based on the model made above
G = secs3d.make_G(grid, alts_grid, data['lat'][:,:,:], data['lon'][:,:,:], 
                  data['alt'][:,:,:], ext_factor=ext_factor)
full_j = G.dot(m)

#Get the jperp of the ful_j expressed by the model, to be compared to input
br, btheta, bphi = secs3d.make_b_unitvectors(data['Bu'][:,:,:].flatten(), 
                            -data['Bn'][:,:,:].flatten(), data['Be'][:,:,:].flatten())
N = br.size
B = secs3d.make_B(br, btheta, bphi)
P = secs3d.make_P(N)
j_perp = P.T.dot(B.dot(P.dot(full_j)))
jpar = np.sum(np.array([full_j[0:N], full_j[N:2*N], full_j[2*N:3*N]]) * 
                np.array([br, btheta, bphi]), axis=0)
residual_r = d[0:N]-j_perp[0:N]
residual_theta = d[N:2*N]-j_perp[N:2*N]
residual_phi = d[2*N:3*N]-j_perp[2*N:3*N]
data_fac = data['fac'][:,:,:].flatten()
residual_fac = data_fac - jpar

###########################
#Visualize performance
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
ax.view_init(azim=-24, elev=-26)
visualization.spherical_grid(ax, data['lat'], data['lon'], data['alt'], color='blue')
# visualization.field_aligned_grid(ax, grid, data['alts_grid'], color='green')
for kk in range(data['lat'][0,-1,:].size):
    visualization.plot_field_line(ax, data['lat'][0,-1,kk], data['lon'][0,-1,kk], 
                              data['alts_grid'])
cmap = plt.cm.seismic
import matplotlib
clim = 1e-8
vmin = -clim
vmax = clim
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
shape = data['alt'][:,:,:].shape
# param = data_fac.reshape(shape)# - jpar.reshape(shape)
ax.set_title('Modelled fac, jperp input')
ax.set_title('True fac')
param = jpar.reshape(shape)
# param = full_j[2*N:3*N].reshape(shape)
x, y, z = sph_to_car((RE+data['alt'].flatten(), 90-data['lat'].flatten(), 
                      data['lon'].flatten()), deg=True)
sss = [2]
for ss in sss:
    p = ax.plot_surface(x.reshape(shape)[:,:,ss], y.reshape(shape)[:,:,ss], 
                        z.reshape(shape)[:,:,ss], alpha=0.7,
                        facecolors=cmap(norm(param[:,:,ss])), cmap=cmap)
from matplotlib import cm
mm = cm.ScalarMappable(cmap=cmap, norm=norm)
# m.set_array([])
fig.colorbar(mm, label='Arb. $[\mu A/m^2]$')



# Plotting resuduals
# Plot grid and coastlines:
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
from pysymmetry.utils.spherical import sph_to_car, car_to_sph
sitelat = d_grid.projection.position[1]# 67.35 #69.38
sitephi = d_grid.projection.position[0]# 23. #20.30
ski_x, ski_y, ski_z = sph_to_car((RE, 90-sitelat, sitephi), deg=True)
xlim = (ski_x[0]-0.5*d_grid.L*1e-3, ski_x[0]+0.5*d_grid.L*1e-3) 
ylim = (ski_y[0]-0.5*d_grid.L*1e-3, ski_y[0]+0.5*d_grid.L*1e-3) 
zlim = (RE, RE+alts_grid[-1])
ax.scatter(ski_x, ski_y, ski_z, marker='s', color='yellow')
for cl in visualization.get_coastlines():
    x,y,z = sph_to_car((RE, 90-cl[1], cl[0]), deg=True)
    use = (x > xlim[0]-200) & (x < xlim[1]+200) & (y > ylim[0]-200) & (y < ylim[1]+200) & (z > 0)
    ax.plot(x[use], y[use], z[use], color = 'C0')
residual_r = d[0:N]-j_perp[0:N]
residual_theta = d[N:2*N]-j_perp[N:2*N]
residual_phi = d[2*N:3*N]-j_perp[2*N:3*N]
residual_fac = data_fac - jpar
x, y, z = sph_to_car((RE+alt, 90-lat, lon), deg=True)
cmap = plt.cm.seismic
import matplotlib
vmin = -10e-9
vmax = 10e-9
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
# nnn = residual_r.reshape(sh)[:,0,:]/residual_r.reshape(sh)[:,0,:].max()
# p = ax.plot_surface(x.reshape(sh)[:,-1,:], y.reshape(sh)[:,-1,:], z.reshape(sh)[:,-1,:], 
#                     facecolors=cmap(norm(residual_r.reshape(sh)[:,-1,:])), cmap=cmap)
# p = ax.plot_surface(x.reshape(sh)[10,:,:], y.reshape(sh)[10,:,:], z.reshape(sh)[10,:,:], 
#                     facecolors=cmap(norm(residual_r.reshape(sh)[10,:,:])), cmap=cmap)
# p = ax.scatter(x, y, z, c=residual_r[use]/d[0:N][use], cmap='seismic', vmin = -1, vmax=1)
s = np.ones(N)
s[inside] = 4
s[~inside] = 40
p = ax.scatter(x, y, z, c=residual_fac, cmap='seismic', norm=norm, s=s)
# p = ax.scatter(x, y, z, c=jpar, cmap='seismic', norm=norm)

# ax.scatter(x, y, z, c=np.sqrt(d[0:N][use]**2+d[N:2*N][use]**2+d[2*N:3*N][use]**2), cmap='viridis')#, vmin = -1, vmax=1)
# ax.set_title('Altitude > '+str(aaa)+' km')
ax.set_xlim3d(xlim)
ax.set_ylim3d(ylim)
from matplotlib import cm
mm = cm.ScalarMappable(cmap=cmap, norm=norm)
# m.set_array([])
fig.colorbar(mm, label='residual jtheta $[\mu A/m^2]$')
ax.view_init(elev=10., azim=210)


#Scatterplots projected to alt-lat plane
clim=1e-8
plt.figure()
# plt.scatter(lat, alt, c=jpar, s=4, cmap='seismic', vmin=-clim, vmax=clim)
plt.scatter(data['lat'][:,:,jj], data['alt'][:,:,jj], c=data['fac'][:,:,jj], s=4, cmap='seismic', vmin=-clim, vmax=clim)
# plt.colorbar()
plt.xlim(33,38)
plt.ylim(100,220)

plt.figure()
plt.scatter(lat, alt, c=data_fac, s=4, cmap='seismic', vmin=-clim, vmax=clim)
plt.xlim(33,38)
plt.ylim(100,220)

plt.figure()
plt.scatter(lat, alt, c=data_fac-jpar, s=4, cmap='seismic', vmin=-clim, vmax=clim)
plt.xlim(33,38)
plt.ylim(100,220)

#Residual scatterplot
plt.figure()
if jperp:
    plt.scatter(1e6*d[0:N],1e6*j_perp[0:N], s=1, label='j_perp,r')
    plt.scatter(1e6*d[N:2*N],1e6*j_perp[N:2*N], s=1, label='j_perp,theta')
    plt.scatter(1e6*d[2*N:3*N],1e6*j_perp[2*N:3*N], s=1, label='j_perp,phi')
    plt.scatter(1e6*data_fac,1e6*jpar, s=1, label='jpar')
    plt.title('Modelled with jperp input')
else:
    plt.scatter(1e6*d[0:N],1e6*full_j[0:N], s=1, label='j_r')
    plt.scatter(1e6*d[N:2*N],1e6*full_j[N:2*N], s=1, label='j_theta')
    plt.scatter(1e6*d[2*N:3*N],1e6*full_j[2*N:3*N], s=1, label='j_phi')
    plt.scatter(1e6*data_fac,1e6*jpar, s=1, label='jpar')
    plt.title('Modelled with full j input')
plt.legend()
plt.xlabel('Synthetic $[\mu A/m^2]$')
plt.ylabel('SECS $[\mu A/m^2]$')
# plt.title(sss)

#Residual scatterplot of inside vs outside points
from gemini3d.grid import convert
import apexpy
a = apexpy.Apex(2022)
mlat_, mlon_ = a.geo2apex(lat, lon, alt)
m_glat, m_glon, _ = a.apex2geo(mlat_, mlon_, height)
inside = grid.ingrid(m_glon, m_glat, ext_factor=-1)
plt.figure()
plt.scatter(1e6*d[N:2*N][inside],1e6*j_perp[N:2*N][inside], s=1, label='j_perp,theta inside')
plt.scatter(1e6*d[N:2*N][~inside],1e6*j_perp[N:2*N][~inside], s=1, label='j_perp,theta outside')
# plt.scatter(1e6*d[N:2*N],1e6*full_j[N:2*N], s=1, label='jtheta')
# plt.scatter(1e6*d[2*N:3*N],1e6*full_j[2*N:3*N], s=1, label='jphi')
plt.legend()
# plt.title(sss)
plt.xlabel('GEMINI $[\mu A/m^2]$')
plt.ylabel('SECS $[\mu A/m^2]$')

plt.figure()
plt.hist(residual_r/d[0:N], bins=50, range=(-4,4))
plt.xlabel('$j_r$ residual / $j_r$ GEMINI')
plt.ylabel('#')