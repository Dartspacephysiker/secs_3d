#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:41:07 2022

@author: jone

Investigate 3D SECS inversion scheme taylored towards EISCAT_3D measurements, 
as outlined by Kalle

"""
import gemini3d.read as read
import numpy as np
import helpers
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


#Global variables
path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/gemini_output/"
height = 80 #km. height of secs CS grid at bottom boundary. To be used for SECS representation of current
RE = 6371.2 #Earth radius in km

#Define CS grid
xg = read.grid(path)
extend=1
grid, grid_ev = helpers.make_csgrid(xg, height=height, crop_factor=0.7, 
                                    resolution_factor=0.2, extend=extend) #outer (secs), inner (evaluation) grid
# alts = np.concatenate((np.arange(80,170,2),np.arange(170,400,10),np.arange(400,850,50)))
alts = np.arange(100,120,2)
altres = np.diff(alts)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
#Grid dimensions
K = alts.shape[0] #Number of vertival layers
I = grid.shape[0] #Number of cells in eta direction
J = grid.shape[1]  #Number of cells in xi direction 

#Open simulation output files
var = ["v1", "v2", "v3", "Phi", "J1", "J2", "J3", "ne"]
cfg = read.config(path)
xg = read.grid(path)
dims = xg['lx']
times = cfg["time"][-1:]
t = times[0]
dat = read.frame(path, t, var=var)
dat = helpers.compute_enu_components(xg, dat)

# Sample some data in 3D
datadict = helpers.sample_at_alt(xg, dat, alt=alts, altres=altres, 
                                 var=["je","jn","ju"], resfac=0.2)
# alt = np.ones(datadict['je'].shape[1:])*alt_

#Gemini input
jphi = datadict['je'].flatten()
jtheta = -datadict['jn'].flatten()
jr = datadict['ju'].flatten()
use = np.isfinite(jphi)
jphi = jphi[use]
jtheta = jtheta[use]
jr = jr[use]

lat = datadict['glatmesh'].flatten()[use]
lon = datadict['glonmesh'].flatten()[use]
alt = datadict['altmesh'].flatten()[use]

# Remove data/evaluation points outside secs_grid, using the ingrid function
use = grid.ingrid(lon.flatten(), lat.flatten())
lat = lat[use]
lon = lon[use]
alt = alt[use]
d = np.hstack((jr[use], jtheta[use], jphi[use]))

#Retrieve matrices to compute m from GEMINI data
Ge_cf, Gn_cf, Ge_df, Gn_df = helpers.get_SECS_J_G_matrices_3D(grid, alts, 
                lat, lon, alt)
S = helpers.get_jr_matrix(grid, alts, lat, lon, alt)
O = np.zeros(S.shape)
Gcf = np.vstack((S, -Gn_cf, Ge_cf))
Gdf = np.vstack((O, -Gn_df, Ge_df))
G = np.hstack((Gcf, Gdf))

#Solve
m = lstsq(G, d, cond=0.01)[0]


# Evaluate model FIX: make it possible to specify evaluation/data location
# outside 3D domain spanned by the secs and vertical grid. At present it chrashes
eval_alts = np.linspace(100,118,10)
lat_ev = np.tile(grid_ev.lat[:,:,np.newaxis],eval_alts.size)
lat_ev = np.swapaxes(np.swapaxes(lat_ev,2,1),0,1)
lon_ev = np.tile(grid_ev.lon[:,:,np.newaxis],eval_alts.size)
lon_ev = np.swapaxes(np.swapaxes(lon_ev,2,1),0,1)
alt_ev, _, _ = np.meshgrid(eval_alts, np.ones(grid_ev.shape[0]),
                            np.ones(grid_ev.shape[1]), indexing='ij')


Ge_cf, Gn_cf, Ge_df, Gn_df = helpers.get_SECS_J_G_matrices_3D(grid, alts, 
                                    lat_ev, lon_ev, alt_ev)
S = helpers.get_jr_matrix(grid, alts, lat_ev, lon_ev, alt_ev)
O = np.zeros(S.shape)
Gcf = np.vstack((S, -Gn_cf, Ge_cf))
Gdf = np.vstack((O, -Gn_df, Ge_df))
G = np.hstack((Gcf, Gdf))

#Evaluate
full_j = G.dot(m)
N_ev = lat_ev.size
jr_ev = full_j[:N_ev].reshape(lat_ev.shape)
jtheta_ev = full_j[N_ev:2*N_ev].reshape(lat_ev.shape)
jphi_ev = full_j[2*N_ev:3*N_ev].reshape(lat_ev.shape)

#Plotting of input vs output
vmin = -30
vmax = 30
p = 'ju'
input_param = datadict[p]*1e6
model_param = jr_ev*1e6
for alt_i in range(len(alts)):
    # alt_i = 2 #altitude index
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.pcolormesh(datadict['glonmesh'][alt_i,:,:], datadict['glatmesh'][alt_i,:,:], 
                   input_param[alt_i,:,:], cmap='seismic', vmin=vmin, vmax=vmax)
    plt.title('GEMINI: '+p+' @ '+str(alts[alt_i])+' km')
    plt.xlim(4,37)
    plt.ylim(62,74)
    plt.xlabel('glon')
    plt.ylabel('glat')
    
    plt.subplot(1,2,2)
    plt.pcolormesh(lon_ev[alt_i,:,:], lat_ev[alt_i,:,:], 
                   model_param[alt_i,:,:], cmap='seismic', vmin=vmin, vmax=vmax)
    plt.colorbar(label='$\mu$A/m$^2$')
    plt.title('3D SECS: '+p+' @ '+str(alts[alt_i])+' km')
    plt.xlim(4,37)
    plt.ylim(62,74)
    plt.xlabel('glon')
    plt.ylabel('glat')
    
    plt.savefig('./plots/tmp/snap_'+p+'_%03i.png' % alt_i)
