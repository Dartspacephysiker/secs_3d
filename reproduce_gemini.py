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
# import helpers
import gemini_tools
import secs3d
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import apexpy


#Global variables
path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/gemini_output/"
height = 80 #km. height of secs CS grid at bottom boundary. To be used for SECS representation of current
RE = 6371.2 #Earth radius in km
apex = apexpy.Apex(2022)
interpolate = True #More accurate treatment of observing/evaluation locations
interpolate_S = True # control the integration matrix separately
jperp = True

#Define CS grid
xg = read.grid(path)
extend=1
grid, grid_ev = gemini_tools.make_csgrid(xg, height=height, crop_factor=0.4, #0.45
                                    resolution_factor=0.18, extend=extend) # 0.3 outer (secs), inner (evaluation) grid
singularity_limit=grid.Lres
# alts_grid = np.concatenate((np.arange(80,170,2),np.arange(170,400,10),np.arange(400,850,50)))
alts_grid = np.arange(90,126,2)
alts_grid_obs = np.arange(90,126,1)
altres = np.diff(alts_grid)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
#Grid dimensions
K = alts_grid.shape[0] #Number of vertival layers
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
dat = gemini_tools.compute_enu_components(xg, dat)

# Sample some data in 3D Here from resampled (interpolated) in spherical coords. 
# Should try to use input directly from GEMINI grid
if jperp:
    var=["je", "jn", "ju", "Be", "Bn", "Bu"]
else:
    var=["je", "jn", "ju"]
    
# datadict = gemini_tools.sample_at_alt(xg, dat, alt=alts_grid, altres=altres, 
                                 # var=var, resfac=0.25)
datadict = gemini_tools.sample_eiscat(xg, dat, alts_grid_obs)


#Prepare the sampled GEMINI data for inversion
jj, lat, lon, alt = gemini_tools.prepare_model_data(grid, datadict, alts_grid, jperp=jperp)
d = np.hstack((jj[0], jj[1], jj[2])) # (r, theta, phi components)


#Retrieve matrices to compute m from GEMINI data
Ge_cf, Gn_cf, Ge_df, Gn_df = secs3d.get_SECS_J_G_matrices_3D(grid, alts_grid, 
                lat, lon, alt, interpolate=interpolate, 
                singularity_limit=singularity_limit)
S = secs3d.get_jr_matrix(grid, alts_grid, lat, lon, alt, interpolate=interpolate_S)
O = np.zeros(S.shape)
Gcf = np.vstack((S, -Gn_cf, Ge_cf))
Gdf = np.vstack((O, -Gn_df, Ge_df))
G = np.hstack((Gcf, Gdf))
if jperp:
    br, btheta, bphi = secs3d.make_b_unitvectors(jj[3],jj[4],jj[5])
    B = secs3d.make_B(br, btheta, bphi)    
    P = secs3d.make_P(br.size)
    G = P.T.dot(B.dot(P.dot(G)))
    
    
GTG = G.T.dot(G)
R = np.diag(np.ones(GTG.shape[0])) * np.median(np.diag(GTG)) * 0.1
GTd = G.T.dot(d)

#Solve
m = lstsq(GTG + R, GTd, cond=0.)[0]

############################################################
############################################################

# Evaluate model FIX: make it possible to specify evaluation/data location
# outside 3D domain spanned by the secs and vertical grid. At present it chrashes
Nxi = 50 #new evaluation resolution: "j" index
Neta = 40 #new evaluation resolution: "i" index
alts__ = alts_grid[1:]-altres[1:]
etas = np.linspace(grid.eta[1,0],grid.eta[-2,0],Neta)
xis = np.linspace(grid.xi[0,1],grid.xi[0,-2],Nxi)
xi_ev, eta_ev = np.meshgrid(xis, etas, indexing = 'xy')
alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, etas, xis, indexing='ij')
lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev)
sh = lon_ev.shape
#OR
# eval_alts = alts_grid #np.linspace(100,118,10)
# lat_ev = np.tile(grid_ev.lat[:,:,np.newaxis],eval_alts.size)
# lat_ev = np.swapaxes(np.swapaxes(lat_ev,2,1),0,1)
# lon_ev = np.tile(grid_ev.lon[:,:,np.newaxis],eval_alts.size)
# lon_ev = np.swapaxes(np.swapaxes(lon_ev,2,1),0,1)
# alt_ev, _, _ = np.meshgrid(eval_alts, np.ones(grid_ev.shape[0]),
#                             np.ones(grid_ev.shape[1]), indexing='ij')


# Remove evaluation points outside the perimiter defined by the secs_grid nodes
# using the ingrid function
lat, lon, alt = gemini_tools.remove_outside(grid, alts_grid, lat_ev.flatten(), 
                                       lon_ev.flatten(), alt_ev.flatten(), 
                                       ext_factor=-1, params=None)
if lat.size != lat_ev.flatten().size:
    print('Evaluation locations outside secs grid. Can not evaluate.')
    print(1/0)
    
Ge_cf, Gn_cf, Ge_df, Gn_df = secs3d.get_SECS_J_G_matrices_3D(grid, alts_grid, 
                                    lat, lon, alt, interpolate=interpolate, 
                                    singularity_limit=singularity_limit)
S = secs3d.get_jr_matrix(grid, alts_grid, lat_ev, lon_ev, alt_ev, interpolate=interpolate_S)
O = np.zeros(S.shape)
Gcf = np.vstack((S, -Gn_cf, Ge_cf))
Gdf = np.vstack((O, -Gn_df, Ge_df))
G = np.hstack((Gcf, Gdf))

#Evaluate
full_j = G.dot(m)
N_ev = lat_ev.size
jr_ev = full_j[:N_ev].reshape(sh)
jtheta_ev = full_j[N_ev:2*N_ev].reshape(sh)
jphi_ev = full_j[2*N_ev:3*N_ev].reshape(sh)
lon_ev = lon_ev.reshape(sh)
lat_ev = lat_ev.reshape(sh)
alt_ev = alt_ev.reshape(sh)

# jr_ev = full_j[:N_ev].reshape(alts_grid.size-1,grid_ev.shape[0],grid_ev.shape[1])
# jtheta_ev = full_j[N_ev:2*N_ev].reshape(alts_grid.size-1,grid_ev.shape[0],grid_ev.shape[1])
# jphi_ev = full_j[2*N_ev:3*N_ev].reshape(alts_grid.size-1,grid_ev.shape[0],grid_ev.shape[1])
# lon_ev = lon_ev.reshape(alts_grid.size-1,grid_ev.shape[0],grid_ev.shape[1])
# lat_ev = lat_ev.reshape(alts_grid.size-1,grid_ev.shape[0],grid_ev.shape[1])

#Plotting of input vs output
vmin = -20
vmax = 20
p = 'ju'
input_param = datadict[p]*1e6
model_param = jr_ev*1e6
m_cf = m[0:m.size//2].reshape((alts_grid.size,grid.lat.shape[0],grid.lat.shape[1]))

#Divergence of horizontal currents as expressed on the CS grid
je_cs, jn_cs, ju_cs = gemini_tools.grid_param_at_alt(datadict,grid,param='j')
D = grid.divergence(S=1)


# for alt_i in range(len(alts_grid)-1):
for alt_i in range(sh[0]-1):

    # alt_i = 2 #altitude index
    plt.figure(figsize=(10,9))
    plt.subplot(2,2,1)
    glat_secs = grid.lat#.flatten()
    glon_secs = grid.lon#.flatten()
    if datadict['glonmesh'][alt_i,:,:].max() > 360:
        datadict['glonmesh'][alt_i,:,:] = datadict['glonmesh'][alt_i,:,:] - 360
    plt.pcolormesh(datadict['glonmesh'][alt_i,:,:], datadict['glatmesh'][alt_i,:,:], 
                    input_param[alt_i,:,:], cmap='seismic', vmin=vmin, vmax=vmax)
    plt.plot(glon_secs[extend:-extend,extend], glat_secs[extend:-extend,extend], color='black')
    plt.plot(glon_secs[extend:-extend,-extend-1], glat_secs[extend:-extend,-extend], color='black')
    plt.plot(glon_secs[extend,extend:-extend], glat_secs[extend,extend:-extend], color='black')
    plt.plot(glon_secs[-extend-1,extend:-extend], glat_secs[-extend,extend:-extend], color='black')
    plt.scatter(lon, lat)
    plt.title('GEMINI: '+p+' @ '+str(alts_grid[alt_i])+' km')
    plt.xlim(4+0,37+0)
    plt.ylim(62,74)
    plt.xlabel('glon')
    plt.ylabel('glat')
        
    plt.subplot(2,2,2)
    plt.pcolormesh(lon_ev[alt_i,:,:], lat_ev[alt_i,:,:], 
                   model_param[alt_i,:,:], cmap='seismic', vmin=vmin, vmax=vmax)
    plt.title('3D SECS: '+p+' @ '+str(alt_ev[alt_i,0,0])+' km')
    plt.xlim(4,37)
    plt.ylim(62,74)
    plt.xlabel('glon')
    plt.ylabel('glat')
    plt.colorbar(label='$\mu$A/m$^2$')
    
    # plt.subplot(2,2,3)
    # div = D.dot(np.hstack((je_cs[alt_i,:,:].flatten()*altres[alt_i]*2e3, 
    #                        jn_cs[alt_i,:,:].flatten()*altres[alt_i]*2e3)))*1e6
    
    # # plt.pcolormesh(grid.lon, grid.lat, m_cf[alt_i,:,:], cmap='seismic', 
    # #                 vmin=vmin/10, vmax=vmax/10)
    # plt.pcolormesh(grid.lon, grid.lat, div.reshape(grid.lat.shape), cmap='seismic', 
    #                vmin=vmin/10, vmax=vmax/10)
    #     # plt.title('3D SECS: CF amp. @ '+str(alts_grid[alt_i])+' km')
    # plt.title('GEMINI: $\\nabla \\cdot \\vec{J_{hor}}$ on CS grid @ '+str(alt_ev[alt_i,0,0])+' km')
    # plt.xlim(4,37)
    # plt.ylim(62,74)
    # plt.xlabel('glon')
    # plt.ylabel('glat')
    
    plt.subplot(2,2,4)
    # div = D.dot(np.hstack((je_cs[alt_i,:,:].flatten()*altres[alt_i]*2e3, 
                            # jn_cs[alt_i,:,:].flatten()*altres[alt_i]*2e3)))*1e6
    
    plt.pcolormesh(grid.lon, grid.lat, m_cf[alt_i,:,:], cmap='seismic', 
                    vmin=vmin/10, vmax=vmax/10)
    plt.colorbar(label='$\mu$A/m$^2$')
    plt.title('3D SECS: CF amp. @ '+str(alts_grid[alt_i])+' km')
    plt.xlim(4,37)
    plt.ylim(62,74)
    plt.xlabel('glon')
    plt.ylabel('glat')
    
    if jperp:
        fix = 'jperp_'
    else:
        fix = ''
        
    filename = './plots/tmp/'+fix+'snap_'+p+'_%03i.png' % alt_i

    plt.savefig(filename)

import glob
files = glob.glob('./plots/tmp/'+fix+'snap_ju*.png')
files.sort()
gifname = './plots/tmp/'+fix+'ju_plot.gif'   
from cmodel.helpers import make_gif
make_gif(files, filename=gifname, fps=2)
