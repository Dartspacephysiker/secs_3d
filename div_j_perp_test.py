#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:41:34 2022

@author: jone
"""

#Investigate divergence of j_perp and how this differs from the divegence
# og j_perp projected on horizontal plane (what is convenient to do with EISCAT)

import gemini3d.read as read
import helpers
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from gemini3d.grid.convert import unitvecs_geographic


#Global variables
path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/gemini_output/"

#Open simulation output files
var = ["v1", "v2", "v3", "Phi", "J1", "J2", "J3", "ne"]
cfg = read.config(path)
xg = read.grid(path)
dims = xg['lx']
times = cfg["time"][-1:]
t = times[0]
dat = read.frame(path, t, var=var)
# dat = helpers.compute_enu_components(xg, dat)

#Compute divergence of j_perp
d = helpers.divergence(xg, dat, param='j_perp')
dat['divjperp'] = xr.DataArray(d, dims=('x1','x2','x3'))
dat['fac'] = dat.J1

#Compute divergence of j_horizontal
#Will project (J2,J3) onto horizontal plane, and express in (east,north) components
[egalt,eglon,eglat]=unitvecs_geographic(xg)    
je2 = np.sum(xg["e2"]*eglon*dat.J2.values[...,np.newaxis],3)
jn2 = np.sum(xg["e2"]*eglat*dat.J2.values[...,np.newaxis],3)
# ju2 = np.sum(xg["e2"]*egalt*dat.J2.values[...,np.newaxis],3)
je3 = np.sum(xg["e3"]*eglon*dat.J3.values[...,np.newaxis],3)
jn3 = np.sum(xg["e3"]*eglat*dat.J3.values[...,np.newaxis],3)
# ju3 = np.sum(xg["e3"]*egalt*dat.J3.values[...,np.newaxis],3)
je = je2 + je3
jn = jn2 + jn3
dat['je'] = xr.DataArray(je, dims=('x1','x2','x3'))
dat['jn'] = xr.DataArray(jn, dims=('x1','x2','x3'))

#Sample the divergence at fixed altitude
alts = np.flip(np.concatenate((np.arange(80,170,2),np.arange(170,400,10),np.arange(400,850,50))))

# height= 150 # km
altres = np.diff(alts)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
for (i, alt) in enumerate(alts): 
    
    #Divergence of perp current
    sampledict_perp = helpers.sample_at_alt(xg, dat, var=['divjperp','fac'], alt=alt, altres=altres[i])
    divjperp = sampledict_perp['divjperp'][0,:,:] * 2*altres[i]*1e3
    
    #Divergence of perp current projected to horizontal plane
    sampledict_hor = helpers.sample_at_alt(xg, dat, grid = None, alt=alt, altres=altres[i], 
                                   time_ind = -1, var = ["je", "jn"], path=path, 
                                   resfac=0.8)
    divjh = helpers.divergence_spherical(sampledict_hor, alt=alt, param=['je', 'jn']) * 2*altres[i]*1e3
    
    #Plotting
    plt.figure(figsize=(10,3.5))
    plt.subplot(1,3,1)
    plt.pcolormesh(sampledict_perp['glonmesh'], sampledict_perp['glatmesh'],divjperp*1e6, cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar(label='$\mu$A/m$^2$', location='left')
    plt.title('$\\nabla \\cdot \\vec{j}_{\\perp} \cdot '+str(round(2*altres[i]))+'$ km @ '+str(alt)+' km')
    plt.xlabel('glon')
    plt.ylabel('glat')

    plt.subplot(1,3,2)
    plt.pcolormesh(sampledict_hor['glonmesh'], sampledict_hor['glatmesh'],divjh*1e6, cmap='seismic', vmin=-1, vmax=1)
    plt.title('$\\nabla \\cdot \\vec{j}_{\\perp->hor} \cdot '+str(round(2*altres[i]))+'$ km @ '+str(alt)+' km')
    plt.xlabel('glon')
    plt.ylabel('glat')
    
    plt.subplot(1,3,3)
    plt.pcolormesh(sampledict_perp['glonmesh'], sampledict_perp['glatmesh'], sampledict_perp['fac'][0,:,:]*1e6, cmap='seismic', vmin=-10, vmax=10)
    plt.colorbar(label='$\mu$A/m$^2$')
    plt.title('FAC km @ '+str(alt)+' km')
    plt.xlabel('glon')
    plt.ylabel('glat')
    
    filename = './plots/div_jperp/div_j_perp_%03ikm.png' % alt
    plt.savefig(filename)

from cmodel.helpers import make_gif
import glob
files = glob.glob('./plots/div_jperp/*.png')
files.sort()
make_gif(files, filename='./plots/div_jperp/div_j_perp.gif', fps=2)


