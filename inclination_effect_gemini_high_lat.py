#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:59:27 2022

@author: jone

Investigate inclination effect in GEMINI.
Try to center a grid with sufficiently high resolution in a region with structure
in J, to see how the fits may be affected by field lines mapping out of the 
analysis volume.

"""

import gemini3d.read as read
import numpy as np
import gemini_tools
import secs3d
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import apexpy
from secsy import cubedsphere
import visualization
from pysymmetry.utils.spherical import sph_to_car, car_to_sph
import matplotlib

#Global variables
path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/gemini_output/"
height = 90 #km. height of secs CS grid at bottom boundary. To be used for SECS representation of current
RE = 6371.2 #Earth radius in km
apex = apexpy.Apex(2022)
interpolate = True #More accurate treatment of observing/evaluation locations
jperp = True #Use only perp components of J as inpur to inversion

#Define CS grid
xg = read.grid(path)
extend=1
grid, grid_ev = gemini_tools.make_csgrid(xg, height=height, crop_factor=0.2, #0.45
                                    resolution_factor=0.3, extend=extend, 
                                    dlat = -1.) # dlat = 2.0
singularity_limit=grid.Lres
alts_grid = np.concatenate((np.arange(90,140,2),np.arange(140,170,5),np.arange(170,230,10),np.arange(230,830,50)))
# alts_grid = np.array([100,150,200,500])
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
    var=["je", "jn", "ju", "Be", "Bn", "Bu"]#, "Phitop"]
else:
    var=["je", "jn", "ju"]
Nxi = int(J*1.5) #new evaluation resolution: "j" index
Neta = int(I*1.5) #new evaluation resolution: "i" index
alts__ = alts_grid# alts_grid[1:]-altres[1:]
etas = np.linspace(grid.eta_mesh[1,0]+0.01*grid.deta,grid.eta_mesh[-2,0]-0.01*grid.deta,Neta)
xis = np.linspace(grid.xi_mesh[0,1]+0.01*grid.dxi,grid.xi_mesh[0,-2]-0.01*grid.dxi,Nxi)
xi_ev, eta_ev = np.meshgrid(xis, etas, indexing = 'xy')
alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, etas, xis, indexing='ij')
lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev)
sh = lon_ev.shape
datadict = gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)
br, btheta, bphi = secs3d.make_b_unitvectors(datadict['Bu'], 
                -datadict['Bn'], datadict['Be'])
data_fac = np.sum(np.array([datadict['ju'], -datadict['jn'], datadict['je']]) * 
                np.array([br, btheta, bphi]), axis=0)

#Plotting of grid placement
fig = plt.figure(figsize = (20, 10))
ax = fig.add_subplot(121, projection='3d')
ax.set_axis_off()
ax.view_init(azim=-26, elev=7)
visualization.spherical_grid(ax, lat_ev, lon_ev, alt_ev, color='blue')
visualization.field_aligned_grid(ax, grid, alts_grid, color='green')
kwargs={'linewidth':3}
for kk in range(lat_ev[0,-1,:].size):
    visualization.plot_field_line(ax, lat_ev[0,-1,kk], lon_ev[0,-1,kk], 
                              alts__, color='orange', **kwargs, dipole=True)
    visualization.plot_field_line(ax, lat_ev[0,sh[1]//2,kk], lon_ev[0,sh[1]//2,kk], 
                              alts__, color='orange', **kwargs, dipole=True)
x, y, z = sph_to_car((RE+datadict['alt'].flatten(), 90-datadict['lat'].flatten(), 
                      datadict['lon'].flatten()), deg=True)
cmap = plt.cm.seismic
clim = 1e-5
norm = matplotlib.colors.Normalize(vmin=-clim, vmax=clim)
p = ax.plot_surface(x.reshape(sh)[-1,:,:], y.reshape(sh)[-1,:,:], 
                    z.reshape(sh)[-1,:,:], alpha=0.5,
                    facecolors=cmap(norm(data_fac.reshape(sh)[-1,:,:])), cmap=cmap)
p = ax.plot_surface(x.reshape(sh)[:,:,sh[2]//2], y.reshape(sh)[:,:,sh[2]//2], 
                    z.reshape(sh)[:,:,sh[2]//2], alpha=0.5,
                    facecolors=cmap(norm(data_fac.reshape(sh)[:,:,sh[2]//2])), cmap=cmap)
x0, y0, z0 = sph_to_car((RE+0, 90-grid.projection.position[1], grid.projection.position[0]), deg=True)
range_ =  alts_grid[-1]*0.3
ax.set_xlim(x0-range_, x0+range_)
ax.set_ylim(y0-range_, y0+range_)
ax.set_zlim(z0, z0+2*range_)
ax.set_title('FAC from GEMINI')

#Prepare the sampled GEMINI data for inversion
jj, lat, lon, alt = gemini_tools.prepare_model_data(grid, datadict, alts_grid, jperp=jperp)
d = np.hstack((jj[0], jj[1], jj[2])) # (r, theta, phi components)
G = secs3d.make_G(grid, alts_grid, lat, lon, alt, interpolate=interpolate)
if jperp:
    br, btheta, bphi = secs3d.make_b_unitvectors(jj[3],jj[4],jj[5])
    B = secs3d.make_B(br, btheta, bphi)    
    P = secs3d.make_P(br.size)
    G = P.T.dot(B.dot(P.dot(G)))
GT = G.T
GTG = GT.dot(G)
Reg = np.diag(np.ones(GTG.shape[0])) * np.median(np.diag(GTG)) * 0.01
GTd = GT.dot(d)

#Make resolution matrix
if True:
    Cd  = np.diag(np.ones(G.shape[0])) #Data covariance matrix
    Gt = np.linalg.pinv(GTG+Reg).dot(GT)
    R = Gt.dot(G)

# Investigate resolution matrix
position = grid.projection.position
k, i, j = secs3d.get_indices_kij(grid, alts_grid, position[1], position[0], np.array([500]))
kij = np.ravel_multi_index((k[0],i[0],j[0]), (K,I,J))
NN=1 # 0 if CF parameters, 1 if DF parameters
psf = R[:,NN*K*I*J+kij] #psf
# psf = R[kij,:] # averaging function
clim = 1e-2
absolute=True

fig = plt.figure(figsize = (30, 10))
ax = fig.add_subplot(131, projection='3d')
ax.set_axis_off()
visualization.plot_resolution(ax, grid, alts_grid, kij, psf[NN*K*I*J:(NN+1)*K*I*J], clim=clim, 
                             planes=[0], absolute=absolute)
ax = fig.add_subplot(132, projection='3d')
ax.set_axis_off()
visualization.plot_resolution(ax, grid, alts_grid, kij, psf[NN*K*I*J:(NN+1)*K*I*J], clim=clim, 
                             planes=[1], az=40, el=5, absolute=absolute)
ax = fig.add_subplot(133, projection='3d')
ax.set_axis_off()
visualization.plot_resolution(ax, grid, alts_grid, kij, psf[NN*K*I*J:(NN+1)*K*I*J], clim=clim, 
                             planes=[2], absolute=absolute)


#Solve
m = lstsq(GTG + Reg, GTd, cond=0.)[0]

position = grid.projection.position
if jperp:
    sss = 'm_perp_inclination_test_gemini_%4.2f_%4.2f.npy' % (position[0],position[1])
else:
    sss = 'm_inclination_test_gemini_%4.2f_%4.2f.npy' % (position[0],position[1])
keep = {'datadict':datadict, 'alts_grid':alts_grid, 'grid':grid, 'shape':sh, 'm':m}
np.save(sss, keep)


############################################################
############################################################
# Investigate working principle in GEMINI, without inversion
shape = lat_ev.shape
datadict['shape'] = shape
divjh_sph = gemini_tools.divergence_spherical(datadict, hor=True, perp=False)
H = np.sqrt(datadict['Be']**2+datadict['Bn']**2)
inclination = np.degrees(np.arctan(H/np.abs(datadict['Bu']))).reshape(datadict['shape'])
altres__ = np.diff(datadict['alt'][:,0,0])
altres_ = np.ones(datadict['shape']) * np.hstack((altres__, altres__[-1]))[:,np.newaxis,np.newaxis]
jr_sph = datadict['ju'][0,:,:] + np.cumsum(-divjh_sph*((altres_*1e3)/np.cos(np.radians(inclination))), axis=0)

# Performance plotting
# fig = plt.figure(figsize = (10, 10))
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_axis_off()
ax2.view_init(azim=-26, elev=7)
visualization.spherical_grid(ax2, lat_ev, lon_ev, alt_ev, color='blue') #Data locations
visualization.field_aligned_grid(ax2, grid, alts_grid, color='green') # Base SECS grid
kwargs={'linewidth':3}
for kk in range(datadict['lat'].reshape(shape)[0,-1,:].size): # Plot some field-lines
    visualization.plot_field_line(ax2, datadict['lat'].reshape(shape)[0,-1,kk], 
                datadict['lon'].reshape(shape)[0,-1,kk], 
                datadict['alt'].reshape(shape)[:,-1,0], 
                color='orange', **kwargs, dipole=True)
    visualization.plot_field_line(ax2, datadict['lat'].reshape(shape)[0,shape[1]//2,kk], 
                datadict['lon'].reshape(shape)[0,shape[1]//2,kk], 
                datadict['alt'].reshape(shape)[:,shape[1]//2,0],
                color='orange', **kwargs, dipole=True)
    # visualization.plot_field_line(ax, lat_ev[0,8,kk], lon_ev[0,8,kk], 
                              # alts__)
x, y, z = sph_to_car((RE+datadict['alt'].flatten(), 90-datadict['lat'].flatten(), 
                      datadict['lon'].flatten()), deg=True)
cmap = plt.cm.seismic
clim = 1e-5
norm = matplotlib.colors.Normalize(vmin=-clim, vmax=clim)
p = ax2.plot_surface(x.reshape(shape)[-1,:,:], y.reshape(shape)[-1,:,:], 
                    z.reshape(shape)[-1,:,:], alpha=0.5,
                    facecolors=cmap(norm((jr_sph)[-1,:,:])), cmap=cmap)
p = ax2.plot_surface(x.reshape(shape)[:,:,shape[2]//2], y.reshape(shape)[:,:,shape[2]//2], 
                    z.reshape(shape)[:,:,shape[2]//2], alpha=0.5,
                    facecolors=cmap(norm((jr_sph)[:,:,shape[2]//2])), cmap=cmap)
x0, y0, z0 = sph_to_car((RE+0, 90-grid.projection.position[1], grid.projection.position[0]), deg=True)
range_ =  alts_grid[-1]*0.3
ax2.set_xlim(x0-range_, x0+range_)
ax2.set_ylim(y0-range_, y0+range_)
ax2.set_zlim(z0, z0+2*range_)
ax2.set_title('Jr from current continuity')

plt.figure()
plt.plot(datadict['ju'].flatten()*1e6, jr_sph.flatten()*1e6)
plt.xlabel('GEMINI Jr $[\mu A/m^2]$')
plt.ylabel('int(div_h(jh) + j0 ( $[\mu A/m^2]$')

############################################################
############################################################
# Evaluate model and investigate performance
jperp = True
# position = np.array([23.39,69.25])
# position = np.array([23.39, 68.25])
# position = np.array([23.39, 67.25])
position = np.array([23.39, 66.25])
if jperp:
    savestr='m_perp_inclination_test_gemini_%4.2f_%4.2f.npy' % (position[0],position[1])
else:
    savestr = 'm_inclination_test_gemini_%4.2f_%4.2f.npy' % (position[0],position[1])
m_ = np.load(savestr, allow_pickle=True).item()
m = m_['m']
try:
    datadict = m_['datadict'].item()
except:
    datadict = m_['datadict']
jj, lat, lon, alt = gemini_tools.prepare_model_data(m_['grid'], datadict,
                                                    m_['alts_grid'], jperp=jperp)
d = np.hstack((jj[0], jj[1], jj[2])) # (r, theta, phi components)
shape = m_['shape']
 
#Make G to evaluate for full j based on the model made above
G = secs3d.make_G(m_['grid'], m_['alts_grid'], lat, lon, alt)
full_j = G.dot(m)

#Get the jperp and fac of the ful_j expressed by the model, to be compared to input
br, btheta, bphi = secs3d.make_b_unitvectors(datadict['Bu'], 
                -datadict['Bn'], datadict['Be'])
N = br.size
B = secs3d.make_B(br, btheta, bphi)
P = secs3d.make_P(N)
j_perp = P.T.dot(B.dot(P.dot(full_j)))
jpar = np.sum(np.array([full_j[0:N], full_j[N:2*N], full_j[2*N:3*N]]) * 
                np.array([br, btheta, bphi]), axis=0)
data_fac = np.sum(np.array([datadict['ju'], -datadict['jn'], datadict['je']]) * 
                np.array([br, btheta, bphi]), axis=0)
residual_r = d[0:N]-j_perp[0:N]
residual_theta = d[N:2*N]-j_perp[N:2*N]
residual_phi = d[2*N:3*N]-j_perp[2*N:3*N]
residual_fac = data_fac - jpar

##################################3
# Performance plotting
# fig = plt.figure(figsize = (10, 10))
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_axis_off()
ax2.view_init(azim=-26, elev=7)
visualization.spherical_grid(ax2, lat_ev, lon_ev, alt_ev, color='blue') #Data locations
visualization.field_aligned_grid(ax2, m_['grid'], m_['alts_grid'], color='green') # Base SECS grid
kwargs={'linewidth':3}
for kk in range(lat.reshape(shape)[0,-1,:].size): # Plot some field-lines
    visualization.plot_field_line(ax2, lat.reshape(shape)[0,-1,kk], 
                lon.reshape(shape)[0,-1,kk], alt.reshape(shape)[:,-1,0], 
                color='orange', **kwargs, dipole=True)
    visualization.plot_field_line(ax2, lat.reshape(shape)[0,shape[1]//2,kk], 
                lon.reshape(shape)[0,shape[1]//2,kk], 
                alt.reshape(shape)[:,shape[1]//2,0],
                color='orange', **kwargs, dipole=True)
    # visualization.plot_field_line(ax, lat_ev[0,8,kk], lon_ev[0,8,kk], 
                              # alts__)
x, y, z = sph_to_car((RE+datadict['alt'].flatten(), 90-datadict['lat'].flatten(), 
                      datadict['lon'].flatten()), deg=True)
cmap = plt.cm.seismic
clim = 1e-5
norm = matplotlib.colors.Normalize(vmin=-clim, vmax=clim)
p = ax2.plot_surface(x.reshape(shape)[-1,:,:], y.reshape(shape)[-1,:,:], 
                    z.reshape(shape)[-1,:,:], alpha=0.5,
                    facecolors=cmap(norm((jpar).reshape(shape)[-1,:,:])), cmap=cmap)
p = ax2.plot_surface(x.reshape(shape)[:,:,shape[2]//2], y.reshape(shape)[:,:,shape[2]//2], 
                    z.reshape(shape)[:,:,shape[2]//2], alpha=0.5,
                    facecolors=cmap(norm((jpar).reshape(shape)[:,:,shape[2]//2])), cmap=cmap)
x0, y0, z0 = sph_to_car((RE+0, 90-grid.projection.position[1], grid.projection.position[0]), deg=True)
range_ =  alts_grid[-1]*0.3
ax2.set_xlim(x0-range_, x0+range_)
ax2.set_ylim(y0-range_, y0+range_)
ax2.set_zlim(z0, z0+2*range_)
ax2.set_title('FAC from 3D SECS')
# from matplotlib import cm
# mm = cm.ScalarMappable(cmap=cmap, norm=norm)
# # m.set_array([])
# fig.colorbar(mm, label='FAC $[\mu A/m^2]$')


# Scatterplots
#Residual scatterplot
plt.figure()
from gemini3d.grid import convert
mlon_, mtheta_ = convert.geog2geomag(lon,lat)
m_theta = np.arcsin(np.sqrt((RE+alts_grid[0])/(RE+alt))*np.sin(mtheta_)) #sjekk - ok!
m_mlon = mlon_
m_glon, m_glat = convert.geomag2geog(m_mlon, m_theta)
inside = grid.ingrid(m_glon, m_glat, ext_factor=-1)
if jperp:
    plt.scatter(1e6*d[0:N],1e6*j_perp[0:N], s=1, label='j_perp,r')
    plt.scatter(1e6*d[N:2*N],1e6*j_perp[N:2*N], s=1, label='j_perp,theta')
    plt.scatter(1e6*d[2*N:3*N],1e6*j_perp[2*N:3*N], s=1, label='j_perp,phi')
    # plt.scatter(1e6*data_fac,1e6*jpar, s=1, label='jpar')
    plt.scatter(1e6*data_fac[inside],1e6*jpar[inside], s=1, label='jpar inside', 
                zorder=100)
    plt.scatter(1e6*data_fac[~inside],1e6*jpar[~inside], s=1, label='jpar outide', 
                zorder=100)
    plt.title('Modelled with jperp input')
else:
    plt.scatter(1e6*d[0:N],1e6*full_j[0:N], s=1, label='j_r')
    plt.scatter(1e6*d[N:2*N],1e6*full_j[N:2*N], s=1, label='j_theta')
    plt.scatter(1e6*d[2*N:3*N],1e6*full_j[2*N:3*N], s=1, label='j_phi')
    # plt.scatter(1e6*data_fac,1e6*jpar, s=1, label='jpar')
    plt.scatter(1e6*data_fac[inside],1e6*jpar[inside], s=1, label='jpar inside', 
                zorder=100)
    plt.scatter(1e6*data_fac[~inside],1e6*jpar[~inside], s=1, label='jpar outide', 
                zorder=100)
    plt.title('Modelled with full j input')
plt.legend()
plt.xlabel('GEMINI $[\mu A/m^2]$')
plt.ylabel('SECS $[\mu A/m^2]$')

sum(np.abs(residual_r/d[0:N])<0.2)/N
