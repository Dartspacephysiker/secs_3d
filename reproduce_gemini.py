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
from secsy import cubedsphere
import visualization


#Global variables
path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/gemini_output/"
height = 90 #km. height of secs CS grid at bottom boundary. To be used for SECS representation of current
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
alts_grid = np.concatenate((np.arange(90,140,2),np.arange(140,170,5),np.arange(170,230,10)))
# alts_grid = np.arange(height,126,2)
# alts_grid_obs = np.arange(height,126,1)
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


# Get input measurement datapoints    
# datadict = gemini_tools.sample_at_alt(xg, dat, alt=alts_grid, altres=altres, 
#                                   var=var, resfac=0.25)
# datadict = gemini_tools.sample_eiscat(xg, dat, alts_grid_obs)
Nxi = 18#6 #new evaluation resolution: "j" index
Neta = 24#8 #new evaluation resolution: "i" index
alts__ = alts_grid[1:]-altres[1:]
etas = np.linspace(grid.eta_mesh[1,0]+0.01*grid.deta,grid.eta_mesh[-2,0]-0.01*grid.deta,Neta)
xis = np.linspace(grid.xi_mesh[0,1]+0.01*grid.dxi,grid.xi_mesh[0,-2]-0.01*grid.dxi,Nxi)
xi_ev, eta_ev = np.meshgrid(xis, etas, indexing = 'xy')
alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, etas, xis, indexing='ij')
lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev)
sh_sample = lon_ev.shape
datadict = gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)

#Prepare the sampled GEMINI data for inversion
jj, lat, lon, alt = gemini_tools.prepare_model_data(grid, datadict, alts_grid, jperp=jperp)
d = np.hstack((jj[0], jj[1], jj[2])) # (r, theta, phi components)
G = secs3d.make_G(grid, alts_grid, lat, lon, alt)
if jperp:
    br, btheta, bphi = secs3d.make_b_unitvectors(jj[3],jj[4],jj[5])
    B = secs3d.make_B(br, btheta, bphi)    
    P = secs3d.make_P(br.size)
    G = P.T.dot(B.dot(P.dot(G)))
GTG = G.T.dot(G)
R = np.diag(np.ones(GTG.shape[0])) * np.median(np.diag(GTG)) * 0.01
GTd = G.T.dot(d)

#Solve
m = lstsq(GTG + R, GTd, cond=0.)[0]

if jperp:
    sss = 'm_perp.npy'
else:
    sss = 'm.npy'
np.save(sss, m)

############################################################
############################################################


m = np.load(sss)

# Evaluate model FIX: make it possible to specify evaluation/data location
# outside 3D domain spanned by the secs and vertical grid. At present it chrashes
Nxi = 50#50 #new evaluation resolution: "j" index
Neta = 40#40 #new evaluation resolution: "i" index
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
    print('Evaluation locations outside secs grid. Will stop.')
    print(1/0)
G = secs3d.make_G(grid, alts_grid,lat, lon, alt, interpolate=interpolate)


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


# Fixed altitude plotting
for alt_i in range(sh[0]-1):
    if jperp:
        fix = 'jperp_'
    else:
        fix = ''    
    filename = './plots/tmp/'+fix+'snap_'+p+'_%03i.png' % alt_i
    
    #Make 1x3 plot with following panels:
    #panel 1: GEMINI-input, in muA/m2
    xi, eta = grid.projection.geo2cube(datadict['lon'].reshape(sh_sample)[alt_i,:,:], datadict['lat'].reshape(sh_sample)[alt_i,:,:])
    values = input_param.reshape(sh_sample)[alt_i,:,:]
    glat = datadict['lat'].reshape(sh_sample)[alt_i,:,:]
    glon = datadict['lon'].reshape(sh_sample)[alt_i,:,:]
    d1 = {'xi':xi, 'eta':eta, 'values':values, 
          'title':'GEMINI: '+p+' @ '+str(alts_grid[alt_i])+' km', 'glat':glat, 
          'glon':glon, 'alt':alts_grid[alt_i], 'filename':filename, 
          'xirange':(grid.xi_min,grid.xi_max), 'etarange':(grid.eta_min,grid.eta_max)}
    
    #panel 2: Reconstructed ju, in muA/m2
    xi, eta = grid.projection.geo2cube(lon_ev[alt_i,:,:], lat_ev[alt_i,:,:])
    values = model_param[alt_i,:,:]
    d2 = {'xi':xi, 'eta':eta, 'values':values, 
          'title':'3D SECS: '+p+' @ '+str(alt_ev[alt_i,0,0])+' km', 
          'glat':lat_ev[alt_i,:,:], 'glon':lon_ev[alt_i,:,:], 'alt':alt_ev[alt_i,0,0],
          'xirange':(grid.xi_min,grid.xi_max), 'etarange':(grid.eta_min,grid.eta_max), 
          'plotgrid':grid, 'filename':filename}
    
    #panel 3: Delta jr in layer, in muA/m2
    Ar = grid.A * (alts_grid[alt_i]+RE)**2/(grid.R*1e-3)**2
    djr = -1e6*10*m_cf[alt_i,:,:] * altres[alt_i]*1e3 / Ar
    # plt.pcolormesh(grid.xi, grid.eta, djr*1e6, cmap='seismic', 
    #                 vmin=vmin/10, vmax=vmax/10)
    values = model_param[alt_i,:,:]
    d3 = {'xi':grid.xi, 'eta':grid.eta, 'values':djr, 
          'title':'3D SECS: $10 \cdot \Delta j_u$ @ '+str(alts_grid[alt_i])+' km', 
          'glat':grid.lat, 'glon':grid.lon, 'alt':alts_grid[alt_i], 
          'xirange':(grid.xi_min,grid.xi_max), 'etarange':(grid.eta_min,grid.eta_max), 
          'plotgrid':grid, 'filename':filename}
        
    visualization.fixed_alt((d1,d2,d3), cbartitle = '$[\mu A/m^2]$')

    


import glob
files = glob.glob('./plots/tmp/'+fix+'snap_'+p+'*.png')
files.sort()
gifname = './plots/tmp/'+fix+p+'_plot.gif'   
from cmodel.helpers import make_gif
make_gif(files, filename=gifname, fps=2)

#Altitude profile
visualization.altitude_profile(m, K, I, J, alts_grid, i=6, j=6)

################
#Residual plots

# Evaluation grid
Nxi = 20 #new evaluation resolution: "j" index
Neta = 14 #new evaluation resolution: "i" index
alts__ = alts_grid[1:]-altres[1:]
# alts__ = np.arange(90,200,2)
etas = np.linspace(grid.eta_mesh[1,0]+0.01*grid.deta,grid.eta_mesh[-2,0]-0.01*grid.deta,Neta)
xis = np.linspace(grid.xi_mesh[0,1]+0.01*grid.dxi,grid.xi_mesh[0,-2]-0.01*grid.dxi,Nxi)
xi_ev, eta_ev = np.meshgrid(xis, etas, indexing = 'xy')
alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, etas, xis, indexing='ij')
lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev)
sh = lon_ev.shape

#Input data
datadict = gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)
jj, lat, lon, alt = gemini_tools.prepare_model_data(grid, datadict, alts_grid, 
                                                    jperp=False, ext_factor=-1)
d = np.hstack((jj[0], jj[1], jj[2])) # (r, theta, phi components) observed j_perp

#Full j at input locations
G = secs3d.make_G(grid, alts_grid, lat, lon, alt, ext_factor=-1)
full_j = G.dot(m)
N = lat.size

# Plot grid and coastlines:
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
from pysymmetry.utils.spherical import sph_to_car, car_to_sph
sitelat = 67.35 #69.38
sitephi = 23. #20.30
ski_x, ski_y, ski_z = sph_to_car((RE, 90-sitelat, sitephi), deg=True)
xlim = (ski_x[0]-0.5*grid.L*1e-3, ski_x[0]+0.5*grid.L*1e-3) 
ylim = (ski_y[0]-0.5*grid.L*1e-3, ski_y[0]+0.5*grid.L*1e-3) 
zlim = (RE, RE+alts_grid[-1])
ax.scatter(ski_x, ski_y, ski_z, marker='s', color='yellow')
for cl in visualization.get_coastlines():
    x,y,z = sph_to_car((RE, 90-cl[1], cl[0]), deg=True)
    use = (x > xlim[0]-200) & (x < xlim[1]+200) & (y > ylim[0]-200) & (y < ylim[1]+200) & (z > 0)
    ax.plot(x[use], y[use], z[use], color = 'C0')
residual_r = d[0:N]-full_j[0:N]
residual_theta = d[N:2*N]-full_j[N:2*N]
residual_phi = d[2*N:3*N]-full_j[2*N:3*N]
x, y, z = sph_to_car((RE+alt, 90-lat, lon), deg=True)
cmap = plt.cm.seismic
import matplotlib
vmin = -10e-6
vmax = 10e-6
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
# nnn = residual_r.reshape(sh)[:,0,:]/residual_r.reshape(sh)[:,0,:].max()
p = ax.plot_surface(x.reshape(sh)[:,-1,:], y.reshape(sh)[:,-1,:], z.reshape(sh)[:,-1,:], 
                    facecolors=cmap(norm(residual_r.reshape(sh)[:,-1,:])), cmap=cmap)
p = ax.plot_surface(x.reshape(sh)[10,:,:], y.reshape(sh)[10,:,:], z.reshape(sh)[10,:,:], 
                    facecolors=cmap(norm(residual_r.reshape(sh)[10,:,:])), cmap=cmap)
# p = ax.scatter(x, y, z, c=residual_r[use]/d[0:N][use], cmap='seismic', vmin = -1, vmax=1)
# p = ax.scatter(x, y, z, c=residual_r, cmap='seismic', vmin=-10e-6, vmax=10e-6)
# ax.scatter(x, y, z, c=np.sqrt(d[0:N][use]**2+d[N:2*N][use]**2+d[2*N:3*N][use]**2), cmap='viridis')#, vmin = -1, vmax=1)
# ax.set_title('Altitude > '+str(aaa)+' km')
ax.set_xlim3d(xlim)
ax.set_ylim3d(ylim)
from matplotlib import cm
mm = cm.ScalarMappable(cmap=cmap, norm=norm)
# m.set_array([])
fig.colorbar(mm, label='residual jr $[\mu A/m^2]$')


#Residual scatterplot
br, btheta, bphi = secs3d.make_b_unitvectors(jj[3],jj[4],jj[5])
d_jpar = np.sum(np.array([jj[0], jj[1], jj[2]]) * np.array([br, btheta, bphi]), axis=0)
m_jpar = np.sum(np.array([full_j[0:N], full_j[N:2*N], full_j[2*N:3*N]]) * 
                np.array([br, btheta, bphi]), axis=0)
plt.figure()
plt.scatter(1e6*d[0:N],1e6*full_j[0:N], s=1, label='jr')
plt.scatter(1e6*d[N:2*N],1e6*full_j[N:2*N], s=1, label='jtheta')
plt.scatter(1e6*d[2*N:3*N],1e6*full_j[2*N:3*N], s=1, label='jphi')
plt.scatter(1e6*d_jpar,1e6*m_jpar, s=1, label='jpar')
plt.legend()
plt.xlabel('GEMINI $[\mu A/m^2]$')
plt.ylabel('SECS $[\mu A/m^2]$')
plt.title(sss)


#Residual scatterplot of inside vs outside points
from gemini3d.grid import convert
mlon_, mtheta_ = convert.geog2geomag(lon,lat)
m_theta = np.arcsin(np.sqrt((RE+alts_grid[0])/(RE+alt))*np.sin(mtheta_)) #sjekk
m_mlon = mlon_
m_glon, m_glat = convert.geomag2geog(m_mlon, m_theta)
inside = grid.ingrid(m_glon, m_glat, ext_factor=-1)
plt.figure()
plt.scatter(1e6*d_jpar[inside],1e6*m_jpar[inside], s=1, label='jpar inside', zorder=100)
plt.scatter(1e6*d_jpar[~inside],1e6*m_jpar[~inside], s=1, label='jpar outide', zorder=100)
# plt.scatter(1e6*d[N:2*N],1e6*full_j[N:2*N], s=1, label='jtheta')
# plt.scatter(1e6*d[2*N:3*N],1e6*full_j[2*N:3*N], s=1, label='jphi')
plt.legend()
plt.title(sss)
plt.xlabel('GEMINI $[\mu A/m^2]$')
plt.ylabel('SECS $[\mu A/m^2]$')

plt.figure()
plt.hist(residual_r/d[0:N], bins=50, range=(-4,4))
plt.xlabel('$j_r$ residual / $j_r$ GEMINI')
plt.ylabel('#')

sum(np.abs(residual_r/d[0:N])<0.2)/N

#####
# Investigate div(j) 
# SECS Model div(j)
Nlon = grid.shape[1] #new evaluation resolution: "j" index
Nlat = grid.shape[0] #new evaluation resolution: "i" index
alts__ = np.arange(90,200,2)
lons = np.linspace(19, 25, Nlon)
lats = np.linspace(67., 69.1, Nlat)
alt_ev, lat_ev, lon_ev = np.meshgrid(alts__, lats, lons, indexing='ij')
shape = lon_ev.shape
G = secs3d.make_G(grid, alts_grid, lat_ev.flatten(), lon_ev.flatten(), 
                  alt_ev.flatten(), ext_factor=0)
full_j = G.dot(m)
N = lat_ev.size
sampledict = {'glat':lat_ev, 'glon':lon_ev, 'alt':alt_ev, 
              'je':full_j[2*N:3*N].reshape(shape), 
              'jn':-full_j[1*N:2*N].reshape(shape), 
              'ju':full_j[0*N:1*N].reshape(shape)}
divj_sph = gemini_tools.divergence_spherical(sampledict, hor=False)
for i in range(0,50, 1):
    plt.figure()
    plt.pcolormesh(lon_ev[i,:,:], lat_ev[i,:,:], divj_sph[i,:,:], 
                   vmin=-1e-9, vmax=1e-9, cmap='seismic')
    # plt.xlim(-4,3)
    # plt.ylim(26,46)
    plt.title('Altitude = ' + str(alts__[i]))

# SECS Model div(j_perp)
Nlon = grid.shape[1] #new evaluation resolution: "j" index
Nlat = grid.shape[0] #new evaluation resolution: "i" index
alts__ = np.arange(90,200,2)
lons = np.linspace(19, 25, Nlon)
lats = np.linspace(67., 69.1, Nlat)
alt_ev, lat_ev, lon_ev = np.meshgrid(alts__, lats, lons, indexing='ij')
shape = lon_ev.shape
G = secs3d.make_G(grid, alts_grid, lat_ev.flatten(), lon_ev.flatten(), 
                  alt_ev.flatten(), ext_factor=0)
full_j = G.dot(m)
datadict = gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)
br, btheta, bphi = secs3d.make_b_unitvectors(datadict['Bu'].reshape(shape), 
                    -datadict['Bn'].reshape(shape), datadict['Be'].reshape(shape))
N = br.size
B = secs3d.make_B(br, btheta, bphi)
P = secs3d.make_P(N)
j_perp = P.T.dot(B.dot(P.dot(full_j)))
sampledict = {'glat':lat_ev, 'glon':lon_ev, 'alt':alt_ev, 
              'je':j_perp[2*N:3*N].reshape(shape), 
              'jn':j_perp[1*N:2*N].reshape(shape), 
              'ju':j_perp[0*N:1*N].reshape(shape)}
divjperp_sph = gemini_tools.divergence_spherical(sampledict, hor=False)
for i in range(0,50, 1):
    plt.figure()
    plt.pcolormesh(lon_ev[i,:,:], lat_ev[i,:,:], divjperp_sph[i,:,:], 
                   vmin=-1e-9, vmax=1e-9, cmap='seismic')
    # plt.xlim(-4,3)
    # plt.ylim(26,46)
    plt.title('Altitude = ' + str(alts__[i]))

# GEMINI div(j)
datadict = gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)
sampledict = {'glat':datadict['lat'].reshape(shape), 
              'glon':datadict['lon'].reshape(shape), 
              'alt':datadict['alt'].reshape(shape), 
              'je':datadict['je'].reshape(shape), 
              'jn':datadict['jn'].reshape(shape), 
              'ju':datadict['ju'].reshape(shape)}
divj_sph = gemini_tools.divergence_spherical(sampledict, hor=False)
for i in range(0,30, 1):
    plt.figure()
    plt.pcolormesh(lon_ev[i,:,:], lat_ev[i,:,:], divj_sph[i,:,:], 
                   vmin=-1e-9, vmax=1e-9, cmap='seismic')
    # plt.xlim(-4,3)
    # plt.ylim(26,46)
    plt.title('Altitude = ' + str(alts__[i]))