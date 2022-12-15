#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:24:53 2022

@author: jone

Script to make sample dataset of ionospheric 3D currents in spherical geometry.
The intension is to investigate the effect of inclination by settiung up a very
simple situation (yet physically self consistent and realistic), from which we
can sample from when applying the 3D-SECS technique, and see how well our 
reconstruction technique performs in various regions, with various inclinations.

Neutral atmosphere densities and tempoerature are taken from MSIS (through CCMC)
Ion temperature profile from IRI (through CCMC).

We consider the following neutrals: N2, O2, O
We consider the following ions: NO+, O2+, O+

We consider altitudes up to 800 km
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import secs_3d.gemini_tools as gemini_tools
from secs_3d import visualization

#Constants
c_c = 1.546e-16 # From https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JA021396 eq 13
mp = 1.67e-27 #proton mass in kg
me = 9.11e-31
B = 50000e-9
e = 1.6e-19
ZN = 7
ZO = 8
m_n2 = 2*2*ZN * mp
m_no = (2*ZN + 2*ZO) * mp
m_o2 = 2*2*ZO * mp
m_o = 2*ZO * mp
omega_no = e*B/(m_no) #gyro freq
omega_o2 = e*B/(m_o2) #gyro freq
omega_o = e*B/(m_o) #gyro freq
omega_e = e*B/(me) #gyro freq

#Open msis and iri profiles (CCMC output)
msis = pd.read_csv('msis-90.txt', header=7, names=['Height','O', 'N2', 'O2', 
                 'Temperature_neutral', 'Temperature_exospheric'], 
                   delim_whitespace=True)
iri = pd.read_csv('iri2016.txt', header=6, names=['Height','ne', 'Tn', 'Ti', 
                 'Te'], delim_whitespace=True)


# Collission frequencies
# Use forula 13 from https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JA021396
# Dominating species are taken from eq A4 in https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JA027128
c_no_n2 = c_c * msis.N2.values*1e6 * ((2*ZN)/(ZN+ZO + 2*ZN)) * np.sqrt(iri.Ti.values/(ZN+ZO) + msis.Temperature_neutral.values/(2*ZN))
c_no_o2 = c_c * msis.O2.values*1e6 * ((2*ZO)/(ZN+ZO + 2*ZO)) * np.sqrt(iri.Ti.values/(ZN+ZO) + msis.Temperature_neutral.values/(2*ZO))
c_no_o = c_c * msis.O.values*1e6 * ((ZO)/(ZN+ZO + ZO)) * np.sqrt(iri.Ti.values/(ZN+ZO) + msis.Temperature_neutral.values/(ZO))
n_n = np.mean(np.vstack((msis.O.values,msis.N2.values,msis.O2.values)), axis=0)
n_i = iri['ne'].values
kn2 = 4.34e-16
ko2 = 4.28e-16
ko = 2.44e-16
# c_brekke = 2.6e-15 * (n_n+n_i) * (30.7 * mp)**(-.5)
c_brekke = kn2 * msis.N2.values*1e6 + ko2 * msis.O2.values*1e6 + ko * msis.O.values*1e6


# #Mobility plot, to be compared with Figure 5.1 in Brekke book
# plt.plot(omega_no/c_no_n2,msis.Height, label='NO-N2 collission')
# plt.plot(omega_no/c_brekke,msis.Height, label='Average (Brekke formula)')
# plt.plot(omega_no/(c_no_n2+c_no_o2+c_no_o), msis.Height, label='N2+O2+O collissions')
# plt.xscale('log')
# plt.ylim([0,400])
# plt.xlim([1e-4,1e8])
# plt.xlabel('NO+ mibility')
# plt.legend()

# Compute conductivity using eqs 29 and 30 in https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2000JA000423
sp = e*iri['ne']/B * (c_brekke * omega_no)/(omega_no**2 + c_brekke**2)
sh = e*iri['ne']/B * (c_brekke**2)/(omega_no**2 + c_brekke**2)

#Conductivity plot
# plt.plot(sp, msis.Height, label='Pedersen')
# plt.plot(sh, msis.Height, label='Hall')
# plt.xscale('log')
# plt.ylim([0,400])
# plt.xlim([1e-12,1e-4])
# plt.legend()
# plt.xlabel('[S/m]')

# Make grid at top boundary
from secsy import cubedsphere
from secs_3d.gemini_tools import RE
import apexpy
import ppigrf
import datetime as dt
from pysymmetry.utils.spherical import sph_to_car, car_to_sph


# position = (0,35) # approx 45 deg inclination
position = (20,70) #Close to Skibotn, small inclination
height = 220
Be, Bn, Bu = ppigrf.igrf(position[0],position[1],height, date=dt.datetime.now())
H = np.sqrt(Be**2+Bn**2)
I = np.degrees(np.arctan(H/np.abs(Bu)))
a = apexpy.Apex(2022)
f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = a.basevectors_apex(position[1], position[0], height=height)
orientation = 0 # np.degrees(np.arctan2(f2[0],f2[1]))
L, W, Lres, Wres = 200e3,700e3, 10e3, 10e3 #L will be in magnetic east-west direction
grid = cubedsphere.CSgrid(cubedsphere.CSprojection(position, 
            -orientation), L, W, Lres, Wres, R = (RE+height)*1e3) 

#Define hortizontal flow channel at top boundary
mu = -0.012 #center location of channel in grid, in eta location
sigma = grid.eta_max/10 #Width of channel, units of eta coordinate
A = 500 #amplitude of velicity, m/s
g = A * np.exp(-0.5*((grid.eta[:,0]-mu)/sigma)**2)
ve = np.tile(g[:,np.newaxis],grid.shape[1])
vn = np.zeros(ve.shape)
vu = np.zeros(ve.shape)
v = np.vstack((ve.flatten(),vn.flatten(),vu.flatten()))

#Electric field at top boundary
Be, Bn, Bu = ppigrf.igrf(grid.lon,grid.lat,height, date=dt.datetime.now())
B = np.vstack((Be.flatten(),Bn.flatten(),Bu.flatten()))*1e-9 # in T
E = -np.cross(v,B, axis=0) #in V/m

#Altitude grid
alts_grid = np.linspace(100,height,(height-100)//2+1)
# alts_grid = np.concatenate((np.arange(100,150,1),np.arange(150,190,4),np.arange(190,height+10,10)))
altres = np.diff(alts_grid)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))

#Interpolate conductivity onto grid
from scipy.interpolate import interp1d
f = interp1d(iri.Height.values, sh.values)
sh_grid = f(alts_grid)
f = interp1d(iri.Height.values, sp.values)
sp_grid = f(alts_grid)

#Mapped E-field and B-field at each height
import secs_3d.secs3d
alat, alon = a.geo2apex(grid.lat, grid.lon, height)
Em = np.zeros((3, alts_grid.size, grid.shape[0], grid.shape[1]))
bm = np.zeros((3, alts_grid.size, grid.shape[0], grid.shape[1]))
Bm = np.zeros((3, alts_grid.size, grid.shape[0], grid.shape[1]))
glon = np.zeros((alts_grid.size, grid.shape[0], grid.shape[1]))
glat = np.zeros((alts_grid.size, grid.shape[0], grid.shape[1]))
alt = np.zeros((alts_grid.size, grid.shape[0], grid.shape[1]))
for i in range(alts_grid.size):
    Em_ = a.map_E_to_height(alat.flatten(), alon.flatten(), height, alts_grid[i], E)
    Em[:,i,:,:] = Em_.reshape((3,grid.shape[0], grid.shape[1]))
    glat_, glon_, _ = a.apex2geo(alat, alon, alts_grid[i])
    glat[i,:,:] = glat_
    glon[i,:,:] = glon_
    alt[i,:,:] = alts_grid[i]
    Be, Bn, Bu = ppigrf.igrf(glon_,glat_,alts_grid[i], date=dt.datetime(2022,1,1))
    B = np.vstack((Be.flatten(),Bn.flatten(),Bu.flatten()))*1e-9 # in T
    be, bn, bu = secs_3d.secs3d.make_b_unitvectors(B[0],B[1],B[2])
    bhat = np.vstack((be,bn,bu))    
    bm[:,i,:,:] = bhat.reshape((3,grid.shape[0], grid.shape[1]))
    Bm[:,i,:,:] = B.reshape((3,grid.shape[0], grid.shape[1]))

# Calculate perpendiciular current (e,n,u components) (Inclination grid)
jperp = sp_grid[np.newaxis,:,np.newaxis,np.newaxis]*Em + \
            sh_grid[np.newaxis,:,np.newaxis,np.newaxis] * np.cross(bm, Em, axis=0)

# Interpolate jperp onto a regular grid (not shifted along B). Use spherical grid
# and scipys griddata function to do the interpolation onto this grid. (The 
# mapped grid is not regular)
Nlon = grid.shape[1] #new evaluation resolution: "j" index
Nlat = grid.shape[0] #new evaluation resolution: "i" index
alts__ = alts_grid#[1:]-altres[1:]
# lons = np.linspace(glon.min(), glon.max(),Nlon)
# lats = np.linspace(glat.min(), glat.max(),Nlat)
# lons = np.linspace(-0.85, 0.72, Nlon)
# lats = np.linspace(33.5, 36., Nlat)
lons = np.linspace(18, 22, Nlon)
lats = np.linspace(68, 72., Nlat)
alt_ev, lat_ev, lon_ev = np.meshgrid(alts__, lats, lons, indexing='ij')
shape = lon_ev.shape
jperp_e_sph = np.ones(shape) * np.nan
jperp_n_sph = np.ones(shape) * np.nan
jperp_u_sph = np.ones(shape) * np.nan
Be, Bn, Bu = ppigrf.igrf(lon_ev,lat_ev,alt_ev, date=dt.datetime(2022,1,1))
alat, alon = a.geo2apex(lat_ev, lon_ev, alt_ev)
m_glat, m_glon, _ = a.apex2geo(alat, alon, 100)
# use = ((m_glat>lats[0]) & (m_glat<lats[-1]) & (m_glon>lons[0]) & (m_glon<lons[-1]))
for i in range(len(alts__)):
    # xi, eta = grid.projection.geo2cube(lon_ev[i,:,:], lat)
    evaluate = np.array((lat_ev[i,:,:].flatten(), lon_ev[i,:,:].flatten())).T
    jperp_e_sph[i,:,:] = scipy.interpolate.griddata((glat[i,:,:].flatten(),
                    glon[i,:,:].flatten()), jperp[0,i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(shape[1:])
    jperp_n_sph[i,:,:] = scipy.interpolate.griddata((glat[i,:,:].flatten(),
                    glon[i,:,:].flatten()), jperp[1,i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(shape[1:])
    jperp_u_sph[i,:,:] = scipy.interpolate.griddata((glat[i,:,:].flatten(),
                    glon[i,:,:].flatten()), jperp[2,i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(shape[1:])


#Compute divergence of j_perp on the regular spherical grid
sampledict = {'glat':lat_ev, 'glon':lon_ev, 'alt':alt_ev, 'je':jperp_e_sph, 
              'jn':jperp_n_sph, 'ju':jperp_u_sph, 'Be':Be[0,:,:,:], 
              'Bn':Bn[0,:,:,:], 'Bu':Bu[0,:,:,:]}
divjperp_sph = gemini_tools.divergence_spherical(sampledict, hor=False, perp=False)

#Interpolate back to original grid (layeres preserving field-line identity)
divjperp_B = np.ones(shape) * np.nan
for i in range(len(alts_grid)):
    evaluate = np.array((glat[i,:,:].flatten(), glon[i,:,:].flatten())).T
    divjperp_B[i,:,:] = scipy.interpolate.griddata((lat_ev[i,:,:].flatten(),
                    lon_ev[i,:,:].flatten()), divjperp_sph[i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(glat.shape[1:])

# Apply current cuntinuity (Assumes that FAC is zero at bottom boundary)
# This will only work in the region within the sph grid where the field lines
# extend from top to bottom
fac = np.cumsum(divjperp_B*((altres*2*1e3)/np.cos(np.radians(I[0])))[:,np.newaxis,np.newaxis], axis=0)
# fac = np.cumsum(divjperp_B*((altres*2*1e3))[:,np.newaxis,np.newaxis], axis=0)
jr_B = jperp[2,:,:,:] + fac * bm[2,:,:,:]
jtheta_B = -jperp[1,:,:,:] - fac * bm[1,:,:,:]
jphi_B = jperp[0,:,:,:] + fac * bm[0,:,:,:]
j = np.stack((jr_B, jtheta_B, jphi_B))
data = {'lat':glat, 'lon':glon, 'alt':alt, 'jperp':jperp, 
        'vetop':ve, 'B':Bm, 'E':Em, 'fac':fac, 'grid':grid, 'j':j, 
        'alts_grid':alts_grid}
np.save('inclination_dataset_B.npy', data)

# Test that div(j) = 0 
# Interpolate the r, theta, phi components on the B-following grid onto a 
# regular spherical grid
jr_sph = np.ones(shape) * np.nan
jtheta_sph = np.ones(shape) * np.nan
jphi_sph = np.ones(shape) * np.nan
fac_sph = np.ones(shape) * np.nan

for i in range(len(alts_grid)):
    # xi, eta = grid.projection.geo2cube(lon_ev[i,:,:], lat)
    evaluate = np.array((lat_ev[i,:,:].flatten(), lon_ev[i,:,:].flatten())).T
    jr_sph[i,:,:] = scipy.interpolate.griddata((glat[i,:,:].flatten(),
                    glon[i,:,:].flatten()), jr_B[i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(shape[1:])
    jtheta_sph[i,:,:] = scipy.interpolate.griddata((glat[i,:,:].flatten(),
                    glon[i,:,:].flatten()), jtheta_B[i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(shape[1:])
    jphi_sph[i,:,:] = scipy.interpolate.griddata((glat[i,:,:].flatten(),
                    glon[i,:,:].flatten()), jphi_B[i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(shape[1:])
    fac_sph[i,:,:] = scipy.interpolate.griddata((glat[i,:,:].flatten(),
                    glon[i,:,:].flatten()), fac[i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(shape[1:])

# Compute divergence of full current field (should be zero)
sampledict = {'glon':lon_ev, 'glat':lat_ev, 'alt':alt_ev,'je':jphi_sph, 
              'jn':-jtheta_sph, 'ju':jr_sph}
divj_sph = gemini_tools.divergence_spherical(sampledict, hor=False)

# Interpolate divj onto field-aligned grid
divj_B = np.ones(shape) * np.nan
for i in range(len(alts_grid)):
    evaluate = np.array((glat[i,:,:].flatten(), glon[i,:,:].flatten())).T
    divj_B[i,:,:] = scipy.interpolate.griddata((lat_ev[i,:,:].flatten(),
                    lon_ev[i,:,:].flatten()), divj_sph[i,:,:].flatten(), 
                    (evaluate[:,0],evaluate[:,1])).reshape(glat.shape[1:])


###################################
# Illustrate the different volumes involved
# 1) Largest volume where jperp is known, in the B-following grid
# 2) 3D Spherical segment in sph coords. Jprep_B is interpolated onto this grid
#    to compute divjperp_sph
# 3) Cropped version of 2) where full j is known. jpar is estimated from 
#    current continuity, after divjperp_B is estimated from interpolating 
#    divjperp_sph onto B-following grid. jpar_sph is found from interpolating
#    jpar_B back to the spherical grid. Since jpar_B is only defined where divjpar_B
#    has a value from the bottom of the domain and along the field line, the 
#    region having a finite jpar value will be this cropped version of 2), cropped
#    along the last field line that reaches all the way to the bottom.

# Initialize figure
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-129, azim=-1)

# Plot 1) volume
visualization.field_aligned_grid(ax, grid, alts_grid, color='green')

# Plot 2) volume
visualization.spherical_grid(ax, lat_ev, lon_ev, alt_ev, color='red')

# Plot 3) volume
# plt.pcolormesh(np.sum(fac_sph, axis=0))
imin =  2#3
jmin = 2#3
imax = 64#40
jmax = 15#16
lat_ev2 =lat_ev[:,imin:imax+1, jmin:jmax+1]
lon_ev2 =lon_ev[:,imin:imax+1, jmin:jmax+1]
alt_ev2 =alt_ev[:,imin:imax+1, jmin:jmax+1]
visualization.spherical_grid(ax, lat_ev2, lon_ev2, alt_ev2, color='blue')

# Plot imposed flow channel on top of grid
cmap = plt.cm.viridis
import matplotlib
vmin = 0
vmax = A
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
shape = alt.shape
x, y, z = sph_to_car((RE+alt.flatten(), 90-glat.flatten(), glon.flatten()), deg=True)
p = ax.plot_surface(x.reshape(shape)[-1,:,:], y.reshape(shape)[-1,:,:], 
                    z.reshape(shape)[-1,:,:], alpha=0.5,
                    facecolors=cmap(norm(ve)), cmap=cmap)

#Save dataset within the blue box in where all components are known (spherical 
# componenbts)
Be, Bn, Bu = ppigrf.igrf(lon_ev2, lat_ev2, alt_ev2, date=dt.datetime(2022,1,1))
B = np.vstack((Bu,-Bn,Be)) # r, theta, phi components
data = {'lat':lat_ev2,'lon':lon_ev2,'alt':alt_ev2,'jr':jr_sph[:,imin:imax+1,jmin:jmax+1], 
        'jtheta':jtheta_sph[:,imin:imax+1, jmin:jmax+1], 
        'jphi':jphi_sph[:,imin:imax+1, jmin:jmax+1], 
        'fac':fac_sph[:,imin:imax+1, jmin:jmax+1], 
        'jperp_r':jperp_u_sph[:,imin:imax+1, jmin:jmax+1], 
        'jperp_theta':-jperp_n_sph[:,imin:imax+1, jmin:jmax+1], 
        'jperp_phi':jperp_e_sph[:,imin:imax+1, jmin:jmax+1], 'B':B, 
        'grid':grid, 'alts_grid':alts_grid,
        've_top':ve[imin:imax+1, jmin:jmax+1]}
np.save('inclination_dataset_sph_'+str(position[0])+'-'+str(position[1])+'.npy', data)

# #
# plt.figure()
# ii = 30
# jj=8
# plt.plot(alts_grid,divjperp_B[:,ii,jj])
# plt.plot(alts_grid,divj_B[:,ii,jj])
# # plt.plot(alts_grid,divjperp_sph[:,ii,jj])
# # plt.plot(alts_grid,divj_sph[:,ii,jj])

# plt.figure()
# plt.plot(alts_grid, np.abs(divjperp_B[:,ii,jj]/divj_B[:,ii,jj]))
# plt.ylim([0,20])

# plt.figure()
# i = 0
# plt.pcolormesh(glon[i,:,:], glat[i,:,:], divjperp_B[i,:,:])
# plt.xlim(-4,3)
# plt.ylim(26,46)

# plt.pcolormesh(divjperp_sph[:,:,10], vmin=-1e-12, vmax=1e-12,cmap='seismic')

# for i in range(10,30, 1):
#     plt.figure()
#     # i = 20
#     # plt.pcolormesh(lon_ev[i,:,:], lat_ev[i,:,:], divjperp_sph[i,:,:]/divj_sph[i,:,:], vmin=-10, vmax=10, cmap='seismic')
#     plt.pcolormesh(lon_ev[i,:,:], lat_ev[i,:,:], divj_sph[i,:,:], vmin=-1e-12, vmax=1e-12, cmap='seismic')
#     # plt.pcolormesh(lon_ev[i,:,:], lat_ev[i,:,:], divjperp_sph[i,:,:], vmin=-1e-12, vmax=1e-12, cmap='seismic')
#     # plt.pcolormesh(lon_ev[i,:,:], lat_ev[i,:,:], d0[i,:,:], vmin=-1e-12, vmax=1e-12, cmap='seismic')
#     # plt.pcolormesh(lon_ev[i,:,:], lat_ev[i,:,:], jperp_u_sph[i,:,:], vmin=-1e-8, vmax=1e-8, cmap='seismic')
#     plt.xlim(-4,3)
#     plt.ylim(26,46)
#     plt.title('Altitude = ' + str(100+i))

# for i in range(10,23, 1):
#     plt.figure()
#     # i = 17
#     # plt.pcolormesh(glon[i,:,:], glat[i,:,:], fac[i,:,:], vmin=-1e-8, vmax=1e-8, cmap='seismic')
#     plt.pcolormesh(glon[i,:,:], glat[i,:,:], divjperp_B[i,:,:], vmin=-1e-13, vmax=1e-13, cmap='seismic')
#     # plt.pcolormesh(glon[i,:,:], glat[i,:,:], jperp[1,i,:,:], vmin=-1e-8, vmax=1e-8, cmap='seismic')
#     plt.xlim(-4,3)
#     plt.ylim(26,46)
#     plt.title('Altitude = ' + str(100+i))

#############################################3
# Test of divergence function
# Nlon = 20 #new evaluation resolution: "j" index
# Nlat = 20 #new evaluation resolution: "i" index
# alts__ = np.linspace(100,400,1000) # alts_grid#[1:]-altres[1:]
# lons = np.linspace(0, 90, Nlon)
# lats = np.linspace(0, 60, Nlat)
# alt_ev, lat_ev, lon_ev = np.meshgrid(alts__, lats, lons, indexing='ij')
# shape = lon_ev.shape
# # Be, Bn, Bu = ppigrf.igrf(lon_ev,lat_ev,alt_ev, date=dt.datetime.now())

# jr = np.zeros(shape)
# jtheta = np.cos(10*np.radians(90-lat_ev))
# jphi = np.sin(10*np.radians(lon_ev))
# sampledict = {'glon':lon_ev, 'glat':lat_ev, 'alt':alt_ev,'je':jphi, 'jn':-jtheta, 'ju':jr}
# div = gemini_tools.divergence_spherical(sampledict)
# r = (RE + alt_ev)*1e3
# true_div = (np.cos(10*np.radians(90-lat_ev))*np.cos(np.radians(90-lat_ev)))/(np.sin(np.radians(90-lat_ev)) * r) - \
#             10*np.sin(10*np.radians(90-lat_ev))/r + \
#             10*np.cos(10*np.radians(lon_ev))/(r * np.sin(np.radians(90-lat_ev)))

# from pysymmetry.utils.spherical import enu_to_ecef, sph_to_car
# j = np.hstack((jr.flatten()[:,np.newaxis], jtheta.flatten()[:,np.newaxis], 
#                jphi.flatten()[:,np.newaxis]))
# lon = glon.flatten()
# lat = glat.flatten()
# ecef = enu_to_ecef(j, lon, lat)
# xyz = sph_to_car((RE+alt.flatten(), 90-lat, lon), deg=True)
# jx = ecef[:,0].reshape(glon.shape)
# jy = ecef[:,1].reshape(glon.shape)
# jz = ecef[:,2].reshape(glon.shape)
# sampledict = {'x':xyz[0,:].reshape(glon.shape), 'y':xyz[1,:].reshape(glon.shape), 
#               'z':xyz[2,:].reshape(glon.shape), 'jx':jx, 'jy':jy, 'jz':jz}


# # use projection matrix to retrieve full j
# jperparr = np.hstack((jperp[2,:,:,:].flatten(), -jperp[1,:,:,:].flatten(), jperp[0,:,:,:].flatten()))
# B = secs3d.make_B(bm[2,:,:,:].flatten(), -bm[1,:,:,:].flatten(), bm[0,:,:,:].flatten())
# P = secs3d.make_P(glat.size)
# G = P.T.dot(B.dot(P))
# from scipy.linalg import lstsq


