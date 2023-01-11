#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:27:54 2023

@author: jone

Make sketch of EISCAT_3D FOV

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
height = 100 #km. height of secs CS grid at bottom boundary. To be used for SECS representation of current
RE = 6371.2 #Earth radius in km
apex = apexpy.Apex(2022)
jperp = True #Use only perp components of J as inpur to inversion

#Define CS grid. Note: Not defined with respect to a TX site on ground
xg = read.grid(path)
extend=1
grid, grid_ev = gemini_tools.make_csgrid(xg, height=height, crop_factor=0.2, #0.45
                                    resolution_factor=0.3, extend=extend, 
                                    dlat = 0.) # dlat = 2.0

#Open simulation output files
var = ["v1", "v2", "v3", "Phi", "J1", "J2", "J3", "ne"]
cfg = read.config(path)
xg = read.grid(path)
dims = xg['lx']
times = cfg["time"][-1:]
t = times[0]
dat = read.frame(path, t, var=var)
dat = gemini_tools.compute_enu_components(xg, dat)

#Sample EISCAT at some specific beams
alts_grid = np.array([100,300])
datadict = gemini_tools.sample_eiscat(xg, dat, alts_grid) #"Beams" are hardcoded here for now
lat0 = datadict['lat'][::2]
lon0 = datadict['lon'][::2]
alt0 = datadict['alt'][::2]
lat1 = datadict['lat'][1:][::2]
lon1 = datadict['lon'][1:][::2]
alt1 = datadict['alt'][1:][::2]

####################3
# Make the E3D FOV Figure
az=-26
el=7
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
ax.view_init(azim=az, elev=el)
kwargs={'label':'Field-aligned grid'}
visualization.field_aligned_grid(ax, grid, alts_grid, color='green', 
                    fullbox=True, verticalcorners=True, **kwargs)
kwargs={'label':'30$^\circ$ elevation'}
visualization.plot_e3dfov(ax, lat0, lon0, alt0, lat1, lon1, alt1, **kwargs)
ax.legend()