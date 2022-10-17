#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:55:55 2022

@author: jone


Reproduce GEMINI 3D current density with realistic EISCAT3D sampling coverage
using the 3D SECS technique

"""

import gemini3d.read as read
import numpy as np
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
grid, grid_ev = gemini_tools.make_csgrid(xg, height=height, crop_factor=0.5, #0.45
                                    resolution_factor=0.1, extend=extend) # 0.3 outer (secs), inner (evaluation) grid
singularity_limit=grid.Lres
# alts_grid = np.concatenate((np.arange(80,170,2),np.arange(170,400,10),np.arange(400,850,50)))
# alts_grid = np.arange(90,126,2)
alts_grid = np.hstack((np.arange(80,170,2),np.arange(170,400,10))) #km
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
# ju, je, jn = gemini_tools.model_vec2geo_vec(xg, dat, param='J', perp=True)
# dat['je_perp'] = je
# dat['jn_perp'] = jn
# dat['ju_perp'] = ju

# Sample some data in 3D at realistic EISCAT3D locations
datadict = gemini_tools.sample_eiscat(xg, dat, alts_grid) #0.5

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