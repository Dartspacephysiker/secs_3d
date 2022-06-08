#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 06:39:21 2022

@author: jone
"""

from gemini3d.grid.convert import unitvecs_geographic
import numpy as np
from lompe.secsy import cubedsphere
from lompe.data_tools.dataloader import getbearing
import great_circle_calculator.great_circle_calculator as gcc


######
#Contain helper functions needed in the 3D SECS representation




def model_vec2geo_vec(xg, dat, param='v'):
    '''
    Function to convert model vector components into geographic conponents. 
    Code provided by M. Zettergren, and put into this function by JPR.

    Parameters
    ----------
    xg : GEMINI grid object
        contain all info about the grid used in the simulation
    dat : GAMERA data object (xarray) at specific time
        As returned from the read.frame() function.
    param : 'str'
        'v' (default) or 'J', refering to velocity or current density
        
    Returns
    -------
    (radial, south, east) geographic components of velocity (need to check the directions to be sure about signs)

    '''
    ###############################################################################
    # read in a vector quantity, rotate into geographic components and then grid
    ###############################################################################
    # v1=dat["v1"]; v2=dat["v2"]; v3=dat["v3"];
    [egalt,eglon,eglat]=unitvecs_geographic(xg)     #up, east, north
    #^ returns a set of geographic unit vectors on xg; these are in ECEF geomag comps
    #    like all other unit vectors in xg

    # each of the components in models basis projected onto geographic unit vectors
    vgalt=( np.sum(xg["e1"]*egalt,3)*dat[param+"1"] + np.sum(xg["e2"]*egalt,3)*dat[param+"2"] + 
        np.sum(xg["e3"]*egalt,3)*dat[param+"3"] )
    vglat=( np.sum(xg["e1"]*eglat,3)*dat[param+"1"] + np.sum(xg["e2"]*eglat,3)*dat[param+"2"] +
        np.sum(xg["e3"]*eglat,3)*dat[param+"3"] )
    vglon=( np.sum(xg["e1"]*eglon,3)*dat[param+"1"] + np.sum(xg["e2"]*eglon,3)*dat[param+"2"] + 
        np.sum(xg["e3"]*eglon,3)*dat[param+"3"] )
    
    return [vgalt, vglon, vglat] # (up, east, north)??


def make_csgrid(xg, height = 500, crop_factor = 0.6, resolution_factor = 0.5):
    '''
    Put a CubedSphere grid inside GEMINI model domain at specified height

    Parameters
    ----------
    xg : GEMINI grid object
        contain all info about the grid used in the simulation
    height : float or int
        Height in km of where to make the CS grid
    crop_factor : float
        How much to reduce the CS grid compared to GEMINI grid
    resolution_factor : float
        How much to reduce the spatial resolution compared to GEMINI. 0.5 will double the spacing of grid cells

    Returns
    -------
    CS grid object.

    '''

    RE = 6371.2 #Earth radius in km


    #GEMINI grid is field-aligned and orthogonal. These limits are thus not exactly the limits at the fixed height
    #The crop factor is used to make the CS grid fit inside the model domain. Should be changed to something more robust
    # and intuitive in the future
    
    # Find index of height matching the desired height in centre of grid
    dims = xg['lx']
    diff = xg['alt'][:,dims[1]//2, dims[2]//2] - height*1e3
    ii = np.argmin(np.abs(diff))
   
    #Find the orientation of the model grid
    x0 = (xg['glat'][ii,dims[1]//2,dims[2]//2], xg['glon'][ii,dims[1]//2,dims[2]//2])
    x1 = (xg['glat'][ii,1+dims[1]//2,dims[2]//2], xg['glon'][ii,1+dims[1]//2,dims[2]//2])
    orientation = np.degrees(getbearing(np.array([x0[0]]), np.array([x0[1]]), np.array([x1[0]]), np.array([x1[1]])))
    
    #Centre location of CS grid
    position = (xg['glon'][ii,dims[1]//2,dims[2]//2], xg['glat'][ii,dims[1]//2,dims[2]//2]) 

    #Dimensions of CS grid
    p0 = (xg['glon'][ii,0,dims[2]//2],xg['glat'][ii,0,dims[2]//2])
    p1 = (xg['glon'][ii,-1,dims[2]//2],xg['glat'][ii,-1,dims[2]//2])
    d2 = gcc.distance_between_points(p0, p1) #distance in dimension 2 (magntic northsouth ish)
    L = d2 * crop_factor
    Lres = d2/dims[1] / resolution_factor
    p0 = (xg['glon'][ii,dims[1]//2,0],xg['glat'][ii,dims[1]//2,0])
    p1 = (xg['glon'][ii,dims[1]//2,-1],xg['glat'][ii,dims[1]//2,-1])    
    d3 = gcc.distance_between_points(p0, p1) #distance in dimension 3 (magntic east ish)
    W = d3 * crop_factor
    Wres = d3/dims[2] / resolution_factor

    #Make CS grid object
    grid_ev = cubedsphere.CSgrid(cubedsphere.CSprojection(position, -orientation[0]), L, W, Lres, Wres, R = (RE+height)*1e3) #inner grid, to evaluate on
    xi_e  = np.hstack((grid_ev.xi_mesh[0]    , grid_ev.xi_mesh [0 , - 1] + grid_ev.dxi )) - grid_ev.dxi /2 
    eta_e = np.hstack((grid_ev.eta_mesh[:, 0], grid_ev.eta_mesh[-1,   0] + grid_ev.deta)) - grid_ev.deta/2 
    grid = cubedsphere.CSgrid(cubedsphere.CSprojection(grid_ev.projection.position, grid_ev.projection.orientation),
                       grid_ev.L + grid_ev.Lres, grid_ev.W + grid_ev.Wres, grid_ev.Lres, grid_ev.Wres, 
                       edges = (xi_e, eta_e), R = grid_ev.R) #outer grid that represent E-field
    
    return grid, grid_ev
    
    # #plot grid to compare with model grid
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.pcolormesh(xg['glon'][ii,:,:], xg['glat'][ii,:,:], xg['alt'][ii,:,:])
    # ax.pcolormesh(grid.lon, grid.lat, grid.lat)
    # ax.set_xlim(-20,50)

def dipole_B(theta, height = 500):
    #theta must be in radians
    mu0 = 4*np.pi * 10**(-7)
    m = 7.94e22 #magnetic dipole moment, taken from GEMINI documentation
    RE = 6371.2 * 1e3
    Bu = -2 * mu0 * m * np.cos(theta) / (4*np.pi*(RE+height*1e3)**3)
    Btheta =  -mu0 * m * np.sin(theta) / (4*np.pi*(RE+height*1e3)**3)
    Bn = -Btheta
    return (Bn, Bu)