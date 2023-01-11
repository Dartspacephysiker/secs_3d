#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:33:22 2022

@author: jone

Tools for working with output of GEMINI model. Tools are developed to enable 
benchmarking the secs3d representation of the current density.

"""


from gemini3d.grid.convert import unitvecs_geographic
import numpy as np
import lompe
from lompe.data_tools.dataloader import getbearing
import great_circle_calculator.great_circle_calculator as gcc
import xarray as xr
from gemini3d.grid.gridmodeldata import model2geogcoords, model2pointsgeogcoords
import pandas as pd
from secsy import cubedsphere
import secs_3d.secs3d as secs3d

RE = 6371.2 #Earth radius in km

def model_vec2geo_vec(xg, dat, param='v', perp=False):
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
    perp : Boolean
        Specifies if only the perpendicular component (2 and 3) of param is to
        be projected to (r, theta, phi) components. Default is False.
    Returns
    -------
    (radial, south, east) geographic components of velocity (need to check the
    directions to be sure about signs)

    '''
    #########################################################################
    # read in a vector quantity, rotate into geographic components and then
    # grid
    ###########################################################################

    [egalt,eglon,eglat]=unitvecs_geographic(xg)     #up, east, north
    #^ returns a set of geographic unit vectors on xg; these are in ECEF geomag
    # comps like all other unit vectors in xg

    # each of the components in models basis projected onto geographic unit 
    # vectors
    if perp:
        vgalt=(np.sum(xg["e2"]*egalt,3)*dat[param+"2"] + 
               np.sum(xg["e3"]*egalt,3)*dat[param+"3"] )
        vglat=(np.sum(xg["e2"]*eglat,3)*dat[param+"2"] +
               np.sum(xg["e3"]*eglat,3)*dat[param+"3"] )
        vglon=(np.sum(xg["e2"]*eglon,3)*dat[param+"2"] + 
               np.sum(xg["e3"]*eglon,3)*dat[param+"3"] )
    else:
        vgalt=( np.sum(xg["e1"]*egalt,3)*dat[param+"1"] + 
               np.sum(xg["e2"]*egalt,3)*dat[param+"2"] + 
               np.sum(xg["e3"]*egalt,3)*dat[param+"3"] )
        vglat=( np.sum(xg["e1"]*eglat,3)*dat[param+"1"] + 
               np.sum(xg["e2"]*eglat,3)*dat[param+"2"] +
               np.sum(xg["e3"]*eglat,3)*dat[param+"3"] )
        vglon=( np.sum(xg["e1"]*eglon,3)*dat[param+"1"] + 
               np.sum(xg["e2"]*eglon,3)*dat[param+"2"] + 
               np.sum(xg["e3"]*eglon,3)*dat[param+"3"] )
    
    return [vgalt, vglon, vglat] # (up, east, north)


def make_csgrid(xg, height = 500, crop_factor = 0.6, resolution_factor = 0.5, 
                extend = 1, dlon = 0., dlat = 0.):
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
        How much to reduce the spatial resolution compared to GEMINI. 0.5 will 
        double the spacing of grid cells
    extend : int
        How many secs poles to pad with on each side compared to the evaluation 
        grid
    dlon : float
        How much th shift the centre longitude of grid, in degrees. Default is
        0
    dlat : float
        How much th shift the centre latitude of grid, in degrees. Default is
        0        
    Returns
    -------
    Tuple containing two CS grid object, first the SECS grid, second is the 
    corresponding evaluation grid.

    '''

    #GEMINI grid is field-aligned and orthogonal. These limits are thus not 
    # exactly the limits at the fixed height
    # The crop factor is used to make the CS grid fit inside the model domain. 
    # Should be changed to something more robust and intuitive in the future
    
    # Find index of height matching the desired height in centre of grid
    dims = xg['lx']
    diff = xg['alt'][:,dims[1]//2, dims[2]//2] - height*1e3
    ii = np.argmin(np.abs(diff))
   
    #Find the orientation of the model grid
    x0 = (xg['glat'][ii,dims[1]//2,dims[2]//2], 
          xg['glon'][ii,dims[1]//2,dims[2]//2])
    x1 = (xg['glat'][ii,1+dims[1]//2,dims[2]//2], 
          xg['glon'][ii,1+dims[1]//2,dims[2]//2])
    orientation = np.degrees(getbearing(np.array([x0[0]]), np.array([x0[1]]), 
                                        np.array([x1[0]]), np.array([x1[1]])))
    
    #Centre location of CS grid
    position = (xg['glon'][ii,dims[1]//6,dims[2]//2] + dlon, 
                xg['glat'][ii,(dims[1]//6),dims[2]//2] + dlat) #Added the //6 hack

    #Dimensions of CS grid
    p0 = (xg['glon'][ii,0,dims[2]//2],xg['glat'][ii,0,dims[2]//2])
    p1 = (xg['glon'][ii,-1,dims[2]//2],xg['glat'][ii,-1,dims[2]//2])
    # d2: distance in dimension 2 (magntic northsouth ish)
    d2 = gcc.distance_between_points(p0, p1) 
    L = d2 * crop_factor
    Lres = d2/dims[1] / resolution_factor # For secs grid (grid)
    p0 = (xg['glon'][ii,dims[1]//2,0],xg['glat'][ii,dims[1]//2,0])
    p1 = (xg['glon'][ii,dims[1]//2,-1],xg['glat'][ii,dims[1]//2,-1])
    # d3 is distance in dimension 3 (magntic east ish)    
    d3 = gcc.distance_between_points(p0, p1) 
    W = d3 * crop_factor
    Wres = d3/dims[2] / resolution_factor # For secs grid (grid)

    #Make CS grid object
    #inner grid, to evaluate on    
    grid_ev = cubedsphere.CSgrid(cubedsphere.CSprojection(position, 
                -orientation[0]), L, W, Lres, Wres, R = (RE+height)*1e3) 
    xi_e  = np.hstack((grid_ev.xi_mesh[0,0]-np.flip([i*grid_ev.dxi for i in range(1,extend)]), grid_ev.xi_mesh[0], grid_ev.xi_mesh[0,-1] + np.array([i*grid_ev.dxi for i in range(1,extend+1)]) )) - grid_ev.dxi /2 
    eta_e = np.hstack((grid_ev.eta_mesh[0,0]-np.flip([i*grid_ev.deta for i in range(1,extend)]), grid_ev.eta_mesh[:, 0], grid_ev.eta_mesh[-1,   0] + np.array([i*grid_ev.deta for i in range(1,extend+1)]) )) - grid_ev.deta/2 
    # outer grid that represent E-field
    grid = cubedsphere.CSgrid(cubedsphere.CSprojection(grid_ev.projection.position,
                grid_ev.projection.orientation), grid_ev.L + extend*2*grid_ev.Lres, 
                grid_ev.W + extend*2*grid_ev.Wres, grid_ev.Lres, grid_ev.Wres, 
                edges = (xi_e, eta_e), R = grid_ev.R)
    
    return grid, grid_ev


def dipole_B(theta, height = 500):
    #theta must be in radians
    #Compute the dipole components at theta location. Not used after I figured 
    # out how to pull out B field from GEMINI
    mu0 = 4*np.pi * 10**(-7)
    m = 7.94e22 #magnetic dipole moment, taken from GEMINI documentation
    RE = 6371.2 * 1e3
    Bu = -2 * mu0 * m * np.cos(theta) / (4*np.pi*(RE+height*1e3)**3)
    Btheta =  -mu0 * m * np.sin(theta) / (4*np.pi*(RE+height*1e3)**3)
    Bn = -Btheta
    return (Bn, Bu)


def compute_enu_components(xg, dat):
    """
    Add ENU components of V, J and B to xarray dataset

    Parameters
    ----------
    xg : GEMINI grid object
        Read by gemini read function from config file
    dat : GEMINI data object (xarray dataset)
        Containing GEMINI output data for specified variables at specified time.

    Returns
    -------
    xarray dataset where the geographic ENU components of V, J and B is added.

    """
    
    #Convert velocity and current to grographic components, use ENU notation
    vu, ve, vn = model_vec2geo_vec(xg, dat, param='v')
    ju, je, jn = model_vec2geo_vec(xg, dat, param='J')
    
    #B vectors from model output, project on geo ENU frame
    [egalt,eglon,eglat]=unitvecs_geographic(xg)    
    Be = np.sum(xg["e1"]*eglon*xg['Bmag'][...,np.newaxis],3)
    Bn = np.sum(xg["e1"]*eglat*xg['Bmag'][...,np.newaxis],3)
    Bu = np.sum(xg["e1"]*egalt*xg['Bmag'][...,np.newaxis],3)
    
    dat['ve'] = ve
    dat['vn'] = vn
    dat['vu'] = vu
    dat['je'] = je
    dat['jn'] = jn
    dat['ju'] = ju
    dat['Be'] = xr.DataArray(Be, dims=('x1','x2','x3'))
    dat['Bn'] = xr.DataArray(Bn, dims=('x1','x2','x3'))
    dat['Bu'] = xr.DataArray(Bu, dims=('x1','x2','x3'))
    
    return dat

def sample_at_alt(xg, dat, grid = None, alt=800, altres=1, time_ind = -1, 
                          var = ["v1", "v2", "v3", "Phi", "J1", "J2", "J3", 
                                 "ne", "Be", "Bn", "Bu"], 
                          path='./', resfac = 1.):
    """

    Parameters
    ----------
    xg : GEMINI grid object
        Read by gemini read function from config file
    dat : GEMINI data object
        Containing GEMINI output data for specified variables at specified time.
    grid : CS grid object, optional
        Generated to fit inside GEMINI grid. This should be the grid_ev grid
        to make the SECS padding work as intended. (made with the function in
        this file). If not provided, sample within entire GEMINI domain.
    alt : INT, or array-like, optional
        Altitude (in km) to sample from. The default is 800 km.
    altres : INT or array-like, optional
        Half height of the altitude range to sample from, in km. The default is 1.
    time_ind : INT
        Index specifying the time to use from the simulation. The default is -1.
    var : list of strings
        Containing the names of the variables to sample. The default is 
        ["v1", "v2", "v3", "Phi", "J1", "J2", "J3"]
    path : STR optional
        Path to where the GEMINI output is stored. Default is './'
    resfac : float, optional
        how much to shrink the sample resolution compared to native grid

    Returns
    -------
    Dictionary containing the sampled variables at specified altitude range.

    """
    dims = xg['lx']
    
    if grid == None:
        glatlims = (np.percentile(xg['glat'][-1,:,:].flatten(),5),
                    np.percentile(xg['glat'][-1,:,:].flatten(),90)) 
        glonlims = (np.percentile(xg['glon'][-1,:,:].flatten(),15),
                    np.percentile(xg['glon'][-1,:,:].flatten(),90))         
    else:
        glatlims=(grid.lat.min(),grid.lat.max())
        glonlims=(grid.lon.min(),grid.lon.max())
    
    #Resample on spherical shell specific altitude
    ddd = {}
    if type(alt) == int:
        alt = np.array(alt)
    
    for v in var:
        if v[-1] in ['1','2','3']: #is a vector component
            dde = []
            ddn = []
            ddu = []
            for (ii,aa) in enumerate(alt):
                galti, gloni, glati, vve = model2geogcoords(xg, dat[v[0]+'e'], 
                            1, round(resfac*dims[2]), round(resfac*dims[1]), wraplon=True, 
                            altlims=((aa-altres[ii])*1e3, (aa+altres[ii])*1e3), 
                            glatlims=glatlims, glonlims=glonlims)
                galti, gloni, glati, vvn = model2geogcoords(xg, dat[v[0]+'n'], 
                            1, round(resfac*dims[2]), round(resfac*dims[1]), wraplon=True, 
                            altlims=((aa-altres[ii])*1e3, (aa+altres[ii])*1e3), 
                            glatlims=glatlims, glonlims=glonlims)          
                galti, gloni, glati, vvu = model2geogcoords(xg, dat[v[0]+'u'], 
                            1, round(resfac*dims[2]), resfac*dims[1], wraplon=True, 
                            altlims=((aa-altres[ii])*1e3, (aa+altres[ii])*1e3), 
                            glatlims=glatlims, glonlims=glonlims)            
                dde.append(vve)
                ddn.append(vvn)
                ddu.append(vvu)
                
            ddd[v[0]+'e'] = np.array(dde)[:,0,:,:]
            ddd[v[0]+'n'] = np.array(ddn)[:,0,:,:]
            ddd[v[0]+'u'] = np.array(ddu)[:,0,:,:]
        else: #scalar quantity
            dd = []
            for (ii,aa) in enumerate(alt):
                if v == 'Phitop':
                    lx1 = xg["lx"][0]
                    inds1 = range(2, lx1 + 2)
                    x1 = xg["x1"][inds1]
                    dat[v] = dat[v].expand_dims(x1=x1)
                galti, gloni, glati, vvs = model2geogcoords(xg, dat[v], 1, 
                            round(resfac*dims[2]), round(resfac*dims[1]), wraplon=True, 
                            altlims=((aa-altres[ii])*1e3, (aa+altres[ii])*1e3), 
                            glatlims=glatlims,glonlims=glonlims)                        
                dd.append(vvs)    
            ddd[v] = np.array(dd)[:,0,:,:]
    
    
        
        # if 'Be' in var:
        #     mag = []
        #     for (ii,aa) in enumerate(alt):
                
                
        #         ddd['Bmag'] = np.sqrt(ddd['Be']**2+ddd['Bn']**2+ddd['Bu']**2).flatten()
            
    altmesh, glonmesh, glatmesh = np.meshgrid(alt, gloni,glati, indexing='ij')
    ddd['glatmesh'] = glatmesh
    ddd['glonmesh'] = glonmesh
    ddd['altmesh'] = altmesh
        
    return ddd #Append E-field mapped to alt

def sample_points(xg, dat, lat, lon, alt):
    # Sample GEMINI at input locations
    je = model2pointsgeogcoords(xg, dat['je'], alt*1e3, lon, lat)
    jn = model2pointsgeogcoords(xg, dat['jn'], alt*1e3, lon, lat)
    ju = model2pointsgeogcoords(xg, dat['ju'], alt*1e3, lon, lat)
    Be = model2pointsgeogcoords(xg, dat['Be'], alt*1e3, lon, lat)
    Bn = model2pointsgeogcoords(xg, dat['Bn'], alt*1e3, lon, lat)
    Bu = model2pointsgeogcoords(xg, dat['Bu'], alt*1e3, lon, lat)

    datadict = {'lat':lat.flatten(), 'lon':lon.flatten(), 'alt':alt.flatten(), 
                'je':je, 'jn':jn, 'ju':ju, 'Be':Be, 'Bn':Bn, 'Bu':Bu}
    
    return datadict

def sample_eiscat(xg, dat, alts_grid):
    
    #Implement 27 beam (monostatic config) configuration as sketched by Ogawa (2021)
    
    # el1 = np.array([64,61,60,58,57,55,54,54,57,59,61,61])
    # az1 = np.array([0,35,69,101,130,156,180,204,231,258,288,323])
    # el2 = np.array([30,30,30,30,30,30,30,30,30,30,30,30])
    # az2 = np.array([0,30,60,90,120,150,180,210,240,270,300,330])
    # el3 = np.array([66,77.8,90])
    # az3 = np.array([180,180,180])
    # el = np.hstack((el1,el2,el3)) #deg
    # az = np.hstack((az1,az2,az3)) #deg
    NN = 100
    el = np.ones(NN) * 30
    az = np.linspace(0,360,NN)
    # el = np.array([30,30,30,30,30,30,30,30])
    # az = np.array([0,45,90,135,180,225,270,315])

    sitelat = 67.35 #69.38
    sitetheta = 90-sitelat
    sitephi = 23. #20.30
    O_sph = np.array([RE, 0, 0]) #site location vector
    R = sph2car(sitetheta,sitephi) #sph2ecef rotation matrix cor vector components
    O_ecef = R.dot(O_sph) #Site location vecor in ecef
    r = []
    theta = []
    phi = []
    for i in range(len(az)):
        uhat_sph = get_uhat(az[i],el[i]) # beam direction unit vector in sph
        uhat_ecef = R.dot(uhat_sph) #Beam direction unit vector in ecef
        #Line sphere intersecion formula from https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        dot = uhat_ecef.dot(O_ecef)# RE * np.cos(np.radians(90+el[i]))
        root = np.sqrt(dot**2 - (RE**2-(RE+alts_grid)**2))
        #d0 = -dot - root # through earth
        d = -dot + root # the distances from site corresponding to selected altitudes
        deltas = (uhat_ecef.reshape(3,1) * d) #Vector to be added to O_ecef
        pos = O_ecef[:,np.newaxis] + deltas
        r_, theta_, phi_ = car2sph(pos[0,:], pos[1,:], pos[2,:])
        r.append(r_)
        theta.append(theta_)
        phi.append(phi_)
    
    poss = np.vstack((np.array(r).flatten(), np.array(theta).flatten(),
                      np.array(phi).flatten())).T  
    
    je = model2pointsgeogcoords(xg, dat['je'],poss[:,0]-RE,
                                     poss[:,2],90-poss[:,1])
    jn = model2pointsgeogcoords(xg, dat['jn'],poss[:,0]-RE,
                                     poss[:,2],90-poss[:,1])
    ju = model2pointsgeogcoords(xg, dat['ju'],poss[:,0]-RE,
                                     poss[:,2],90-poss[:,1])
    Be = model2pointsgeogcoords(xg, dat['Be'],poss[:,0]-RE,
                                     poss[:,2],90-poss[:,1])
    Bn = model2pointsgeogcoords(xg, dat['Bn'],poss[:,0]-RE,
                                     poss[:,2],90-poss[:,1])
    Bu = model2pointsgeogcoords(xg, dat['Bu'],poss[:,0]-RE,
                                     poss[:,2],90-poss[:,1])

    datadict = {'lat':90-poss[:,1], 'lon':poss[:,2], 'alt':poss[:,0]-RE, 
                'je':je, 'jn':jn, 'ju':ju, 'Be':Be, 'Bn':Bn, 'Bu':Bu}
    
    return datadict
 
    
def sph2car(theta, phi):
    #Theta and phi in degrees. Make rotation matrix to convert a vector with 
    # spherical components at location (theta,phi) to ECEF cartesian components
    theta = np.radians(theta)
    phi = np.radians(phi)
    R = np.array([[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)], 
                  [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi) ],
                  [np.cos(theta),             -np.sin(theta),            0           ]])
    return R

def car2sph(x,y,z):
    # Convert the cartesian ECEF location to spherical (r, theta, phi) coordinates
    # Returns components in degrees.
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arccos(z/r))
    phi = np.degrees(np.arctan2(y,x))
    return (r, theta, phi)
    
def get_uhat(az, el):
    # Unit vector of beam direction defined by azimuth and elevation angle, 
    # expressed in spherical components. Refer to the radar site location.
    norm = np.sqrt(1-(np.sin(np.radians(el)))**2)
    ur = np.sin(np.radians(el))
    utheta = -np.cos(np.radians(az))*norm
    uphi = np.sin(np.radians(az))*norm
    return np.array([ur, utheta, uphi])



def grid_e_field_at_alt(datadict, grid, return_B=False):
    """
    
    Parameters
    ----------
    datadict : dictionary
        The GEMINI output in ENU components at specific altitude.
    grid : CS grid object
        The field-aligned CS grid, defined at the top boundary. Should be the 
        evaluation grid (grid_ev object made with the function above in this 
        file).
    return_B : bool
        Use to rather returned the gridded B-field

    Returns
    -------
    Tuple containing the ENU components of the E-field deduced from model 
    output of Phi, gridded on the CS grid.

    """
    
    use = np.isfinite(datadict['ju'].flatten()) & grid.ingrid(datadict['glonmesh'].flatten(),
                    datadict['glatmesh'].flatten()) 
    ii, jj = grid.bin_index(datadict['glonmesh'].flatten()[use],datadict['glatmesh'].flatten()[use])
    ii1d = grid._index(ii, jj)
    df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 'phi': datadict['phi'].flatten()[use], 
                    'Be': datadict['Be'].flatten()[use], 'Bn': datadict['Bn'].flatten()[use], 
                    'Bu': datadict['Bu'].flatten()[use]})
    gdf = df.groupby('i1d').mean()
    df_sorted = gdf.reindex(index=pd.Series(np.arange(0,len(grid.lon.flatten()))), method='nearest', tolerance=0.1)
    phi = df_sorted.phi.values.reshape(grid.shape) #The "top boundary condition" for layerded SECS below
    Be = df_sorted.Be.values.reshape(grid.shape)
    Bn = df_sorted.Bn.values.reshape(grid.shape)
    Bu = df_sorted.Bu.values.reshape(grid.shape)
    Le, Ln = grid.get_Le_Ln()
    
    Ee = -Le.dot(phi.flatten()).reshape(grid.shape)    
    En = -Ln.dot(phi.flatten()).reshape(grid.shape)
    Eu = (Ee*Be + En*Bn)/Bu #Using E.dot(B) = 0
    
    if not return_B:
        datadict['Ee'] = Ee
        datadict['En'] = En
        datadict['Eu'] = Eu
        
        return (Ee, En, Eu)
    else:
        return (Be, Bn, Bu)
    
def grid_param_at_alt(datadict, grid, param='v'):
    """
    
    Parameters
    ----------
    datadict : dictionary
        The GEMINI output in ENU components (in 3D).
    grid : CS grid object
        The field-aligned CS grid, defined at the top boundary. Should be the 
        evaluation grid (grid_ev object made with the function above in this 
        file).
    param : str
        Select type of parameter to grid. : v, j, n

    Returns
    -------
    Tuple containing the ENU components of the param from model 
    output, gridded on the CS grid. Each component is a 3D array.

    """
    alts_ = datadict['altmesh'][:,0,0]
    use = np.isfinite(datadict['ju'][0,:,:].flatten()) & grid.ingrid(datadict['glonmesh'][0,:,:].flatten(),
                datadict['glatmesh'][0,:,:].flatten())
    if param == 'n':
        p = []
    else:
        pe = []
        pn = []
        pu = []

    for (i,alt) in enumerate(alts_):
        ii, jj = grid.bin_index(datadict['glonmesh'][i,:,:].flatten()[use],datadict['glatmesh'][i,:,:].flatten()[use])
        ii1d = grid._index(ii, jj)
        if param == 'n':
            df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 'n': datadict['n'].flatten()[use]})
        else:
            df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 
                               param+'e': datadict[param+'e'][i,:,:].flatten()[use], 
                               param+'n': datadict[param+'n'][i,:,:].flatten()[use], 
                               param+'u': datadict[param+'u'][i,:,:].flatten()[use]})        
        gdf = df.groupby('i1d').mean()
        df_sorted = gdf.reindex(index=pd.Series(np.arange(0,len(grid.lon.flatten()))), method='nearest', tolerance=0.1)
        
        if param == 'n':
            p.append(df_sorted[param].values.reshape(grid.shape))
        else:
            pe.append(df_sorted[param+'e'].values.reshape(grid.shape))
            pn.append(df_sorted[param+'n'].values.reshape(grid.shape))
            pu.append(df_sorted[param+'u'].values.reshape(grid.shape))
        
    if param == 'n':
        return np.array(p)
    else:
        return (np.array(pe),np.array(pn), np.array(pu))
    
def divergence(xg, dat, param = 'j_perp'):
    """
    

    Parameters
    ----------
    xg : GEMINI grid object
        Read by gemini read function from config file
    dat : GEMINI data object
        Containing GEMINI output data for specified variables at specified time.
    param : STR
        Which vector field to compute divergence of. Default is j_perp

    Returns
    -------
    Divergence of param on GEMINI grid.

    """
    RE = 6371.2e3 #Earth radius in m
    dims = xg['lx']


    #Metric factors defined in eqs 114-116 in GEMINI documentation
    h1 = xg['r']**3/(RE**2*np.sqrt(1+3*(np.cos(xg['theta']))**2))
    h2 = RE*(np.sin(xg['theta']))**3/np.sqrt(1+3*(np.cos(xg['theta']))**2)
    h3 = xg['r'] * np.sin(xg['theta'])
    scale = 1./(h1*h2*h3)
    
    #The three components (1,2,3 gemini grid) of the vector field to work with
    if param == 'j_perp':
        a1 = np.zeros(dat.J1.values.shape)
        a2 = dat.J2.values
        a3 = dat.J3.values
    elif param == 'j':
        a1 = dat.J1.values
        a2 = dat.J2.values
        a3 = dat.J3.values    
      
    else:
        print(param + ' not implemented yet.')
        print(1/0)
    
    #Differentials
    dx1__ = np.diff(dat.x1)
    dx1_ = np.concatenate((dx1__,[dx1__[-1]]))
    dx1 =  np.array([np.ones(dims[1:])*i for i in dx1_])
    dx2__ = np.diff(dat.x2)
    dx2_ = np.concatenate((dx2__,[dx2__[-1]]))
    dx2 =  np.array([np.ones([dims[0],dims[2]])*i for i in dx2_])
    dx2 = np.swapaxes(dx2,1,0)
    dx3__ = np.diff(dat.x3)
    dx3_ = np.concatenate((dx3__,[dx3__[-1]]))
    dx3 =  np.array([np.ones([dims[0],dims[1]])*i for i in dx3_])
    dx3 = np.swapaxes(dx3,1,0)
    dx3 = np.swapaxes(dx3,1,2)


    #The three terms of the divergence
    d1_ = np.append(np.diff(h2*h3*a1, axis=0),np.zeros([1,dims[1],dims[2]]), axis=0)
    d1 = scale*d1_/dx1
    d2_ = np.append(np.diff(h1*h3*a2, axis=1),np.zeros([dims[0],1,dims[2]]), axis=1)
    d2 = scale*d2_/dx2
    d3_ = np.append(np.diff(h1*h2*a3, axis=2),np.zeros([dims[0],dims[1],1]), axis=2)
    d3 = scale*d3_/dx3
    
    #Add the three terms to make the divergence of param
    d = d1 + d2 + d3

    return d    

def divergence_spherical(sampledict, hor=False, alt = None, perp=False):
    """
    

    Parameters
    ----------
    sampledict : dictionary
        Dictoinary containing coordinates and data sampled with sampling 
        function in this file (spherical components). Lat/lon in degrees is 
        required, and altitude in km. Dictionary also contain EN(U) components 
        of the vector field which must be named je, jn, (and ju).
    perp : boolean
        Whether only horizontal (lon/lat) divergence is computed
    alt : int
        Altitude in km at where divergence is evaluated. Needed when input is 2D.

    Returns
    -------
    Divergence of param on computed from spherical components.

    """
    RE = 6371.2e3 #Earth radius in m
    
    if 'shape' in sampledict.keys():
        dims = sampledict['shape']
        sampledict['glon'] =sampledict['lon'].reshape(dims)
        sampledict['glat'] =sampledict['lat'].reshape(dims)
        sampledict['alt'] =sampledict['alt'].reshape(dims)
        sampledict['je'] =sampledict['je'].reshape(dims)
        sampledict['jn'] =sampledict['jn'].reshape(dims)
        sampledict['ju'] =sampledict['ju'].reshape(dims)
    else:
        dims = sampledict['glat'].shape
    ndims = len(dims)
    if ndims == 2: #Does not use np.gradient(). Should be implemented.
        if alt is None:
            print('Must specify altitude')
            print(1/0)
        r = RE + alt*1e3

        #Vector components
        jtheta = -sampledict['jn'][0,:,:]
        jphi = sampledict['je'][0,:,:]
        
        #Grid differentials
        glons = np.radians(sampledict['glonmesh'])
        glats = np.radians(sampledict['glatmesh'])
        thetas = np.pi/2. - glats
        dphi_ = np.diff(glons, axis=0)
        dphi = np.concatenate((dphi_,[dphi_[-1]]))
        dtheta_ = np.diff(thetas, axis=1)
        dtheta = np.hstack((dtheta_,dtheta_[:,-1][:,np.newaxis]))
        
        #vector part differentials
        cphi_ = np.diff(jphi, axis=0)
        cphi = np.concatenate((cphi_,[cphi_[-1]]))
        ctheta_ = np.diff(jtheta*np.sin(thetas), axis=1)
        ctheta = np.hstack((ctheta_,ctheta_[:,-1][:,np.newaxis]))
        
        #Divergence terms
        d1 = 1./(r*np.sin(thetas)) * (ctheta / dtheta)
        d2 = 1./(r*np.sin(thetas)) * (cphi / dphi)
        
        return d1 + d2
    
    if ndims == 3:

        r = RE + sampledict['alt']*1e3

        #Vector components
        jr = sampledict['ju']
        jtheta = -sampledict['jn']
        jphi = sampledict['je']
        
        # Grid representation
        rs = RE + sampledict['alt'][:,0,0]*1e3
        glats = np.radians(sampledict['glat'])
        thetas = np.pi/2 - glats
        glons = np.radians(sampledict['glon'])
        # rs = RE + sampledict['alt'][:,0,0]*1e3
        # glats = np.radians(sampledict['glat'][0,:,:])
        # thetas = np.pi/2 - glats
        # glons = np.radians(sampledict['glon'][0,:,:])
        
        # #Alternative way
        # glats = np.radians(sampledict['glat'])
        # thetas = np.pi/2 - glats
        # glons = np.radians(sampledict['glon'])
        # dphi_ = np.diff(np.radians(sampledict['glon'][0,0,:]))#[0]
        # dphi = np.concatenate((dphi_,[dphi_[-1]]))
        # dtheta_ = np.diff(np.radians(90.-sampledict['glat'][0,:,0]))#[0]
        # dtheta = np.concatenate((dtheta_,[dtheta_[-1]]))
        # rs = RE + sampledict['alt'][:,0,0]*1e3
        # dr_ = np.diff(rs)
        # dr = np.concatenate((dr_,[dr_[-1]]))

        # d0 = (1/r**2) * np.gradient(r**2 * jr, dr, axis=0)
        # d1 = 1./(r*np.sin(thetas)) * np.gradient(jtheta * np.sin(thetas), dtheta, axis=1)
        # d2 = 1./(r*np.sin(thetas)) * np.gradient(jphi, dphi, axis=2)
        
        #Grid differentials
        dphi_ = np.diff(glons, axis=1)
        dphi = np.hstack((dphi_,dphi_[:,-1][:,np.newaxis]))
        dtheta_ = np.diff(thetas, axis=1)
        # dtheta = np.vstack((dtheta_,dtheta_[-1,:][np.newaxis,:]))
        dtheta = np.hstack((dtheta_,dtheta_[:,-1,:][:,np.newaxis,:]))
        dr_ = np.diff(rs)
        dr = np.concatenate((dr_,[dr_[-1]]))
        
        #vector part differentials
        cphi_ = np.diff(jphi, axis=2)
        cphi = np.dstack((cphi_,cphi_[:,:,-1][:,:,np.newaxis]))
        ctheta_ = np.diff(jtheta*np.sin(thetas), axis=1)
        ctheta = np.hstack((ctheta_,ctheta_[:,-1,:][:,np.newaxis,:]))
        cr_ = np.diff(r**2 * jr, axis=0)
        cr = np.vstack((cr_, cr_[-1,:,:][np.newaxis,:,:]))
        
        #Divergence terms
        d0 = (1./r**2) * (cr / dr[:,np.newaxis, np.newaxis])
        d1 = 1./(r*np.sin(thetas)) * (ctheta / dtheta)
        d2 = 1./(r*np.sin(thetas)) * (cphi / dphi)
        if hor:
            return d1 + d2
        else:
            if perp: # Have a look at this before use!
                print('Have a look at this before use!')
                br, btheta, bphi = secs3d.make_b_unitvectors(sampledict['Bu'], 
                                    -sampledict['Bn'], sampledict['Be'])
                return br*d0 + btheta*d1 + bphi*d2
            else: 
                return d0 + d1 + d2            



from scipy.interpolate import RectBivariateSpline
def get_interpolated_amplitudes(m, secs_grid, grid_ev, alts_grid):
    # instead of using the 
    m_cf = m[0:m.size//2].reshape((alts_grid.size, 
                        secs_grid.lat.shape[0],secs_grid.lat.shape[1]))
    x = secs_grid.eta[:,0]
    y = secs_grid.xi[0,:]    
    
    K = alts_grid.size
    for k in range(K):
        f = RectBivariateSpline(x, y, m_cf[k,:,:], kx = 2, ky = 2, s = 0)
        eval_xi = grid_ev.xi[0,:]
        eval_eta = grid_ev.eta[:,0]
        gridded = f(eval_eta, eval_xi)


def remove_outside(secs_grid, alts_grid, lat, lon, alt, params=None, ext_factor=-1):
    """
    Parameters
    ----------
    secs_grid : CS grid object
        The grid we use to compute gridded indices of data/evaluation locations
        Dimension is (I,J). Assumes secs_grid.A refer to bottom layer.
    alts_grid : array-like, 1D or 2D
        Altitude grid to use together with secs_grid. Length is K. Values 
        represent the height in km of the centre of the voxels of each layer. 
        Should be in increasing order.
    lat : array-like
        Output latitude after filtering. Flattened, in degrees.
    lon : array-like
        Output longitude after filtering. Flattened, in degrees.
    alt : array-like
        Output altitude after filtering. Flattened, in km.
    params : tuple, optional
        Apply the same filtering also to the variables as contained in this
        tuple. Must be of same size as lat/lon/alt. The default is None.
    ext_factor : int, optional
        To control how to filter out locations based on their proximity to the 
        grid. The default is -1, removing points closer than 1 grid cell from 
        the edge of the grid (mesh).

    Returns
    -------
    Tuple.
        Filtered lat,lon,alt arrays, flattened. If params is provided, the first
        element in the returned tuple will be tuple of the different params 
        provided.
    """
    # Remove data/evaluation points outside the perimiter of secs nodes
    use = secs_grid.ingrid(lon.flatten(), lat.flatten(), ext_factor = ext_factor)
    lat = lat[use]
    lon = lon[use]
    alt = alt[use]
    if params is not None:
        ps = []
        for p in params:
            p_ = p[use]
            ps.append(p_)
            
    # Remove data/evaluation points outside grid in vertical direction
    k0 = secs3d.get_alt_index(alts_grid, alt, returnfloat=True)
    inside = (k0 >= 0) & (k0<alts_grid.size-1)
    lat = lat.flatten()[inside]
    lon = lon.flatten()[inside]
    alt = alt.flatten()[inside]
    
    if params is not None:
        pss = []
        for p in ps:
            p__ = p[inside]
            pss.append(p__)
    
        return (pss, lat, lon, alt)

    return (lat, lon, alt)            
    

def prepare_model_data(secs_grid, datadict, alts_grid, jperp=False, 
                       ext_factor=-1):
    """
    Prepare GEMINI data sampled at spherical shells at different heights to
    input in 3D inversion. Remove nans and data outside the inner grid region,
    controlled by the ext_factor keyword, which is here hard-coded to -1.

    Parameters
    ----------
    secs_grid : CS grid object
        The grid we use to compute gridded indices of data/evaluation locations
        Dimension is (I,J). Assumes secs_grid.A refer to bottom layer.
    datadict : dictionary
        The GEMINI output in ENU components (in 3D).
    alts_grid : array-like, 1D or 2D
        Altitude grid to use together with secs_grid. Length is K. Values 
        represent the height in km of the centre of the voxels of each layer. 
        Should be in increasing order.
    jperp : Boolean, optional
        Specifies whether to return the (r, theta, phi) components of the perp
        current. If false(default) the full current is returned (r, theta, phi)
    ext_factor : int, optional
        To control how to filter out locations based on their proximity to the 
        grid. The default is -1, removing points closer than 1 grid cell from 
        the edge of the grid (mesh).        

    Returns
    -------
    jj : list
        First three elements contain jr,  jtheta and jphi components after 
        filtering. Last three elements will be Br, Btheta and Bphi if this is 
        provided in datadict.
    lat : array-like
        Output latitude after filtering. Flattened, in degrees.
    lon : array-like
        Output longitude after filtering. Flattened, in degrees.
    alt : array-like
        Output altitude after filtering. Flattened, in km.

    """
    jphi = datadict['je'].flatten()
    jtheta = -datadict['jn'].flatten()
    jr = datadict['ju'].flatten()
    use = np.isfinite(jphi)
    jphi = jphi[use]
    jtheta = jtheta[use]
    jr = jr[use]
    if 'glatmesh' in datadict.keys():
        lat = datadict['glatmesh'].flatten()[use]
        lon = datadict['glonmesh'].flatten()[use]
        alt = datadict['altmesh'].flatten()[use]
    else:
        lat = datadict['lat'].flatten()[use]
        lon = datadict['lon'].flatten()[use]
        alt = datadict['alt'].flatten()[use]
    
    if "Be" in datadict.keys():
        Br = datadict['Bu'].flatten()[use]
        Btheta = -datadict['Bn'].flatten()[use]
        Bphi = datadict['Be'].flatten()[use]
        params=(jr, jtheta, jphi, Br, Btheta, Bphi)
        if "fac" in datadict.keys():
            fac = datadict['fac'].flatten()[use]
            params = params + (fac,)
    else:
        params=(jr, jtheta, jphi)            
    
    # Remove data/evaluation points outside the perimiter of secs nodes
    jj, lat, lon, alt = remove_outside(secs_grid, alts_grid, lat, lon, alt, 
                                      params=params, ext_factor=ext_factor)
    
    if jperp:
        br, btheta, bphi = secs3d.make_b_unitvectors(jj[3],jj[4],jj[5])
        N = br.size
        B = secs3d.make_B(br, btheta, bphi)
        P = secs3d.make_P(N)
        jjj = np.hstack((jj[0],jj[1],jj[2]))
        j_perp = P.T.dot(B.dot(P.dot(jjj)))
        jj[0] = j_perp[0:N]
        jj[1] = j_perp[N:2*N]
        jj[2] = j_perp[2*N:3*N]
        
        #Alternative way to get jperp, by mapping potential mapping
        

    return jj, lat, lon, alt


