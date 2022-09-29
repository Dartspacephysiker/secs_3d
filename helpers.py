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
import xarray as xr
from gemini3d.grid.gridmodeldata import model2geogcoords
import pandas as pd
from lompe.secsy.secsy import utils as secsy


######
#Contain helper functions needed in the 3D SECS representation


RE = 6371.2 #Earth radius in km


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


def make_csgrid(xg, height = 500, crop_factor = 0.6, resolution_factor = 0.5, 
                extend = 3):
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
    extend : int
        How many secs poles to pad with on each side compared to the evaluation grid

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
    position = (xg['glon'][ii,dims[1]//4,dims[2]//2], xg['glat'][ii,(dims[1]//4),dims[2]//2]) #Added the -10 hack (or //4)

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
    # extend = 3 #How many extra secs poles on each side to pad with
    
    xi_e  = np.hstack((grid_ev.xi_mesh[0,0]-np.flip([i*grid_ev.dxi for i in range(1,extend)]), grid_ev.xi_mesh[0], grid_ev.xi_mesh[0,-1] + np.array([i*grid_ev.dxi for i in range(1,extend+1)]) )) - grid_ev.dxi /2 
    eta_e = np.hstack((grid_ev.eta_mesh[0,0]-np.flip([i*grid_ev.deta for i in range(1,extend)]), grid_ev.eta_mesh[:, 0], grid_ev.eta_mesh[-1,   0] + np.array([i*grid_ev.deta for i in range(1,extend+1)]) )) - grid_ev.deta/2 
    grid = cubedsphere.CSgrid(cubedsphere.CSprojection(grid_ev.projection.position,
                grid_ev.projection.orientation), grid_ev.L + extend*2*grid_ev.Lres, 
                grid_ev.W + extend*2*grid_ev.Wres, grid_ev.Lres, grid_ev.Wres, 
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


def compute_enu_components(xg, dat):
    """
    

    Parameters
    ----------
    xg : GEMINI grid object
        Read by gemini read function from config file
    dat : GEMINI data object
        Containing GEMINI output data for specified variables at specified time.

    Returns
    -------
    xarray dataset where the geographic ENU components of V, J and B

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
                            altlims=((aa-altres[ii])*1e3, (aa-altres[ii])*1e3), 
                            glatlims=glatlims, glonlims=glonlims)
                galti, gloni, glati, vvn = model2geogcoords(xg, dat[v[0]+'n'], 
                            1, round(resfac*dims[2]), round(resfac*dims[1]), wraplon=True, 
                            altlims=((aa-altres[ii])*1e3, (aa-altres[ii])*1e3), 
                            glatlims=glatlims, glonlims=glonlims)          
                galti, gloni, glati, vvu = model2geogcoords(xg, dat[v[0]+'u'], 
                            1, round(resfac*dims[2]), resfac*dims[1], wraplon=True, 
                            altlims=((aa-altres[ii])*1e3, (aa-altres[ii])*1e3), 
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
                galti, gloni, glati, vvs = model2geogcoords(xg, dat[v], 1, 
                            round(resfac*dims[2]), round(resfac*dims[1]), wraplon=True, 
                            altlims=((aa-altres[ii])*1e3, (aa-altres[ii])*1e3), 
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
        The GEMINI output in ENU components at specific altitude.
    grid : CS grid object
        The field-aligned CS grid, defined at the top boundary. Should be the 
        evaluation grid (grid_ev object made with the function above in this 
        file).
    param : str
        Select type of parameter to grid. : v, j, n

    Returns
    -------
    Tuple containing the ENU components of the param from model 
    output, gridded on the CS grid.

    """
    
    use = np.isfinite(datadict['ju'].flatten()) & grid.ingrid(datadict['glonmesh'].flatten(),
                    datadict['glatmesh'].flatten()) 
    ii, jj = grid.bin_index(datadict['glonmesh'].flatten()[use],datadict['glatmesh'].flatten()[use])
    ii1d = grid._index(ii, jj)
    if param == 'n':
        df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 'n': datadict['n'].flatten()[use]})
    else:
        df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, param+'e': datadict[param+'e'].flatten()[use], 
                           param+'n': datadict[param+'n'].flatten()[use], 
                           param+'u': datadict[param+'u'].flatten()[use]})        
    gdf = df.groupby('i1d').mean()
    df_sorted = gdf.reindex(index=pd.Series(np.arange(0,len(grid.lon.flatten()))), method='nearest', tolerance=0.1)
    
    if param == 'n':
        p = df_sorted[param].values.reshape(grid.shape)
        return p
    else:
        pe = df_sorted[param+'e'].values.reshape(grid.shape)
        pn = df_sorted[param+'n'].values.reshape(grid.shape)
        pu = df_sorted[param+'u'].values.reshape(grid.shape)
        
        return (pe,pn, pu)
    
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

def divergence_spherical(sampledict, param = ['je','jn'], alt = None):
    """
    

    Parameters
    ----------
    sampledict : dictionary
        Dictoinary containing coordinates and data sampled with sampling 
        function in this file (spherical components)
     param : list of STR
        Which vector field to compute divergence of. Default is je, jh
        (horizontal current density)
    alt : int
        Altitude in km at where divergence is evaluated. Needed when input is 2D.

    Returns
    -------
    Divergence of param on computed from spherical components.

    """
    RE = 6371.2e3 #Earth radius in m
    ndims = len(param)
    dims = sampledict['glatmesh'].shape
    if ndims == 2:
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
        thetas = 90. - glats
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

def get_SECS_J_G_matrices_3D(secs_grid, alts_grid, lat, lon, alt, constant = 
                             1./(4.*np.pi), singularity_limit=None):
    ''' 

        Calculate SECS J_G matrices for 3D representation using CS grid at fixed
        (lat,lon) locations at different heights. For now we assume sampling 
        in each altitude layer is identical, allowing us to scale the SECS G 
        matrices with height. 
    
    (I, J, K) shapes: I is number of SECS nodes in eta (CS) direction, 
    J is number of SECS poles in xi (CS) direction, and K is number of hor. 
    layers.

    Parameters
    ----------
    secs_grid : CS grid object
        The grid we use to compute gridded indices of data/evaluation locations
        Dimension is (I,J)
    alts_grid : array-like, 1D or 2D
        Altitude grid to use together with secs_grid. Length is K. Values represent the height in km of the centre of the
        voxels of each layer. Should be in increasing order.
    lat : array-like, 1D or 2D
       latitudes [deg] of the data/evaluation locations. Flattened to 1D.
    lon : array-like, 1D
       longitudes [deg] of the data/evaluation locations. Flattened to 1D.       
    alt : array-like, 1D
       altitude [km] of the data/evaluation locations. Flattened to 1D.              
    constant : float, optional
        Passed to the underlying secsy function. The default is 1./(4.*np.pi).
    singularity_limit : float, optional
        Passed to the underlying secsy function. The default is 0. [m] meaning
        that the modified SECS functions are not used. Typically ~half a CS grid 
        cell is used. Default (None) will use 0.5*grid.Lres

    Returns
    -------
    SECS CF and DF G matrices stacked in the appropriate way to represent its
    value in the 3D domain.

    '''
    
    # Evaluation locations test
    if not lat.shape[0]==lon.shape[0]==alt.shape[0]:
        print('Inconsistent dimensions of data/evaluation locations (lat,lon,alt)')
        print(1/0)  
    RE = 6371.2 #Earth radius in km
    
    if singularity_limit == None:
        singularity_limit=secs_grid.Lres*0.5
        
    # Remove data/evaluation points outside (2D) secs_grid, using the ingrid function
    use = secs_grid.ingrid(lon.flatten(), lat.flatten())
    lat = lat.flatten()[use]
    lon = lon.flatten()[use]
    alt = alt.flatten()[use]
    
    # Remove data/evaluation points outside grid in vertical direction
    k0 = get_alt_index(alts_grid, alt)
    inside = k0 != -1
    lat = lat.flatten()[inside]
    lon = lon.flatten()[inside]
    alt = alt.flatten()[inside]     

    #Grid dimensions
    K = alts_grid.shape[0] #Number of vertival layers
    I = secs_grid.shape[0] #Number of cells in eta direction
    J = secs_grid.shape[1]  #Number of cells in xi direction  
    IJK = I*J*K
    N = lat.shape[0] # Number of data/evaluation points  
           
    #Compute SECS matrices at bottom layer, using all observational locations (N). 
    # We will apply the singularity threshold
    # defined as a length on the bottom layer, and use the same corresponding
    # theta0 = singularity_limit/RI on every layer, although this corresponds
    # to a different lenght. I think this makes the mose sense, and is easier
    # to implement (we dont need to keep track of which nodes needs to be modified
    # for each layer).
    alt_ = alts_grid[0]
    Ge_cf_, Gn_cf_ = secsy.get_SECS_J_G_matrices(lat.flatten(), 
                lon.flatten(), secs_grid.lat.flatten(), 
                secs_grid.lon.flatten(), constant = 1./(4.*np.pi), 
                RI=RE * 1e3 + alt_ * 1e3, current_type = 'curl_free', 
                singularity_limit=singularity_limit)
    Ge_df_, Gn_df_ = secsy.get_SECS_J_G_matrices(lat.flatten(), 
                lon.flatten(), secs_grid.lat.flatten(), 
                secs_grid.lon.flatten(), constant = 1./(4.*np.pi), 
                RI=RE * 1e3 + alt_ * 1e3, current_type = 'divergence_free', 
                singularity_limit=singularity_limit)
    
    #Indices of each evaluation point in 3D
    k, i, j = get_indices_kij(secs_grid, alts_grid, lat, lon, alt)  
    # kij = np.ravel_multi_index((k,i,j), (K,I,J)) #flattened index

    altmask = np.zeros(IJK)
    altmask = []
    r_ = (RE + alt_)/(RE + alts_grid) #Altitude correction factors
    for kkk in range(N):
        mask_ = np.zeros(IJK)
        kij_start = np.ravel_multi_index((k[kkk],0,0), (K,I,J)) #flattened index
        kij_stop = kij_start + I*J
        mask_[kij_start:kij_stop] = 1 * r_[k[kkk]]
        altmask.append(mask_)
    altmask = np.array(altmask)

    #Make stacked SECS G matrices, not corrected for altitude nor vertical 
    # observational / evaluation location
    Ge_cf_k = np.tile(Ge_cf_,K)
    Gn_cf_k = np.tile(Gn_cf_,K)
    Ge_df_k = np.tile(Ge_df_,K)
    Gn_df_k = np.tile(Gn_df_,K)

    Ge_cf = Ge_cf_k * altmask
    Gn_cf = Gn_cf_k * altmask
    Ge_df = Ge_df_k * altmask
    Gn_df = Gn_df_k * altmask
    
    
    # #Kalles broadcasting way
    # r, theta, phi = np.meshgrid(r1d, theta1d, phi1d, indexing = 'ij')
    # r = np.array([np.meshgrid(np.ones(secs_grid.shape[0])*alts_grid[i], np.ones(secs_grid.shape[1])*alts_grid[i])[0] for i in range(len(alts_grid))])
    # Ge_cf_kalle = (Ge_cf_.reshape((N, -1, secs_grid.lon.shape[0] * secs_grid.lon.shape[1])) / 
    #          r.reshape((-1, alts_grid.size, secs_grid.lon.shape[0] * secs_grid.lon.shape[1]))).reshape(N, I*J*K)        



    
    # #Altitude correction factors, make into array
    # r_ = (RE + alt_)/(RE + alts_grid)
    # r_scale = np.tile(np.array([np.ones(I*J)*i for i in r_]).flatten(),np.array([N,1]))
    
    # # Correction of G matrices to express 3D current density [A/m2] instead of
    # # 2D current density [A/m] as initially designed for by O. Amm. (Slab correction)
    # altres = np.diff(alts_grid)
    # altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
    # r_2 = np.array([1./(np.ones(I*J)*i*1e3) for i in altres])
    # r_scale2 = np.tile(r_2.flatten(), np.array([N,1]))
    
    # #Apply altitude and slab corrections on hstacked (3rd dimesion) SECS G matrices
    # Ge_cf = Ge_cf_k * r_scale# * r_scale2
    # Gn_cf = Gn_cf_k * r_scale * r_scale2
    # Ge_df = Ge_df_k * r_scale * r_scale2
    # Gn_df = Gn_df_k * r_scale * r_scale2    
    
    return (Ge_cf, Gn_cf, Ge_df, Gn_df)



def get_alt_index(alts_grid, alt):
    """

    Parameters
    ----------
    alts_grid : array-like, 1D or 2D
        Altitude grid to use together with secs_grid. Length is K. Values represent the height in km of the centre of the
        voxels of each layer. Should be in increasing order.
    alt : array-like, 1D or 2D
        altitude [km] of the data/evaluation locations. Flattened to 1D of 
        length N
    Returns
    -------
    Array of length N of the index in vertical direction of each altitude in 
    alt. Altitudes outside the range spefified by alts_grid is given index -1

    """
    altres = np.diff(alts_grid)*0.5
    altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))

    edges = np.concatenate((alts_grid-altres, np.array([alts_grid[-1]+altres[-1]])))
    k = (np.digitize(alt, edges) - 1)
    overthetop = k == alts_grid.shape[0]
    k[overthetop] = -1

    return k    
    
def get_indices_kij(secs_grid, alts_grid, lat, lon, alt):
    """

    Parameters
    ----------
    secs_grid : CS grid object
        The grid we use to compute gridded indices of data/evaluation locations
        Dimension is (I,J)
    alts_grid : array-like, 1D or 2D
        Altitude grid to use together with secs_grid. Length is K. Values represent the height in km of the centre of the
        voxels of each layer. Should be in increasing order.
    lat : array-like, 1D or 2D
       latitudes [deg] of the data/evaluation locations. Length N when 
       flattened to 1D
    lon : array-like, 1D or 2D
       longitudes [deg] of the data/evaluation locations. Length N when 
       flattened to 1D
    alt : array-like, 1D or 2D
        altitude [km] of the data/evaluation locations. Length N when 
        flattened to 1D
    Returns
    -------
    Tuple of indices (k,i,j). k is index in altitude, i is index in CS eta 
    direction, j is index in CS xi direction, refering to alts_grid. (k,i,j)
    is flattened before return.

    """
    
    if (alt.flatten().shape[0] != lon.flatten().shape[0]) | (alt.flatten().shape[0] != lat.flatten().shape[0]):
        print('Dimension mismathc in evaluation locations')
        print (1/0)
    
    binnumber = secs_grid.bin_index(lon, lat)
    k = get_alt_index(alts_grid, alt).flatten()
    i = binnumber[0].flatten()
    j = binnumber[1].flatten()
    return (k, i, j)

    
def get_flattened_index_from_ijk(secs_grid, alts_grid, i, j, k):
    return  np.ravel_multi_index((k, i, j), (alts_grid.size, secs_grid.NL, secs_grid.NW)).flatten() #Same as in CS class
    
    
def get_jr_matrix(secs_grid, alts_grid, lat, lon, alt):
    """

    TODO: Add the thickness of SECS layer, such that solution amplitudes has 
            units of A/m, i.e. reflect the sheet current density, and SECS representation
            therefore describe volume current density A/m2

    Parameters
    ----------
    secs_grid : CS grid object
        The grid we use to compute gridded indices of data/evaluation locations
        Dimension is (I,J). Assumes secs_grid.A refer to bottom layer.
    alts_grid : array-like, 1D or 2D
        Altitude grid to use together with secs_grid. Length is K. Values represent the height in km of the centre of the
        voxels of each layer. Should be in increasing order.
    lat : array-like, 1D or 2D
       latitudes [deg] of the data/evaluation locations. Flattened to 1D of 
       length N
    lon : array-like, 1D or 2D
       longitudes [deg] of the data/evaluation locations. Flattened to 1D of 
       length N
    alt : array-like, 1D or 2D
        altitude [km] of the data/evaluation locations. Flattened to 1D of 
        length N
    Returns
    -------
    Tuple of indices (i,j,k). i is index in CS eta direction, j is index in 
    CS xi direction, k is index in altitude, refering to alts_grid.

    """
    # Remove data/evaluation points outside (2D) secs_grid, using the ingrid function
    use = secs_grid.ingrid(lon.flatten(), lat.flatten())
    lat = lat.flatten()[use]
    lon = lon.flatten()[use]
    alt = alt.flatten()[use]

    # Remove data/evaluation points outside grid in vertical direction
    k0 = get_alt_index(alts_grid, alt)
    inside = k0 != -1
    lat = lat.flatten()[inside]
    lon = lon.flatten()[inside]
    alt = alt.flatten()[inside]  
    
    # Evaluation locations.
    if not lat.shape[0]==lon.shape[0]==alt.shape[0]:
        print('Inconsistent dimensions of data/evaluation locations (lat,lon,alt)')
        print(1/0)        
        
    # Dimensions of 3D SECS grid
    N = lat.shape[0]
    I = secs_grid.shape[0]
    J = secs_grid.shape[1]
    K = alts_grid.shape[0]
    IJK = I*J*K
    
    #Vertical resolution
    altres = np.diff(alts_grid)
    altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
    
    # Horizontal area of each 3D grid cell
    A0 = secs_grid.A #in m2
    
    #Kalles broadcasting stuff. Maybe try this later to speed up
    # A = (A0.reshape((N, -1, I*J)))# / r.reshape((-1, r1d.size, phi1d.size * theta1d.size))).reshape(N, r.size)    
    # Ge = (Ge.reshape((N, -1, phi1d.size * theta1d.size)) / r.reshape((-1, r1d.size, phi1d.size * theta1d.size))).reshape(N, r.size)        

    #Indices of each evaluation point in 3D
    k, i, j = get_indices_kij(secs_grid, alts_grid, lat, lon, alt)

    # Corresponding flattened index    
    kij = np.ravel_multi_index((k,i,j), (K,I,J)) #flattened index
    
    # Make the vertical integration matrix S. S is very sparse. Should take advantage if that.
    S = np.zeros((N, IJK))
    for (counter,idx) in enumerate(kij): #each evaluation/data point        
        temp = np.zeros(IJK)
        k_n = np.arange(0, k[counter]+1)
        alt_n = alts_grid[0:k[counter]+1]
        i_n = np.ones(k[counter]+1).astype(int)*i[counter]
        j_n = np.ones(k[counter]+1).astype(int)*j[counter]
        fill =  np.ravel_multi_index((k_n, i_n, j_n), (K, I, J)) #flattened index   
        dr = altres[k_n] * 1e3 #altitude range of layer in m
        temp[fill] = dr/(A0[i[counter],j[counter]] * (alt_n+RE)/(alt_n[0]+RE))
        S[counter,:] = temp
        
    return S

