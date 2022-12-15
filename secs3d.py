#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:38:11 2022

@author: jone

Tools for expressing vector fields in 3D using spherical geometry and SECS
representation in horizontal layers. Developed for describing electric current 
density field in 3D, with the built in constraint of current continuity and
zero current at bottom side of domain. Description is developed with intention
to be applied to EISCAT 3D data in the future, from which the perpendicular
current density can be obtained under the assumtion of equipotential field lines.

Conceptual development is largely building on what has been outlined by Kalle,
described in this document: https://www.overleaf.com/project/6329caf229ba68180bc4152f

"""

import numpy as np
from secsy.secsy import utils as secsy
import secs_3d.gemini_tools

RE = 6371.2 #Earth radius in km


def get_alt_index(alts_grid, alt, returnfloat=False):
    """

    Parameters
    ----------
    alts_grid : array-like, 1D or 2D
        Altitude grid to use together with secs_grid. Length is K. Values 
        represent the height in km of the centre of the
        voxels of each layer. Should be in increasing order.
    alt : array-like, 1D or 2D
        altitude [km] of the data/evaluation locations. Flattened to 1D of 
        length N
    returnfloat : Boolean
        If one wants to know how close it is to a neighboring layer. Default is
        False
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
    
    if returnfloat:
        use = k != -1
        res_ = (alt[use] - edges[k[use]])/(edges[k[use]+1]-edges[k[use]])
        # res_ = (alt[use] - alts_grid[k[use]])/(2*altres[k[use]])
        k = k.astype(float)
        k[use] = k[use] - 0.5 + res_

    return k    
    
def get_indices_kij(secs_grid, alts_grid, lat, lon, alt, returnfloat=False):
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
       flattened to 1D. Must be inside secs_grid (not checked).
    lon : array-like, 1D or 2D
       longitudes [deg] of the data/evaluation locations. Length N when 
       flattened to 1D. Must be inside secs_grid (not checked).
    alt : array-like, 1D or 2D
        altitude [km] of the data/evaluation locations. Length N when 
        flattened to 1D.
    returnfloat : boolean, optional
        return the exact index of the input location. Defailt is False.
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
    k = get_alt_index(alts_grid, alt, returnfloat=returnfloat).flatten()
    
    i = binnumber[0].flatten()
    j = binnumber[1].flatten()
    if returnfloat:
        xi_obs, eta_obs = secs_grid.projection.geo2cube(lon, lat)
        xi_grid = secs_grid.xi[i,j]
        eta_grid = secs_grid.eta[i,j]
        i_frac = (eta_obs-eta_grid)/secs_grid.deta
        j_frac = (xi_obs-xi_grid)/secs_grid.dxi
        return (k, i+i_frac, j+j_frac)    
    else:
        return (k, i, j)


def get_SECS_J_G_matrices_3D(secs_grid, alts_grid, lat, lon, alt, constant = 
                             1./(4.*np.pi), singularity_limit=None, 
                             interpolate=False, ext_factor=-1):
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
    interpolate : Boolean, optional
        If True: Each row in G (observation/evaloation, N)
        will affect the two closest layers of SECS nodes (above and below its 
        location). Its vertical placement reflect the relative weight of the 
        influence of each layer. S matrix should be constructed with the same
        option. Default is False
    TODO:
    interpolation : Boolean, optional
        If True: Each row in G (observation/evaloation, N)
        will affect the two closest layers of SECS nodes (above and below its 
        location). Its vertical placement reflect the relative weight of the 
        influence of each layer. S matrix should be constructed with the same
        option.
    ext_factor : int, optional
        To control how to filter out locations based on their proximity to the 
        grid. The default is -1, removing points closer than 1 grid cell from 
        the edge of the grid (mesh).        

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
    use = secs_grid.ingrid(lon.flatten(), lat.flatten(), ext_factor=ext_factor)
    lat = lat.flatten()[use]
    lon = lon.flatten()[use]
    alt = alt.flatten()[use]
    
    # Remove data/evaluation points outside grid in vertical direction
    k0 = get_alt_index(alts_grid, alt, returnfloat=True)
    inside = (k0 >= 0) & (k0<alts_grid.size-1)
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
    k, i, j = get_indices_kij(secs_grid, alts_grid, lat, lon, alt, 
                              returnfloat=interpolate)  
    # kij = np.ravel_multi_index((k,i,j), (K,I,J)) #flattened index

    
    if interpolate:
        k_under = np.floor(k).astype(int)
        k_over = np.ceil(k).astype(int)
        same = k_over == k_under
        k_over[same] = k_over[same] + 1
        overthetop = k_over >= len(alts_grid)
        k_over[overthetop] = len(alts_grid)-1
        kij_start_under = np.ravel_multi_index((k_under, 
                                                np.zeros(len(k)).astype(int),
                                                np.zeros(len(k)).astype(int)), 
                                                (K,I,J)) #flattened
        kij_stop_under = kij_start_under + I*J
        kij_start_over = np.ravel_multi_index((k_over, 
                                                np.zeros(len(k)).astype(int),
                                                np.zeros(len(k)).astype(int)), 
                                                (K,I,J)) #flattened
        kij_stop_over = kij_start_over + I*J
        k_frac = k-np.round(k).astype(int)
        negs = k_frac<0
        w_under = np.zeros(len(k))
        w_under[negs] = -k_frac[negs]
        poss = k_frac>=0
        w_under[poss] = -k_frac[poss]+1
        w_over = np.zeros(len(k))
        w_over[negs] = k_frac[negs]+1
        w_over[poss] = k_frac[poss]

    #Do the altitude correction
    altmask = []
    r_ = (RE + alt_)/(RE + alts_grid) #Altitude correction factors        
    for kkk in range(N):
        mask_ = np.zeros(IJK)
        if interpolate:
            mask_[kij_start_under[kkk]:kij_stop_under[kkk]] = w_under[kkk] * r_[k_under[kkk]]   
            if not overthetop[kkk]:
                mask_[kij_start_over[kkk]:kij_stop_over[kkk]] = w_over[kkk] * r_[k_over[kkk]]    
        else:
            kij_start = np.ravel_multi_index((k[kkk],0,0), (K,I,J)) #flattened
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

    return (Ge_cf, Gn_cf, Ge_df, Gn_df)


    
def get_jr_matrix(secs_grid, alts_grid, lat, lon, alt, interpolate=None, 
                  ext_factor=-1):
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
    lat : array-like, 1D or 2D
       latitudes [deg] of the data/evaluation locations. Flattened to 1D of 
       length N
    lon : array-like, 1D or 2D
       longitudes [deg] of the data/evaluation locations. Flattened to 1D of 
       length N
    alt : array-like, 1D or 2D
        altitude [km] of the data/evaluation locations. Flattened to 1D of 
        length N. Need to be within the range specified by alts_grid. Points 
        outside will be removed.
    highres : Boolean, optional
        Specifies whether to take into account the vertical placement of the
        data/evaluation location, and distribute the effect of the data into
        two layers: The layer it falls within, and the closest one. 
        Default is False
    interpolate : Boolean, optional
        Affect two things if True: 1) Vertical location of observation/evaluation
        location is implemented as a weighted contribution of the above and below
        node in the construction of S. 2) When making S, the closest
        four grid cells (in each layer) are considered, weighted by their 
        distance to the location in question (n). Default is False.
    ext_factor : int, optional
        To control how to filter out locations based on their proximity to the 
        grid. The default is -1, removing points closer than 1 grid cell from 
        the edge of the grid (mesh).        
    Returns
    -------
    Tuple of indices (i,j,k). i is index in CS eta direction, j is index in 
    CS xi direction, k is index in altitude, refering to alts_grid.

    """
    # Remove data/evaluation points outside (2D) secs_grid, using the ingrid function
    use = secs_grid.ingrid(lon.flatten(), lat.flatten(), ext_factor=ext_factor)
    lat = lat.flatten()[use]
    lon = lon.flatten()[use]
    alt = alt.flatten()[use]

    # Remove data/evaluation points outside grid in vertical direction
    k0 = get_alt_index(alts_grid, alt, returnfloat=True)
    inside = (k0 >= 0) & (k0<alts_grid.size-1)
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
    KIJ = K*I*J
    
    #Vertical resolution
    altres = np.diff(alts_grid)
    altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
    
    # Horizontal area of each 3D grid cell
    A0 = secs_grid.A #in m2
    
    #Kalles broadcasting stuff. Maybe try this later to speed up
    # A = (A0.reshape((N, -1, I*J)))# / r.reshape((-1, r1d.size, phi1d.size * theta1d.size))).reshape(N, r.size)    
    # Ge = (Ge.reshape((N, -1, phi1d.size * theta1d.size)) / r.reshape((-1, r1d.size, phi1d.size * theta1d.size))).reshape(N, r.size)        


    #Indices of each evaluation point in 3D
    k, i, j = get_indices_kij(secs_grid, alts_grid, lat, lon, alt, 
                              returnfloat=interpolate)
    # Corresponding flattened index    
    kij = np.ravel_multi_index((np.round(k).astype(int),np.round(i).astype(int),
                                np.round(j).astype(int)), (K,I,J)) #flattened index
    
    # Make the vertical integration matrix S. S is very sparse. TO BE IMPLEMENTED
    S = np.zeros((N, KIJ)) #The final integration matrix
    for (counter,idx) in enumerate(kij): #each evaluation/data point  
        # print(counter)#k[counter], i[counter], j[counter])      
        temp = np.zeros(KIJ) #The part of S corresponding to observation idx
        if interpolate:
            k_n = np.arange(0, np.ceil(k[counter])+1).astype(int)
            if k[counter] % 1 == 0: #add above layer, but set to 0 weight
                k_n = np.hstack((k_n,k_n[-1]+1))
            ks = k_n.size
            alt_n = alts_grid[0:np.floor(k[counter]).astype(int)+2]            
            if (i[counter]<=1) | (i[counter]>=I-1) | (j[counter]<=1) | (j[counter]>=J-1):
                # Do not interpolate horizontally on points close to edge
                i_n = np.ones(k_n.size).astype(int)*np.round(i[counter]).astype(int)
                j_n = np.ones(k_n.size).astype(int)*np.round(j[counter]).astype(int)
                fill =  np.ravel_multi_index((k_n, i_n, j_n), (K, I, J)) #flattened index   
                dr = altres[k_n] * 1e3 #altitude range of layer in m
                temp[fill] = -dr/(A0[i_n,j_n] * (alt_n+RE)/(alt_n[0]+RE))
                #negative sign due to the sign convention of amplituded and FAC defined by Amm     

                #Apply the linear interpolation scheme in vertical direction
                k_frac = k[counter] % 1
                w_over = k_frac
                w_under = 1 - w_over
                under_ = np.take(fill,np.array([ks-2]))
                over_ = np.take(fill,np.array([ks-1]))
                temp[under_] = temp[under_] * w_under
                temp[over_] = temp[over_] * w_over
            else: #Point is in interior             
                # Identify the four neighboring secs nodes.
                xi1 = secs_grid.xi[np.floor(i[counter]).astype(int),
                                   np.floor(j[counter]).astype(int)]
                eta1 = secs_grid.eta[np.floor(i[counter]).astype(int),
                                   np.floor(j[counter]).astype(int)]            
                xi2 = secs_grid.xi[np.ceil(i[counter]).astype(int),
                                   np.floor(j[counter]).astype(int)]
                eta2 = secs_grid.eta[np.ceil(i[counter]).astype(int),
                                   np.floor(j[counter]).astype(int)]                        
                xi3 = secs_grid.xi[np.ceil(i[counter]).astype(int),
                                   np.ceil(j[counter]).astype(int)]
                eta3 = secs_grid.eta[np.ceil(i[counter]).astype(int),
                                   np.ceil(j[counter]).astype(int)]                        
                xi4 = secs_grid.xi[np.floor(i[counter]).astype(int),
                                   np.ceil(j[counter]).astype(int)]
                eta4 = secs_grid.eta[np.floor(i[counter]).astype(int),
                                   np.ceil(j[counter]).astype(int)]
                xi_obs, eta_obs = secs_grid.projection.geo2cube(lon[counter], lat[counter])
                #Bilinear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
                w1 = (xi4-xi_obs)*(eta2-eta_obs) / ((xi4-xi1)*(eta2-eta1)) #w11
                w2 = (xi4-xi_obs)*(eta_obs-eta1) / ((xi4-xi1)*(eta2-eta1)) #w12
                w3 = (xi_obs-xi1)*(eta_obs-eta1) / ((xi4-xi1)*(eta2-eta1)) #w22
                w4 = (xi_obs-xi1)*(eta2-eta_obs) / ((xi4-xi1)*(eta2-eta1)) #w21
                wij = np.hstack((np.tile(w1,ks),np.tile(w2,ks),np.tile(w3,ks),np.tile(w4,ks)))
                #Where to fill in temp array for this observation/evaluation location
                #Here all layers are treated the same way, just scaled with area
                k_ns = np.tile(k_n,4) 
                i_ns = np.hstack((np.ones(k_n.size).astype(int)*np.floor(i[counter]).astype(int),
                                np.ones(k_n.size).astype(int)*np.ceil(i[counter]).astype(int),
                                np.ones(k_n.size).astype(int)*np.ceil(i[counter]).astype(int),
                                np.ones(k_n.size).astype(int)*np.floor(i[counter]).astype(int)))
                j_ns = np.hstack((np.ones(k_n.size).astype(int)*np.floor(j[counter]).astype(int),
                                np.ones(k_n.size).astype(int)*np.floor(j[counter]).astype(int),
                                np.ones(k_n.size).astype(int)*np.ceil(j[counter]).astype(int),
                                np.ones(k_n.size).astype(int)*np.ceil(j[counter]).astype(int)))       
                fill =  np.ravel_multi_index((k_ns, i_ns, j_ns), (K, I, J)) #flattened index   
                dr = np.tile(altres[k_n],4) * 1e3 #altitude range of layer in m
                temp[fill] = -dr*wij/(A0[i_ns,j_ns] * (np.tile(alt_n,4)+RE)/(alt_n[0]+RE))            
                #negative sign due to the sign convention of amplituded and FAC defined by Amm
    
                #Apply the linear interpolation scheme in vertical direction
                k_frac = k[counter] % 1
                w_over = k_frac
                w_under = 1 - w_over
                under_ = np.take(fill,np.array([ks-2, 2*ks-2, 3*ks-2, 4*ks-2]))
                over_ = np.take(fill,np.array([ks-1, 2*ks-1, 3*ks-1, 4*ks-1]))
                temp[under_] = temp[under_] * w_under
                temp[over_] = temp[over_] * w_over          

        else:
            k_n = np.arange(0, k[counter]+1)
            alt_n = alts_grid[0:np.floor(k[counter]).astype(int)+1]
            i_n = np.ones(k_n.size).astype(int)*np.round(i[counter]).astype(int)
            j_n = np.ones(k_n.size).astype(int)*np.round(j[counter]).astype(int)
            fill =  np.ravel_multi_index((k_n, i_n, j_n), (K, I, J)) #flattened index   
            dr = altres[k_n] * 1e3 #altitude range of layer in m
            temp[fill] = -dr/(A0[i[counter],j[counter]] * (alt_n+RE)/(alt_n[0]+RE))
            #negative sign due to the sign convention of amplituded and FAC defined by Amm

        S[counter,:] = temp
        
    return S



def make_b_unitvectors(Br, Btheta, Bphi):
    """
    

    Parameters
    ----------
    Br : array-like
        1D array of radial magnetic field strength.
    Btheta : array-like
        1D array of magnetic field strength in theta direction (towards south).
    Bphi : array-like
        1D array of magnetic field strength in eta direction (towards east).

    Returns
    -------
    tuple containing the following:
    
    br : array-like
        Radial component of unit vector of magnetic field.
    btheta : array-like
        Theta component of unit vector of magnetic field.
    bphi : array-like
        Phi component of unit vector of magnetic field.

    """
    Bmag = np.sqrt(Br**2 + Btheta**2 + Bphi**2)
    br = Br/Bmag
    btheta = Btheta/Bmag
    bphi = Bphi/Bmag
    return (br, btheta, bphi)


def make_B(br, btheta, bphi):
    """
    Make matrix that project 3D vectors (r, theta, phi components) to the plane
    perpendicular to the magnetic field as given in input. The returned 
    matrix need to act on a 1D array of N vectors (length of 3N) sorted 
    vectorwise, i.e. [r1, theta1, phi1, ... rN, thetaN, phiN].
    
    Parameters
    ----------
    br : array-like
        Radial component of unit vector of magnetic field.
    btheta : array-like
        Theta component of unit vector of magnetic field.
    bphi : array-like
        Phi component of unit vector of magnetic field.

    Returns
    -------
    B : Projection matrix. 3Nx3N array, N is length of input array
       
    """
    br = br.flatten()
    btheta = btheta.flatten()
    bphi = bphi.flatten()
    N = br.flatten().size
    
    # The nine components of the projection matrix
    brr = btheta**2+bphi**2
    brtheta = -br*btheta
    brphi = -br*bphi
    bthetar = -br*btheta
    bthetatheta = br**2+bphi**2
    bthetaphi = -btheta*bphi
    bphir = -br*bphi
    bphitheta = -btheta*bphi
    bphiphi = br**2+btheta**2
    # from scipy import sparse    
    # B = sparse.csc_matrix((3*N,3*N))
    B = np.zeros((3*N,3*N))
    for n in range(N):
        B[3*n,3*n] = brr[n]
        B[3*n,3*n+1] = brtheta[n]
        B[3*n,3*n+2] = brphi[n]
        B[3*n+1,3*n] = bthetar[n]
        B[3*n+1,3*n+1] = bthetatheta[n]
        B[3*n+1,3*n+2] = bthetaphi[n]
        B[3*n+2,3*n] = bphir[n]
        B[3*n+2,3*n+1] = bphitheta[n]
        B[3*n+2,3*n+2] = bphiphi[n]        
    
    return B

def make_P(N):
    """
    Function to make permutation matrix that act on a 1D array of length 3N 
    with a vectorwise representation i.e. 
    v = [r1, theta1, phi1, ... rN, thetaN, phiN]. Result of P.dot(v) is a 3N 1D
    array with componentwise representation, 
    i.e. [r1...rN, theta1...thetaN, phi1...phiN]. P is orthogonal, hence 
    P**-1 = P.T

    Parameters
    ----------
    N : int
        Number of measurements.

    Returns
    -------
    P : 2D array
        Permutation matrix that will produce component wise representation
        when acting on a vectorized representation.

    """
    P = np.zeros((3*N,3*N))
    for n in range(N):
        P[3*n+0, n+0] = 1
        P[3*n+1, n+0+N] = 1
        P[3*n+2, n+0+2*N] = 1
    return P

def make_G(grid, alts_grid, lat, lon, alt, interpolate=True, ext_factor=-1):
    Ge_cf, Gn_cf, Ge_df, Gn_df = get_SECS_J_G_matrices_3D(grid, alts_grid, 
                    lat, lon, alt, interpolate=interpolate, 
                    singularity_limit=grid.Lres, ext_factor=ext_factor)
    S = get_jr_matrix(grid, alts_grid, lat, lon, alt, interpolate=interpolate, 
                      ext_factor=ext_factor)
    O = np.zeros(S.shape)
    Gcf = np.vstack((S, -Gn_cf, Ge_cf))
    Gdf = np.vstack((O, -Gn_df, Ge_df))
    G = np.hstack((Gcf, Gdf))
    
    return G