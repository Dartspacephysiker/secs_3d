#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:51:24 2022

@author: jone

Functions to produce various types of plots in 3D secs analysis, and its validation
using GEMINI output.

"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import cartopy.io.shapereader as shpreader
from pysymmetry.utils.spherical import sph_to_car, car_to_sph
from secs_3d.gemini_tools import RE
import apexpy
import matplotlib



def fixed_alt(data, shape = None, crange=(-20,20), cbartitle='Arb.', **kwargs):
    '''
    

    Parameters
    ----------
    data : tuple
        Each element is a dictionary containing what to plot in each panel.
    shape : tuple
        Shape of plot. Default is None, leading to a 1xN layout
    crange : tuple
        Size 2. Values used in color range. Units according to data values.
    cbartitle : str
        Title to go on the colorbar. Should indicate units.

    Returns
    -------
    None.

    '''
    if shape == None:
        shape = (1,len(data))
    ccc = 7
    fig = plt.figure(figsize = (0.5*ccc*shape[1],0.5*ccc*shape[0]))
    plotshape = (ccc*shape[0], ccc*shape[1]+1)
    
    #Colorbar
    cbarax = plt.subplot2grid(plotshape, (0, plotshape[1]-1), rowspan = ccc, colspan = 1)
    cmap = mpl.cm.seismic
    norm = mpl.colors.Normalize(vmin=crange[0], vmax=crange[1])
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label(cbartitle)
 
    row_i = -1
    col_i = -1
    # plt.figure(figsize=figsize)
    for i in range(len(data)):
        if (i % shape[1] == 0) & (shape[1]>1):
            col_i = col_i + 1
            row_i = 0
        elif (i % shape[1] == 0) & (shape[1]==1):
            col_i = 0
            row_i = row_i + 1
        else:
            col_i = col_i + 1
        ax = plt.subplot2grid(plotshape, (ccc*row_i, ccc*col_i), rowspan = ccc, colspan = ccc)
        ddd = data[i]
        # dxi = np.diff(ddd['xi'][0,:])[0]
        # deta = np.diff(ddd['eta'][:,0])[0]
        
        # plt.subplot(shape[0],shape[1],i+1)
        ax.set_axis_off()
        # plt.axis('off')
        ax.pcolormesh(ddd['xi'], ddd['eta'], ddd['values'], cmap='seismic', 
                      vmin=crange[0], vmax=crange[1])
        c1 = ax.contour(ddd['xi'], ddd['eta'], ddd['glat'], levels=[64,66,68,70], 
                         colors='grey', linewidths=0.5, **kwargs)
        ax.clabel(c1, inline=1, fontsize=10, fmt = '%1.0f$^\circ$')
        c2 = ax.contour(ddd['xi'], ddd['eta'], ddd['glon'], levels=[15,20,25,30], 
                         colors='grey', linewidths=0.5, **kwargs)
        ax.clabel(c2, inline=1, fontsize=10, fmt = '%1.0f$^\circ$')
        ax.set_title(ddd['title'])
        ax.set_xlim(ddd['xirange'][0], ddd['xirange'][1])
        ax.set_ylim(ddd['etarange'][0], ddd['etarange'][1])
        
        if 'plotgrid' in ddd.keys():
            for xi, eta in get_grid_boundaries(ddd['plotgrid'].xi_mesh, 
                        ddd['plotgrid'].eta_mesh, ddd['plotgrid'].NL, 
                        ddd['plotgrid'].NW):
                ax.plot(xi, eta, color = 'grey', linewidth = .4)
    
    plt.savefig(data[0]['filename'])
    


def altitude_profile(m, K, I, J, alts_grid, i = 6, j = 6):
    use_i = np.ones(K).astype(int)*i
    use_j = np.ones(K).astype(int)*j
    use_k = np.arange(K).astype(int)
    kijs = np.ravel_multi_index((use_k, use_i, use_j), (K,I,J))
    kijs_df = kijs + K*I*J
    plt.plot(m[kijs],alts_grid, label='CF')
    plt.plot(m[kijs_df],alts_grid, label='DF')
    plt.xlabel('SECS amplitude [A/m]')
    plt.ylabel('Altitude [km]')
    plt.title('i='+str(i)+', j='+str(j))
    plt.legend()    
    
def get_grid_boundaries(lon, lat, NL, NW):
    """ 
    Get grid boundaries for plotting 
        
    Yields tuples of (lon, lat) arrays that outline
    the grid cell boundaries. 

    Example:
    --------
    for c in obj.get_grid_boundaries():
        lon, lat = c
        plot(lon, lat, 'k-', transform = ccrs.Geocentric())
    """
    x, y = lon, lat

    for i in range(NL + NW + 2):
        if i < NL + 1:
            yield (x[i, :], y[i, :])
        else:
            i = i - NL - 1
            yield (x[:, i], y[:, i])
            

def get_coastlines(**kwargs):
    """ generate coastlines in projected coordinates """

    if 'resolution' not in kwargs.keys():
        kwargs['resolution'] = '50m'
    if 'category' not in kwargs.keys():
        kwargs['category'] = 'physical'
    if 'name' not in kwargs.keys():
        kwargs['name'] = 'coastline'

    shpfilename = shpreader.natural_earth(**kwargs)
    reader = shpreader.Reader(shpfilename)
    coastlines = reader.records()
    multilinestrings = []
    for coastline in coastlines:
        if coastline.geometry.geom_type == 'MultiLineString':
            multilinestrings.append(coastline.geometry)
            continue
        lon, lat = np.array(coastline.geometry.coords[:]).T 
        yield (lon, lat)

    for mls in multilinestrings:
        for ls in mls:
            lon, lat = np.array(ls.coords[:]).T 
            yield (lon, lat)    


    # # plt.plot(glon_secs[extend:-extend,extend], glat_secs[extend:-extend,extend], color='black')
    # # plt.plot(glon_secs[extend:-extend,-extend-1], glat_secs[extend:-extend,-extend], color='black')
    # # plt.plot(glon_secs[extend,extend:-extend], glat_secs[extend,extend:-extend], color='black')
    # # plt.plot(glon_secs[-extend-1,extend:-extend], glat_secs[-extend,extend:-extend], color='black')



def plot_field_aligned_segment(ax, mlon, mlat, alts_grid, color='green'):
    apex = apexpy.Apex(2022)
    xs = []
    ys = []
    zs = []
    # alts = np.linspace(0,500, 20)
    for alt in alts_grid:
        glat_, glon_, e = apex.apex2geo(mlat, mlon, alt)
        x,y,z = sph_to_car((RE+alt, 90-glat_, glon_), deg=True)
        xs.append(x[0])
        ys.append(y[0])
        zs.append(z[0])
    ax.plot(xs,ys,zs, color=color)    

def plot_hor_segment(ax, mlons, mlats, alt, color='green'):
    apex = apexpy.Apex(2022)
    glat_, glon_, e = apex.apex2geo(mlats, mlons, alt)
    x,y,z = sph_to_car((RE+alt, 90-glat_, glon_), deg=True)
    ax.plot(x,y,z, color=color)


def field_aligned_grid(ax, grid, alts_grid, color='green', showlayers=False, 
                       showbase=True, fullbox=False):
    '''
    Make 3D plot of volume spanned by CS grid following a field line from its
    central location
    
    Parameters
    ----------
    ax : matplotlib 3D axis object
        To plot on
    grid : CS grid object
        The CS grid ato top or bottom boundary to extend along field line.
    alts_grid : 1-D array type
        Center location of altitude layers. In km.

    Returns
    -------
    None.

    '''
    # Plot grid and coastlines:
    # fig = plt.figure(figsize = (10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    #ax.set_axis_off()
    
    #Calculate ecef grid boundaries
    apex = apexpy.Apex(2022)
    L = grid.L*1e-3
    Lres = grid.Lres*1e-3
    pos = grid.projection.position
    lat_ = pos[1] # in degrees
    lon_ = pos[0] # in degrees
    site_mlat, site_mlon = apex.geo2apex(lat_, lon_, 0*0.001)
    x_, y_, z_ = sph_to_car((RE, 90-lat_, lon_), deg=True)
    xlim = (x_[0]-L-10*Lres, x_[0]+L+10*Lres) 
    ylim = (y_[0]-L-10*Lres, y_[0]+L+10*Lres) 
    zlim = (RE, RE+alts_grid[-1]+1)
    zlim = (z_[0], z_[0]+ alts_grid[-1])
    #Plot coastlines in ecef frame
    for cl in get_coastlines():
        x,y,z = sph_to_car((RE, 90-cl[1], cl[0]), deg=True)
        use = (x > xlim[0]-L/2) & (x < xlim[1]+L/2) & (y > ylim[0]-L/2) & (y < ylim[1]+L/2) & (z > 0)
        ax.plot(x[use], y[use], z[use], color = 'C0')
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_zlim(5600,6400)
    ax.set_zlim(zlim)
    ######
    
    #Plot field-aligned layers of the SECS grid 
    mlats0, mlons0 = apex.geo2apex(grid.lat_mesh, grid.lon_mesh, alts_grid[0]) #-1?
    # glats = grid.lat_mesh[np.newaxis]
    # glons = grid.lon_mesh[np.newaxis]
    # mlats = mlats0[np.newaxis]
    # mlons = mlons0[np.newaxis]
    if showbase:
        glat_,glon_, _ = apex.apex2geo(mlats0, mlons0, alts_grid[0])
        for lon, lat in get_grid_boundaries(glon_, glat_, grid.NL, grid.NW):
            x,y,z = sph_to_car((RE+alts_grid[0], 90-lat, lon), deg=True)
            ax.plot(x, y, z, color = 'grey', linewidth = .4)   
    
    if showlayers:
        for alt in alts_grid[::4]:
            glat_,glon_, _ = apex.apex2geo(mlats0, mlons0, alt)
            # glats = np.vstack((glats, glat_[np.newaxis]))
            # glons = np.vstack((glons, glon_[np.newaxis]))
            # mlat_, mlon_ = apex.geo2apex(glat_, glon_, alt)
            # mlats = np.vstack((mlats, mlat_[np.newaxis]))
            # mlons = np.vstack((mlons, mlon_[np.newaxis]))
            for lon, lat in get_grid_boundaries(glon_, glat_, grid.NL, grid.NW):
                x,y,z = sph_to_car((RE+alt, 90-lat, lon), deg=True)
                ax.plot(x, y, z, color = 'grey', linewidth = .4)
        

    
    #Horizontal boundary
    plot_hor_segment(ax, mlons0[0,:], mlats0[0,:], alts_grid[0], color=color)
    plot_hor_segment(ax, mlons0[-1,:], mlats0[-1,:], alts_grid[0], color=color)
    plot_hor_segment(ax, mlons0[:,0], mlats0[:,0], alts_grid[0], color=color)
    plot_hor_segment(ax, mlons0[:,-1], mlats0[:,-1], alts_grid[0], color=color)
    
    if fullbox:
        #Horizontal boundary
        plot_hor_segment(ax, mlons0[0,:], mlats0[0,:], alts_grid[-1], color=color)
        plot_hor_segment(ax, mlons0[-1,:], mlats0[-1,:], alts_grid[-1], color=color)
        plot_hor_segment(ax, mlons0[:,0], mlats0[:,0], alts_grid[-1], color=color)
        plot_hor_segment(ax, mlons0[:,-1], mlats0[:,-1], alts_grid[-1], color=color)
    
        #Field-aligned boundary
        plot_field_aligned_segment(ax, mlons0[0,0], mlats0[0,0], alts_grid, color=color)
        plot_field_aligned_segment(ax, mlons0[0,-1], mlats0[0,-1], alts_grid, color=color)
        plot_field_aligned_segment(ax, mlons0[-1,0], mlats0[-1,0], alts_grid, color=color)
        plot_field_aligned_segment(ax, mlons0[-1,-1], mlats0[-1,-1], alts_grid, color=color)  
        

def spherical_grid(ax, lat_ev, lon_ev, alt_ev, color='red'):

    # Vertical lines
    x,y,z = sph_to_car((RE+alt_ev[:,0,0], 90-lat_ev[:,0,0], lon_ev[:,0,0]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[:,-1,0], 90-lat_ev[:,-1,0], lon_ev[:,-1,0]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[:,0,-1], 90-lat_ev[:,0,-1], lon_ev[:,0,-1]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[:,-1,-1], 90-lat_ev[:,-1,-1], lon_ev[:,-1,-1]), deg=True)
    ax.plot(x,y,z, color=color)

    #Horizontal lines
    x,y,z = sph_to_car((RE+alt_ev[0,:,0], 90-lat_ev[0,:,0], lon_ev[0,:,0]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[0,:,-1], 90-lat_ev[0,:,-1], lon_ev[0,:,-1]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[0,0,:], 90-lat_ev[0,0,:], lon_ev[0,0,:]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[0,-1,:], 90-lat_ev[0,-1,:], lon_ev[0,-1,:]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[-1,:,0], 90-lat_ev[-1,:,0], lon_ev[-1,:,0]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[-1,:,-1], 90-lat_ev[-1,:,-1], lon_ev[-1,:,-1]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[-1,0,:], 90-lat_ev[-1,0,:], lon_ev[-1,0,:]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = sph_to_car((RE+alt_ev[-1,-1,:], 90-lat_ev[-1,-1,:], lon_ev[-1,-1,:]), deg=True)
    ax.plot(x,y,z, color=color)

def plot_field_line(ax, lat0, lon0, alts_grid, color='grey', dipole=False, **kwargs):
    if dipole:
        from gemini3d.grid import convert
        mlon_, mtheta_ = convert.geog2geomag(lon0,lat0)
        m_theta = np.arcsin(np.sqrt((RE+alts_grid)/(RE+alts_grid[0]))*np.sin(mtheta_))
        m_mlon = np.ones(alts_grid.size)*mlon_
        m_glon, m_glat = convert.geomag2geog(m_mlon, m_theta)
        x,y,z = sph_to_car((RE+alts_grid, 90-m_glat, m_glon), deg=True)
        ax.plot(x,y,z, color=color, alpha=0.5, **kwargs)

    else:
        apex = apexpy.Apex(2022)
        mlat0, mlon0 = apex.geo2apex(lat0, lon0, alts_grid[0])
        xs = []
        ys = []
        zs = []
        for alt in alts_grid:
            glat_, glon_, e = apex.apex2geo(mlat0, mlon0, alt)
            x,y,z = sph_to_car((RE+alt, 90-glat_, glon_), deg=True)
            xs.append(x[0])
            ys.append(y[0])
            zs.append(z[0])
        ax.plot(xs,ys,zs, color=color, alpha=0.5, **kwargs)


def plot_resolution(ax, grid, alts_grid, kij, psf, az=-26, el=7, clim=1e-6, 
                   planes=[0,1], dipole=True, alpha=0.5):
    '''

    Parameters
    ----------
    ax : matplotlib axis object
        Axis to plot on
    grid : TYPE
        DESCRIPTION.
    alts_grid : TYPE
        DESCRIPTION.
    kij : TYPE
        DESCRIPTION.
    psf : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    #Grid dimensions
    K = alts_grid.shape[0] #Number of vertival layers
    I = grid.shape[0] #Number of cells in eta direction
    J = grid.shape[1]  #Number of cells in xi direction 
    k ,i, j = np.unravel_index(kij, (K,I,J))

    ax.view_init(azim=az, elev=el)
    field_aligned_grid(ax, grid, alts_grid, color='green')
    kwargs={'linewidth':3}
    # for kk in range(lat_ev[0,-1,:].size):
    #     visualization.plot_field_line(ax, lat_ev[0,-1,kk], lon_ev[0,-1,kk], 
    #                               alts__, color='orange', **kwargs, dipole=True)
    #     visualization.plot_field_line(ax, lat_ev[0,sh[1]//2,kk], lon_ev[0,sh[1]//2,kk], 
    #                               alts__, color='orange', **kwargs, dipole=True)
    xis = grid.xi[0,:]
    etas = grid.eta[:,0]
    xi_, eta_ = np.meshgrid(xis, etas, indexing = 'xy')
    alt_, eta_, xi_ = np.meshgrid(alts_grid, etas, xis, indexing='ij')
    alt_, etas_, xis_ = np.meshgrid(alts_grid, etas, xis, indexing='ij')
    lon_, lat_ = grid.projection.cube2geo(xis_, etas_)
    sh = lon_.shape
    x, y, z = sph_to_car((RE+alt_.flatten(), 90-lat_.flatten(), 
                          lon_.flatten()), deg=True)
    cmap = plt.cm.seismic
    norm = matplotlib.colors.Normalize(vmin=-clim, vmax=clim)
    if 0 in planes:
        p = ax.plot_surface(x.reshape(sh)[k,:,:], y.reshape(sh)[k,:,:], 
                            z.reshape(sh)[k,:,:], alpha=alpha, zorder=1,
                            facecolors=cmap(norm(psf.reshape(sh)[k,:,:])), cmap=cmap)
    if 1 in planes:
        p = ax.plot_surface(x.reshape(sh)[:,i,:], y.reshape(sh)[:,i,:], 
                            z.reshape(sh)[:,i,:], alpha=alpha, zorder=3,
                            facecolors=cmap(norm(psf.reshape(sh)[:,i,:])), cmap=cmap)
    if 2 in planes:
        p = ax.plot_surface(x.reshape(sh)[:,:,j], y.reshape(sh)[:,:,j], 
                        z.reshape(sh)[:,:,j], alpha=alpha, zorder=2,
                        facecolors=cmap(norm(psf.reshape(sh)[:,:,j])), cmap=cmap)    
    ax.scatter(x[kij], y[kij], z[kij], s=50, marker='*', color='green')
    
    #Field lines
    for kk in range(lat_[0,-1,:].size):     
        plot_field_line(ax, lat_[0,-1,kk], lon_[0,-1,kk], 
                                  alts_grid, color='orange', **kwargs, dipole=dipole)
        
    x0, y0, z0 = sph_to_car((RE+0, 90-grid.projection.position[1], grid.projection.position[0]), deg=True)
    range_ =  alts_grid[-1]*0.3
    ax.set_xlim(x0-range_, x0+range_)
    ax.set_ylim(y0-range_, y0+range_)
    ax.set_zlim(z0, z0+2*range_)
    ax.set_title('PSF at k='+str(k)+', i='+str(i)+', j='+str(j))