import gemini3d.read as read
import importlib  
plotcurv = importlib.import_module("pygemini-scripts.gridoutput.plotcurv")
# from pygemini-scripts.gridoutput.plotcurv import plotcurv3D
from matplotlib.pyplot import show
from gemini3d.grid.gridmodeldata import model2magcoords,model2geogcoords, geog2dipole
from gemini3d.grid.convert import unitvecs_geographic, geog2geomag
import numpy as np
import matplotlib.pyplot as plt
import helpers
from lompe.secsy.secsy import utils as secsy
from scipy.linalg import lstsq
from lompe.ppigrf import igrf
import datetime as dt
import xarray as xr
import pandas as pd
import apexpy

#Global variables
path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/gemini_output/"
height = 800 #km. height of secs CS grid at top boundary. Will be mapped down
RE = 6371.2 #Earth radius in km

#Define CS grid
xg = read.grid(path)
extend=1
grid, grid_ev = helpers.make_csgrid(xg, height=height, crop_factor=0.7, 
                                    resolution_factor=0.2, extend=extend) #outer (secs), inner (evaluation) grid


#Open simulation output files
var = ["v1", "v2", "v3", "Phi", "J1", "J2", "J3", "ne"]
cfg = read.config(path)
xg = read.grid(path)
dims = xg['lx']
times = cfg["time"][-1:]
t = times[0]
dat = read.frame(path, t, var=var)
dat = helpers.compute_enu_components(xg, dat)
# var = ["ne", "Ti", "Te", "v1", "v2", "v3", "J1", "J2", "J3", "Phi"]


#################################3
##################################
# Layered SECS representation
#############################
#################################3
# 1) Sampling from model at specific height below top boundary
# alts = [90,100,110,120]#,130,140,150,160,170,180,190,200,225,250,275,300,350,400,450,500,600,700,800]
alts = np.flip(np.concatenate((np.arange(80,170,2),np.arange(170,400,10),np.arange(400,850,50))))
altres = np.diff(alts)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
prev_fac = 0
model_facs = []
prev_model = 0
model_layer_facs = []
prev_eiscat = 0
eiscat_layer_facs = []
apex = apexpy.Apex(2022)
for (i, alt) in enumerate(alts):    
    #Sample from layer
    datadict = helpers.sample_at_alt(xg, dat, grid=grid_ev, alt=alt, altres=altres[i], 
                                     time_ind = -1, path=path)

    # 1) Mapping grids from upper boundary to given altitude with apexpy
    mlats_secs, mlons_secs = apex.geo2apex(grid.lat, grid.lon, height) 
    glat_secs, glon_secs, _ = apex.apex2geo(mlats_secs, mlons_secs, alt)
    mlats_secs_ev, mlons_secs_ev = apex.geo2apex(grid_ev.lat, grid_ev.lon, height)
    glat_secs_ev, glon_secs_ev, _ = apex.apex2geo(mlats_secs_ev, mlons_secs_ev, alt)
    
    # 2) Helmmholtz decomposition of horizontal currents at alt, as expressed in 
    # the model (The ground truth, not what EISCAT will observe)
    Ge_cf_alt, Gn_cf_alt = secsy.get_SECS_J_G_matrices(datadict['glatmesh'].flatten(), 
                datadict['glonmesh'].flatten(), glat_secs.flatten(), 
                glon_secs.flatten(), constant = 1./(4.*np.pi), 
                RI=RE * 1e3 + alt * 1e3, current_type = 'curl_free', 
                singularity_limit=grid.Lres*0.5)
    Gcf_alt = np.vstack((Ge_cf_alt, Gn_cf_alt))
    Ge_df_alt, Gn_df_alt = secsy.get_SECS_J_G_matrices(datadict['glatmesh'].flatten(), 
                datadict['glonmesh'].flatten(), glat_secs.flatten(), 
                glon_secs.flatten(), constant = 1./(4.*np.pi), 
                RI=RE * 1e3 + alt * 1e3, current_type = 'divergence_free', 
                singularity_limit=grid.Lres*0.5)
    Gdf_alt = np.vstack((Ge_df_alt, Gn_df_alt))
    G = np.hstack((Gcf_alt,Gdf_alt))
    d = np.hstack((datadict['je'].flatten()*altres[i]*2e3, datadict['jn'].flatten()*altres[i]*2e3))
    use = np.isfinite(d)
    m_alt_model = lstsq(G[use,:], d[use], cond=0.01)[0]
    m_cf_alt_model = m_alt_model[0:m_alt_model.shape[0]//2]
    m_df_alt_model = m_alt_model[m_alt_model.shape[0]//2:]
    #Evaluate CF SECS representation.
    Ge_, Gn_ = secsy.get_SECS_J_G_matrices(glat_secs_ev.flatten(), 
                glon_secs_ev.flatten(), glat_secs.flatten(), glon_secs.flatten(), 
                constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + height * 1e3, 
                current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
    G_pred = np.vstack((Ge_, Gn_))
    Jcf = G_pred.dot(m_cf_alt_model)    
    
    # 3) Calculate the perpendicular CF (and DF) current from mapped E and ion 
    # velocities using j = ne(vi-ve)) (What EISCAT will observe)
    (Ee, En, Eu) = helpers.grid_e_field_at_alt(datadict, grid_ev)
    (Be, Bn, Bu) = helpers.grid_e_field_at_alt(datadict, grid_ev, return_B=True)
    n = helpers.grid_param_at_alt(datadict, grid_ev, param='n')
    (ve, vn, vu) = helpers.grid_param_at_alt(datadict, grid_ev, param='v')
    (je, jn, ju) = helpers.grid_param_at_alt(datadict, grid_ev, param='j')
    Bmag = np.sqrt(Be.flatten()**2+Bn.flatten()**2+Bu.flatten()**2)
    mappedE = np.vstack((Ee.flatten(),En.flatten(),Eu.flatten()))
    B = np.vstack((Be.flatten(),Bn.flatten(),Bu.flatten()))
    v_electrons = np.cross(mappedE, B, axis=0)/Bmag**2
    jE = n.flatten() * 1.6e-19 * (ve.flatten() - v_electrons[0,:])
    jN = n.flatten() * 1.6e-19 * (vn.flatten() - v_electrons[1,:])
    jU = n.flatten() * 1.6e-19 * (vu.flatten() - v_electrons[2,:])
    #Initiate the layered SECS representation at the altitude below top boundary
    Ge_cf_alt, Gn_cf_alt = secsy.get_SECS_J_G_matrices(glatmesh.flatten(), glonmesh.flatten(), 
                glat_secs.flatten(), glon_secs.flatten(), constant = 1./(4.*np.pi), 
                RI=6371.2 * 1e3 + alt * 1e3, current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
    Gcf_alt = np.vstack((Ge_cf_alt, Gn_cf_alt))
    Ge_df_alt, Gn_df_alt = secsy.get_SECS_J_G_matrices(glatmesh.flatten(), glonmesh.flatten(), 
                glat_secs.flatten(), glon_secs.flatten(), constant = 1./(4.*np.pi), 
                RI=6371.2 * 1e3 + alt * 1e3, current_type = 'divergence_free', singularity_limit=grid.Lres*0.5)
    Gdf_alt = np.vstack((Ge_df_alt, Gn_df_alt))
    G = np.hstack((Gcf_alt,Gdf_alt))
    d = np.hstack((jE*altres[i]*2e3, jN*altres[i]*2e3))
    use = np.isfinite(d)
    # m_cf_alt = lstsq(Gcf_alt[use,:], d[use], cond=0.01)[0]
    m_alt = lstsq(G[use,:], d[use], cond=0.01)[0]
    m_cf_alt = m_alt[0:m_alt.shape[0]//2]
    # m_df_alt = m_alt[m_alt.shape[0]//2:]
    
    

    use = np.isfinite(vei.flatten()) & grid_ev.ingrid(glonmesh.flatten(), glatmesh.flatten())
    
    # 2) Mapping from upper boundary to given altitude
    apex = apexpy.Apex(2022)
    #Map the sampeled GEMINI locations to specified altitude
    mlats0, mlons0 = apex.geo2apex(glatmesh0, glonmesh0, height) #Magnetic coordinates of init grid at height=height
    glat_, glon_, _ = apex.apex2geo(mlats0, mlons0, alt) # new geographic locations at a mapped lower altitude than top boundary
    #Map SECS grid to specific altitude
    mlats_secs, mlons_secs = apex.geo2apex(grid.lat, grid.lon, height) #Magnetic coordinates of init grid at height=height
    glat_secs, glon_secs, _ = apex.apex2geo(mlats_secs, mlons_secs, alt) # new geographic locations at a mapped lower altitude than top boundary
    #Evaluate electric field at all sample locations at top boundary
    Ge_cf, Gn_cf = secsy.get_SECS_J_G_matrices(glatmesh0.flatten(), glonmesh0.flatten(), 
                grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
                RI=6371.2 * 1e3 + height * 1e3, current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
    G_pred = np.vstack((Ge_cf, Gn_cf))
    Ecf = G_pred.dot(m_cf)
    Ecf_e = Ecf[0:Ecf.shape[0]//2] 
    Ecf_n = Ecf[Ecf.shape[0]//2:]
    mappedE = apex.map_E_to_height(mlats0.flatten(), mlons0.flatten(), height, alt, np.vstack((Ecf_e, Ecf_n, np.zeros(Ecf_e.flatten().shape[0]))))
    # #Inspect how the mapping has affected E-field
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.pcolormesh(glonmesh0, glatmesh0, Ecf_e.reshape(glatmesh0.shape), cmap='seismic', vmin=-0.04, vmax=0.04)
    # plt.quiver(glonmesh0[::9,::9].flatten(), glatmesh0[::9,::9].flatten(), Ecf_e.reshape(glatmesh0.shape)[::9,::9].flatten(), Ecf_n.reshape(glatmesh0.shape)[::9,::9].flatten(), color='green', scale=0.4, width=0.005)
    # plt.xlim(-4,37)
    # plt.ylim(62,74)
    # plt.xlabel('glon')
    # plt.ylabel('glat')
    # plt.title('E$_{East}$ @ '+str(height)+' km: CF SECS')
    # plt.subplot(1,2,2)
    # plt.pcolormesh(glon_, glat_, mappedE[0,:].reshape(glatmesh0.shape), cmap='seismic', vmin=-0.04, vmax=0.04)
    # plt.colorbar(label='V/m')
    # plt.quiver(glon_[::9,::9].flatten(), glat_[::9,::9].flatten(), mappedE[0,:].reshape(glatmesh0.shape)[::9,::9].flatten(), mappedE[1,:].reshape(glatmesh0.shape)[::9,::9].flatten(), color='green', scale=0.4, width=0.005)
    # plt.xlim(-4,37)
    # plt.ylim(62,74)
    # plt.xlabel('glon')
    # plt.ylabel('glat')
    # plt.title('E$_{East}$ @ '+str(alt)+' km: Mapped')
    
    
    # 2) Calculate the perpendicular CF (and DF) current from mapped E and ion velocities using j = ne(vi-ve)) (What EISCAT will observe)
    Bmag_alt = np.sqrt(Bei_alt**2+Bni_alt**2+Bui_alt**2).flatten()
    v_electrons = np.cross(mappedE, np.vstack((Bei_alt.flatten(),Bni_alt.flatten(),Bui_alt.flatten())), axis=0)/Bmag_alt**2
    jE = ni.flatten() * 1.6e-19 * (vei.flatten() - v_electrons[0,:])
    jN = ni.flatten() * 1.6e-19 * (vni.flatten() - v_electrons[1,:])
    jU = ni.flatten() * 1.6e-19 * (vui.flatten() - v_electrons[2,:])
    #Initiate the layered SECS representation at the altitude below top boundary
    Ge_cf_alt, Gn_cf_alt = secsy.get_SECS_J_G_matrices(glatmesh.flatten(), glonmesh.flatten(), 
                glat_secs.flatten(), glon_secs.flatten(), constant = 1./(4.*np.pi), 
                RI=6371.2 * 1e3 + alt * 1e3, current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
    Gcf_alt = np.vstack((Ge_cf_alt, Gn_cf_alt))
    Ge_df_alt, Gn_df_alt = secsy.get_SECS_J_G_matrices(glatmesh.flatten(), glonmesh.flatten(), 
                glat_secs.flatten(), glon_secs.flatten(), constant = 1./(4.*np.pi), 
                RI=6371.2 * 1e3 + alt * 1e3, current_type = 'divergence_free', singularity_limit=grid.Lres*0.5)
    Gdf_alt = np.vstack((Ge_df_alt, Gn_df_alt))
    G = np.hstack((Gcf_alt,Gdf_alt))
    d = np.hstack((jE*altres[i]*2e3, jN*altres[i]*2e3))
    use = np.isfinite(d)
    # m_cf_alt = lstsq(Gcf_alt[use,:], d[use], cond=0.01)[0]
    m_alt = lstsq(G[use,:], d[use], cond=0.01)[0]
    m_cf_alt = m_alt[0:m_alt.shape[0]//2]
    # m_df_alt = m_alt[m_alt.shape[0]//2:]
    
    
    #Helmmholtz decomposition of horizontal currents at alt, as expressed in the model (The ground truth, not what EISCAT will observe)
    G = G#Gcf_alt
    d = np.hstack((jei_alt.flatten()*altres[i]*2e3, jni_alt.flatten()*altres[i]*2e3))
    use = np.isfinite(d)
    m_alt_model = lstsq(G[use,:], d[use], cond=0.01)[0]
    m_cf_alt_model = m_alt_model[0:m_alt_model.shape[0]//2]
    m_df_alt_model = m_alt_model[m_alt_model.shape[0]//2:]
    
    #Evaluate CF SECS representation. First, map the evaluation grid to same altitude
    mlats_secs_ev, mlons_secs_ev = apex.geo2apex(grid_ev.lat, grid_ev.lon, height) #Magnetic coordinates of init grid at height=height
    glat_secs_ev, glon_secs_ev, _ = apex.apex2geo(mlats_secs_ev, mlons_secs_ev, alt) # new geographic locations at a mapped lower altitude
    Ge_, Gn_ = secsy.get_SECS_J_G_matrices(glat_secs_ev.flatten(), glon_secs_ev.flatten(), glat_secs.flatten(), glon_secs.flatten(), constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + height * 1e3, current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
    G_pred = np.vstack((Ge_, Gn_))
    Jcf = G_pred.dot(m_cf_alt_model)
    
    
    #Inspection plots of currents at different alts etc
    jscale = 4 #How much weaker the current at each layer is
    A_alt = grid.A * np.nanmean(Bmag/Bmag_alt) #Area scaled to the specific altitude
    #Claculate total FAC up
    ii, jj = grid.bin_index(glonmesh.flatten(),glatmesh0.flatten())
    ii1d = grid._index(ii, jj)
    df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 'ju': jui_alt.flatten()})
    facs = df.groupby('i1d').ju.mean() #Should make proper FAC, not just use ju
    df_sorted = facs.reindex(index=pd.Series(np.arange(0,len(grid.lon.flatten()))), method='nearest', tolerance=0.1)
    grid_fac = df_sorted.values.reshape(grid.shape) #FAC from model at alt, intepolated to grid
    ups_model = grid_fac > 0
    Iup_total = np.sum(grid_fac[ups_model] * A_alt[ups_model])*1e-3
    model_facs.append(Iup_total)
    diff_fac = Iup_total - prev_fac
    prev_fac = Iup_total
    ups_model_layer = (m_cf_alt_model < 0) & grid_ev.ingrid(glon_secs.flatten(), glat_secs.flatten())
    Iup_model_layer = np.sum(-m_cf_alt_model[ups_model_layer])*1e-3
    model_layer_facs.append(Iup_model_layer)
    diff_model = Iup_model_layer - prev_model
    prev_model = Iup_model_layer
    ups_layer = (m_cf_alt < 0) & grid_ev.ingrid(glon_secs.flatten(), glat_secs.flatten())
    Iup_layer = np.sum(-m_cf_alt[ups_layer])*1e-3
    eiscat_layer_facs.append(Iup_layer)
    diff_eiscat = Iup_layer - prev_eiscat
    prev_eiscat = Iup_layer

    plt.figure(figsize=(10,3.5))
    plt.subplot(1,3,1)
    plt.pcolormesh(glonmesh, glatmesh, jui_alt[0,:,:]*1e6, cmap='seismic', vmin=-4, vmax=4)
    # plt.colorbar(label='$\mu$A/m$^2$')
    plt.title('FAC @ '+str(alt)+' km')
    plt.plot(glon_secs[extend:-extend,extend], glat_secs[extend:-extend,extend], color='black')
    plt.plot(glon_secs[extend:-extend,-extend-1], glat_secs[extend:-extend,-extend], color='black')
    plt.plot(glon_secs[extend,extend:-extend], glat_secs[extend,extend:-extend], color='black')
    plt.plot(glon_secs[-extend-1,extend:-extend], glat_secs[-extend,extend:-extend], color='black')
    plt.xlim(-4,37)
    plt.ylim(62,74)
    plt.xlabel('glon')
    plt.ylabel('glat')
    plt.text(0, 73, 'Iup = %4i kA' % Iup_total)
    if alt != alts[0]:
        plt.text(0, 72, 'diff = %4i kA' % diff_fac)
    
    plt.subplot(1,3,2)
    ju_alt_model = -(m_cf_alt_model.reshape(grid.shape)/A_alt)*1e6
    plt.pcolormesh(glon_secs, glat_secs, ju_alt_model*1e0, cmap='seismic', vmin=-jscale, vmax=jscale)
    plt.plot(glon_secs[extend:-extend,extend], glat_secs[extend:-extend,extend], color='black')
    plt.plot(glon_secs[extend:-extend,-extend-1], glat_secs[extend:-extend,-extend], color='black')
    plt.plot(glon_secs[extend,extend:-extend], glat_secs[extend,extend:-extend], color='black')
    plt.plot(glon_secs[-extend-1,extend:-extend], glat_secs[-extend,extend:-extend], color='black')
    plt.xlim(-4,37)
    plt.ylim(62,74)
    plt.yticks(color='w')
    plt.xlabel('glon')
    # plt.ylabel('glat')
    plt.title('CF SECS amp. @ '+str(alt)+' km')
    plt.text(0, 73, 'Iup (layer) = %5i kA' % Iup_model_layer)
    if alt != alts[0]:
        plt.text(0, 72, 'diff = %5i A' % diff_model)
    
    plt.subplot(1,3,3)
    # SHOULD SCALE AREA WITH HEIGHT
    ju_alt = -(m_cf_alt.reshape(grid.shape)/A_alt)*1e6
    plt.pcolormesh(glon_secs, glat_secs, ju_alt, cmap='seismic', vmin=-jscale, vmax=jscale)
    plt.colorbar(label='$\mu$A/m$^2$')
    plt.plot(glon_secs[extend:-extend,extend], glat_secs[extend:-extend,extend], color='black')
    plt.plot(glon_secs[extend:-extend,-extend-1], glat_secs[extend:-extend,-extend], color='black')
    plt.plot(glon_secs[extend,extend:-extend], glat_secs[extend,extend:-extend], color='black')
    plt.plot(glon_secs[-extend-1,extend:-extend], glat_secs[-extend,extend:-extend], color='black')
    plt.xlim(-4,37)
    plt.ylim(62,74)
    plt.yticks(color='w')
    plt.xlabel('glon')
    # plt.ylabel('glat')
    plt.title('EISCAT equivalent')
    plt.text(0, 73, 'Iup (layer) = %5i kA' % Iup_layer)
    if alt != alts[0]:
        plt.text(0, 72, 'diff = %5i kA' % diff_eiscat)
    filename = './plots/layered_currents_%03ikm.png' % alt
    plt.savefig(filename)

    # #Test that the sampled je amd jn currents are reproduced by estimating them with j = ne(vi-ve)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.pcolormesh(glonmesh0, glatmesh0, jei_alt[0,:,:], vmin=-2e-6, vmax=2e-6, cmap='seismic')
    # nn = 9
    # plt.quiver(glonmesh0[::nn,::nn].flatten(), glatmesh0[::nn,::nn].flatten(), jei_alt[0,::nn,::nn].flatten(), jni_alt[0,::nn,::nn].flatten(), color='green', scale=1e-5, width=0.005)
    # plt.subplot(1,2,2)
    # plt.pcolormesh(glonmesh0, glatmesh0, jE.reshape(glatmesh0.shape), vmin=-2e-6, vmax=2e-6, cmap='seismic')
    # nn = 9
    # plt.colorbar()
    # plt.quiver(glonmesh0[::nn,::nn].flatten(), glatmesh0[::nn,::nn].flatten(), jE.reshape(glatmesh0.shape)[::nn,::nn].flatten(), jN.reshape(glatmesh0.shape)[::nn,::nn].flatten(), color='green', scale=1e-5, width=0.005)

from cmodel.helpers import make_gif
import glob
files = glob.glob('./plots/*.png')
files.sort()
make_gif(files, filename='./plots/secs_layers.gif', fps=2)













##########################3
#Old stuff, 
#Get E-field

# #######################################3
# ###########################################
# #SECS description of E-field at top boundary
# #############################################

# glatmesh0 = datadict['glatmesh']
# glonmesh0 = datadict['glonmesh']

# use = np.isfinite(datadict['ve'].flatten()) & grid_ev.ingrid(glonmesh0.flatten(), glatmesh0.flatten())
# Ge_cf, Gn_cf = secsy.get_SECS_J_G_matrices(glatmesh0.flatten()[use], glonmesh0.flatten()[use], 
#             grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
#             RI=6371.2 * 1e3 + height * 1e3, current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
# Ge_df, Gn_df = secsy.get_SECS_J_G_matrices(glatmesh0.flatten()[use], glonmesh0.flatten()[use], 
#             grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), 
#             RI=6371.2 * 1e3 + height * 1e3, current_type = 'divergence_free', singularity_limit=grid.Lres*0.5)

# #Convert to E-field "measurement" by invoking corss product E = - v x B. ENU components are in geographical, spherical frame.
# Ee_model = -(datadict['vn'].flatten() * datadict['Bu'].flatten() - datadict['vu'].flatten() * datadict['Bn'].flatten())
# En_model = (datadict['ve'].flatten() * datadict['Bu'].flatten() - datadict['vu'].flatten() * datadict['Be'].flatten())
# Eu_model = -(datadict['ve'].flatten() * datadict['Bn'].flatten() - datadict['vn'].flatten() * datadict['Be'].flatten())
# Ee_model_mesh = -(datadict['vn'] * datadict['Bu'] - datadict['vu'] * datadict['Bn'])
# En_model_mesh = (datadict['ve'] * datadict['Bu'] - datadict['vu'] * datadict['Be'])
# d = np.hstack((Ee_model[use],En_model[use])) #Append to data vector

# Gcf = np.vstack((Ge_cf,Gn_cf)) 
# Gdf = np.vstack((Ge_df,Gn_df)) 
# # G = np.hstack((Gcf,Gdf)) 

# #Solve
# m_cf = lstsq(Gcf, d, cond=0.01)[0] 
# m_df = lstsq(Gdf, d, cond=0.01)[0] 
# # m = lstsq(G, d, cond=0.01)[0] #Solve combined
# # m_cf = m[0:m.shape[0]//2]
# # m_df = m[:m.shape[0]//2]

# #Plotting for verification purposes
# #Compare SECS representation of convection with input data
# plt.figure(figsize=(10,4))
# plt.subplot(1,3,1)
# # plt.scatter(glonmesh.flatten()[use], glatmesh.flatten()[use], c = Ee_model[use], cmap='seismic', vmin=-0.04, vmax=0.04)
# plt.pcolormesh(glonmesh0, glatmesh0, Ee_model_mesh[0,:,:], cmap='seismic', vmin=-0.04, vmax=0.04)
# # plt.colorbar()
# # plt.quiver(glonmesh.flatten()[use][::19], glatmesh.flatten()[use][::19], Ee_model[use][::19], En_model[use][::19], color='green')
# plt.quiver(glonmesh0[::9,::9].flatten(), glatmesh0[::9,::9].flatten(), Ee_model_mesh[0,:,:][::9,::9].flatten(), En_model_mesh[0,:,:][::9,::9].flatten(), color='green', scale=0.4, width=0.005)
# # plt.scatter(glonmesh.flatten()[use], glatmesh.flatten()[use], c = vei[0,:,:].flatten()[use], cmap='seismic', vmin=-1000, vmax=1000)
# # plt.pcolormesh(glonmesh, glatmesh, vni[0,:,:], vmin=-1000,vmax=1000, cmap='seismic')
# plt.xlim(-4,37)
# plt.ylim(62,74)
# plt.xlabel('glon')
# plt.ylabel('glat')
# plt.title('E$_{East}$ @ '+str(height)+' km: GEMINI')

# #SECS prediction
# # grid_ev = helpers.make_csgrid(xg, height=height, crop_factor=0.9)
# Ge_, Gn_ = secsy.get_SECS_J_G_matrices(grid_ev.lat.flatten(), grid_ev.lon.flatten(), grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + height * 1e3, current_type = 'curl_free', singularity_limit=grid.Lres*0.5)
# G_pred = np.vstack((Ge_, Gn_))
# Ecf = G_pred.dot(m_cf)
# Ge_, Gn_ = secsy.get_SECS_J_G_matrices(grid_ev.lat.flatten(), grid_ev.lon.flatten(), grid.lat.flatten(), grid.lon.flatten(), constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + height * 1e3, current_type = 'divergence_free', singularity_limit=grid.Lres*0.5)
# G_pred = np.vstack((Ge_, Gn_))
# Edf = G_pred.dot(m_df)
# # phi, theta = geog2geomag(grid_ev.lon.flatten(),grid_ev.lat.flatten()) #phi, theta are the magnetic centered dipole coords of the resampled locations
# # Be, Bn, Bu = igrf(grid_ev.lon.flatten(), grid_ev.lat.flatten(), height, dtime) # in nT in ENU
# Ecf_e = Ecf[0:Ecf.shape[0]//2] 
# Ecf_n = Ecf[Ecf.shape[0]//2:]
# Edf_e = Edf[0:Edf.shape[0]//2] 
# Edf_n = Edf[Edf.shape[0]//2:]

# # Ve = (En / (Bu.flatten()*1e-9))# - (Eu / (Bn.flatten()*1e-9))
# # Vn = -Ee / (Bu.flatten()*1e-9)


# ######
# #SECS decomposition plot
# plt.subplot(1,3,2)
# # plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Vn, cmap='seismic', vmin=-1000, vmax=1000)
# # plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Ecf_e+Edf_e, cmap='seismic', vmin=-0.04, vmax=0.04)
# plt.pcolormesh(grid_ev.lon, grid_ev.lat, (Ecf_e).reshape(grid_ev.lat.shape), cmap='seismic', vmin=-0.04, vmax=0.04)
# # plt.quiver()
# # plt.colorbar()
# nn = 3
# EEe = Ecf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()# + Edf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
# EEn = Ecf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()# + Edf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
# plt.quiver(grid_ev.lon[::nn,::nn].flatten(), grid_ev.lat[::nn,::nn].flatten(), EEe, EEn, color='green', scale=0.4, width=0.005)
# plt.xlim(-4,37)
# plt.ylim(62,74)
# plt.xlabel('glon')
# plt.ylabel('glat')
# plt.title('E$_{East}$ @ '+str(height)+' km: SECS CF')

# plt.subplot(1,3,3)
# # plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Vn, cmap='seismic', vmin=-1000, vmax=1000)
# # plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Ecf_e+Edf_e, cmap='seismic', vmin=-0.04, vmax=0.04)
# plt.pcolormesh(grid_ev.lon, grid_ev.lat, (Edf_e).reshape(grid_ev.lat.shape), cmap='seismic', vmin=-0.04, vmax=0.04)
# # plt.quiver()
# plt.colorbar(label='V/m')
# nn = 3
# EEe = Edf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()# + Edf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
# EEn = Edf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()# + Edf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
# plt.quiver(grid_ev.lon[::nn,::nn].flatten(), grid_ev.lat[::nn,::nn].flatten(), EEe, EEn, color='green', scale=0.4, width=0.005)
# plt.xlim(-4,37)
# plt.ylim(62,74)
# plt.xlabel('glon')
# plt.ylabel('glat')
# plt.title('E$_{East}$ @ '+str(height)+' km: SECS DF')

# # plt.subplot(1,3,3)
# # # plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Vn, cmap='seismic', vmin=-1000, vmax=1000)
# # # plt.scatter(grid_ev.lon.flatten(), grid_ev.lat.flatten(), c = Ecf_e+Edf_e, cmap='seismic', vmin=-0.04, vmax=0.04)
# # plt.pcolormesh(grid_ev.lon, grid_ev.lat, (Ecf_e+Edf_e).reshape(grid_ev.lat.shape), cmap='seismic', vmin=-0.04, vmax=0.04)
# # # plt.quiver()
# # plt.colorbar()
# # nn = 3
# # EEe = Ecf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten() + Edf_e.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
# # EEn = Ecf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten() + Edf_n.reshape(grid_ev.lat.shape)[::nn,::nn].flatten()
# # plt.quiver(grid_ev.lon[::nn,::nn].flatten(), grid_ev.lat[::nn,::nn].flatten(), EEe, EEn, color='green', scale=0.4, width=0.005)
# # plt.xlim(-4,37)
# # plt.ylim(65,77)
# # plt.xlabel('glon')
# # plt.ylabel('glat')
# # plt.title('SECS: CF+DF Eastward electric field at 500 km')
# #As pointed out by Heikki, the sum of the independent CF + DF field does not necesarily 
# # reflect the real field, as the Laplacian part mey project into both comopnents, due 
# # to the Helmholtz decomposition not being unique locally I suppose. We also see the same 
# # as one of the Vanhamäki papers, that the Laplacian parts is confined to the padded 
# #boundary SECS nodes. Hence, we stick with our decomposition inside our domain/grid (SECS)


# ##############
# #Solution amplitude plots
# plt.figure()
# plt.subplot(1,2,1)
# plt.pcolormesh(grid.lon, grid.lat, m_cf.reshape(grid.shape), cmap='seismic', vmin=-800, vmax=800)
# plt.plot(grid.lon[extend:-extend,extend], grid.lat[extend:-extend,extend], color='black')
# plt.plot(grid.lon[extend:-extend,-extend-1], grid.lat[extend:-extend,-extend], color='black')
# plt.plot(grid.lon[extend,extend:-extend], grid.lat[extend,extend:-extend], color='black')
# plt.plot(grid.lon[-extend-1,extend:-extend], grid.lat[-extend,extend:-extend], color='black')
# plt.xlabel('glon')
# plt.ylabel('glat')
# # plt.colorbar()
# plt.title('CF SECS amplitudes')
# plt.subplot(1,2,2)
# plt.pcolormesh(grid.lon, grid.lat, m_df.reshape(grid.shape), cmap='seismic', vmin=-800, vmax=800)
# plt.plot(grid.lon[extend:-extend,extend], grid.lat[extend:-extend,extend], color='black')
# plt.plot(grid.lon[extend:-extend,-extend-1], grid.lat[extend:-extend,-extend], color='black')
# plt.plot(grid.lon[extend,extend:-extend], grid.lat[extend,extend:-extend], color='black')
# plt.plot(grid.lon[-extend-1,extend:-extend], grid.lat[-extend,extend:-extend], color='black')
# plt.xlabel('glon')
# plt.ylabel('glat')
# plt.title('DF SECS amplitudes')
# plt.colorbar()
# # End plotting of E field representation at top boundary
# ######################################


# #####################################
# ###############3#####################
# #SECS representation of FACs above dynamo region
# #############################################
# use = np.isfinite(jui.flatten()) & grid_ev.ingrid(glonmesh0.flatten(), glatmesh0.flatten())
# ii, jj = grid.bin_index(glonmesh0.flatten()[use],glatmesh0.flatten()[use])
# ii1d = grid._index(ii, jj)
# df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 'ju': jui.flatten()[use]})
# facs = df.groupby('i1d').ju.mean()
# df_sorted = facs.reindex(index=pd.Series(np.arange(0,len(grid.lon.flatten()))), method='nearest', tolerance=0.1)
# grid_fac = df_sorted.values.reshape(grid.shape) #The "top boundary condition" for layerded SECS below
# # Inspect result of new binning
# plt.figure()
# plt.subplot(1,2,1)
# plt.pcolormesh(glonmesh0, glatmesh0, jui[0,:,:]*1e6, cmap='seismic', vmin=-4, vmax=4)
# plt.title('GEMINI FAC @ '+str(height)+' km')
# plt.plot(grid.lon[extend:-extend,extend], grid.lat[extend:-extend,extend], color='black')
# plt.plot(grid.lon[extend:-extend,-extend-1], grid.lat[extend:-extend,-extend], color='black')
# plt.plot(grid.lon[extend,extend:-extend], grid.lat[extend,extend:-extend], color='black')
# plt.plot(grid.lon[-extend-1,extend:-extend], grid.lat[-extend,extend:-extend], color='black')
# plt.subplot(1,2,2)
# plt.pcolormesh(grid.lon, grid.lat, grid_fac*1e6, cmap='seismic', vmin=-4, vmax=4)
# plt.title('CS grid FAC @ '+str(height)+' km')
# plt.colorbar(label='$\mu$ A/m$^2$')
# plt.plot(grid.lon[extend:-extend,extend], grid.lat[extend:-extend,extend], color='black')
# plt.plot(grid.lon[extend:-extend,-extend-1], grid.lat[extend:-extend,-extend], color='black')
# plt.plot(grid.lon[extend,extend:-extend], grid.lat[extend,extend:-extend], color='black')
# plt.plot(grid.lon[-extend-1,extend:-extend], grid.lat[-extend,extend:-extend], color='black')






    
    # plt.figure(figsize=(10,4))
    # plt.subplot(1,3,1)
    # plt.pcolormesh(glonmesh0, glatmesh0, jui[0,:,:]*1e6, cmap='seismic', vmin=-4, vmax=4)
    # plt.quiver(glonmesh0[::9,::9].flatten(), glatmesh0[::9,::9].flatten(), jei[0,::9,::9].flatten()*1e6, jni[0,::9,::9].flatten()*1e6, color='green', scale=10, width=0.005)
    # plt.xlim(-4,37)
    # plt.ylim(62,74)
    # plt.xlabel('glon')
    # plt.ylabel('glat')
    # plt.title('J @ '+str(height)+' km: GEMINI J')
    # plt.subplot(1,3,2)
    # plt.pcolormesh(glon_, glat_, jU.reshape(glatmesh0.shape)*1e6, cmap='seismic', vmin=-4, vmax=4)
    # plt.quiver(glon_[::9,::9].flatten(), glat_[::9,::9].flatten(), jE.reshape(glatmesh0.shape)[::9,::9].flatten()*1e6, jN.reshape(glatmesh0.shape)[::9,::9].flatten()*1e6, color='green', scale=10, width=0.005)
    # plt.xlim(-4,37)
    # plt.ylim(62,74)
    # plt.xlabel('glon')
    # plt.ylabel('glat')
    # plt.title('J @ '+str(alt)+' km: v$_{ion}$ + frozen e-')
    # plt.subplot(1,3,3)
    # plt.pcolormesh(glon_, glat_, jui_alt[0,:,:]*1e6, cmap='seismic', vmin=-4, vmax=4)
    # plt.colorbar(label='$\mu$A/m$^2$')
    # plt.quiver(glon_[::9,::9].flatten(), glat_[::9,::9].flatten(), jei_alt[0,::9,::9].flatten()*1e6, jni[0,::9,::9].flatten()*1e6, color='green', scale=10, width=0.005)
    # plt.xlim(-4,37)
    # plt.ylim(62,74)
    # plt.xlabel('glon')
    # plt.ylabel('glat')
    # plt.title('J @ '+str(alt)+' km: GEMINI J')
    
    
    
    
    
    
    
    # ############
    # #Variability plots
    # plt.subplots(1,1)
    # sss = np.nanmean(Ve_model_meshs[:,:,1:],axis=2)
    # pcolormesh(glonmesh, glatmesh,sss, vmin=-1000,vmax=1000)
    # plt.colorbar()
    # plt.xlabel('glon')
    # plt.ylabel('glat')
    # plt.title('Eastward velocity (geo) mean [m/s]')
    
    
    # plt.subplots(1,1)
    # sss = np.nanstd(Ve_model_meshs[:,:,1:],axis=2)
    # pcolormesh(glonmesh, glatmesh,sss, vmin=0,vmax=12)
    # plt.colorbar()
    # plt.xlabel('glon')
    # plt.ylabel('glat')
    # plt.title('Eastward velocity (geo) std [m/s]')
    
    
    # #####################################################################
    # #Some sample plotting examples from Matt
    # ################################
    
    
    
    # # these plotting functions will internally grid data
    # print("Plotting...")
    # plotcurv.plotcurv3D(xg, dat[parm], cfg, lalt=128, llon=128, llat=128, coord="geographic")
    
    
    # ###############################################################################
    # # produce gridded dataset arrays from model output for user
    # ###############################################################################
    # lalt=256; llon=128; llat=256;
    
    # # regrid data in geographic
    # print("Sampling in geographic coords...")
    # galti, gloni, glati, parmgi = model2geogcoords(xg, dat[parm], lalt, llon, llat, wraplon=True)
    
    # # regrid in geomagnetic
    # print("Sampling in geomagnetic coords...")
    # malti, mloni, mlati, parmmi = model2magcoords(xg, dat[parm], lalt, llon, llat)
    
    
    # # bring up plot
    # show(block=False)
    
    # ###############################################################################
    # # read in a vector quantity, rotate into geographic components and then grid
    # ###############################################################################
    # v1=dat["v1"]; v2=dat["v2"]; v3=dat["v3"];
    # [egalt,eglon,eglat]=unitvecs_geographic(xg)    
    # #^ returns a set of geographic unit vectors on xg; these are in ECEF geomag comps
    # #    like all other unit vectors in xg
    
    # # each of the components in models basis projected onto geographic unit vectors
    # vgalt=( np.sum(xg["e1"]*egalt,3)*dat["v1"] + np.sum(xg["e2"]*egalt,3)*dat["v2"] + 
    #     np.sum(xg["e3"]*egalt,3)*dat["v3"] )
    # vglat=( np.sum(xg["e1"]*eglat,3)*dat["v1"] + np.sum(xg["e2"]*eglat,3)*dat["v2"] +
    #     np.sum(xg["e3"]*eglat,3)*dat["v3"] )
    # vglon=( np.sum(xg["e1"]*eglon,3)*dat["v1"] + np.sum(xg["e2"]*eglon,3)*dat["v2"] + 
    #     np.sum(xg["e3"]*eglon,3)*dat["v3"] )
    
    # # must grid each (geographic) vector components separately
    # print("Sampling vector compotnents in geographic...")
    # galti, gloni, glati, vgalti = model2geogcoords(xg, vgalt, lalt, llon, llat, wraplon=True)
    # galti, gloni, glati, vglati = model2geogcoords(xg, vglat, lalt, llon, llat, wraplon=True)
    # galti, gloni, glati, vgloni = model2geogcoords(xg, vglon, lalt, llon, llat, wraplon=True)
    
    # # for comparison also grid the flows in the model coordinate system componennts
    # galti, gloni, glati, v1i = model2geogcoords(xg, dat["v1"], lalt, llon, llat, wraplon=True)
    # galti, gloni, glati, v2i = model2geogcoords(xg, dat["v2"], lalt, llon, llat, wraplon=True)
    # galti, gloni, glati, v3i = model2geogcoords(xg, dat["v3"], lalt, llon, llat, wraplon=True)
    
    # # quickly compare flows in model components vs. geographic as a meridional slice
    # plt.subplots(1,3)
    
    # plt.subplot(2,3,1)
    # plt.pcolormesh(glati,galti,v1i[:,64,:])
    # plt.xlabel("glat")
    # plt.ylabel("glon")
    # plt.title("$v_1$")
    # plt.colorbar()
    
    # plt.subplot(2,3,2)
    # plt.pcolormesh(glati,galti,v2i[:,64,:])
    # plt.xlabel("glat")
    # plt.ylabel("glon")
    # plt.colorbar()
    # plt.title("$v_2$")
    
    # plt.subplot(2,3,3)
    # plt.pcolormesh(glati,galti,v3i[:,64,:])
    # plt.xlabel("glat")
    # plt.ylabel("glon")
    # plt.colorbar()
    # plt.title("$v_3$")
    
    # plt.subplot(2,3,4)
    # plt.pcolormesh(glati,galti,vgalti[:,64,:])
    # plt.xlabel("glat")
    # plt.ylabel("glon")
    # plt.title("$v_r$")
    # plt.colorbar()
    
    # plt.subplot(2,3,5)
    # plt.pcolormesh(glati,galti,vglati[:,64,:])
    # plt.xlabel("glat")
    # plt.ylabel("glon")
    # plt.colorbar()
    # plt.title("$v_{mer}$")
    
    # plt.subplot(2,3,6)
    # plt.pcolormesh(glati,galti,vgloni[:,64,:])
    # plt.xlabel("glat")
    # plt.ylabel("glon")
    # plt.colorbar()
    # plt.title("$v_{zon}$")
