#%%
# IMPORT PACKAGES
#==================================
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import landlab
from landlab import RasterModelGrid, imshow_grid
from landlab.components import (OverlandFlowTransporter, 
                               FlowAccumulator, 
                               TruckPassErosion, 
                               DepressionFinderAndRouter)
np.set_printoptions(threshold=np.inf)
from utilities.erodible_grid import Erodible_Grid

#%% 
# DEFINING PARAMETERS
#==================================
#==================================
# SPECIFY RUN DURATION
#==================================
run_duration = 120 #[days]; ~4 months

#==================================
# PHYSICAL CONSTANTS
#==================================
rho_w=1000 #[kg/m^3]
rho_s=2650 #[kg/m^3]
g=9.81     #[m/s^2]

#==================================
# DEFINE SITE AND RAIN GAUGE
#==================================
site_name = "MEL12"
rain_gauge_list = ["RG_BISH05_mm","RG_BISH12_mm","RG_DEL0103_mm",\
    "RG_KID1316_mm","RG_KID46_mm","RG_MEL05_mm","RG_MEL14_mm",\
    "RG_NASE0104_mm","RG_NASE05i_mm","RG_NEWS1920_mm"]
rain_gauge = rain_gauge_list[6]

#==================================
# SPECIFY SITE PARAMETERS
#==================================
#Call parameters from .csv file
parameters = pd.read_csv("input/parameters_WY2024.csv")
site_params = parameters.loc[parameters["Site Name"] == site_name].iloc[0]
S = site_params["Road Gradient"]/100
porosity_c = 0.35
porosity_f = 0.35

#==================================
# MODEL RUN STARTING INDEX
#==================================
intensity_index = 64

#==================================
# SET RANDOM SEED
#==================================
seed = 0
np.random.seed(seed)

#==================================
# RAINFALL INTENSITY AND DURATION
#==================================
#Read in actual data and index for specified rain gauge
#and starting index
intensity = pd.read_csv("input/WY2023_RG_daily_intensity.csv")
intensity_run_dur = intensity[rain_gauge].iloc[intensity_index:].values
dt_hours = pd.read_csv("input/WY2023_RG_daily_dt.csv")
dt_hours_run_dur = dt_hours[rain_gauge].iloc[intensity_index:].values 

#Rainfall data for equilibrium run
#(uncomment lines 71-74; comment other rainfall data lines)
# intensity = np.ones(run_duration)*2
# intensity_run_dur = intensity.copy()
# dt_hours = np.array([4 if x!=0 else 0 for x in intensity])
# dt_hours_run_dur = dt_hours.copy()

#Stochastically generated rainfall run
#(uncomment lines 82-88)
    #size = length of array; 
    #zero_prob = probability there is no rainfall; 
    #lam = mean of distribution
# def create_array_poisson(size, zero_prob, lam):  
#     is_zero = np.random.random(size) < zero_prob
#     arr = np.random.poisson(lam, size=size)
#     arr[is_zero] = 0
#     return arr
# intensity_run_dur = create_array_poisson(run_duration, 0.6, 3)
#     #change first value in np.random.poisson to change average storm duration [hrs]
# dt_hours_run_dur = np.array([np.random.poisson(4,1).item() if x!=0 else 0 for x in intensity_run_dur])

#Convert to dt to days
dt = np.array(dt_hours_run_dur)/24

#==================================
# INITIALIZE ROAD LAYER DEPTHS
#==================================
Sa_ini = 0.019 # active depth in m
Ss_ini = 0.23 # surfacing depth in m
Sb_ini = 2    # ballast depth in m

#==================================
# NUMBER OF TRUCK PASSES
#==================================
truck_num_ini = 7

#==================================
# ROUGHNESS VALUES
#==================================
n_c = 0.05
n_f = 0.015

#==================================
# DEFINE D50, D95, AND TAU_C
#==================================
d50_road = 0.0001 # [m] 
tau_c_road = 0.146
d95=0.019

# %% 
# CREATE GRID; ADD FIELDS; ADD RANDOM NOISE
#==================================
#==================================
#Parameters for grid creation
cell_spacing = 0.1475 # cell width or length dimension in meters
cell_area = cell_spacing**2
nrows = 540 # number of rows in the grid
ncols = 65  # number of columns in the grid (NO DITCH)

#We're using half tire width for node spacing
center = int(np.ceil(ncols/2-1))
half_width = 7 
full_tire = False

eg = Erodible_Grid(nrows=nrows, ncols=ncols,\
    spacing=cell_spacing, full_tire=full_tire, \
    long_slope=S, road_peak=center, ditch=False)

mg, z, road_flag, n = eg() 

noise_amplitude=0.005
road = road_flag==1

random=np.random.rand(
    len(z[road])
)
z[road] += noise_amplitude * random #z is the road elevation

#Add depth fields that will update in the components; these are the initial conditions
active_depth = mg.add_ones('active__depth', at='node')
active_depth *= Sa_ini
surf_depth = mg.add_ones('surfacing__depth', at='node')
surf_depth *= Ss_ini
ball_depth = mg.add_ones('ballast__depth', at='node')
ball_depth *= Sb_ini

# %% Pre-define location indices
ruts = [mg.nodes[1:-1, 26-8:40-8], mg.nodes[1:-1, 41-8:55-8]]
half_road = mg.nodes[1:-1, 1:33]
full_road = mg.core_nodes

# %% DEM plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
imshow_grid(mg, z, plot_name='Synthetic road', var_name='Elevation', var_units='m',\
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=5)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
plt.tight_layout()
plt.show()

# %% Save some pre-run variables for comparison later
X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

xsec_pre = mg.at_node['topographic__elevation'][mg.nodes[100,:].flatten()].copy()
#%% 
# PREP OUTPUT ARRAYS/LISTS
#==================================
mask = road_flag
intensity_arr=[]
dt_arr = []
dz_arr=[]
dz_arr_cum = []

#Depths for each layer
sa_arr = np.zeros(run_duration)
ss_arr = np.zeros(run_duration)
sb_arr = np.zeros(run_duration)

Ma = np.zeros(run_duration)
Maf = np.zeros(run_duration)
Mac = np.zeros(run_duration)

Ms = np.zeros(run_duration)
Msf = np.zeros(run_duration)
Msc = np.zeros(run_duration)

Mb = np.zeros(run_duration)
Mbf = np.zeros(run_duration)
Mbc = np.zeros(run_duration)

#Flux/mass arrays
mass_fillslope_inflow = np.zeros(run_duration)
mass_fillslope_rut_outflow = np.zeros(run_duration)
mass_ditch_inflow = np.zeros(run_duration)
mass_ditch_rut_outflow = np.zeros(run_duration)
total_road_mass = np.zeros(run_duration)
road_mass_change_oft = np.zeros(run_duration)

#
road_shear_frac_arr = np.zeros(run_duration)
road_shear_cum_arr = np.zeros(run_duration)
road_shear_i = np.zeros(mg.number_of_nodes, dtype=bool)
road_shear_cum = np.zeros(mg.number_of_nodes, dtype=bool)

# shear stresses (averages)
avg_shear_stress_ruts = []
avg_shear_stress_road = []

# manning's roughness averages
avg_n_ruts = []
avg_n_road = []

# sediment load in the ruts due to TPE
tpe_load_ruts = []

# shear stress partitioning coefficient averages
fs_avg_ruts = []
fs_avg_road = []

truck_num = 0     

# %% 
# INITIALIZE LANDLAB COMPONENTS
#==================================
df = DepressionFinderAndRouter(mg, reroute_flow = True)
df.map_depressions()

fa = FlowAccumulator(
                    mg, 
                    surface='topographic__elevation',
                    flow_director="FlowDirectorD8", 
                    runoff_rate=intensity_run_dur[0]*2.77778e-7,
                    )

#==================================
# UNCOMMENT FOR EQUILIBRIUM RUN
#==================================
# fa.accumulate_flow()
# q = mg.at_node['surface_water__discharge']
# qo = (mg.at_node['surface_water__discharge']/mg.dx)
# So = mg.at_node["topographic__steepest_slope"]
# M_c = porosity_c*d95*(1-porosity_f)*rho_s*mg.dx*mg.dy

# mg.at_node['active__mass_fines'] = ((((rho_w*g*So**0.7*qo**0.6*n_f**1.5)/tau_c_road)**(10/9)-n_c)/(n_f-n_c))*M_c

# mg.at_node['active__mass_fines'] = [mg.at_node['active__mass_fines'][i] if mg.at_node['active__mass_fines'][i]>=0 else 0.0 for i in range(len(mg.at_node['active__mass_fines']))]
# mg.at_node['active__mass_fines'] = [mg.at_node['active__mass_fines'][i] if mg.at_node['active__mass_fines'][i]<=M_c else M_c for i in range(len(mg.at_node['active__mass_fines']))]

tpe = TruckPassErosion(
                    mg, 
                    center, 
                    half_width,
                    full_tire, 
                    truck_num=truck_num_ini,
                    F_af0 = 0.85, 
                    F_sf0 = 1, 
                    F_bc0 = 0.5, 
                    scat_loss=8e-4, 
                    porosity_c=porosity_c, 
                    porosity_f=porosity_f,
                    rand_seed=seed,
                    )

oft = OverlandFlowTransporter(
                            mg, 
                            porosity_f=porosity_f, 
                            porosity_c=porosity_c, 
                            d50=d50_road, 
                            longitudinal_slope=S, 
                            tau_c=tau_c_road, 
                            n_c=n_c, 
                            n_f=n_f,
                            flow_accumulator=fa,
                            depression_finder=df
                            )

#%% 
# PER STORM INTENSITY DIST
#==================================
p = np.linspace(0.0001, 35, 1000)
def pdf(p, p_mean):
    p_prime = p/p_mean
    pdf = 1/p_mean*np.exp(-p/p_mean)
    return pdf, p_prime

def cdf(p, p_mean):
    p_prime = p/p_mean
    cdf = 1-np.exp(-p_prime)
    return cdf, p_prime

def p_prime_arr(prob):
    p_prime_arr = -np.log(1-prob)
    return p_prime_arr

prob_arr = np.concatenate((np.array([0.25, 0.5]), np.linspace(0.6, 0.99, 8)))
frac_arr = np.abs(np.concatenate((np.array([0.25]), np.diff(1-prob_arr))))

pdf, p_prime = pdf(p, 5)
cdf, p_prime = cdf(p, 5)
p_prime_arr = p_prime_arr(prob_arr)

# Plots are helpful, save for later
# fig, ax = plt.subplots()
# ax.plot(p, pdf)
# ax.annotate(r'$f(p)=\frac{1}{p_{mean}}exp(-\frac{p}{p_{mean}})$', xy=(10,0.1), xycoords='data',
#             xytext=(0.5, 0.5), textcoords='axes fraction',
#             va='top', ha='center',size=12)
# ax.set_xlim(0,35)
# ax.set_ylim(0,0.2)
# ax.set_xlabel('$p$ [mm/hr]')
# ax.set_ylabel('$f(p)$')
# ax.set_title('PDF w.r.t. precipitation')
# plt.show()

# fig, ax = plt.subplots(1,2, figsize=(14,4.5))
# ax[0].plot(p, cdf)
# ax[0].annotate(r'$F(p)=1-exp{(-\frac{p}{p_{mean}})}$', xy=(10,0.5), xycoords='data',
#                xytext=(0.5, 0.5), textcoords='axes fraction',
#                va='top', ha='center',size=12)
# ax[0].set_xlim(0,35)
# ax[0].set_ylim(0,1)
# ax[0].set_xlabel('$p$ [mm/hr]')
# ax[0].set_ylabel('$F(p)$')
# ax[0].set_title('CDF w.r.t. precipitation')

# ax[1].plot(p_prime, cdf)
# ax[1].annotate("$F(p\')=1-exp{(-p\')}$", xy=(2,0.5), xycoords='data',
#                xytext=(0.5, 0.5), textcoords='axes fraction',
#                va='top', ha='center',size=12)
# ax[1].set_xlim(0,7)
# ax[1].set_ylim(0,1)
# ax[1].set_xlabel('$p\' =\\frac{p}{p_{mean}}$')
# ax[1].set_ylabel('$F(p\')$')
# ax[1].set_title('CDF w.r.t. nondimensionalized precipitation')
# plt.show()

# fig, ax = plt.subplots()
# for i in range(len(prob_arr)):
#     plt.plot([0, max(p_prime)],[prob_arr[i],prob_arr[i]], '--', color='gray')
# ax.plot(p_prime, cdf)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[0], xy=(5, prob_arr[0]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[1], xy=(5, prob_arr[1]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[2], xy=(5, prob_arr[2]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[3], xy=(5, prob_arr[3]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[4], xy=(5, prob_arr[4]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[5], xy=(5, prob_arr[5]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[6], xy=(5, prob_arr[6]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[7], xy=(5, prob_arr[7]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[8], xy=(5, prob_arr[8]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[9], xy=(5, prob_arr[9]-0.035), size=8)
# ax.set_xlim(0,7)
# ax.set_ylim(0,1)
# ax.set_xlabel('$p\'$')
# ax.set_ylabel('$F(p\')$')
# ax.set_title('CDF w.r.t. nondimensionalized precipitation')
# plt.show()

#%% 
# MAIN MODEL LOOP
#==================================
#==================================
# INITIAL DEPTHS AND MASSES FOR DIFF
#==================================
z_ini_cum = mg.at_node['topographic__elevation'].copy()
Ma_ini_cum = mg.at_node['active__mass'].copy()
Ms_ini_cum = mg.at_node['surfacing__mass'].copy()
Mb_ini_cum = mg.at_node['ballast__mass'].copy()
active_init = mg.at_node['active__depth'][full_road].copy()
surfacing_init = mg.at_node['surfacing__depth'][full_road].copy()
ballast_init = mg.at_node['ballast__depth'][full_road].copy()

#==================================
# CREATE OUTPUT FOLDER
#==================================
import os
import shutil

dir = 'output/'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

#==================================
# START MODEL LOOP
#==================================
start = time.time()
for i in range(0, run_duration): # daily time step
    z_ini = mg.at_node['topographic__elevation'].copy()
    #==================================
    # UNCOMMENT FOR EQUILIBRIUM RUN
    # AND TAB IN LINES 386-387
    #==================================
    # if i >= 30 and i < run_duration-30:
    tpe.run_one_step()
    truck_num += tpe.truck_num
    
    intensity = intensity_run_dur[i]  # use the i-th day's intensity
    dt_day = dt[i] # use the i-th day's time step

    #==================================
    # NO RAINFALL
    #==================================
    if intensity <= 0:
        print(f"Day {i}: No rainfall")
        rain_m_per_s = intensity * 2.77778e-7 # conversion to m/s
        mg.at_node['water__unit_flux_in'] = np.ones(mg.number_of_nodes) * rain_m_per_s
        fa.accumulate_flow()
        df.map_depressions()

        intensity_arr.append(0)
        dt_arr.append(0)

        dz = z - z_ini
        dz_arr.append(sum(dz[full_road.flatten()]))

        dz_cum = z - z_ini_cum
        dz_arr_cum.append(sum(dz_cum[full_road.flatten()]))

        road_shear_cum_arr[i] = road_shear_cum_arr[i-1]

        #Append shear stress vectors
        avg_shear_stress_ruts.append(0)
        avg_shear_stress_road.append(0)

        #Append manning's roughness vectors
        avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][ruts]))
        avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][full_road]))

        #Append shear stress partitioning coefficient
        fs_avg_ruts.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ruts]))
        fs_avg_road.append(np.nanmean(mg.at_node['shear_stress__partitioning'][full_road]))

        #Append TPE loading
        tpe_load_ruts.append((tpe.sed_added[full_road]).sum())
        
        #For plotting layer depths
        sa_arr[i] = np.sum(mg.at_node['active__depth'][full_road])
        ss_arr[i] = np.sum(mg.at_node['surfacing__depth'][full_road])
        sb_arr[i] = np.sum(mg.at_node['ballast__depth'][full_road])
        Ma[i] = np.sum(mg.at_node['active__mass'])
        Maf[i] = np.sum(mg.at_node['active__mass_fines'])
        Mac[i] = np.sum(mg.at_node['active__mass_coarse'])
        Ms[i] = np.sum(mg.at_node['surfacing__mass'])
        Msf[i] = np.sum(mg.at_node['surfacing__mass_fines'])
        Msc[i] = np.sum(mg.at_node['surfacing__mass_coarse'])
        Mb[i] = np.sum(mg.at_node['ballast__mass'])
        Mbf[i] = np.sum(mg.at_node['ballast__mass_fines'])
        Mbc[i] = np.sum(mg.at_node['ballast__mass_coarse'])

    #==================================
    # RAINFALL
    #==================================
    else:
        intensity_arr.append(intensity)
        print(f"Day {i}: Average Intensity = {intensity:.2f} mm/hr")
        dt_arr.append(dt_day)
        print(f"Day {i}: Length of Storm = {dt_day:.2f} day")
        
        road_shear_i[:] = 0

        #==================================
        # DISCRETIZE RAINFALL
        # ==================================
        for j, storm_frac in enumerate(frac_arr):
            intensity_dist = p_prime_arr[j]*intensity
            dt_frac = storm_frac*dt_day
            #=================================Calculate overland flow transport=================================
            rain_m_per_s = intensity_dist * 2.77778e-7 # conversion to m/s
            mg.at_node['water__unit_flux_in'] = np.ones(mg.number_of_nodes) * rain_m_per_s
            fa.accumulate_flow()
            df.map_depressions()
            oft.run_one_step(dt_frac)

            mass_ditch_rut_outflow_i = (mg.at_node["sediment__mass_influx"][mg.nodes[0,0:center]]).sum()*dt_frac*86400
            mass_ditch_inflow_i = (mg.at_node["sediment__mass_influx"][mg.nodes[1:,0]]).sum()*dt_frac*86400
            mass_fillslope_inflow_i = (mg.at_node["sediment__mass_influx"][mg.nodes[1:,(ncols-1)]]).sum()*dt_frac*86400
            mass_fillslope_rut_outflow_i = (mg.at_node["sediment__mass_influx"][mg.nodes[0,(center+1):ncols]]).sum()*dt_frac*86400

            mass_ditch_rut_outflow[i] += mass_ditch_rut_outflow_i
            mass_ditch_inflow[i] += mass_ditch_inflow_i
            mass_fillslope_inflow[i] += mass_fillslope_inflow_i
            mass_fillslope_rut_outflow[i] += mass_fillslope_rut_outflow_i

            mg.at_node['shear_stress'] = oft.shear_stress
            road_shear = oft.shear_stress[full_road].flatten()

            road_shear_i = np.array([True if shear_stress>=tau_c_road or road_shear_i[x]==True \
                 else False for x, shear_stress in enumerate(road_shear)])
            road_shear_cum = np.array([True if shear_stress>=tau_c_road or road_shear_cum[x]==True\
                 else False for x, shear_stress in enumerate(road_shear)])

        road_shear_frac_arr[i] = road_shear_i[road_shear_i==True].sum()/len(road_shear_i)
        road_shear_cum_arr[i] = road_shear_cum[road_shear_cum==True].sum()/len(road_shear_cum)

        mg.at_node['shear_stress'] = oft.shear_stress
        avg_shear_stress_ruts.append(np.nanmean(mg.at_node['shear_stress'][ruts]))
        avg_shear_stress_road.append(np.nanmean(mg.at_node['shear_stress'][full_road]))

        road_mass_change_oft[i] = mass_ditch_inflow[i] + mass_ditch_rut_outflow[i]
        total_road_mass[i] = mass_ditch_inflow[i] + mass_ditch_rut_outflow[i] +\
             mass_fillslope_inflow[i] + mass_fillslope_rut_outflow[i]
        
        dz = z-z_ini # calculate elevation change at each daily time step
        dz_arr.append(sum(dz[full_road.flatten()]))

        dz_cum = z-z_ini_cum # calculate cumulative elevation change
        dz_arr_cum.append(sum(dz_cum[full_road.flatten()])) 

        #==================================
        # FIGURES FOR GIF CREATION - 1ST GIF
        #==================================
        fig = plt.figure(figsize=(9,6))
        mg.add_field('fine_frac', mg.at_node['active__depth_fines']/mg.at_node['active__depth_coarse'], at='node', units='m-', clobber=True)
        plt.subplot(131)
        im = imshow_grid(mg,'fine_frac', var_name='Faf', var_units='-', 
                        plot_name='$S_{af}:S_{ac}$ in\nactive layer, t = %i days' %i,
                        grid_units=('m','m'), cmap='Dark2', vmin=0, vmax=1.6, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')

        plt.subplot(132)
        im = imshow_grid(mg,'surface_water__discharge', var_name='Discharge', 
                        plot_name='Discharge, t = %i days' %i,
                        var_units='$m/s^3$', grid_units=('m','m'), 
                        cmap='Blues', vmin=0, vmax=0.00001, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')

        mg.add_field('dz_cum', dz_cum, at='node', units='m', clobber=True)
        plt.subplot(133)
        im = imshow_grid(mg,'dz_cum', var_name='Cumulative dz', var_units='m', 
                        plot_name='Elevation change, t = %i days' %i,
                        grid_units=('m','m'), cmap='RdBu', vmin=-0.0009, vmax=0.0009, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')

        plt.tight_layout()
        plt.savefig('output/plots_%i_days.png' %i, \
            bbox_inches='tight', dpi=300) #full road plots for Saf:Sac, dz_cum, and discharge
        plt.close()

        #==================================
        # FIGURES FOR GIF CREATION - 2ND GIF
        #==================================
        fig = plt.figure(figsize=(4,7))
        plt.subplot(321,)
        im1 = imshow_grid(mg,'dz_cum', cmap='RdBu', plot_name="Upper Left",
            vmin=-0.0009, vmax=0.0009, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(73.75,79.65)

        plt.subplot(323,)
        im2 = imshow_grid(mg,'dz_cum', cmap='RdBu', plot_name="Middle Left",
            vmin=-0.0009, vmax=0.0009, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(33.925,39.825)

        plt.subplot(325)
        im3 = imshow_grid(mg,'dz_cum', cmap='RdBu', plot_name="Lower Left",
            vmin=-0.0009, vmax=0.0009, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(0,5.9)

        plt.subplot(322)
        im4 = imshow_grid(mg,'dz_cum', cmap='RdBu', plot_name="Upper Right",
            vmin=-0.0009, vmax=0.0009, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(73.75,79.65)

        plt.subplot(324)
        im5 = imshow_grid(mg,'dz_cum', cmap='RdBu', plot_name="Middle Right",
            vmin=-0.0009, vmax=0.0009, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(33.925,39.825)

        plt.subplot(326)
        im6 = imshow_grid(mg,'dz_cum', cmap='RdBu', plot_name="Lower Right",
            vmin=-0.0009, vmax=0.0009, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(0,5.9)

        fig.suptitle('Elevation change,\nt = %i days' %i)
        plt.tight_layout()
        cax = plt.axes((1, 0.075, 0.075, 0.8))
        plt.colorbar(cax=cax, extend="both", label="Cumulative dz ($m$)")
        plt.savefig('output/subplots_%i_days.png' %i, bbox_inches='tight') #zoomed in subplots for dz_cum
        plt.close()

        #==================================
        # FIGURES FOR GIF CREATION - 3RD GIF
        #==================================
        fig = plt.figure(figsize=(4,7))
        plt.subplot(321,)
        im1 = imshow_grid(mg,'fine_frac', cmap='Dark2', plot_name="Upper Left",
            vmin=0, vmax=1.6, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(73.75,79.65)

        plt.subplot(323,)
        im2 = imshow_grid(mg,'fine_frac', cmap='Dark2', plot_name="Middle Left",
            vmin=0, vmax=1.6, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(33.925,39.825)

        plt.subplot(325)
        im3 = imshow_grid(mg,'fine_frac', cmap='Dark2', plot_name="Lower Left",
            vmin=0, vmax=1.6, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(0,5.9)

        plt.subplot(322)
        im4 = imshow_grid(mg,'fine_frac', cmap='Dark2', plot_name="Upper Right",
            vmin=0, vmax=1.6, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(73.75,79.65)

        plt.subplot(324)
        im5 = imshow_grid(mg,'fine_frac', cmap='Dark2', plot_name="Middle Right",
            vmin=0, vmax=1.6, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(33.925,39.825)

        plt.subplot(326)
        im6 = imshow_grid(mg,'fine_frac', cmap='Dark2', plot_name="Lower Right",
            vmin=0, vmax=1.6, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(0,5.9)

        fig.suptitle('$S_{af}:S_{ac}$,\nt = %i days' %i)
        plt.tight_layout()
        cax = plt.axes((1, 0.075, 0.075, 0.8))
        plt.colorbar(cax=cax, extend="max", label="Ratio of fines to coarse in active layer ($-$)")
        plt.savefig('output/fubplots_%i_days.png' %i, bbox_inches='tight') #zoomed in subplots for Saf:Sac
        plt.close()

        #==================================
        # FIGURES FOR GIF CREATION - 4TH GIF
        #==================================
        fig = plt.figure(figsize=(4,7))
        plt.subplot(321,)
        im1 = imshow_grid(mg,'surface_water__discharge', cmap='Blues', plot_name="Upper Left",
            vmin=0, vmax=1e-5, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(73.75,79.65)

        plt.subplot(323,)
        im2 = imshow_grid(mg,'surface_water__discharge', cmap='Blues', plot_name="Middle Left",
            vmin=0, vmax=1e-5, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(33.925,39.825)

        plt.subplot(325)
        im3 = imshow_grid(mg,'surface_water__discharge', cmap='Blues', plot_name="Lower Left",
            vmin=0, vmax=1e-5, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(0,4.72)
        plt.ylim(0,5.9)

        plt.subplot(322)
        im4 = imshow_grid(mg,'surface_water__discharge', cmap='Blues', plot_name="Upper Right",
            vmin=0, vmax=1e-5, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(73.75,79.65)

        plt.subplot(324)
        im5 = imshow_grid(mg,'surface_water__discharge', cmap='Blues', plot_name="Middle Right",
            vmin=0, vmax=1e-5, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(33.925,39.825)

        plt.subplot(326)
        im6 = imshow_grid(mg,'surface_water__discharge', cmap='Blues', plot_name="Lower Right",
            vmin=0, vmax=1e-5, grid_units=('m','m'), allow_colorbar=False)
        plt.xlim(4.72,9.44)
        plt.ylim(0,5.9)

        fig.suptitle('Surface Water Discharge,\nt = %i days' %i)
        plt.tight_layout()
        cax = plt.axes((1, 0.075, 0.075, 0.8))
        plt.colorbar(cax=cax, extend="max", label="Discharge ($m^3/s$)")
        plt.savefig('output/hubplots_%i_days.png' %i, bbox_inches='tight') #zoomed in subplots for discharge
        plt.close()

        #Append manning's n vectors
        avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][ruts]))
        avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][full_road]))

        #Append shear stress partitioning coefficient vectors        
        fs_avg_ruts.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ruts]))
        fs_avg_road.append(np.nanmean(mg.at_node['shear_stress__partitioning'][full_road]))
        
        #Append TPE loading
        tpe_load_ruts.append((tpe.sed_added[full_road]).sum())
    
        #For plotting layer depths
        sa_arr[i] = np.sum(mg.at_node['active__depth'][full_road])
        ss_arr[i] = np.sum(mg.at_node['surfacing__depth'][full_road])
        sb_arr[i] = np.sum(mg.at_node['ballast__depth'][full_road])

        Ma[i] = np.sum(mg.at_node['active__mass'])
        Maf[i] = np.sum(mg.at_node['active__mass_fines'])
        Mac[i] = np.sum(mg.at_node['active__mass_coarse'])
        Ms[i] = np.sum(mg.at_node['surfacing__mass'])
        Msf[i] = np.sum(mg.at_node['surfacing__mass_fines'])
        Msc[i] = np.sum(mg.at_node['surfacing__mass_coarse'])
        Mb[i] = np.sum(mg.at_node['ballast__mass'])
        Mbf[i] = np.sum(mg.at_node['ballast__mass_fines'])
        Mbc[i] = np.sum(mg.at_node['ballast__mass_coarse'])

wall_time = time.time() - start
print("Wall time for run:", wall_time, "s")

#%% 
# FINAL CALCS FOR PLOTS
#==================================
road_elev_change_dz = np.divide(dz_arr,2)/(540*64/2)
cum_road_elev_change_dz = np.divide(dz_arr_cum,2)/(540*64/2)

cum_road_mass_change_oft = road_mass_change_oft.cumsum()
#%% 
# GIF CREATION
#==================================
from PIL import Image
import glob

path = "output/"
images_p=[]
images_s=[]
images_h=[]
images_f=[]
for file in glob.glob(path+'p*.png'):
    images_p.append(file)
for file in glob.glob(path+'s*.png'):
    images_s.append(file)
for file in glob.glob(path+'h*.png'):
    images_h.append(file)
for file in glob.glob(path+'f*.png'):
    images_f.append(file)

def number_p(filename):
    return int(filename[13:-9])

def number_s(filename):
    return int(filename[16:-9])

def number_h(filename):
    return int(filename[16:-9])

def number_f(filename):
    return int(filename[16:-9])

images_p = sorted(images_p, key=number_p)
images_s = sorted(images_s, key=number_s)
images_h = sorted(images_h, key=number_h)
images_f = sorted(images_f, key=number_f)

# Create a list of image objects
image_list_p = [Image.open(file) for file in images_p]
image_list_s = [Image.open(file) for file in images_s]
image_list_h = [Image.open(file) for file in images_h]
image_list_f = [Image.open(file) for file in images_f]

# Save the first image as a GIF file
image_list_p[0].save(
            'plots.gif',
            save_all=True,
            append_images=image_list_p[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

image_list_s[0].save(
            'subplots.gif',
            save_all=True,
            append_images=image_list_s[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

image_list_h[0].save(
            'subplots_h.gif',
            save_all=True,
            append_images=image_list_h[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

image_list_f[0].save(
            'subplots_f.gif',
            save_all=True,
            append_images=image_list_f[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)


#%%
# CROSS SECTION PLOT
#==================================
xsec_active = mg.at_node['topographic__elevation'][mg.nodes[100,:].flatten()]
xsec_surf =  mg.at_node['surfacing__elevation'][mg.nodes[100,:].flatten()] 
xsec_ball = mg.at_node['ballast__elevation'][mg.nodes[100,:].flatten()]

plt.figure(figsize=(8,3), layout='tight')
plt.plot(X[36], xsec_pre, color='gray', linestyle='-.', label='Before')
plt.plot(X[36], xsec_active, color ='black', linestyle='-', label = 'After - Active elevation')
plt.plot(X[36], xsec_surf, color ='magenta', linestyle='-', label = 'After - Surfacing elevation')
plt.plot(X[36], xsec_ball, color ='cyan', linestyle='-', label = 'After - Ballast elevation ')
plt.xlim(0,9.5)
plt.xlabel('Road width (m)')
plt.ylabel('Elevation (m)')
plt.legend()
plt.show()

#%% 
# RAINFALL PLOTS
#==================================
fig, ax = plt.subplots(2,1)
ax[0].bar(range(0,run_duration), np.multiply(intensity_arr,np.multiply(dt_arr,24)))
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Rainfall [mm]')
ax[0].set_xlim(0,run_duration)

ax[1].plot(range(0,run_duration), intensity_arr)
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Rainfall intensity [mm/hr]')
ax[1].set_xlim(0,run_duration)
plt.suptitle(r'%s' %site_name)
plt.tight_layout()
plt.show()

#%% 
# IMPORTANT MODEL OUTPUT
#==================================
print(
    "Total rainfall,",
    run_duration, "days:", 
    np.round(sum(np.multiply(intensity_arr,np.multiply(dt_arr,24)),2)), 'mm'
    )
print(
    'Fine sediment pumped:', 
    np.round(mg.at_node['sediment__pumped'].sum(),2), 'kg'
    )
print(
    'Fine sediment made available due to scattering:', 
    np.round(mg.at_node['sediment__scattered'].sum(),2), 'kg'
    )
print(
    'Total sediment added to the active layer:', 
    np.round(mg.at_node['sediment__added'].sum(),2), 'kg'
    )
print(
    'Cumulative sediment load from cutslope side of road:', 
    np.round((mass_ditch_inflow + mass_ditch_rut_outflow).sum(),2), 'kg' 
    )
print(
    'Cumulative sediment load from fillslope side of road:', 
    np.round((mass_fillslope_inflow + mass_fillslope_rut_outflow).sum(),2), 'kg' 
    )
print(
    'Cumulative sediment load from both ruts:', 
    np.round((mass_fillslope_rut_outflow + mass_ditch_rut_outflow).sum(),2), 'kg' 
    )
print(
    'Cumulative sediment load from both sides:', 
    np.round((mass_fillslope_inflow + mass_ditch_inflow).sum(),2), 'kg' 
    )
print(
    'Cumulative sediment load from full road:', 
    np.round((total_road_mass).sum(),2), 'kg' 
    )

#%% 
# DELTA MASS AND ELEV BTWN TIME STEPS
#==================================
fig, ax = plt.subplots(1,2, figsize=(9,4))

# plot total mass change between time steps on the road
ax[0].plot(range(0,run_duration), -road_mass_change_oft) # added porosity consideration
ax[0].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[0].set_xlabel('Day')
ax[0].set_ylabel(r'$\Delta$ mass between time steps [$kg$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title('(a) Half Road (OFT)')

# plot total elevation change between time steps between time steps on the road
ax[1].plot(range(0,run_duration), road_elev_change_dz)
ax[1].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[1].set_xlabel('Day')
ax[1].set_ylabel(r'$\Delta$ elevation between time steps [$m$]')
ax[1].set_xlim(0,run_duration)
ax[1].set_title('(b) Half Road (dz)')
plt.tight_layout()
plt.show()

#%% 
# CUMULATIVE MASS AND ELEV CHANGE
#==================================
fig, ax = plt.subplots(1,2, figsize=(9,4))

ax[0].plot(range(0,run_duration), -cum_road_mass_change_oft)
ax[0].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Cumulative mass change - \nhalf road [$kg$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title('(a) Half Road (OFT)')

ax[1].plot(range(0,run_duration), cum_road_elev_change_dz)
ax[1].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Cumulative elevation change - \nhalf road [$m$]')
ax[1].set_xlim(0,run_duration)
ax[1].set_title('(b) Half Road (dz)')

plt.tight_layout()
plt.show()

#%% 
# TPE loading
#==================================
# plot sediment load to the active layer in the ruts from truck passes
plt.plot(range(0,run_duration), tpe_load_ruts)
plt.xlabel('Day')
plt.ylabel('Cumulative sediment load to the active layer in the ruts \nfrom TPE [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%% 
# AVERAGE LAYER DEPTHS
#================================== 
fig, ax = plt.subplots(3,1, figsize=(4,7))

ax[0].set_title(r'%s ($n_{f_{road}} = %0.3f$)' %(site_name, n_f))
ax[0].plot(range(0,run_duration), (-active_init.sum()+sa_arr)/(nrows*ncols))
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Active Depth change\n(averaged over full road) [$m$]')
ax[0].set_xlim(0,run_duration)

ax[1].plot(range(0,run_duration), (-surfacing_init.sum()+ss_arr)/(nrows*ncols))
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Surfacing Depth change\n(averaged over full road) [$m$]')
ax[1].set_xlim(0,run_duration)

ax[2].plot(range(0,run_duration), (-ballast_init.sum()+sb_arr)/(nrows*ncols))
ax[2].set_xlabel('Day')
ax[2].set_ylabel('Ballast Depth change\n(averaged over full road) [$m$]')
ax[2].set_xlim(0,run_duration)
plt.tight_layout()
plt.show()

#%% 
# AVERAGE LAYER MASSES
#==================================  
fig, ax = plt.subplots(3,1, figsize=(4,7))
ax[0].plot(range(0,run_duration), (Ma)/(nrows*ncols))
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Active Mass\naverage [$kg$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title(r'%s ($n_{f_{road}} = %0.3f$)' %(site_name, n_f))

ax[1].plot(range(0,run_duration), (Maf)/(nrows*ncols))
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Active Mass - fines\naverage [$kg$]')
ax[1].set_xlim(0,run_duration)

ax[2].plot(range(0,run_duration), (Mac)/(nrows*ncols))
ax[2].set_xlabel('Day')
ax[2].set_ylabel('Active Mass - coarse\naverage [$kg$]')
ax[2].set_xlim(0,run_duration)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3,1, figsize=(4,7))
ax[0].plot(range(0,run_duration), (Ms)/(nrows*ncols))
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Surfacing Mass\naverage [$kg$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title(r'%s ($n_{f_{road}} = %0.3f$)' %(site_name, n_f))

ax[1].plot(range(0,run_duration), (Msf)/(nrows*ncols))
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Surfacing Mass - fines\naverage [$kg$]')
ax[1].set_xlim(0,run_duration)

ax[2].plot(range(0,run_duration), (Msc)/(nrows*ncols))
ax[2].set_xlabel('Day')
ax[2].set_ylabel('Surfacing Mass - coarse\naverage [$kg$]')
ax[2].set_xlim(0,run_duration)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3,1, figsize=(4,7))
ax[0].plot(range(0,run_duration), (Mb)/(nrows*ncols))
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Ballast Mass\naverage [$kg$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title(r'%s ($n_{f_{road}} = %0.3f$)' %(site_name, n_f))

ax[1].plot(range(0,run_duration), (Mbf)/(nrows*ncols))
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Ballast Mass - fines\naverage [$kg$]')
ax[1].set_xlim(0,run_duration)

ax[2].plot(range(0,run_duration), (Mbc)/(nrows*ncols))
ax[2].set_xlabel('Day')
ax[2].set_ylabel('Ballast Mass - coarse\naverage [$kg$]')
ax[2].set_xlim(0,run_duration)
plt.tight_layout()
plt.show()

# %%
# plt.bar(range(0,run_duration), np.multiply(road_shear_frac_arr, 100))
# plt.xlabel('Day')
# plt.ylabel('Percentage of Road $\\geq$ Critical \nShear Stress')
# plt.xlim(0,run_duration)
# plt.title(r'%s' %site_name)
# plt.show()

# plt.plot(range(0,run_duration), np.multiply(road_shear_cum_arr, 100))
# plt.xlabel('Day')
# plt.ylabel('Cumulative Percentage of Road $\\geq$ Critical \nShear Stress')
# plt.xlim(0,run_duration)
# plt.title(r'%s' %site_name)
# plt.show()

# # plot shear stress partitioning coefficient over time
# plt.plot(range(0,run_duration), fs_avg_road, label='Full Road')
# plt.plot(range(0,run_duration), fs_avg_ruts, label='Ruts')
# plt.xlabel('Day')
# plt.ylabel('Average Shear Stress Partitioning \nCoefficient: $f_s$')
# plt.xlim(0,run_duration)
# plt.legend()
plt.show()