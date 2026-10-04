#%%
#%load_ext autoreload
#%autoreload 2

# %%
import os
import shutil
import sys
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import landlab
from landlab import RasterModelGrid, imshow_grid
from landlab.components import (
    OverlandFlowTransporter,
    FlowAccumulator,
    TruckPassErosion,
    DepressionFinderAndRouter,
)

np.set_printoptions(threshold=np.inf)

# Make `utilities` importable regardless of where this file lives or what
# directory the notebook/kernel's working directory happens to be. Adjust
# the number of ".." if you move this file to a different folder depth --
# each ".." goes up one directory level from THIS file's own location.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
_historical_csv_confirmed = set()
from utilities.erodible_grid import Erodible_Grid

# %%
# ==========================================================================
# CONSTANTS / CONFIG THAT DO NOT DEPEND ON THE RANDOM SEED
# (computed once, reused by every realization)
# ==========================================================================

# ------ default run duration (can be overridden per call) ------
run_duration_default = 30  # ~4 months=120 days

# ------ physical constants ------
rho_w = 1000
rho_s = 2650
g = 9.81

# ------ site and rain gauge ------
site_name = "MEL12"
rain_gauge_list = [
    "RG_BISH05_mm", "RG_BISH12_mm", "RG_DEL0103_mm",
    "RG_KID1316_mm", "RG_KID46_mm", "RG_MEL05_mm", "RG_MEL14_mm",
    "RG_NASE0104_mm", "RG_NASE05i_mm", "RG_NEWS1920_mm",
]
rain_gauge = rain_gauge_list[6]

# ------ site parameters (read once, not per run) ------
parameters = pd.read_csv(os.path.join(PROJECT_ROOT, "input/parameters_WY2024.csv"))
site_params = parameters.loc[parameters["Site Name"] == site_name].iloc[0]
S = 0.05
porosity_c = 0.35
porosity_f = 0.35

# ------ model run starting index (only relevant if using historical data) ------
intensity_index = 64

# ------ road layer initial depths ------
Sa_ini = 0.019  # active depth in m
Ss_ini = 0.23   # surfacing depth in m
Sb_ini = 2      # ballast depth in m

# ------ truck passes ------
truck_num_ini = 2

# ------ roughness values ------
n_c = 0.05
n_f = 0.015

# ------ D50, D95, tau_c ------
d50_road = 0.0001  # [m]
tau_c_road = 0.146
d95 = 0.019

# ------ grid geometry ------
cell_spacing = 0.1475  # cell width/length, m
cell_area = cell_spacing ** 2
nrows = 540  # NO DITCH
ncols = 64
center = 32
half_width = 7
full_tire = False

#%%
# ------ storm sub-fraction distribution (deterministic, same every run) ------
# Splits each day's total rainfall depth into several within-day intensity
# "chunks" following an exponential distribution, rather than applying the
# average intensity uniformly across the whole day.
p = np.linspace(0.0001, 35, 1000)


def _pdf(p, p_mean):
    p_prime = p / p_mean
    pdf = 1 / p_mean * np.exp(-p / p_mean)
    return pdf, p_prime


def _cdf(p, p_mean):
    p_prime = p / p_mean
    cdf = 1 - np.exp(-p_prime)
    return cdf, p_prime


def _p_prime_arr(prob):
    return -np.log(1 - prob)


prob_arr = np.concatenate((np.array([0.25, 0.5]), np.linspace(0.6, 0.99, 8)))
frac_arr = np.abs(np.concatenate((np.array([0.25]), np.diff(1 - prob_arr))))
p_prime_arr = _p_prime_arr(prob_arr)


# ==========================================================================
# STOCHASTIC RAINFALL GENERATOR (Poisson model)
# ==========================================================================
def create_array_poisson(size, zero_prob, lam, rng):
    """
    size      = length of array
    zero_prob = probability there is no rainfall on a given day
    lam       = mean of the Poisson distribution used for intensity
    rng       = a numpy.random.Generator (e.g. from np.random.default_rng(seed))
                -- passed explicitly so rainfall randomness can be
                controlled independently of other random draws (like the
                road-surface noise), instead of sharing the global
                np.random state with everything else.
    """
    is_zero = rng.random(size) < zero_prob
    arr = rng.poisson(lam, size=size)
    arr[is_zero] = 0
    return arr

#%%
# ==========================================================================
# SINGLE-REALIZATION SIMULATION
# ==========================================================================
def run_realization(seed, run_duration=run_duration_default,
                     save_plots=False, verbose=True, return_grid=False,
                     rainfall_source="historical", rain_seed=None, grid_seed=None,
                     porosity_c=porosity_c, porosity_f=porosity_f,
                     truck_num_ini=truck_num_ini, S=S, u_ps=2.18e-4, u_pb=2.3e-6,
                     F_af0=0.50, F_sf0=1, F_bc0=0.5, scat_loss=8e-4, compression=7e-4,
                     tau_c_road=tau_c_road, n_c=n_c, n_f=n_f, d50_road=d50_road,
                     rain_zero_prob=0.6, rain_lam=7, rain_duration_mean=8,
                     historical_csv_path="/Users/goddamnit/github/roads_model/input/Case Low.csv"):
    """
    Run one full realization of the road erosion model.

    Parameters
    ----------
    seed : int
        Default random seed for this realization, used for BOTH rainfall
        and grid noise unless rain_seed/grid_seed are given explicitly.
    run_duration : int
        Number of days to simulate.
    save_plots : bool
        If True, saves the daily diagnostic maps/images and builds GIFs, and
        also creates the initial DEM plot. Leave False for ensemble runs.
    verbose : bool
        If True, prints per-day rainfall/status messages.
    return_grid : bool
        If True, includes the Landlab grid object and a few extra arrays in
        the returned dict. Automatically turned on if save_plots=True.
    rainfall_source : "poisson" | "historical" | "equilibrium"
        Which rainfall generator to use. Only "poisson" varies with the seed.
    rain_seed : int, optional
        Seed for the rainfall random draws specifically. Defaults to `seed`.
    grid_seed : int, optional
        Seed for the road-surface noise specifically. Defaults to `seed`.
    porosity_c, porosity_f : float
        Coarse/fine sediment porosity, passed to both TruckPassErosion and
        OverlandFlowTransporter. Defaults match the module-level constants.
    truck_num_ini : int
        Number of truck passes per day, passed to TruckPassErosion.
    F_af0, F_sf0, F_bc0, scat_loss : float
        TruckPassErosion parameters (initial active-layer fines fraction,
        surfacing fines fraction, ballast coarse fraction, and scatter-loss
        rate). Defaults match what was previously hardcoded in the
        TruckPassErosion(...) call.
    tau_c_road, n_c, n_f, d50_road : float
        OverlandFlowTransporter parameters (critical shear stress, coarse
        and fine Manning's roughness, and median grain size). Defaults
        match the module-level constants.
    compression : float
        TruckPassErosion parameter: amount by which porosity decreases
        per truck pass, in layers where phi_c >= phi_limit (0.31).
    S : float
        Road/longitudinal slope gradient, used by both Erodible_Grid (to
        build the grid) and OverlandFlowTransporter. Defaults to the
        module-level value read from parameters_WY2024.csv for site_name.
    u_ps : float
        TruckPassErosion parameter -- pumping rate from the surfacing
        layer to the active layer, per truck pass [kg/truck].
    u_pb : float
        TruckPassErosion parameter -- pumping rate from the ballast layer
        to the surfacing layer, per truck pass [kg/truck].
    rain_zero_prob : float
        Probability a given day has no rainfall at all (rainfall_source=
        "poisson" only). Lower value = more days with rain.
    rain_lam : float
        Mean of the Poisson distribution used for rainfall intensity on
        rainy days [mm/hr] (rainfall_source="poisson" only).
    rain_duration_mean : float
        Mean of the Poisson distribution used for storm duration on rainy
        days [hr] (rainfall_source="poisson" only).
    historical_csv_path : str
        Path to the CSV file used when rainfall_source="historical". Every
        call prints the resolved path, the file's last-modified timestamp,
        and its column names/first few values -- use this to directly
        confirm which file and version is actually being read.

    Returns
    -------
    dict of results/summary arrays for this realization. Also includes the
    parameter values actually used, for traceability across a sweep.
    """
    return_grid = return_grid or save_plots
    if rain_seed is None:
        rain_seed = seed
    if grid_seed is None:
        grid_seed = seed

    # independent generator for rainfall -- does NOT touch the global
    # np.random state, so it can vary (or stay fixed) completely separately
    # from the grid noise below
    rng_rain = np.random.default_rng(rain_seed)

    # ==================================
    # RAINFALL INTENSITY AND DURATION
    # ==================================
    #IF USING INTENSITY INDEX FROM KNOWN DATA UNMARK THIS
    #if rainfall_source == "historical":
        #intensity = pd.read_csv("/Users/goddamnit/github/roads_model/input/WY2023_RG_daily_intensity.csv")
        #intensity_run_dur = intensity[rain_gauge].iloc[intensity_index:].values
        #dt_hours = pd.read_csv("input/WY2023_RG_daily_dt.csv")
        #dt_hours_run_dur = dt_hours[rain_gauge].iloc[intensity_index:].values

    if rainfall_source == "historical":
        data = pd.read_csv(historical_csv_path)

        def find_col(df, keyword):
            matches = [c for c in df.columns if keyword.lower() in c.lower()]
            if not matches:
                raise KeyError(f"No column containing '{keyword}' found. Actual columns: {df.columns.tolist()}")
            return matches[0]

        intensity_col = find_col(data, "intensity")
        dt_col_name = find_col(data, "dt")
        day_col = find_col(data, "day")

        data["_date"] = pd.to_datetime(data[day_col]).dt.date
        step_hours = data[dt_col_name].iloc[0] / 60  # this file's row spacing (60 -> 1 hour/row)

        daily = data.groupby("_date").apply(
            lambda g: pd.Series({
                "depth_mm": (g[intensity_col] * step_hours).sum(),
                "wet_hours": (g[intensity_col] > 0).sum() * step_hours,
            })
        ).reset_index()

        daily["avg_intensity"] = (daily["depth_mm"] / daily["wet_hours"].replace(0, np.nan)).fillna(0)

        intensity_run_dur = daily["avg_intensity"].values
        dt_hours_run_dur = daily["wet_hours"].values

        if historical_csv_path not in _historical_csv_confirmed:
            mtime = os.path.getmtime(historical_csv_path)
            mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[historical rainfall] reading: {historical_csv_path}")
            print(f"[historical rainfall] file last modified: {mtime_str}")
            print(f"[historical rainfall] aggregated {len(data)} rows -> {len(intensity_run_dur)} days")
            print(f"[historical rainfall] first 3 daily avg intensities: {intensity_run_dur[:3]}")
            _historical_csv_confirmed.add(historical_csv_path)

    elif rainfall_source == "equilibrium":
        # Constant-intensity equilibrium run. Also identical every
        # realization.
        intensity_run_dur = np.ones(run_duration) * 10
        dt_hours_run_dur = np.array([4 if x != 0 else 0 for x in intensity_run_dur])

    elif rainfall_source == "poisson":
        # stochastically generated rainfall run
        #   change the 3rd argument (lam) to change average intensity [mm/hr]
        #   change the first argument to rng_rain.poisson below to change
        #   average storm duration [hrs]
        intensity_run_dur = create_array_poisson(run_duration, rain_zero_prob, rain_lam, rng_rain)
        dt_hours_run_dur = np.array(
            [rng_rain.poisson(rain_duration_mean, 1).item() if x != 0 else 0 for x in intensity_run_dur]
        )

    else:
        raise ValueError(
            f"Unrecognized rainfall_source: {rainfall_source!r}. "
            f"Must be exactly 'historical', 'equilibrium', or 'poisson' "
            f"(check for typos, extra whitespace, or wrong case)."
        )

    dt = np.array(dt_hours_run_dur) / 24  # convert dt to days

    # ==================================
    # CREATE GRID; ADD FIELDS; ADD RANDOM NOISE
    # ==================================
    # NOTE: Erodible_Grid (and the road-noise line below) rely on the
    # legacy global np.random state, not an injected Generator -- seeding
    # it here with grid_seed keeps this fully independent of rain_seed
    # above (which uses its own separate Generator, rng_rain).
    np.random.seed(grid_seed)

    eg = Erodible_Grid(
        nrows=nrows, ncols=ncols, spacing=cell_spacing,
        full_tire=full_tire, long_slope=S, road_peak=center, ditch=False,
    )
    mg, z, road_flag, n = eg()

    noise_amplitude = 0.005
    road = road_flag == 1
    random_noise = np.random.rand(len(z[road]))
    z[road] += noise_amplitude * random_noise  # z is the road elevation

    active_depth = mg.add_ones('active__depth', at='node')
    active_depth *= Sa_ini
    surf_depth = mg.add_ones('surfacing__depth', at='node')
    surf_depth *= Ss_ini
    ball_depth = mg.add_ones('ballast__depth', at='node')
    ball_depth *= Sb_ini

    # ------ location indices ------
    ruts = [mg.nodes[1:-1, 26 - 8:40 - 8], mg.nodes[1:-1, 41 - 8:55 - 8]]
    half_road = mg.nodes[1:-1, 1:33]
    full_road = mg.core_nodes

    if save_plots:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        imshow_grid(mg, z, plot_name='Synthetic road', var_name='Elevation', var_units='m',
                    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=5)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        plt.show()

    # ------ pre-run variables, for later comparison ------
    X = mg.node_x.reshape(mg.shape)
    Y = mg.node_y.reshape(mg.shape)
    Z = z.reshape(mg.shape)
    xsec_pre = mg.at_node['topographic__elevation'][mg.nodes[100, :].flatten()].copy()

    # ------ prep arrays/lists ------
    mask = road_flag
    intensity_arr = []
    dt_arr = []
    dz_arr = []
    dz_arr_cum = []

    sa_arr = np.zeros(run_duration)
    ss_arr = np.zeros(run_duration)
    sb_arr = np.zeros(run_duration)

    phi_c_a_ruts = np.zeros(run_duration)
    phi_f_a_ruts = np.zeros(run_duration)
    phi_c_s_ruts = np.zeros(run_duration)
    phi_f_s_ruts = np.zeros(run_duration)
    phi_c_a_ruts_min = np.zeros(run_duration)   # worst-case (most compacted) node anywhere on the road
    phi_c_s_ruts_min = np.zeros(run_duration) 
    phi_f_a_ruts_min = np.zeros(run_duration)
    phi_f_s_ruts_min = np.zeros(run_duration)

    Ma = np.zeros(run_duration)
    Maf = np.zeros(run_duration)
    Mac = np.zeros(run_duration)
    Ms = np.zeros(run_duration)
    Msf = np.zeros(run_duration)
    Msc = np.zeros(run_duration)
    Mb = np.zeros(run_duration)
    Mbf = np.zeros(run_duration)
    Mbc = np.zeros(run_duration)

    mass_fillslope_inflow = np.zeros(run_duration)
    mass_fillslope_rut_outflow = np.zeros(run_duration)
    mass_ditch_inflow = np.zeros(run_duration)
    mass_ditch_rut_outflow = np.zeros(run_duration)
    total_road_mass = np.zeros(run_duration)
    road_mass_change_oft = np.zeros(run_duration)

    road_shear_frac_arr = np.zeros(run_duration)
    road_shear_cum_arr = np.zeros(run_duration)
    road_shear_i = np.zeros(mg.number_of_nodes, dtype=bool)
    road_shear_cum = np.zeros(mg.number_of_nodes, dtype=bool)

    avg_shear_stress_ruts = []
    avg_shear_stress_road = []
    avg_n_ruts = []
    avg_n_road = []
    tpe_load_ruts = []
    fs_avg_ruts = []
    fs_avg_road = []

    truck_num = 0

    # ------ initialize Landlab components (fresh per realization) ------
    df = DepressionFinderAndRouter(mg, reroute_flow=True)
    df.map_depressions()

    fa = FlowAccumulator(
        mg,
        surface='topographic__elevation',
        flow_director="FlowDirectorD8",
        runoff_rate=intensity_run_dur[0] * 2.77778e-7,
    )

    tpe = TruckPassErosion(
        mg, center, half_width, full_tire,
        truck_num=truck_num_ini, u_ps=u_ps, u_pb=u_pb,
        F_af0=F_af0, F_sf0=F_sf0, F_bc0=F_bc0,
        scat_loss=scat_loss, compression=compression,
        porosity_c=porosity_c, porosity_f=porosity_f,
    )

    oft = OverlandFlowTransporter(
        mg,
        porosity_f=porosity_f, porosity_c=porosity_c,
        d50=d50_road, longitudinal_slope=S,
        tau_c=tau_c_road, n_c=n_c, n_f=n_f,
        flow_accumulator=fa,
        depression_finder=df,
    )

    # ------ main loop bookkeeping ------
    z_ini_cum = mg.at_node['topographic__elevation'].copy()
    Ma_ini_cum = mg.at_node['active__mass'].copy()
    Ms_ini_cum = mg.at_node['surfacing__mass'].copy()
    Mb_ini_cum = mg.at_node['ballast__mass'].copy()
    active_init = mg.at_node['active__depth'][full_road].copy()
    surfacing_init = mg.at_node['surfacing__depth'][full_road].copy()
    ballast_init = mg.at_node['ballast__depth'][full_road].copy()

    if save_plots:
        out_dir = os.path.join(PROJECT_ROOT, 'output/')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    start = time.time()
    for i in range(0, run_duration):  # daily time step
        z_ini = mg.at_node['topographic__elevation'].copy()

        tpe.run_one_step()
        truck_num += tpe.truck_num

        phi_c_a_ruts[i] = np.nanmean(mg.at_node['phi_c_a'][ruts])
        phi_f_a_ruts[i] = np.nanmean(mg.at_node['phi_f_a'][ruts])
        phi_c_s_ruts[i] = np.nanmean(mg.at_node['phi_c_s'][ruts])
        phi_f_s_ruts[i] = np.nanmean(mg.at_node['phi_f_s'][ruts])
        phi_c_a_ruts_min[i] = np.nanmin(mg.at_node['phi_c_a'][ruts])
        phi_c_s_ruts_min[i] = np.nanmin(mg.at_node['phi_c_s'][ruts])
        phi_f_a_ruts_min[i] = np.nanmin(mg.at_node['phi_f_a'][ruts])
        phi_f_s_ruts_min[i] = np.nanmin(mg.at_node['phi_f_s'][ruts])

        intensity = intensity_run_dur[i]  # use the i-th day's intensity
        dt_day = dt[i]  # use the i-th day's time step

        if intensity <= 0:
            if verbose:
                print(f"Day {i}: No rainfall")
            rain_m_per_s = intensity * 2.77778e-7
            mg.at_node['water__unit_flux_in'] = np.ones(mg.number_of_nodes) * rain_m_per_s
            fa.accumulate_flow()
            df.map_depressions()

            intensity_arr.append(0)
            dt_arr.append(0)

            dz = z - z_ini
            dz_arr.append(sum(dz[full_road.flatten()]))
            dz_cum = z - z_ini_cum
            dz_arr_cum.append(sum(dz_cum[full_road.flatten()]))

            road_shear_cum_arr[i] = road_shear_cum_arr[i - 1]

            avg_shear_stress_ruts.append(0)
            avg_shear_stress_road.append(0)
            avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][ruts]))
            avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][full_road]))
            fs_avg_ruts.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ruts]))
            fs_avg_road.append(np.nanmean(mg.at_node['shear_stress__partitioning'][full_road]))
            tpe_load_ruts.append((tpe.sed_added[full_road]).sum())

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

        else:
            intensity_arr.append(intensity)
            if verbose:
                print(f"Day {i}: Average Intensity = {intensity:.2f} mm/hr")
            dt_arr.append(dt_day)
            if verbose:
                print(f"Day {i}: Length of Storm = {dt_day:.2f} day")

            road_shear_i[:] = 0

            for j, storm_frac in enumerate(frac_arr):
                intensity_dist = p_prime_arr[j] * intensity
                dt_frac = storm_frac * dt_day

                rain_m_per_s = intensity_dist * 2.77778e-7
                mg.at_node['water__unit_flux_in'] = np.ones(mg.number_of_nodes) * rain_m_per_s
                fa.accumulate_flow()
                df.map_depressions()
                oft.run_one_step(dt_frac)

                mass_ditch_rut_outflow_i = (mg.at_node["sediment__mass_influx"][mg.nodes[0, 0:33]]).sum() * dt_frac * 86400
                mass_ditch_inflow_i = (mg.at_node["sediment__mass_influx"][mg.nodes[1:, 0]]).sum() * dt_frac * 86400
                mass_fillslope_inflow_i = (mg.at_node["sediment__mass_influx"][mg.nodes[1:, 63]]).sum() * dt_frac * 86400
                mass_fillslope_rut_outflow_i = (mg.at_node["sediment__mass_influx"][mg.nodes[0, 33:64]]).sum() * dt_frac * 86400

                mass_ditch_rut_outflow[i] += mass_ditch_rut_outflow_i
                mass_ditch_inflow[i] += mass_ditch_inflow_i
                mass_fillslope_inflow[i] += mass_fillslope_inflow_i
                mass_fillslope_rut_outflow[i] += mass_fillslope_rut_outflow_i

                mg.at_node['shear_stress'] = oft.shear_stress
                road_shear = oft.shear_stress[full_road].flatten()

                road_shear_i = np.array(
                    [True if shear_stress >= tau_c_road or road_shear_i[x] else False
                     for x, shear_stress in enumerate(road_shear)]
                )
                road_shear_cum = np.array(
                    [True if shear_stress >= tau_c_road or road_shear_cum[x] else False
                     for x, shear_stress in enumerate(road_shear)]
                )

            road_shear_frac_arr[i] = road_shear_i[road_shear_i == True].sum() / len(road_shear_i)
            road_shear_cum_arr[i] = road_shear_cum[road_shear_cum == True].sum() / len(road_shear_cum)

            mg.at_node['shear_stress'] = oft.shear_stress
            avg_shear_stress_ruts.append(np.nanmean(mg.at_node['shear_stress'][ruts]))
            avg_shear_stress_road.append(np.nanmean(mg.at_node['shear_stress'][full_road]))

            road_mass_change_oft[i] = mass_ditch_inflow[i] + mass_ditch_rut_outflow[i]
            total_road_mass[i] = (
                mass_ditch_inflow[i] + mass_ditch_rut_outflow[i]
                + mass_fillslope_inflow[i] + mass_fillslope_rut_outflow[i]
            )

            dz = z - z_ini
            dz_arr.append(sum(dz[full_road.flatten()]))
            dz_cum = z - z_ini_cum
            dz_arr_cum.append(sum(dz_cum[full_road.flatten()]))

            if save_plots:
                fig = plt.figure(figsize=(9, 6))
                mg.add_field('fine_frac', mg.at_node['active__depth_fines'] / mg.at_node['active__depth_coarse'],
                              at='node', units='m-', clobber=True)
                plt.subplot(131)
                imshow_grid(mg, 'fine_frac', var_name='Faf', var_units='-',
                            plot_name='$S_{af}:S_{ac}$ in\nactive layer, t = %i days' % i,
                            grid_units=('m', 'm'), cmap='Dark2', vmin=0, vmax=1.6, shrink=0.9)
                plt.xlabel('Road width (m)')
                plt.ylabel('Road length (m)')

                plt.subplot(132)
                imshow_grid(mg, 'surface_water__discharge', var_name='Discharge',
                            plot_name='Discharge, t = %i days' % i,
                            var_units='$m/s^3$', grid_units=('m', 'm'),
                            cmap='Blues', vmin=0, vmax=0.00001, shrink=0.9)
                plt.xlabel('Road width (m)')
                plt.ylabel('Road length (m)')

                mg.add_field('dz_cum', dz_cum, at='node', units='m', clobber=True)
                plt.subplot(133)
                imshow_grid(mg, 'dz_cum', var_name='Cumulative dz', var_units='m',
                            plot_name='Elevation change, t = %i days' % i,
                            grid_units=('m', 'm'), cmap='RdBu', vmin=-0.0009, vmax=0.0009, shrink=0.9)
                plt.xlabel('Road width (m)')
                plt.ylabel('Road length (m)')

                plt.tight_layout()
                plt.savefig(os.path.join(PROJECT_ROOT, 'output/plots_%i_days.png' % i), bbox_inches='tight', dpi=300)
                plt.show()
                plt.close()

                fig = plt.figure(figsize=(4, 7))
                plt.subplot(321)
                imshow_grid(mg, 'dz_cum', cmap='RdBu', plot_name="Upper Left",
                            vmin=-0.0009, vmax=0.0009, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(73.75, 79.65)
                plt.subplot(323)
                imshow_grid(mg, 'dz_cum', cmap='RdBu', plot_name="Middle Left",
                            vmin=-0.0009, vmax=0.0009, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(33.925, 39.825)
                plt.subplot(325)
                imshow_grid(mg, 'dz_cum', cmap='RdBu', plot_name="Lower Left",
                            vmin=-0.0009, vmax=0.0009, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(0, 5.9)
                plt.subplot(322)
                imshow_grid(mg, 'dz_cum', cmap='RdBu', plot_name="Upper Right",
                            vmin=-0.0009, vmax=0.0009, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(73.75, 79.65)
                plt.subplot(324)
                imshow_grid(mg, 'dz_cum', cmap='RdBu', plot_name="Middle Right",
                            vmin=-0.0009, vmax=0.0009, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(33.925, 39.825)
                plt.subplot(326)
                imshow_grid(mg, 'dz_cum', cmap='RdBu', plot_name="Lower Right",
                            vmin=-0.0009, vmax=0.0009, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(0, 5.9)
                fig.suptitle('Elevation change,\nt = %i days' % i)
                plt.tight_layout()
                cax = plt.axes((1, 0.075, 0.075, 0.8))
                plt.colorbar(cax=cax, extend="both", label="Cumulative dz ($m$)")
                plt.savefig(os.path.join(PROJECT_ROOT, 'output/subplots_%i_days.png' % i), bbox_inches='tight')
                plt.show()

                fig = plt.figure(figsize=(4, 7))
                plt.subplot(321)
                imshow_grid(mg, 'fine_frac', cmap='Dark2', plot_name="Upper Left",
                            vmin=0, vmax=1.6, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(73.75, 79.65)
                plt.subplot(323)
                imshow_grid(mg, 'fine_frac', cmap='Dark2', plot_name="Middle Left",
                            vmin=0, vmax=1.6, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(33.925, 39.825)
                plt.subplot(325)
                imshow_grid(mg, 'fine_frac', cmap='Dark2', plot_name="Lower Left",
                            vmin=0, vmax=1.6, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(0, 5.9)
                plt.subplot(322)
                imshow_grid(mg, 'fine_frac', cmap='Dark2', plot_name="Upper Right",
                            vmin=0, vmax=1.6, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(73.75, 79.65)
                plt.subplot(324)
                imshow_grid(mg, 'fine_frac', cmap='Dark2', plot_name="Middle Right",
                            vmin=0, vmax=1.6, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(33.925, 39.825)
                plt.subplot(326)
                imshow_grid(mg, 'fine_frac', cmap='Dark2', plot_name="Lower Right",
                            vmin=0, vmax=1.6, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(0, 5.9)
                fig.suptitle('$S_{af}:S_{ac}$,\nt = %i days' % i)
                plt.tight_layout()
                cax = plt.axes((1, 0.075, 0.075, 0.8))
                plt.colorbar(cax=cax, extend="max", label="Ratio of fines to coarse in active layer ($-$)")
                plt.savefig(os.path.join(PROJECT_ROOT, 'output/fubplots_%i_days.png' % i), bbox_inches='tight')
                plt.show()

                fig = plt.figure(figsize=(4, 7))
                plt.subplot(321)
                imshow_grid(mg, 'surface_water__discharge', cmap='Blues', plot_name="Upper Left",
                            vmin=0, vmax=1e-5, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(73.75, 79.65)
                plt.subplot(323)
                imshow_grid(mg, 'surface_water__discharge', cmap='Blues', plot_name="Middle Left",
                            vmin=0, vmax=1e-5, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(33.925, 39.825)
                plt.subplot(325)
                imshow_grid(mg, 'surface_water__discharge', cmap='Blues', plot_name="Lower Left",
                            vmin=0, vmax=1e-5, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(0, 4.72); plt.ylim(0, 5.9)
                plt.subplot(322)
                imshow_grid(mg, 'surface_water__discharge', cmap='Blues', plot_name="Upper Right",
                            vmin=0, vmax=1e-5, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(73.75, 79.65)
                plt.subplot(324)
                imshow_grid(mg, 'surface_water__discharge', cmap='Blues', plot_name="Middle Right",
                            vmin=0, vmax=1e-5, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(33.925, 39.825)
                plt.subplot(326)
                imshow_grid(mg, 'surface_water__discharge', cmap='Blues', plot_name="Lower Right",
                            vmin=0, vmax=1e-5, grid_units=('m', 'm'), allow_colorbar=False)
                plt.xlim(4.72, 9.44); plt.ylim(0, 5.9)
                fig.suptitle('Surface Water Discharge,\nt = %i days' % i)
                plt.tight_layout()
                cax = plt.axes((1, 0.075, 0.075, 0.8))
                plt.colorbar(cax=cax, extend="max", label="Discharge ($m^3/s$)")
                plt.savefig(os.path.join(PROJECT_ROOT, 'output/hubplots_%i_days.png' % i), bbox_inches='tight')
                plt.show()

            avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][ruts]))
            avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][full_road]))
            fs_avg_ruts.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ruts]))
            fs_avg_road.append(np.nanmean(mg.at_node['shear_stress__partitioning'][full_road]))
            tpe_load_ruts.append((tpe.sed_added[full_road]).sum())

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
    if verbose:
        print("Wall time for run:", wall_time, "s")

    road_elev_change_dz = np.divide(dz_arr, 2) / (nrows * ncols / 2)
    cum_road_elev_change_dz = np.divide(dz_arr_cum, 2) / (nrows * ncols / 2)
    cum_road_mass_change_oft = road_mass_change_oft.cumsum()

    if save_plots:
        from PIL import Image
        import glob

        path = os.path.join(PROJECT_ROOT, "output/")
        images_p, images_s, images_h, images_f = [], [], [], []
        for file in glob.glob(path + 'p*.png'):
            images_p.append(file)
        for file in glob.glob(path + 's*.png'):
            images_s.append(file)
        for file in glob.glob(path + 'h*.png'):
            images_h.append(file)
        for file in glob.glob(path + 'f*.png'):
            images_f.append(file)

        images_p = sorted(images_p, key=lambda f: int(f[13:-9]))
        images_s = sorted(images_s, key=lambda f: int(f[16:-9]))
        images_h = sorted(images_h, key=lambda f: int(f[16:-9]))
        images_f = sorted(images_f, key=lambda f: int(f[16:-9]))

        image_list_p = [Image.open(file) for file in images_p]
        image_list_s = [Image.open(file) for file in images_s]
        image_list_h = [Image.open(file) for file in images_h]
        image_list_f = [Image.open(file) for file in images_f]

        image_list_p[0].save(os.path.join(PROJECT_ROOT, 'output/plots.gif'), save_all=True, append_images=image_list_p[1:], duration=1000, loop=0)
        image_list_s[0].save(os.path.join(PROJECT_ROOT, 'output/subplots.gif'), save_all=True, append_images=image_list_s[1:], duration=1000, loop=0)
        image_list_h[0].save(os.path.join(PROJECT_ROOT, 'output/subplots_h.gif'), save_all=True, append_images=image_list_h[1:], duration=1000, loop=0)
        image_list_f[0].save(os.path.join(PROJECT_ROOT, 'output/subplots_f.gif'), save_all=True, append_images=image_list_f[1:], duration=1000, loop=0)

    result = {
        "seed": seed,
        "rain_seed": rain_seed,
        "grid_seed": grid_seed,
        "run_duration": run_duration,
        # parameters used for this realization -- kept for traceability
        # across a parameter sweep
        "porosity_c": porosity_c, "porosity_f": porosity_f,
        "truck_num_ini": truck_num_ini,
        "F_af0": F_af0, "F_sf0": F_sf0, "F_bc0": F_bc0, "scat_loss": scat_loss,
        "tau_c_road": tau_c_road, "n_c": n_c, "n_f": n_f, "d50_road": d50_road,
        "compression": compression,
        "intensity_run_dur": np.array(intensity_arr),
        "dt": np.array(dt_arr),
        "wall_time": wall_time,
        "fine_sediment_pumped_kg": mg.at_node['sediment__pumped'].sum(),
        "fine_sediment_scattered_kg": mg.at_node['sediment__scattered'].sum(),
        "sediment_added_kg": mg.at_node['sediment__added'].sum(),
        "mass_ditch_inflow": mass_ditch_inflow,
        "mass_ditch_rut_outflow": mass_ditch_rut_outflow,
        "mass_fillslope_inflow": mass_fillslope_inflow,
        "mass_fillslope_rut_outflow": mass_fillslope_rut_outflow,
        "total_road_mass": total_road_mass,
        "road_mass_change_oft": road_mass_change_oft,
        "cum_road_mass_change_oft": cum_road_mass_change_oft,
        "dz_arr": np.array(dz_arr),
        "dz_arr_cum": np.array(dz_arr_cum),
        "road_elev_change_dz": road_elev_change_dz,
        "cum_road_elev_change_dz": cum_road_elev_change_dz,
        "sa_arr": sa_arr, "ss_arr": ss_arr, "sb_arr": sb_arr,
        "Ma": Ma, "Maf": Maf, "Mac": Mac,
        "Ms": Ms, "Msf": Msf, "Msc": Msc,
        "Mb": Mb, "Mbf": Mbf, "Mbc": Mbc,
        "road_shear_frac_arr": road_shear_frac_arr,
        "road_shear_cum_arr": road_shear_cum_arr,
        "avg_shear_stress_ruts": avg_shear_stress_ruts,
        "avg_shear_stress_road": avg_shear_stress_road,
        "avg_n_ruts": avg_n_ruts,
        "avg_n_road": avg_n_road,
        "tpe_load_ruts": tpe_load_ruts,
        "fs_avg_ruts": fs_avg_ruts,
        "fs_avg_road": fs_avg_road,
        "S": S, "u_ps": u_ps, "u_pb": u_pb,
        "rain_zero_prob": rain_zero_prob, "rain_lam": rain_lam,
        "rain_duration_mean": rain_duration_mean,
        "historical_csv_path": historical_csv_path,
        "phi_c_a_ruts": phi_c_a_ruts, "phi_f_a_ruts": phi_f_a_ruts,
        "phi_c_s_ruts": phi_c_s_ruts, "phi_f_s_ruts": phi_f_s_ruts,
        "phi_c_a_ruts_min": phi_c_a_ruts_min,
        "phi_f_a_ruts_min": phi_f_a_ruts_min,
        "phi_c_s_ruts_min": phi_c_s_ruts_min,
        "phi_f_s_ruts_min": phi_f_s_ruts_min,
    }

    if return_grid:
        result.update({
            "mg": mg,
            "X": X,
            "xsec_pre": xsec_pre,
            "active_init": active_init,
            "surfacing_init": surfacing_init,
            "ballast_init": ballast_init,
        })

    return result

#%%
# ==========================================================================
# FULL DIAGNOSTIC PLOTS FOR ONE REALIZATION
# (equivalent to the original script's bottom section -- call this on a
#  result from run_realization(..., save_plots=True) or return_grid=True)
# ==========================================================================
def plot_cross_section(result, row=100):
    """
    Plot just the road cross-section (before vs. after, by layer) for one
    realization -- lighter-weight than plot_full_diagnostics() when that's
    all you want.

    `result` must come from a call to run_realization(..., return_grid=True)
    -- the grid itself isn't stored in ensemble results (return_grid=False
    there, to save memory across 100+ runs), so to view a specific seed's
    cross-section after the fact, re-run just that one realization with
    return_grid=True. Since results are fully deterministic given the same
    seed (and the same parameter values, if it came from the sweep), this
    reproduces the exact same realization -- it does not require having
    kept anything from the original ensemble run.

    row : which grid row to slice across (default 100, matching the row
        used elsewhere in this script for the road cross-section).
    """
    mg = result["mg"]
    X = result["X"]
    xsec_pre = result["xsec_pre"]

    xsec_active = mg.at_node['topographic__elevation'][mg.nodes[row, :].flatten()]
    xsec_surf = mg.at_node['surfacing__elevation'][mg.nodes[row, :].flatten()]
    xsec_ball = mg.at_node['ballast__elevation'][mg.nodes[row, :].flatten()]

    plt.figure(figsize=(8, 3), layout='tight')
    plt.plot(X[row], xsec_pre, color='gray', linestyle='-.', label='Before')
    plt.plot(X[row], xsec_active, color='black', linestyle='-', label='After - Active elevation')
    plt.plot(X[row], xsec_surf, color='magenta', linestyle='-', label='After - Surfacing elevation')
    plt.plot(X[row], xsec_ball, color='cyan', linestyle='-', label='After - Ballast elevation')
    plt.xlim(0, 9.5)
    plt.xlabel('Road width (m)')
    plt.ylabel('Elevation (m)')
    plt.title(f"Cross section -- seed={result['seed']}, row={row}")
    plt.legend()
    plt.show()


def plot_full_diagnostics(result):
    mg = result["mg"]
    X = result["X"]
    xsec_pre = result["xsec_pre"]
    active_init = result["active_init"]
    surfacing_init = result["surfacing_init"]
    ballast_init = result["ballast_init"]
    run_duration = result["run_duration"]

    xsec_active = mg.at_node['topographic__elevation'][mg.nodes[100, :].flatten()]
    xsec_surf = mg.at_node['surfacing__elevation'][mg.nodes[100, :].flatten()]
    xsec_ball = mg.at_node['ballast__elevation'][mg.nodes[100, :].flatten()]

    plt.figure(figsize=(8, 3), layout='tight')
    plt.plot(X[36], xsec_pre, color='gray', linestyle='-.', label='Before')
    plt.plot(X[36], xsec_active, color='black', linestyle='-', label='After - Active elevation')
    plt.plot(X[36], xsec_surf, color='magenta', linestyle='-', label='After - Surfacing elevation')
    plt.plot(X[36], xsec_ball, color='cyan', linestyle='-', label='After - Ballast elevation')
    plt.xlim(0, 9.5)
    plt.xlabel('Road width (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(2, 1)
    ax[0].bar(range(0, run_duration), np.multiply(result["intensity_run_dur"], np.multiply(result["dt"], 24)))
    ax[0].set_xlabel('Day'); ax[0].set_ylabel('Rainfall [mm]'); ax[0].set_xlim(0, run_duration)
    ax[1].plot(range(0, run_duration), result["intensity_run_dur"])
    ax[1].set_xlabel('Day'); ax[1].set_ylabel('Rainfall intensity [mm/hr]'); ax[1].set_xlim(0, run_duration)
    plt.suptitle(r'%s' % site_name)
    plt.tight_layout()
    plt.show()

    print("Total rainfall,", run_duration, "days:",
          np.round(sum(np.multiply(result["intensity_run_dur"], np.multiply(result["dt"], 24))), 2), 'mm')
    print('Fine sediment pumped:', np.round(result["fine_sediment_pumped_kg"], 2), 'kg')
    print('Fine sediment made available due to scattering:', np.round(result["fine_sediment_scattered_kg"], 2), 'kg')
    print('Total sediment added to the active layer:', np.round(result["sediment_added_kg"], 2), 'kg')
    print('Cumulative sediment load from cutslope side of road:',
          np.round((result["mass_ditch_inflow"] + result["mass_ditch_rut_outflow"]).sum(), 2), 'kg')
    print('Cumulative sediment load from fillslope side of road:',
          np.round((result["mass_fillslope_inflow"] + result["mass_fillslope_rut_outflow"]).sum(), 2), 'kg')
    print('Cumulative sediment load from both ruts:',
          np.round((result["mass_fillslope_rut_outflow"] + result["mass_ditch_rut_outflow"]).sum(), 2), 'kg')
    print('Cumulative sediment load from both sides:',
          np.round((result["mass_fillslope_inflow"] + result["mass_ditch_inflow"]).sum(), 2), 'kg')
    print('Cumulative sediment load from full road:', np.round(result["total_road_mass"].sum(), 2), 'kg')

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(range(0, run_duration), -result["road_mass_change_oft"])
    ax[0].plot(range(0, run_duration), np.zeros(run_duration), '--', color='gray')
    ax[0].set_xlabel('Day'); ax[0].set_ylabel(r'$\Delta$ mass between time steps [$kg$]')
    ax[0].set_xlim(0, run_duration); ax[0].set_title('(a) Half Road (OFT)')
    ax[1].plot(range(0, run_duration), result["road_elev_change_dz"])
    ax[1].plot(range(0, run_duration), np.zeros(run_duration), '--', color='gray')
    ax[1].set_xlabel('Day'); ax[1].set_ylabel(r'$\Delta$ elevation between time steps [$m$]')
    ax[1].set_xlim(0, run_duration); ax[1].set_title('(b) Half Road (dz)')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(range(0, run_duration), -result["cum_road_mass_change_oft"])
    ax[0].plot(range(0, run_duration), np.zeros(run_duration), '--', color='gray')
    ax[0].set_xlabel('Day'); ax[0].set_ylabel('Cumulative mass change - \nhalf road [$kg$]')
    ax[0].set_xlim(0, run_duration); ax[0].set_title('(a) Half Road (OFT)')
    ax[1].plot(range(0, run_duration), result["cum_road_elev_change_dz"])
    ax[1].plot(range(0, run_duration), np.zeros(run_duration), '--', color='gray')
    ax[1].set_xlabel('Day'); ax[1].set_ylabel('Cumulative elevation change - \nhalf road [$m$]')
    ax[1].set_xlim(0, run_duration); ax[1].set_title('(b) Half Road (dz)')
    plt.tight_layout()
    plt.show()

    plt.plot(range(0, run_duration), result["tpe_load_ruts"])
    plt.xlabel('Day')
    plt.ylabel('Cumulative sediment load to the active layer in the ruts \nfrom TPE [$kg$]')
    plt.xlim(0, run_duration)
    plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(4, 7))
    ax[0].set_title(r'%s ($n_{f_{road}} = %0.3f$)' % (site_name, n_f))
    ax[0].plot(range(0, run_duration), (-active_init.sum() + result["sa_arr"]) / (nrows * ncols))
    ax[0].set_xlabel('Day'); ax[0].set_ylabel('Active Depth change\n(averaged over full road) [$m$]')
    ax[0].set_xlim(0, run_duration)
    ax[1].plot(range(0, run_duration), (-surfacing_init.sum() + result["ss_arr"]) / (nrows * ncols))
    ax[1].set_xlabel('Day'); ax[1].set_ylabel('Surfacing Depth change\n(averaged over full road) [$m$]')
    ax[1].set_xlim(0, run_duration)
    ax[2].plot(range(0, run_duration), (-ballast_init.sum() + result["sb_arr"]) / (nrows * ncols))
    ax[2].set_xlabel('Day'); ax[2].set_ylabel('Ballast Depth change\n(averaged over full road) [$m$]')
    ax[2].set_xlim(0, run_duration)
    plt.tight_layout()
    plt.show()

    for label, keys in [
        ("Active", ("Ma", "Maf", "Mac")),
        ("Surfacing", ("Ms", "Msf", "Msc")),
        ("Ballast", ("Mb", "Mbf", "Mbc")),
    ]:
        fig, ax = plt.subplots(3, 1, figsize=(4, 7))
        ax[0].plot(range(0, run_duration), result[keys[0]] / (nrows * ncols))
        ax[0].set_xlabel('Day'); ax[0].set_ylabel(f'{label} Mass\naverage [$kg$]')
        ax[0].set_xlim(0, run_duration)
        ax[0].set_title(r'%s ($n_{f_{road}} = %0.3f$)' % (site_name, n_f))
        ax[1].plot(range(0, run_duration), result[keys[1]] / (nrows * ncols))
        ax[1].set_xlabel('Day'); ax[1].set_ylabel(f'{label} Mass - fines\naverage [$kg$]')
        ax[1].set_xlim(0, run_duration)
        ax[2].plot(range(0, run_duration), result[keys[2]] / (nrows * ncols))
        ax[2].set_xlabel('Day'); ax[2].set_ylabel(f'{label} Mass - coarse\naverage [$kg$]')
        ax[2].set_xlim(0, run_duration)
        plt.tight_layout()
        plt.show()


# ==========================================================================
# SAFE HISTOGRAM HELPER
# np.histogram can fail with "Too many bins for data range" not just when
# a dataset is exactly flat, but also when it varies by an amount too tiny
# for floating-point precision to divide into that many distinct bin edges
# (e.g. values that agree to 10+ significant figures). 
# ==========================================================================
def safe_hist(ax, arr, bins, color, label, xlabel="kg"):
    arr = np.asarray(arr, dtype=float)
    data_range = arr.max() - arr.min()
    scale = max(abs(arr.mean()), 1e-12)  # avoid divide-by-zero if mean is 0

    plotted = False
    if data_range > 1e-8 * scale:  # range is meaningfully larger than float noise
        try:
            ax.hist(arr, bins=bins, color=color)
            plotted = True
        except ValueError:
            plotted = False

    if not plotted:
        ax.axvline(arr.mean(), color=color, linewidth=2)
        ax.set_xlim(arr.mean() - 1, arr.mean() + 1)
        ax.text(0.5, 0.5, "no meaningful variation\nacross realizations",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")

    ax.set_title(label, fontsize=10)
    ax.set_xlabel(xlabel)



# %%
