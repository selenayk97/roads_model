"""
Runs the model-parameter sweep (porosity, TPE settings, etc.) across
multiple mini Monte Carlo ensembles.
Requires oft_model_core.py in the same folder (imports run_realization()
and the plotting/analysis helpers from it).
"""
#%%
from oft_model_core import *

#%%
# ==========================================================================
# PARAMETER SWEEP -- run the ensemble multiple times with DIFFERENT model
# parameters (porosity, TPE truck/scatter settings, roughness, critical
# shear stress, etc.), instead of varying only the rainfall. Each parameter
# setting gets its own mini Monte Carlo ensemble (same pool of seeds
# reused across settings -- "common random numbers" -- so differences
# between settings aren't confounded by also getting different rainfall).
# ==========================================================================
if __name__ == "__main__":
    N_REALIZATIONS_PER_SETTING = 10   # rainfall/grid realizations per parameter setting
    SWEEP_RUN_DURATION = 120

    # Rainfall source for the sweep
    SWEEP_RAINFALL_SOURCE = "historical"

    # =========================================================================
    # SINGLE-PARAMETER RANGE SWEEP -- just set the parameter name and the
    # list (or range) of values to test it at. The sweep is built for you.
    #
    #   SWEEP_PARAM  = the run_realization() keyword argument to vary
    #   SWEEP_VALUES = the values to test it at
    #
    # Examples:
    #   SWEEP_VALUES = [0.5, 0.6, 0.7]            # explicit list
    #   SWEEP_VALUES = np.arange(0.5, 0.8, 0.1)   # CAUTION: floating-point step
    #                                              # sizes can add a stray extra
    #                                              # value (e.g. this can produce
    #                                              # 0.5, 0.6, 0.7, 0.7999999999999999)
    #   SWEEP_VALUES = np.linspace(0.3, 0.7, 5)   # safer: exact count, no float drift
    #
    # Can change the following: porosity_c, porosity_f, scat_loss, tau_c_road,
    # truck_num_ini, n_c, n_f, d50_road, F_af0, F_bc0, F_sf0, compression
    # =========================================================================
    SWEEP_PARAM = "truck_num_ini"  # the run_realization() keyword argument to vary
    #may or may not have to update code with type as int
    SWEEP_VALUES = np.linspace(0, 13, 4).astype(int)  # 4 evenly-spaced values: 0, 4, 8, 12

    PARAM_SWEEP = [
        {"name": f"{SWEEP_PARAM}={v:g}", SWEEP_PARAM: v}
        for v in SWEEP_VALUES
    ]

    # ---- Alternative: comparing several DIFFERENT parameters at once ----
    # instead of one parameter across a range, uncomment this style instead:
    # PARAM_SWEEP = [
    #     {"name": "Baseline"},
    #     {"name": "High porosity", "porosity_c": 0.5, "porosity_f": 0.5},
    #     {"name": "Low porosity", "porosity_c": 0.2, "porosity_f": 0.2},
    #     {"name": "More truck passes", "truck_num_ini": 10},
    #     {"name": "Fewer truck passes", "truck_num_ini": 1},
    #     {"name": "High scatter loss", "scat_loss": 0.02},
    #     {"name": "Low critical shear stress", "tau_c_road": 0.05},
    # ]

    sweep_results = {}
    for params in PARAM_SWEEP:
        name = params.get("name", "unnamed")
        kwargs = {k: v for k, v in params.items() if k != "name"}
        print(f"\n=== Parameter setting: {name} ({kwargs if kwargs else 'defaults'}) ===")

        runs = []
        for i in range(N_REALIZATIONS_PER_SETTING):
            # same seed pool (0..N-1) reused for every parameter setting,
            # so each setting sees the same set of rainfall/noise draws
            res = run_realization(
                seed=i, run_duration=SWEEP_RUN_DURATION,
                save_plots=False, verbose=False, return_grid=False,
                rainfall_source=SWEEP_RAINFALL_SOURCE, **kwargs,
            )
            runs.append(res)
        sweep_results[name] = runs

    # ------ compare total_road_mass across parameter settings ------
    sweep_labels = list(sweep_results.keys())
    sweep_road_mass = [
        np.array([r["total_road_mass"].sum() for r in sweep_results[name]])
        for name in sweep_labels
    ]

    print(f"\n--- Parameter sweep summary ({N_REALIZATIONS_PER_SETTING} realizations each) ---")
    print(f"{'Setting':<28}{'Mean road mass':>16}{'Std':>12}")
    for name, arr in zip(sweep_labels, sweep_road_mass):
        print(f"{name:<28}{arr.mean():>16.2f}{arr.std():>12.2f}")

    # boxplot: distribution of total_road_mass under each parameter setting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(sweep_road_mass, tick_labels=sweep_labels)
    ax.set_ylabel("Total road mass, water-transported [kg]")
    ax.set_title(f"Effect of model parameters on sediment output "
                 f"({N_REALIZATIONS_PER_SETTING} realizations per setting)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    # ------ sediment vs. rainfall, colored by parameter setting ------
    # Since every setting reuses the same seed pool, each point pair across
    # settings saw the SAME rainfall -- this shows whether the relationship
    # between rainfall and sediment output shifts depending on the
    # parameter value, not just whether the totals shift.
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("viridis")
    for i, name in enumerate(sweep_labels):
        runs = sweep_results[name]
        rain = np.array([
            np.sum(np.multiply(r["intensity_run_dur"], np.multiply(r["dt"], 24)))
            for r in runs
        ])
        mass = np.array([r["total_road_mass"].sum() for r in runs])
        color = cmap(i / max(len(sweep_labels) - 1, 1))
        ax.scatter(rain, mass, color=color, label=name, alpha=0.7)
    ax.set_xlabel("Total rainfall [mm]")
    ax.set_ylabel("Total road mass, water-transported [kg]")
    ax.set_title("Sediment vs. rainfall, by parameter setting")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------ paired comparison: same storm sequence across settings ------
    # One line per seed, connecting its result across all parameter values
    # -- directly visualizes "holding the same rainfall fixed, how does
    # sediment output change as the parameter changes?"
    if all(SWEEP_PARAM in params for params in PARAM_SWEEP):
        sweep_param_values = [params[SWEEP_PARAM] for params in PARAM_SWEEP]
        fig, ax = plt.subplots(figsize=(8, 6))
        for seed_idx in range(N_REALIZATIONS_PER_SETTING):
            mass_by_setting = [
                sweep_results[name][seed_idx]["total_road_mass"].sum()
                for name in sweep_labels
            ]
            ax.plot(sweep_param_values, mass_by_setting, color="gray", alpha=0.3, marker="o")
        mean_by_setting = [arr.mean() for arr in sweep_road_mass]
        ax.plot(sweep_param_values, mean_by_setting, color="red", linewidth=2,
                 marker="o", label="Mean")

        # ------ quantify the relationship: linear fit across every individual
        # realization (not just the means), so the fit reflects the full
        # spread of outcomes at each setting, not an artificially smoothed
        # summary. ------
        all_x = np.repeat(sweep_param_values, N_REALIZATIONS_PER_SETTING)
        all_y = np.concatenate(sweep_road_mass)
        slope_fit, intercept_fit = np.polyfit(all_x, all_y, 1)
        y_pred = slope_fit * all_x + intercept_fit
        ss_res = np.sum((all_y - y_pred) ** 2)
        ss_tot = np.sum((all_y - all_y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        x_line = np.array([min(sweep_param_values), max(sweep_param_values)])
        ax.plot(x_line, slope_fit * x_line + intercept_fit, color="black",
                 linestyle="--", linewidth=1.5,
                 label=f"Linear fit (R²={r_squared:.3f})")

        print(f"\n--- Relationship between {SWEEP_PARAM} and total road mass ---")
        print(f"  slope (Δkg per unit {SWEEP_PARAM}): {slope_fit:.4f}")
        print(f"  intercept: {intercept_fit:.4f}")
        print(f"  R² (based on every individual realization, n={len(all_x)}): {r_squared:.4f}")
        if r_squared < 0.3:
            print(f"  NOTE: low R² -- {SWEEP_PARAM} explains only a small share of the "
                  f"variation; rainfall/grid-noise randomness likely dominates over "
                  f"this parameter's effect at the tested values.")

        ax.set_xlabel(SWEEP_PARAM)
        ax.set_ylabel("Total road mass, water-transported [kg]")
        ax.set_title(f"Sediment output vs. {SWEEP_PARAM}, paired by rainfall seed")
        ax.legend()
        plt.tight_layout()
        plt.show()

         
        # ------ log-log version, with a power-law fit ------
        # Log scales need strictly positive values on both axes -- skip if
        # either the parameter or any sediment value isn't positive (e.g.
        # SWEEP_PARAM includes 0 or negative values, or some realization
        # produced exactly 0 kg of sediment).
        if all(v > 0 for v in sweep_param_values) and all(all_y > 0):
            fig, ax = plt.subplots(figsize=(8, 6))
            for seed_idx in range(N_REALIZATIONS_PER_SETTING):
                mass_by_setting = [
                    sweep_results[name][seed_idx]["total_road_mass"].sum()
                    for name in sweep_labels
                ]
                ax.plot(sweep_param_values, mass_by_setting, color="gray", alpha=0.3, marker="o")
            ax.plot(sweep_param_values, mean_by_setting, color="red", linewidth=2,
                     marker="o", label="Mean")
 
            # power-law fit: y = a * x^b  <=>  log(y) = log(a) + b*log(x)
            # -- same idea as the linear fit above, just done in log-log space
            log_x = np.log10(all_x)
            log_y = np.log10(all_y)
            b_fit, log_a_fit = np.polyfit(log_x, log_y, 1)
            a_fit = 10 ** log_a_fit
            log_y_pred = b_fit * log_x + log_a_fit
            ss_res_log = np.sum((log_y - log_y_pred) ** 2)
            ss_tot_log = np.sum((log_y - log_y.mean()) ** 2)
            r_squared_log = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else float("nan")
 
            x_line = np.array([min(sweep_param_values), max(sweep_param_values)])
            ax.plot(x_line, a_fit * x_line ** b_fit, color="black", linestyle="--",
                     linewidth=1.5, label=f"Power-law fit (R²={r_squared_log:.3f})")
 
            print(f"\n--- Power-law fit: total road mass = a * {SWEEP_PARAM}^b ---")
            print(f"  a = {a_fit:.4f}, b (exponent) = {b_fit:.4f}")
            print(f"  R² (log-log space, n={len(all_x)}): {r_squared_log:.4f}")
 
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"{SWEEP_PARAM} (log scale)")
            ax.set_ylabel("Total road mass, water-transported [kg] (log scale)")
            ax.set_title(f"Sediment output vs. {SWEEP_PARAM} -- log-log")
            ax.legend()
            plt.tight_layout()
            plt.show()
        else:
            print(f"\nSkipped log-log plot: {SWEEP_PARAM} or total road mass includes "
                  f"zero/negative values, which can't be shown on a log scale.")
#%%
# ------ optional: view a road cross-section for one specific seed ------
# Works after EITHER the main ensemble or the sweep above (or standalone,
# without running either) -- just re-runs that one seed with
# return_grid=True so the grid is available to plot from.
#
# From the main ensemble (baseline parameters, whatever PARAM_OVERRIDES was):
seed_to_view = 42
detailed = run_realization(seed=seed_to_view, run_duration=RUN_DURATION,
                            save_plots=False, verbose=True, return_grid=True,
                            rainfall_source="poisson", **PARAM_OVERRIDES)
plot_cross_section(detailed)
#%%
# From a specific sweep setting -- pass the SAME parameter value(s) that
# setting used, so it's an exact match to what actually ran:
# seed_to_view = 5
detailed = run_realization(seed=seed_to_view, run_duration=SWEEP_RUN_DURATION,
                             save_plots=False, verbose=True, return_grid=True,
                             rainfall_source=SWEEP_RAINFALL_SOURCE, porosity_c=0.6)
plot_cross_section(detailed)

#%%
# ------ optional: view the daily-map GIFs for one specific seed ------
# save_plots=True is REQUIRED here (unlike the cross-section example
# above) -- the GIF-building code only runs when save_plots=True, since
# it needs the daily PNGs saved to disk first. This is much slower than
# everything else in this script (a matplotlib figure gets saved to
# disk every single simulated day), so keep run_duration short for a
# first look. The four GIFs (plots.gif, subplots.gif, subplots_h.gif,
# subplots_f.gif) get saved to your current working directory, and
# output/ gets wiped and rebuilt each time this runs.
#
from IPython.display import Image, display
#
seed_to_view = 1
detailed = run_realization(seed=seed_to_view, run_duration=30,
                             save_plots=True, verbose=True, return_grid=True,
                             rainfall_source="poisson")

display(Image(filename=os.path.join(PROJECT_ROOT, "output/plots.gif")))
display(Image(filename=os.path.join(PROJECT_ROOT, "output/subplots.gif")))
display(Image(filename=os.path.join(PROJECT_ROOT, "output/subplots_h.gif")))
display(Image(filename=os.path.join(PROJECT_ROOT, "output/subplots_f.gif")))
