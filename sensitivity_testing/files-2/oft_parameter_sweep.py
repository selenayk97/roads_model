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
    SWEEP_RUN_DURATION = 30

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
    # truck_num_ini, n_c, n_f, d50_road, F_af0, F_bc0, F_sf0, compression, u_ps, S, u_pb
    # rain_zero_prob, rain_lam, rain_duration_mean
    # =========================================================================
    RUN_SINGLE_PARAM_SWEEP = True         # set False to skip the SWEEP_PARAM sweep below
    RUN_COMPRESSION_THRESHOLD_GRID = False  # set True to run the compression x truck_num_ini grid sweep
 
    if RUN_SINGLE_PARAM_SWEEP:

        SWEEP_PARAM = "S"  # the run_realization() keyword argument to vary
    #may or may not have to update code with type as int (truck_num for example)
        SWEEP_VALUES = np.linspace(0.05, 0.13, 4)  # the values to test it at

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

    # ------ cutslope-only sediment mass, same structure as above ------
    # cum_road_mass_change_oft is already the cutslope side only (mass_ditch_inflow
    # + mass_ditch_rut_outflow) -- see run_realization for details.
        sweep_cutslope_mass = [
            np.array([r["cum_road_mass_change_oft"][-1] for r in sweep_results[name]])
            for name in sweep_labels
        ]

        print(f"\n--- Parameter sweep summary ({N_REALIZATIONS_PER_SETTING} realizations each) ---")
        print(f"{'Setting':<28}{'Mean total mass':>16}{'Std':>10}{'Mean cutslope mass':>20}{'Std':>10}")
        for name, arr, arr_cut in zip(sweep_labels, sweep_road_mass, sweep_cutslope_mass):
            print(f"{name:<28}{arr.mean():>16.2f}{arr.std():>10.2f}{arr_cut.mean():>20.2f}{arr_cut.std():>10.2f}")

    # boxplot: distribution of total_road_mass under each parameter setting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(sweep_road_mass, tick_labels=sweep_labels)
        ax.set_ylabel("Total road mass, water-transported [kg]")
        ax.set_title(f"Effect of model parameters on sediment output "
                 f"({N_REALIZATIONS_PER_SETTING} realizations per setting)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    # boxplot: cutslope-only sediment mass under each parameter setting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(sweep_cutslope_mass, tick_labels=sweep_labels)
        ax.set_ylabel("Cutslope-side road mass, water-transported [kg]")
        ax.set_title(f"Effect of model parameters on CUTSLOPE-ONLY sediment output "
                    f"({N_REALIZATIONS_PER_SETTING} realizations per setting)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.show()

    # side-by-side comparison: total vs. cutslope-only, per setting
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(sweep_labels))
        width = 0.35
        total_means = [arr.mean() for arr in sweep_road_mass]
        cutslope_means = [arr.mean() for arr in sweep_cutslope_mass]
        ax.bar(x - width/2, total_means, width, label="Total road mass")
        ax.bar(x + width/2, cutslope_means, width, label="Cutslope-only mass")
        ax.set_xticks(x)
        ax.set_xticklabels(sweep_labels, rotation=30, ha="right")
        ax.set_ylabel("Sediment mass [kg]")
        ax.set_title("Total vs. cutslope-only sediment output, by parameter setting")
        ax.legend()
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
                    label=f"Linear fit (slope={slope_fit:.4g}, R²={r_squared:.3f})")

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
                        linewidth=1.5, label=f"Power-law fit (slope/exponent b={b_fit:.3g}, R²={r_squared_log:.3f})")
 
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

                # ------ cutslope-only log-log version, with its own power-law fit ------
            all_y_cutslope = np.concatenate(sweep_cutslope_mass)
            if all(v > 0 for v in sweep_param_values) and all(all_y_cutslope > 0):
                fig, ax = plt.subplots(figsize=(8, 6))
                for seed_idx in range(N_REALIZATIONS_PER_SETTING):
                    mass_by_setting_cut = [
                        sweep_results[name][seed_idx]["cum_road_mass_change_oft"][-1]
                        for name in sweep_labels
                    ]
                    ax.plot(sweep_param_values, mass_by_setting_cut, color="gray", alpha=0.3, marker="o")
                mean_cutslope_by_setting = [arr.mean() for arr in sweep_cutslope_mass]
                ax.plot(sweep_param_values, mean_cutslope_by_setting, color="orange", linewidth=2,
                        marker="o", label="Mean (cutslope only)")

                log_x_cut = np.log10(all_x)
                log_y_cut = np.log10(all_y_cutslope)
                b_fit_cut, log_a_fit_cut = np.polyfit(log_x_cut, log_y_cut, 1)
                a_fit_cut = 10 ** log_a_fit_cut
                log_y_pred_cut = b_fit_cut * log_x_cut + log_a_fit_cut
                ss_res_cut = np.sum((log_y_cut - log_y_pred_cut) ** 2)
                ss_tot_cut = np.sum((log_y_cut - log_y_cut.mean()) ** 2)
                r_squared_cut = 1 - ss_res_cut / ss_tot_cut if ss_tot_cut > 0 else float("nan")

                x_line = np.array([min(sweep_param_values), max(sweep_param_values)])
                ax.plot(x_line, a_fit_cut * x_line ** b_fit_cut, color="black", linestyle="--",
                        linewidth=1.5, label=f"Power-law fit (slope/exponent b={b_fit_cut:.3g}, R²={r_squared_cut:.3f})")

                print(f"\n--- Power-law fit: cutslope-only mass = a * {SWEEP_PARAM}^b ---")
                print(f"  a = {a_fit_cut:.4f}, b (exponent) = {b_fit_cut:.4f}")
                print(f"  R² (log-log space, n={len(all_x)}): {r_squared_cut:.4f}")

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(f"{SWEEP_PARAM} (log scale)")
                ax.set_ylabel("Cutslope-only road mass [kg] (log scale)")
                ax.set_title(f"Cutslope-only sediment output vs. {SWEEP_PARAM} -- log-log")
                ax.legend()
                plt.tight_layout()
                plt.show()
            else:
                print(f"\nSkipped cutslope-only log-log plot: {SWEEP_PARAM} or cutslope mass "
                    f"includes zero/negative values, which can't be shown on a log scale.")
        
        # ------ additional metric sweeps: fines fraction, road condition,
        # shear behavior, TPE load -- same paired-line style as the mass
        # sweep above, just swapping in a different scalar per realization ------
        def plot_metric_sweep(get_value, ylabel, title):
            """
            Line plot of a scalar metric extracted from each realization's
            result dict, across SWEEP_VALUES. Every realization gets its own
            line (colored by seed), so you can see the full spread rather
            than just a mean trend.

            get_value : function(result_dict) -> float
            """
            x = np.array(SWEEP_VALUES)
            values_by_setting = np.array([
                [get_value(res) for res in sweep_results[name]] for name in sweep_labels
            ])  # shape (n_settings, n_seeds)

            n_seeds = values_by_setting.shape[1]
            cmap = plt.get_cmap("viridis")
            fig, ax = plt.subplots(figsize=(7, 5))

            for seed_idx in range(n_seeds):
                color = cmap(seed_idx / max(n_seeds - 1, 1))
                ax.plot(x, values_by_setting[:, seed_idx], color=color, alpha=0.8,
                        marker="o", markersize=3, linewidth=1, label=f"seed {seed_idx}")

            ax.set_xlabel(SWEEP_PARAM)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            # legend gets crowded past ~15 seeds -- drop it and rely on the
            # colorbar-style gradient (low seed = dark purple, high seed =
            # yellow) instead
            if n_seeds <= 15:
                ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.show()

        if all(SWEEP_PARAM in params for params in PARAM_SWEEP):
            # ------ sediment composition / supply ------
            plot_metric_sweep(lambda res: res["Maf"][-1] / res["Ma"][-1],
                               ylabel="Fines fraction in active layer [-]",
                               title=f"Active-layer fines fraction vs. {SWEEP_PARAM}")

            plot_metric_sweep(lambda res: res["fine_sediment_pumped_kg"],
                               ylabel="Fine sediment pumped [kg]",
                               title=f"Fine sediment pumped vs. {SWEEP_PARAM}")

            plot_metric_sweep(lambda res: res["fine_sediment_scattered_kg"],
                               ylabel="Fine sediment scattered [kg]",
                               title=f"Fine sediment scattered vs. {SWEEP_PARAM}")

            plot_metric_sweep(lambda res: res["sediment_added_kg"],
                               ylabel="Sediment added to active layer [kg]",
                               title=f"Sediment added vs. {SWEEP_PARAM}")

            # ------ physical road condition ------
            plot_metric_sweep(lambda res: res["cum_road_elev_change_dz"][-1],
                               ylabel="Cumulative elevation change [m]",
                               title=f"Net road elevation change vs. {SWEEP_PARAM}")

            plot_metric_sweep(lambda res: (res["sa_arr"][-1] - res["sa_arr"][0]) / (nrows * ncols) * 1000,
                               ylabel="Active-layer depth change [mm]",
                               title=f"Active-layer depth change vs. {SWEEP_PARAM}")

            plot_metric_sweep(lambda res: (res["ss_arr"][-1] - res["ss_arr"][0]) / (nrows * ncols) * 1000,
                               ylabel="Surfacing-layer depth change [mm]",
                               title=f"Surfacing-layer depth change vs. {SWEEP_PARAM}")

            plot_metric_sweep(lambda res: (res["sb_arr"][-1] - res["sb_arr"][0]) / (nrows * ncols) * 1000,
                               ylabel="Ballast-layer depth change [mm]",
                               title=f"Ballast-layer depth change vs. {SWEEP_PARAM}")

            # ------ hydraulics / shear behavior ------
            plot_metric_sweep(lambda res: res["road_shear_cum_arr"][-1],
                               ylabel="Fraction of road exceeding $\\tau_c$ [-]",
                               title=f"Shear-stress exceedance vs. {SWEEP_PARAM}")
            # peak shear stress reached at any point during the run
            plot_metric_sweep(lambda res: np.max(res["avg_shear_stress_road"]),
                                ylabel="Peak mean shear stress, full road [Pa]",
                                title=f"Peak mean shear stress vs. {SWEEP_PARAM}")
            plot_metric_sweep(lambda res: res["avg_n_road"][-1],
                               ylabel="Mean roughness, full road [-]",
                               title=f"Roughness vs. {SWEEP_PARAM}")

            plot_metric_sweep(lambda res: res["fs_avg_road"][-1],
                               ylabel="Mean shear-stress partitioning fs, full road [-]",
                               title=f"Shear partitioning vs. {SWEEP_PARAM}")

            # ------ TPE-specific ------
            plot_metric_sweep(lambda res: res["tpe_load_ruts"][-1],
                               ylabel="Cumulative TPE load to ruts [kg]",
                               title=f"TPE sediment load to ruts vs. {SWEEP_PARAM}")
        else:
            print("\nSkipped additional metric sweeps: this PARAM_SWEEP isn't a single "
                  "SWEEP_PARAM range (e.g. you're using the 'several different "
                  "parameters at once' style), so there's no single x-axis to plot against.")
        
    if RUN_COMPRESSION_THRESHOLD_GRID:
        # =========================================================================
        # TWO PARAMETER SWEEP
        # =========================================================================
        GRID_PARAM_1 = "compression"
        GRID_VALUES_1 = [1e-4, 4e-4, 7e-4, 1e-3]
 
        GRID_PARAM_2 = "truck_num_ini"
        GRID_VALUES_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 
        N_REALIZATIONS_PER_GRID_CELL = 20

        def n_realizations_for(v2):
            # more realizations inside the noisy threshold zone, fewer
            # outside it where the plateau is already well-behaved
            return 40 if 1 <= v2 <= 5 else 15
 
        grid_results = {}
        for v1 in GRID_VALUES_1:
            for v2 in GRID_VALUES_2:
                n = n_realizations_for(v2)
                runs = []
                for i in range(n):
                    res = run_realization(
                        seed=i, run_duration=SWEEP_RUN_DURATION,
                        save_plots=False, verbose=False, return_grid=False,
                        rainfall_source=SWEEP_RAINFALL_SOURCE,
                        **{GRID_PARAM_1: v1, GRID_PARAM_2: v2},
                    )
                    runs.append(res)
                grid_results[(v1, v2)] = runs
 
        # positivity check -- log scale needs strictly positive values
        all_grid_y = np.concatenate([
            np.array([r["total_road_mass"].sum() for r in grid_results[(v1, v2)]])
            for v1 in GRID_VALUES_1 for v2 in GRID_VALUES_2
        ])
        if all(v2 > 0 for v2 in GRID_VALUES_2) and all(all_grid_y > 0):
            fig, ax = plt.subplots(figsize=(9, 6))
            cmap = plt.get_cmap("viridis")
            threshold_estimates = {}
            for i1, v1 in enumerate(GRID_VALUES_1):
                means, sems = [], []
                for v2 in GRID_VALUES_2:
                    arr = np.array([r["total_road_mass"].sum() for r in grid_results[(v1, v2)]])
                    means.append(arr.mean())
                    sems.append(arr.std(ddof=1) / np.sqrt(len(arr)))  # standard error of the mean
                means = np.array(means)
                color = cmap(i1 / max(len(GRID_VALUES_1) - 1, 1))
                ax.errorbar(GRID_VALUES_2, means, yerr=sems, marker="o", color=color,
                            capsize=3, elinewidth=1, linestyle="none")
 
                # per-line power-law fit: mass = a * truck_num_ini^b
                log_x = np.log10(GRID_VALUES_2)
                log_y = np.log10(means)
                b_fit, log_a_fit = np.polyfit(log_x, log_y, 1)
                a_fit = 10 ** log_a_fit
                log_y_pred = b_fit * log_x + log_a_fit
                ss_res = np.sum((log_y - log_y_pred) ** 2)
                ss_tot = np.sum((log_y - log_y.mean()) ** 2)
                r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                x_line = np.array([min(GRID_VALUES_2), max(GRID_VALUES_2)])
                ax.plot(x_line, a_fit * x_line ** b_fit, color=color, linestyle="--",
                        label=f"{GRID_PARAM_1}={v1:.1e} (b={b_fit:.3g}, R²={r_sq:.3f})")
 
                diffs = np.diff(means)
                jump_idx = np.argmax(diffs)
                threshold_estimates[v1] = (GRID_VALUES_2[jump_idx] + GRID_VALUES_2[jump_idx + 1]) / 2
 
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"{GRID_PARAM_2} (log scale)")
            ax.set_ylabel("Mean total road mass [kg] (log scale, error bars = SEM)")
            ax.set_title(f"Sediment output vs. {GRID_PARAM_2}, by {GRID_PARAM_1} -- log-log")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.show()
        else:
            print(f"\nSkipped log-log grid plot: {GRID_PARAM_2} or total road mass includes "
                  f"zero/negative values, which can't be shown on a log scale.")
            threshold_estimates = {}
            for i1, v1 in enumerate(GRID_VALUES_1):
                means = [np.mean([r["total_road_mass"].sum() for r in grid_results[(v1, v2)]]) for v2 in GRID_VALUES_2]
                diffs = np.diff(means)
                jump_idx = np.argmax(diffs)
                threshold_estimates[v1] = (GRID_VALUES_2[jump_idx] + GRID_VALUES_2[jump_idx + 1]) / 2
 
        print(f"\n--- Estimated {GRID_PARAM_2} threshold, by {GRID_PARAM_1} ---")
        print(f"{'compression':>14}{'estimated threshold':>22}")
        for v1, thresh in threshold_estimates.items():
            print(f"{v1:>14.1e}{thresh:>22.2f}")
 
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
