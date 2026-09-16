"""
Runs the main 100-realization Poisson-rainfall Monte Carlo ensemble.
Requires oft_model_core.py in the same folder (imports run_realization()
and the plotting/analysis helpers from it).
"""
#%%
from oft_model_core import *

#%%
# ==========================================================================
# MONTE CARLO ENSEMBLE -- run the model 100 times with different rainfall
# realizations, to see how the Poisson stochastic model affects results
# ==========================================================================
if __name__ == "__main__":
    N_RUNS = 10
    RUN_DURATION = 120

    # =========================================================================
    # PARAMETER OVERRIDE -- to test a different parameter value, just add it
    # here. Anything you DON'T list automatically uses its default (baseline)
    # value from run_realization's signature -- you only ever need to set the
    # one thing you're changing.
    #
    # Examples:
    #   PARAM_OVERRIDES = {}                        # baseline, nothing changed
    #   PARAM_OVERRIDES = {"porosity_c": 0.5}       # test one parameter
    #   PARAM_OVERRIDES = {"truck_num_ini": 10}     # test a different one
    # =========================================================================
    PARAM_OVERRIDES = {}

    results = []
    for run_idx in range(N_RUNS):
        print(f"=== Realization {run_idx + 1}/{N_RUNS} (seed={run_idx}) ===")
        res = run_realization(
            seed=run_idx,
            run_duration=RUN_DURATION,
            save_plots=False,          # keep off for the ensemble -- see run_realization docstring
            verbose=False,             # quiet per-day prints during the ensemble
            return_grid=False,         # keep off to save memory across 100 runs
            rainfall_source="poisson", # must be "poisson" for realizations to actually differ
            **PARAM_OVERRIDES,
        )
        results.append(res)

    # ------ aggregate across the ensemble ------
    # PRIMARY metric: total_road_mass is built directly from the per-storm
    # OFT water-transport fluxes, so it's guaranteed to respond to rainfall
    # variability (unlike sediment__pumped, which is dominated by the daily
    # truck-pass erosion process and barely changes between realizations,
    # since truck traffic is identical -- not seed-dependent -- every run).
    road_mass_totals = np.array([r["total_road_mass"].sum() for r in results])
    print(f"\nTotal road mass (water-transported sediment) across {N_RUNS} realizations:")
    print(f"  mean = {road_mass_totals.mean():.2f} kg, std = {road_mass_totals.std():.2f} kg")
    print(f"  range = [{road_mass_totals.min():.2f}, {road_mass_totals.max():.2f}] kg")

    total_rainfall = np.array([
        np.sum(np.multiply(r["intensity_run_dur"], np.multiply(r["dt"], 24))) for r in results
    ])
    print(f"\nTotal rainfall across {N_RUNS} realizations:")
    print(f"  mean = {total_rainfall.mean():.2f} mm, std = {total_rainfall.std():.2f} mm")

    # histogram of the primary (rainfall-sensitive) metric
    fig, ax = plt.subplots()
    safe_hist(ax, road_mass_totals, bins=20, color="steelblue",
              label=f"Sediment output across {N_RUNS} Poisson-rainfall realizations",
              xlabel="Total road mass, water-transported [kg]")
    ax.set_ylabel("Number of realizations")
    plt.show()

    # sediment output vs. total rainfall -- does more rain -> more sediment?
    plt.figure()
    plt.scatter(total_rainfall, road_mass_totals, alpha=0.6)
    plt.xlabel("Total rainfall [mm]")
    plt.ylabel("Total road mass, water-transported [kg]")
    plt.title("Sediment output vs. total rainfall, across realizations")
    plt.show()

    # spaghetti plot: every run's cumulative road mass change, plus the mean
    plt.figure()
    for r in results:
        plt.plot(range(RUN_DURATION), r["cum_road_mass_change_oft"], color="gray", alpha=0.15)
    mean_curve = np.mean([r["cum_road_mass_change_oft"] for r in results], axis=0)
    plt.plot(range(RUN_DURATION), mean_curve, color="red", linewidth=2, label="Mean")
    plt.xlabel("Day")
    plt.ylabel("Cumulative road mass change [kg]")
    plt.legend()
    plt.title(f"Cumulative road mass change across {N_RUNS} realizations")
    plt.show()

    # ------ what about the rainfall actually drives sediment output? ------
    # Total rainfall alone may not explain all the scatter above -- storm
    # frequency and peak intensity could each matter independently of total
    # volume. This decomposes that.
    n_storm_days = np.array([np.sum(r["intensity_run_dur"] > 0) for r in results])
    max_intensity = np.array([r["intensity_run_dur"].max() for r in results])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(n_storm_days, road_mass_totals, alpha=0.6)
    ax[0].set_xlabel("Number of storm days")
    ax[0].set_ylabel("Total road mass [kg]")
    ax[0].set_title("Sediment vs. storm frequency")
    ax[1].scatter(max_intensity, road_mass_totals, alpha=0.6)
    ax[1].set_xlabel("Max single-day intensity [mm/hr]")
    ax[1].set_ylabel("Total road mass [kg]")
    ax[1].set_title("Sediment vs. peak intensity")
    plt.tight_layout()
    plt.show()

    # ------ secondary sediment metrics: printed summary, not plotted ------
    # fine_sediment_pumped/scattered/added may or may not actually respond
    # to rainfall (sediment__pumped in particular is driven mostly by daily
    # truck traffic, not rain -- see prior discussion). A histogram of a
    # metric that doesn't vary is uninformative even once it no longer
    # crashes, so these are reported as a table instead.
    pumped_totals = np.array([r["fine_sediment_pumped_kg"] for r in results])
    scattered_totals = np.array([r["fine_sediment_scattered_kg"] for r in results])
    added_totals = np.array([r["sediment_added_kg"] for r in results])

    print(f"\n--- Sediment metric comparison across {N_RUNS} realizations ---")
    print(f"{'Metric':<40}{'Mean':>12}{'Std':>12}{'Responds to rain?':>20}")
    for label, arr in [
        ("Total road mass (water-transported)", road_mass_totals),
        ("Fine sediment pumped (TPE)", pumped_totals),
        ("Fine sediment scattered", scattered_totals),
        ("Sediment added (active layer)", added_totals),
    ]:
        responds = "Yes" if (arr.max() - arr.min()) > 1e-6 * max(abs(arr.mean()), 1e-12) else "No"
        print(f"{label:<40}{arr.mean():>12.4f}{arr.std():>12.6f}{responds:>20}")

#%%
    # ------ optional: run + fully plot ONE detailed realization ------
detailed = run_realization(seed=0, run_duration=RUN_DURATION,
                             save_plots=True, verbose=True, return_grid=True,
                             rainfall_source="poisson")
plot_full_diagnostics(detailed)
#%%
    # ------ optional diagnostic: isolate rainfall vs. grid-noise effects ------
    # By default (as run above), seed controls BOTH rainfall and grid noise
    # together, so "varies across the ensemble" doesn't tell you WHICH one a
    # metric is actually responding to. Run these two mini-ensembles to find
    # out -- compare the std of any metric (e.g. fine_sediment_scattered_kg)
    # between them.
    #
results_rain_only = [
     run_realization(seed=i, rain_seed=i, grid_seed=0,   # grid noise FIXED
                      run_duration=RUN_DURATION, save_plots=False, verbose=False)
     for i in range(N_RUNS)
 ]
scattered_rain_only = np.array([r["fine_sediment_scattered_kg"] for r in results_rain_only])
print("Scattered sediment, rainfall varying / grid fixed:  std =", scattered_rain_only.std())

results_grid_only = [
     run_realization(seed=i, rain_seed=0, grid_seed=i,   # rainfall FIXED
                      run_duration=RUN_DURATION, save_plots=False, verbose=False)
     for i in range(N_RUNS)
 ]
scattered_grid_only = np.array([r["fine_sediment_scattered_kg"] for r in results_grid_only])
print("Scattered sediment, grid noise varying / rainfall fixed:  std =", scattered_grid_only.std())
    #
    # Whichever of the two stds is much larger identifies the actual driver.


