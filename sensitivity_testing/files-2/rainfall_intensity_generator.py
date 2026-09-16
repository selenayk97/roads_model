#%%
import csv
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%%
## Define variables ##
START_DATE = datetime(2026, 1, 1)     # simulation start date
TOTAL_PERIOD_DAYS = 120                # total simulation length, in days
DT_MINUTES = 60                       # time step: 1440 = daily, 60 = hourly, 15 = 15-min
OUTPUT_NAME = "rainfall_scenario"     # base filename for the plot

# Each scenario = one CSV file + one line plot.
# intensity_mm_hr(I)          constant rainfall intensity DURING each storm
# intensity_step_mm_hr        how much intensity increases with each subsequent
#                             storm (0 = every storm the same)
# storm_duration_days(Tr)     how long a single storm event lasts, in days
# num_storms(V)               number of storm events, evenly spaced across the
#                             full TOTAL_PERIOD_DAYS
SCENARIOS = [
    {"name": "Case Low", "intensity_mm_hr": 0.5, "intensity_step_mm_hr": 0, "storm_duration_days": 0.3, "num_storms": 40},
     {"name": "Case High", "intensity_mm_hr": 11, "intensity_step_mm_hr": 0, "storm_duration_days": 0.3, "num_storms": 40},
     #{"name": "Case High", "intensity_mm_hr": 7, "intensity_step_mm_hr": 0, "storm_duration_days": 0.35, "num_storms": 40},
]

#%%
## Defines functions ##
def build_series(intensity_mm_hr, intensity_step_mm_hr, storm_duration_days, num_storms, total_period_days, dt_minutes, start_date):
    """
    Evenly space `num_storms` storms (each `storm_duration_days` long, at a
    constant `intensity_mm_hr`) across `total_period_days`. Built at fine
    (1-min) resolution first, then averaged down to dt_minutes -- keeps
    short storms accurate even if dt is coarser than the storm itself.
    """
    FINE = 1  # minutes
    fine_steps = int(total_period_days * 24 * 60 / FINE)
    fine = [0.0] * fine_steps

    storm_len = max(1, round(storm_duration_days * 24 * 60 / FINE))
    spacing_days = total_period_days / num_storms if num_storms else total_period_days
    starts = [round((i + 0.5) * spacing_days * 24 * 60 / FINE) for i in range(num_storms)]

    for i, start in enumerate(starts):
            this_intensity = intensity_mm_hr + i * intensity_step_mm_hr
            for j in range(storm_len):
                if start + j < fine_steps:
                    fine[start + j] = this_intensity

    bin_size = max(1, round(dt_minutes / FINE))
    n_bins = fine_steps // bin_size
    intensity = [sum(fine[b * bin_size:(b + 1) * bin_size]) / bin_size for b in range(n_bins)]
    days = [start_date + timedelta(minutes=dt_minutes * i) for i in range(n_bins)]
    return days, intensity


def write_csv(name, days, intensity, dt_minutes):
    path = f"{name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Day", "dt (hr)", "Intensity (mm/hr)"])  #change dt (min or hr) accordingly 
        for d, val in zip(days, intensity):
            writer.writerow([d.strftime("%Y-%m-%d %H:%M"), dt_minutes, round(val, 4)])
    print(f"Saved {path}")
    return path


def main():
    all_series = {}
    for sc in SCENARIOS:
        days, intensity = build_series(
            sc["intensity_mm_hr"], sc.get("intensity_step_mm_hr", 0),sc["storm_duration_days"], sc["num_storms"],
            TOTAL_PERIOD_DAYS, DT_MINUTES, START_DATE
        )
        write_csv(sc["name"], days, intensity, DT_MINUTES)
        all_series[sc["name"]] = (days, intensity)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, (name, (days, intensity)) in enumerate(all_series.items()):
        ax.plot(days, intensity, label=name, linewidth=1)
    ax.set_title("Rainfall Intensity — All Scenarios")
    ax.set_ylabel("Intensity (mm/hr)")
    ax.set_xlabel("Day")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_NAME}.png", dpi=130)
    plt.close(fig)
    print(f"Saved {OUTPUT_NAME}.png")

#%%
## Runs functions ##
if __name__ == "__main__":
    main()

#%%
rain_depth_low = 1.6 * 25.4 #inch to mm/month
rain_depth_high = 37.3 * 25.4 
sim_duration = 4 #months

print(f"Low case total rainfall depth: {(rain_depth_low * sim_duration):.2f} mm")
print(f"High case total rainfall depth: {(rain_depth_high * sim_duration):.2f} mm")
# %%
def solve_intensity_from_depth(total_depth_mm, num_storms, storm_duration_hours):
    """Given a target total depth, and your assumed storm count/duration,
    solve for the constant intensity each storm needs to hold."""
    return total_depth_mm / (num_storms * storm_duration_hours)

# Low case
intensity_low = solve_intensity_from_depth(162.6, num_storms=40, storm_duration_hours=8)
# High case
intensity_high = solve_intensity_from_depth(3789.7, num_storms=40, storm_duration_hours=8)

print(f"Low case intensity: {intensity_low:.2f} mm/hr")
print(f"High case intensity: {intensity_high:.2f} mm/hr")