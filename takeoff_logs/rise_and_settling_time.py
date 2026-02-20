import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Tuple


def _first_crossing_index(
    signal: NDArray[np.float64],
    threshold: float,
    rising: bool = True,
) -> Optional[int]:
    if rising:
        idx = np.where(signal >= threshold)[0]
    else:
        idx = np.where(signal <= threshold)[0]
    return int(idx[0]) if len(idx) > 0 else None


def calculate_metrics(
    data: ArrayLike,
    target: float,
    dt: float = 0.008,
    settling_band: float = 0.02,
) -> Tuple[float, float, float, float]:
    """Compute step-response metrics using standard control definitions.

    - Rise time: between first 10% and 90% crossings of step amplitude.
    - Settling time: first time after which response stays in ±band of target.
    - Overshoot (%): excursion beyond target / step amplitude.
    - Undershoot (%): shortfall from target / step amplitude.
    """
    y = np.asarray(data, dtype=float)
    y0 = float(y[0])
    r = float(target)
    step_amp = r - y0

    if np.isclose(step_amp, 0.0):
        return np.nan, np.nan, 0.0, 0.0

    # 10%-90% rise-time thresholds based on command amplitude
    y10 = y0 + 0.10 * step_amp
    y90 = y0 + 0.90 * step_amp
    rising_step = step_amp > 0

    i10 = _first_crossing_index(y, y10, rising=rising_step)
    i90 = _first_crossing_index(y, y90, rising=rising_step)
    if i10 is None or i90 is None or i90 < i10:
        rise_time = np.nan
    else:
        rise_time = (i90 - i10) * dt

    # Settling-time band around target (commonly 2%)
    band_abs = settling_band * abs(step_amp)
    error = np.abs(y - r)
    outside = np.where(error > band_abs)[0]
    if len(outside) == 0:
        settling_time = 0.0
    elif outside[-1] == len(y) - 1:
        settling_time = np.nan
    else:
        settling_time = (outside[-1] + 1) * dt

    # Overshoot / undershoot normalized by step amplitude.
    # Here undershoot is treated as "did not reach target" shortfall.
    if rising_step:
        peak = float(np.max(y))
        os_pct = max(0.0, (peak - r) / abs(step_amp) * 100.0)
        us_pct = max(0.0, (r - peak) / abs(step_amp) * 100.0)
    else:
        valley = float(np.min(y))
        os_pct = max(0.0, (r - valley) / abs(step_amp) * 100.0)
        us_pct = max(0.0, (valley - r) / abs(step_amp) * 100.0)

    return rise_time, settling_time, os_pct, us_pct


def calculate_steady_state_error(
    data: ArrayLike,
    target: float,
    window: int = 100,
) -> float:
    """Average absolute error to target over the last `window` samples."""
    y = np.asarray(data, dtype=float)
    n = min(window, len(y))
    tail = y[-n:]
    return float(np.mean(np.abs(target - tail)))


if __name__ == "__main__":
    lqr_df = pd.read_csv("lqr_1.csv")
    rvolmea_df = pd.read_csv("rvolmea_1.csv")
    piddob_df = pd.read_csv("piddob_1.csv")

    lqr_x, lqr_y, lqr_z = lqr_df["x"], lqr_df["y"], lqr_df["z"]
    rvolmea_x, rvolmea_y, rvolmea_z = rvolmea_df["x"], rvolmea_df["y"], rvolmea_df["z"]
    piddob_x, piddob_y, piddob_z = piddob_df["x"], piddob_df["y"], piddob_df["z"]

    target_z = 4.0
    z_data = np.asarray(lqr_z, dtype=float)
    z_data_rvolmea = np.asarray(rvolmea_z, dtype=float)
    z_data_piddob = np.asarray(piddob_z, dtype=float)

    # Calculate metrics for Z
    rise_time_s, settling_time_s, overshoot_pct, undershoot_pct = (
        calculate_metrics(z_data, target_z)
    )
    rise_time_s_rvolmea, settling_time_s_rvolmea, overshoot_pct_rvolmea, undershoot_pct_rvolmea = (
        calculate_metrics(z_data_rvolmea, target_z)
    )
    rise_time_s_piddob, settling_time_s_piddob, overshoot_pct_piddob, undershoot_pct_piddob = (
        calculate_metrics(z_data_piddob, target_z)
    )
    steady_state_error = calculate_steady_state_error(
        z_data,
        target_z,
        window=100,
    )
    steady_state_error_rvolmea = calculate_steady_state_error(
        z_data_rvolmea,
        target_z,
        window=100,
    )
    steady_state_error_piddob = calculate_steady_state_error(
        z_data_piddob,
        target_z,
        window=100,
    )
    print(f"Metrics for Z (Target: {target_z}):")
    print(f"Rise Time (10-90%): {rise_time_s:.3f} s")
    print(f"Settling Time (±2%): {settling_time_s:.3f} s")
    print(f"Overshoot: {overshoot_pct:.2f}%")
    print(f"Undershoot: {undershoot_pct:.2f}%")
    print(
        "Steady-State Error (last 100 steps, avg |target-z|): "
        f"{steady_state_error:.6f} m"
    )

    print("\nMetrics for Z (RVolMea):")
    print(f"Rise Time (10-90%): {rise_time_s_rvolmea:.3f} s")
    print(f"Settling Time (±2%): {settling_time_s_rvolmea:.3f} s")
    print(f"Overshoot: {overshoot_pct_rvolmea:.2f}%")
    print(f"Undershoot: {undershoot_pct_rvolmea:.2f}%")
    print(
        "Steady-State Error (last 100 steps, avg |target-z|): "
        f"{steady_state_error_rvolmea:.6f} m"
    )
    print("\nMetrics for Z (PID-DOB):")
    print(f"Rise Time (10-90%): {rise_time_s_piddob:.3f} s")
    print(f"Settling Time (±2%): {settling_time_s_piddob:.3f} s")
    print(f"Overshoot: {overshoot_pct_piddob:.2f}%")
    print(f"Undershoot: {undershoot_pct_piddob:.2f}%")
    print(
        "Steady-State Error (last 100 steps, avg |target-z|): "
        f"{steady_state_error_piddob:.6f} m"
    )
    # Plot response against target
    dt = 0.008
    t = np.arange(len(z_data)) * dt
    t2 = np.arange(len(z_data_rvolmea)) * dt
    t3 = np.arange(len(z_data_piddob)) * dt

    plt.figure(figsize=(10, 5))
    plt.plot(t, z_data, label="LQR", color="tab:blue", linewidth=2)
    plt.plot(t2, z_data_rvolmea, label="RVolMea",
             color="tab:orange", linewidth=2)
    plt.plot(t3, z_data_piddob, label="PID-DOB",
             color="tab:green", linewidth=2)
    plt.axhline(
        y=target_z,
        color="black",
        linestyle="-",
        linewidth=2,
        label="target",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude z [m]")
    plt.title("UAV Altitude Response vs Target")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
