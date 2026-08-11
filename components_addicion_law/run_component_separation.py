"""
Constant/variable polarized component separation via the two-component
addition law, for a blazar observed over several consecutive nights.

For each night (a "day bin"), extracts the variable polarized component via
the Hagen & Thorn (1988) Q/U-vs-I linear regression (`hagen_torn.py`), then
fits the constant (quiescent) component's polarization degree and angle by
minimizing, via a bootstrap of SLSQP fits, the discrepancy between the
addition-law-combined polarization (`addition_law.py`) and the night's
observed polarization degree/angle.

Usage
-----
    python run_component_separation.py

Expects a whitespace-delimited data file (see `DATA_FILE` below) with
columns: #MJD, filter, I, Ierr, p, p_err, theta, theta_err.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize

from addition_law import AdditionLaw
from hagen_torn import HagenTorn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_NAME = "PKS 2155-304"
DATA_FILE = "data_pks2155304_complete.txt"

# MJD boundaries splitting the observations into night-by-night bins:
# bin 0 is MJD <= DAY_BOUNDARIES[0], bin i is DAY_BOUNDARIES[i-1] <= MJD <=
# DAY_BOUNDARIES[i] (bins overlap by one point at each shared boundary,
# matching the original implementation), and the last bin is
# MJD >= DAY_BOUNDARIES[-1].
DAY_BOUNDARIES = [713, 714, 715, 716, 717]

# Bounds on the constant component's polarization degree (%) and position
# angle (deg), used both as SLSQP constraints and as the initial-guess prior;
# adjust these for a different source / expected quiescent polarization.
CONSTANT_P_BOUNDS = (2.4, 8.0)
CONSTANT_THETA_BOUNDS = (92.0, 145.0)
INITIAL_GUESS_P = (4.0, 1.0)  # (mean, std) of the Normal draw for each bootstrap fit
INITIAL_GUESS_THETA = (130.0, 1.0)

N_BOOTSTRAP_FITS_PER_DAY = 500  # number of random-initial-guess SLSQP fits averaged per day

OUTPUT_PLOT_PATH = f"{SOURCE_NAME.lower().replace(' ', '_').replace('-', '')}_addition_law_separation.png"


def load_data(filepath: str) -> pd.DataFrame:
    """Load the photopolarimetric time series from a whitespace-delimited file.

    Uses a regex whitespace separator (`\\s+`) rather than a literal
    two-space separator, so this is robust to files with inconsistent
    single/double/tab spacing between columns.

    Parameters
    ----------
    filepath : str
        Path to the data file (columns: #MJD, filter, I, Ierr, p, p_err,
        theta, theta_err).

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(filepath, sep=r"\s+")


def split_into_day_bins(data: pd.DataFrame, day_boundaries: list[float]) -> list[pd.DataFrame]:
    """Split the data into night-by-night bins at the given MJD boundaries.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset, with an `#MJD` column.
    day_boundaries : list[float]
        MJD boundaries; see `DAY_BOUNDARIES` above for the binning convention.

    Returns
    -------
    list[pd.DataFrame]
        `len(day_boundaries) + 1` day bins.
    """
    bins = [data[data["#MJD"] <= day_boundaries[0]]]
    for lower, upper in zip(day_boundaries[:-1], day_boundaries[1:]):
        bins.append(data[(data["#MJD"] >= lower) & (data["#MJD"] <= upper)])
    bins.append(data[data["#MJD"] >= day_boundaries[-1]])
    return bins


def extract_variable_components(day_bins: list[pd.DataFrame]):
    """Extract each night's variable polarized component via Hagen & Thorn regression.

    Parameters
    ----------
    day_bins : list[pd.DataFrame]
        Per-night data, as returned by `split_into_day_bins`.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays (one entry per night) under keys "p_var", "theta_var",
        "I_var", "p_var_err", "theta_var_err", "I_var_err", "I_cons",
        "I_cons_err", and "time_mean" (the night's mean MJD).
    """
    results = {key: [] for key in ["p_var", "theta_var", "I_var", "p_var_err", "theta_var_err", "I_var_err", "I_cons", "I_cons_err", "time_mean"]}

    for day in day_bins:
        pol_degree = day["p"].to_numpy(dtype=float)
        pol_angle = day["theta"].to_numpy(dtype=float)
        total_intensity = day["I"].to_numpy(dtype=float)
        pol_degree_err = day["p_err"].to_numpy(dtype=float)
        pol_angle_err = day["theta_err"].to_numpy(dtype=float)
        total_intensity_err = day["Ierr"].to_numpy(dtype=float)

        # NOTE: this pre-multiplies total intensity by the polarization
        # fraction before passing it to HagenTorn, which (for
        # is_stokes_params=False) multiplies by the polarization degree
        # again internally to build Stokes Q/U. The net effect is a
        # polarization-degree-squared factor in the derived Stokes
        # parameters, rather than the single factor the standard Hagen &
        # Thorn method calls for. This matches the original implementation;
        # flagged here in case it's worth revisiting against the total
        # (non-polarization-weighted) intensity column instead.
        polarized_intensity = pol_degree * total_intensity / 100

        finite = (
            np.isfinite(pol_degree) & np.isfinite(pol_angle) & np.isfinite(polarized_intensity)
            & np.isfinite(pol_degree_err) & np.isfinite(pol_angle_err) & np.isfinite(total_intensity_err)
        )
        pol_degree, pol_angle, polarized_intensity = pol_degree[finite], pol_angle[finite], polarized_intensity[finite]
        pol_degree_err, pol_angle_err, total_intensity_err = pol_degree_err[finite], pol_angle_err[finite], total_intensity_err[finite]

        night_fit = HagenTorn(
            pol_degree, pol_angle, polarized_intensity, pol_degree_err, pol_angle_err, total_intensity_err,
            is_stokes_params=False,
        )

        results["p_var"].append(night_fit.p_var)
        results["p_var_err"].append(night_fit.p_var_err)
        # The Hagen & Thorn angle convention here is offset by 90 degrees
        # from the addition-law/observed convention used below; the +90
        # reconciles them, matching the original implementation.
        results["theta_var"].append(night_fit.theta_var + 90)
        results["theta_var_err"].append(night_fit.theta_var_err)
        results["I_var"].append(night_fit.I_var)
        results["I_var_err"].append(night_fit.I_var_err)
        results["I_cons"].append(night_fit.I_cons)
        results["I_cons_err"].append(night_fit.I_cons_err)
        results["time_mean"].append(day["#MJD"].to_numpy(dtype=float).mean())

    return {key: np.array(values) for key, values in results.items()}


def fit_constant_component(day_bins: list[pd.DataFrame], variable: dict, n_bootstrap: int = N_BOOTSTRAP_FITS_PER_DAY):
    """Fit each night's constant-component polarization degree and angle.

    For each night, runs `n_bootstrap` independent SLSQP fits (each started
    from an independently drawn random initial guess) minimizing the squared,
    error-weighted discrepancy between the addition-law-combined polarization
    (constant + that night's variable component) and the night's observed
    degree/angle. The night's constant-component estimate is the mean (and
    std, as the uncertainty) across the bootstrap fits.

    Parameters
    ----------
    day_bins : list[pd.DataFrame]
        Per-night data, as returned by `split_into_day_bins`.
    variable : dict
        Output of `extract_variable_components`.
    n_bootstrap : int, default 500
        Number of bootstrap SLSQP fits per night.

    Returns
    -------
    p_cons, p_cons_err, theta_cons, theta_cons_err : np.ndarray
        Constant component's polarization degree (%) and position angle
        (deg), one value per night, with bootstrap uncertainties.
    """
    constraints = (
        {"type": "ineq", "fun": lambda x: CONSTANT_P_BOUNDS[1] - x[0]},
        {"type": "ineq", "fun": lambda x: x[0] - CONSTANT_P_BOUNDS[0]},
        {"type": "ineq", "fun": lambda x: CONSTANT_THETA_BOUNDS[1] - x[1]},
        {"type": "ineq", "fun": lambda x: x[1] - CONSTANT_THETA_BOUNDS[0]},
    )

    p_cons, p_cons_err = [], []
    theta_cons, theta_cons_err = [], []

    for i, day in enumerate(day_bins):
        param_var = [variable["p_var"][i], variable["theta_var"][i], variable["I_var"][i]]
        weights_p = 100**2 / np.array(variable["p_var_err"][i]) ** 2
        weights_theta = 1 / np.array(variable["theta_var_err"][i]) ** 2

        observed_theta = day["theta"].to_numpy(dtype=float)
        observed_p = day["p"].to_numpy(dtype=float)

        def objective(x, i=i, param_var=param_var, weights_p=weights_p, weights_theta=weights_theta, observed_theta=observed_theta, observed_p=observed_p):
            combined = AdditionLaw([x[0], x[1], variable["I_cons"][i]], param_var, is_stokes_params=False)
            error_theta = np.sum(((combined.theta_tot - observed_theta) ** 2) * weights_theta)
            error_p = np.sum(((combined.p_tot - observed_p) ** 2) * weights_p)
            return error_theta + error_p

        night_p_fits, night_theta_fits = [], []
        for _ in range(n_bootstrap):
            x0 = [np.random.normal(*INITIAL_GUESS_P), np.random.normal(*INITIAL_GUESS_THETA)]
            result = minimize(objective, x0, method="SLSQP", constraints=constraints)
            night_p_fits.append(result.x[0])
            night_theta_fits.append(result.x[1])

        p_cons.append(np.average(night_p_fits))
        p_cons_err.append(np.std(night_p_fits))
        theta_cons.append(np.average(night_theta_fits))
        theta_cons_err.append(np.std(night_theta_fits))

    return np.array(p_cons), np.array(p_cons_err), np.array(theta_cons), np.array(theta_cons_err)


def make_spline(x_points: np.ndarray, y_points: np.ndarray):
    """Build a cubic-spline interpolator through `(x_points, y_points)`.

    Parameters
    ----------
    x_points, y_points : np.ndarray

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function evaluating the spline at new x values.
    """
    tck = interpolate.splrep(x_points, y_points)
    return lambda x: interpolate.splev(x, tck)


def plot_separation(time_all, p_all, p_err_all, theta_all, theta_err_all, time_mean,
                     p_result, p_result_err, theta_result, theta_result_err,
                     time_mean_spline_x,
                     p_cons_spline, theta_cons_spline, p_var_spline, theta_var_spline,
                     output_path: str):
    """Plot the observed polarization alongside the fitted component decomposition.

    Parameters
    ----------
    time_all, p_all, p_err_all, theta_all, theta_err_all : np.ndarray
        Full (unbinned) observed time, polarization degree, and angle series.
    time_mean, p_result, p_result_err, theta_result, theta_result_err : np.ndarray
        Per-night addition-law-combined ("result") polarization degree/angle.
    time_mean_spline_x : np.ndarray
        Dense time grid the spline functions below are evaluated on.
    p_cons_spline, theta_cons_spline, p_var_spline, theta_var_spline : dict
        Each a dict with keys "center", "lower", "upper" holding spline
        callables (see `make_spline`) for the constant/variable component
        degree/angle and their uncertainty band.
    output_path : str
        Path the figure is saved to.
    """
    fig, axs = plt.subplots(2, sharex=True)
    fig.subplots_adjust(hspace=0)

    x_spline = time_mean_spline_x

    axs[0].errorbar(time_all, p_all, yerr=p_err_all, fmt="o", label="data", color="blue", markersize=3, capsize=3)
    axs[0].errorbar(time_mean, p_result, yerr=p_result_err, fmt="o", color="black", capsize=5, label="result")
    axs[0].plot(x_spline, p_cons_spline["center"](x_spline), "--", label="constant component", linewidth=1.2, color="black")
    axs[0].fill_between(x_spline, p_cons_spline["lower"](x_spline), p_cons_spline["upper"](x_spline), alpha=0.2)
    axs[0].plot(x_spline, p_var_spline["center"](x_spline), "-", label="variable component", linewidth=1, color="black")
    axs[0].fill_between(x_spline, p_var_spline["lower"](x_spline), p_var_spline["upper"](x_spline), alpha=0.2)
    axs[0].set_ylabel(r"polarization ($\%$)")
    axs[0].grid(linestyle="--", linewidth=0.5)
    axs[0].legend(loc="best", ncol=2)
    axs[0].set_title(f"Component separation (addition law) - {SOURCE_NAME}")

    axs[1].errorbar(time_all, theta_all, yerr=theta_err_all, fmt="o", label="data", color="blue", markersize=3, capsize=3)
    axs[1].errorbar(time_mean, theta_result, yerr=theta_result_err, fmt="o", color="black", capsize=5, label="result")
    axs[1].plot(x_spline, theta_cons_spline["center"](x_spline), "--", label="constant component", linewidth=1.2, color="black")
    axs[1].fill_between(x_spline, theta_cons_spline["lower"](x_spline), theta_cons_spline["upper"](x_spline), alpha=0.2)
    axs[1].plot(x_spline, theta_var_spline["center"](x_spline), "-", label="variable component", linewidth=1, color="black")
    axs[1].fill_between(x_spline, theta_var_spline["lower"](x_spline), theta_var_spline["upper"](x_spline), alpha=0.2)
    axs[1].set_ylabel("position angle (degree)")
    axs[1].grid(linestyle="--", linewidth=0.5)
    axs[1].set_xlabel("time (days)")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main() -> None:
    data = load_data(DATA_FILE)
    day_bins = split_into_day_bins(data, DAY_BOUNDARIES)

    variable = extract_variable_components(day_bins)
    p_cons, p_cons_err, theta_cons, theta_cons_err = fit_constant_component(day_bins, variable)

    # NOTE: the final addition-law combination must use each night's own
    # variable-component values (arrays, one per night) alongside the
    # per-night constant-component arrays just fit above. In the original
    # implementation, this step reused a loop variable that, by the time the
    # day-loop had finished, held only the *last* night's variable-component
    # values — silently combining every night's constant component with the
    # last night's variable component instead of its own. Fixed here by
    # building the full-array variable-component tuple explicitly.
    variable_all_nights = [variable["p_var"], variable["theta_var"], variable["I_var"]]
    combined = AdditionLaw([p_cons, theta_cons, variable["I_cons"]], variable_all_nights, is_stokes_params=False)

    p_result = combined.p_tot
    p_result_err = combined.p_total_err(
        p_cons, variable["p_var"], theta_cons, variable["theta_var"], variable["I_cons"], variable["I_var"],
        p_cons_err, variable["p_var_err"], theta_cons_err, variable["theta_var_err"], variable["I_cons_err"], variable["I_var_err"],
    )
    theta_result = combined.theta_tot
    theta_result_err = combined.theta_total_err(
        p_cons / 100, variable["p_var"] / 100, theta_cons, variable["theta_var"], variable["I_cons"], variable["I_var"],
        p_cons_err / 100, variable["p_var_err"] / 100, theta_cons_err, variable["theta_var_err"], variable["I_cons_err"], variable["I_var_err"],
    )

    print("p_cons:      ", p_cons, p_cons_err)
    print("p_var:       ", variable["p_var"], variable["p_var_err"])
    print("\ntheta_cons:  ", theta_cons, theta_cons_err)
    print("theta_var:   ", variable["theta_var"], variable["theta_var_err"])
    print("\nI_cons:      ", variable["I_cons"], variable["I_cons_err"])
    print("I_var:       ", variable["I_var"], variable["I_var_err"])
    print("\np_total:     ", p_result, p_result_err)
    print("theta_total: ", theta_result, theta_result_err)

    time_mean = variable["time_mean"]
    p_cons_spline = {
        "center": make_spline(time_mean, p_cons),
        "lower": make_spline(time_mean, p_cons - p_cons_err),
        "upper": make_spline(time_mean, p_cons + p_cons_err),
    }
    theta_cons_spline = {
        "center": make_spline(time_mean, theta_cons),
        "lower": make_spline(time_mean, theta_cons - theta_cons_err),
        "upper": make_spline(time_mean, theta_cons + theta_cons_err),
    }
    p_var_spline = {
        "center": make_spline(time_mean, variable["p_var"]),
        "lower": make_spline(time_mean, variable["p_var"] - variable["p_var_err"]),
        "upper": make_spline(time_mean, variable["p_var"] + variable["p_var_err"]),
    }
    theta_var_spline = {
        "center": make_spline(time_mean, variable["theta_var"]),
        "lower": make_spline(time_mean, variable["theta_var"] - variable["theta_var_err"]),
        "upper": make_spline(time_mean, variable["theta_var"] + variable["theta_var_err"]),
    }

    time_all = data["#MJD"].to_numpy()
    x_spline = np.linspace(time_all[0], time_all[-1])

    plot_separation(
        time_all, data["p"].to_numpy(), data["p_err"].to_numpy(), data["theta"].to_numpy(), data["theta_err"].to_numpy(),
        time_mean, p_result, p_result_err, theta_result, theta_result_err,
        x_spline, p_cons_spline, theta_cons_spline, p_var_spline, theta_var_spline,
        OUTPUT_PLOT_PATH,
    )


if __name__ == "__main__":
    main()
