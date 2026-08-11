"""
Bayesian long-term/short-term polarization component separation for a blazar.

Loads polarization degree (PD), angle (PA), and fractional polarization (PF)
time series, builds the corresponding Stokes Q/U/I-like quantities, runs
`BayesianComponentSeparation` (see `bayesian_class.py`) to infer the
long-term (time-independent) polarization component, derives the
short-term (variable) component by subtraction, and plots both the Stokes
parameters and the resulting polarization degree/angle decomposition.

Usage
-----
    python run_bayesian_separation.py

Expects a semicolon-delimited data file (see `DATA_FILE` below) with columns
time; polarization degree (%); polarization angle (deg); fractional
polarization.
"""

import matplotlib.pyplot as plt
import numpy as np

from bayesian_class import BayesianComponentSeparation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_NAME = "OJ287"
DATA_FILE = "data_OJ287.txt"

N_CHAINS = 10
NUM_SAMPLES_PER_CHAIN = 1000

STOKES_PLOT_PATH = f"{SOURCE_NAME.lower()}_bayesian_stokes.png"
PF_PA_PLOT_PATH = f"{SOURCE_NAME.lower()}_bayesian_separation.png"


def load_polarization_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load time, polarization degree, angle, and fractional polarization from disk.

    Also applies a simple correction for the EVPA (electric vector position
    angle) 180-degree ambiguity: angles below 90 degrees are shifted up by
    90 degrees. This is a coarse, dataset-specific fix (not a general
    n*180-degree unwrapping) — inherited as-is from the original
    implementation; revisit if applying this to a dataset with a different
    angle convention or range.

    Parameters
    ----------
    filepath : str
        Path to the semicolon-delimited data file (columns: time, PD, PA, PF).

    Returns
    -------
    time, pd, pa, pf : np.ndarray
        Time, polarization degree (%), polarization angle (deg, ambiguity-
        corrected), and fractional polarization.
    """
    data = np.loadtxt(filepath, delimiter=";")
    time, pd, pa, pf = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    pa = pa.copy()
    pa[pa < 90] += 90
    return time, pd, pa, pf


def build_stokes_parameters(pd: np.ndarray, pa: np.ndarray, pf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized Stokes-like Q, U, I from polarization degree/angle/fraction.

    Parameters
    ----------
    pd, pa, pf : np.ndarray
        Polarization degree (%), angle (deg), and fractional polarization.

    Returns
    -------
    Q, U, I : np.ndarray
        Q = PF * cos(2*PA), U = PF * sin(2*PA), I = PF / PD.
    """
    pa_rad = np.deg2rad(2 * pa)
    Q = pf * np.cos(pa_rad)
    U = pf * np.sin(pa_rad)
    I = pf / pd
    return Q, U, I


def polarization_degree_and_error(Q: np.ndarray, U: np.ndarray, Q_err: np.ndarray, U_err: np.ndarray, error_denominator=None):
    """Fractional polarization P = sqrt(Q^2+U^2) and its propagated uncertainty.

    Parameters
    ----------
    Q, U : np.ndarray
        Stokes Q, U (or a component thereof, e.g. the long-/short-term part).
    Q_err, U_err : np.ndarray
        Uncertainties on `Q` and `U`.
    error_denominator : np.ndarray or None, optional
        Denominator used in the error-propagation formula below. Defaults to
        `P = sqrt(Q**2+U**2)` itself (the standard/self-consistent choice).
        The original implementation's short-term error used the *long-term*
        `P` in the Q-error term but the *short-term* `P` in the U-error term
        — an inconsistency that looks unintentional. Pass the long-term `P`
        here to reproduce that exact original behavior if needed; the
        default reproduces the (more standard) self-consistent formula.

    Returns
    -------
    P, P_err : np.ndarray
        Fractional polarization and its 1-sigma uncertainty (standard error
        propagation, ignoring any Q-U covariance).
    """
    P = np.sqrt(Q**2 + U**2)
    denom = P if error_denominator is None else error_denominator
    P_err = np.sqrt((Q_err * Q / denom) ** 2 + (U_err * U / P) ** 2)
    return P, P_err


def polarization_angle_and_error(Q: np.ndarray, U: np.ndarray):
    """Polarization angle (deg) from Stokes Q, U, and its propagated uncertainty.

    Note this uses `arctan(U/Q)` (not `arctan2`), matching the original
    implementation; it does not resolve the quadrant on its own; the `+180`
    offset below is an empirical correction for this dataset/convention
    rather than a general-purpose EVPA calculation.

    Parameters
    ----------
    Q, U : np.ndarray
        Stokes Q, U (or a component thereof).

    Returns
    -------
    angle_deg, angle_err_deg : np.ndarray
        Polarization angle (deg) and its 1-sigma uncertainty.
    """
    angle_deg = np.rad2deg(np.arctan(U / Q) / 2.0) + 180.0

    d_dQ = (0.5 / Q) / (1 + (0.5 * U / Q) ** 2)
    d_dU = (0.5 * U / Q**2) / (1 + (0.5 * U / Q) ** 2)
    angle_err_deg = np.sqrt(d_dQ**2 + d_dU**2)
    return angle_deg, angle_err_deg


def plot_stokes_components(time, Q_obs, U_obs, long_term, long_term_std, short_term, short_term_std, output_path):
    """Plot observed Q/U alongside the fitted long-term (constant) and short-term components."""
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(time, Q_obs, "o", color="blue", label="data")
    ax[0].plot([time[0], time[-1]], [long_term[0], long_term[0]], "--", color="black", label="long term")
    ax[0].fill_between(
        [time[0], time[-1]], long_term[0] - long_term_std[0], long_term[0] + long_term_std[0], alpha=0.2
    )
    ax[0].plot(time, short_term[0], "-", color="black", label="short term")
    ax[0].fill_between(time, short_term[0] - short_term_std[0], short_term[0] + short_term_std[0], alpha=0.2)
    ax[0].set_ylabel("Q")
    ax[0].grid(linestyle="--", linewidth=0.6)
    ax[0].legend(loc="best", ncol=2)
    ax[0].set_title(f"Bayesian Component Separation - {SOURCE_NAME}")

    ax[1].plot(time, U_obs, "o", color="blue", label="data")
    ax[1].plot([time[0], time[-1]], [long_term[1], long_term[1]], "--", color="black", label="long term")
    ax[1].fill_between(
        [time[0], time[-1]], long_term[1] - long_term_std[1], long_term[1] + long_term_std[1], alpha=0.2
    )
    ax[1].plot(time, short_term[1], "-", color="black", label="short term")
    ax[1].fill_between(time, short_term[1] - short_term_std[1], short_term[1] + short_term_std[1], alpha=0.2)
    ax[1].set_ylabel("U")
    ax[1].grid(linestyle="--", linewidth=0.6)
    ax[1].set_xlabel("Time")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_pf_pa_decomposition(time, pf_obs, pa_obs, pf_l, pf_l_err, pf_s, pf_s_err, pa_l, pa_l_err, pa_s, pa_s_err, output_path):
    """Plot the observed fractional polarization and angle alongside their long-/short-term decomposition."""
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(time, pf_obs, "o", color="blue", label="data")
    ax[0].plot([time[0], time[-1]], [pf_l, pf_l], "--", color="black", label="long term")
    ax[0].fill_between([time[0], time[-1]], pf_l - pf_l_err, pf_l + pf_l_err, alpha=0.2)
    ax[0].plot(time, pf_s, "-", color="black", label="short term")
    ax[0].fill_between(time, pf_s - pf_s_err, pf_s + pf_s_err, alpha=0.2)
    ax[0].set_ylabel("PF")
    ax[0].grid(linestyle="--", linewidth=0.6)
    ax[0].legend(loc="best")

    ax[1].plot(time, pa_obs, "o", color="blue", label="data")
    ax[1].plot([time[0], time[-1]], [pa_l, pa_l], "--", color="black", label="long term")
    ax[1].fill_between([time[0], time[-1]], pa_l - pa_l_err, pa_l + pa_l_err, alpha=0.2)
    ax[1].plot(time, pa_s, "-", color="black", label="short term")
    ax[1].fill_between(time, pa_s - pa_s_err, pa_s + pa_s_err, alpha=0.2)
    ax[1].set_ylabel("PA (deg)")
    ax[1].grid(linestyle="--", linewidth=0.6)
    ax[1].set_xlabel("Time")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main() -> None:
    time, pd_obs, pa_obs, pf_obs = load_polarization_data(DATA_FILE)
    Q_obs, U_obs, I_obs = build_stokes_parameters(pd_obs, pa_obs, pf_obs)

    model = BayesianComponentSeparation(Q_obs, U_obs, I_obs, N_CHAINS, NUM_SAMPLES_PER_CHAIN)

    long_term = np.array(model.long_term)  # [Q, U, I], shape (3,)
    long_term_std = np.array(model.long_term_std)

    print("Long term:\n", long_term)
    print("\nLong term std:\n", long_term_std)

    # NOTE: `long_term` is shape (3,) while `[Q_obs, U_obs, I_obs]` stacks to
    # shape (3, N); reshape to a column vector so the subtraction broadcasts
    # correctly. Without this reshape, this line raises a ValueError - it did
    # in the original implementation, which the person copied `long_term`'s
    # printed values into a hardcoded, differently-shaped array as a
    # workaround rather than fixing the broadcast.
    short_term = np.array([Q_obs, U_obs, I_obs]) - long_term.reshape(-1, 1)
    short_term_std = long_term_std  # matches the original: short-term uncertainty is taken as the long-term fit's std

    pf_l, pf_l_err = polarization_degree_and_error(long_term[0], long_term[1], long_term_std[0], long_term_std[1])
    # Reproduces the original implementation's short-term error formula exactly,
    # including its use of the long-term pf_l (not pf_s) in the Q-error term —
    # see `polarization_degree_and_error`'s docstring for why this default
    # differs from the self-consistent formula used for pf_l_err above.
    pf_s, pf_s_err = polarization_degree_and_error(
        short_term[0], short_term[1], short_term_std[0], short_term_std[1], error_denominator=pf_l
    )

    pa_l, pa_l_err = polarization_angle_and_error(long_term[0], long_term[1])
    pa_s, pa_s_err = polarization_angle_and_error(short_term[0], short_term[1])

    print(f"PF long-term:  {pf_l:.4g} +/- {pf_l_err:.4g}")
    print(f"PA long-term:  {pa_l:.4g} +/- {pa_l_err:.4g} deg")

    plot_stokes_components(time, Q_obs, U_obs, long_term, long_term_std, short_term, short_term_std, STOKES_PLOT_PATH)
    plot_pf_pa_decomposition(
        time, pf_obs, pa_obs, pf_l, pf_l_err, pf_s, pf_s_err, pa_l, pa_l_err, pa_s, pa_s_err, PF_PA_PLOT_PATH
    )


if __name__ == "__main__":
    main()
