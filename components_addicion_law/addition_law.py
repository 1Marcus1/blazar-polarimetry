"""
Two-component polarization vector addition ("addition law") for blazar optical
polarimetry.

Given a "constant" (quiescent jet) and a "variable" polarized component, each
described by a polarization degree, position angle, and intensity, computes
the polarization degree and position angle of their combined (observed) light,
following the standard vector-addition formulae for partially polarized light
(e.g. Bagnulo et al. 2009; the same formulae underlie the Stokes-parameter
addition Q_tot = Q1+Q2, U_tot = U1+U2 for intensity-weighted components).

This is the companion method to `hagen_torn.py`'s linear-regression-based
approach (`components_addition_law` vs. `hagen_torn` in this repository):
here, the variable component is assumed known, and the constant component is
fit (elsewhere, in `run_component_separation.py`) to best reproduce the
observed degree and angle once combined with it via this class.
"""

import numpy as np


class AdditionLaw:
    """Combine a constant and a variable polarized component into the observed polarization.

    Parameters
    ----------
    constant_component : sequence of 3 array-like
        `[p, theta, I]` for the constant (quiescent) component: polarization
        degree (%), position angle (deg), and intensity, unless
        `is_stokes_params=True` (see below).
    variable_component : sequence of 3 array-like
        `[p, theta, I]` for the variable component, same convention as
        `constant_component`.
    is_stokes_params : bool, default False
        If True, `constant_component`/`variable_component` are interpreted as
        `[Q, U, I]` Stokes parameters instead of `[p, theta, I]`, and are
        converted to polarization degree/angle internally before combining.
    normalize : bool, default False
        If True, Z-score standardize each of p, theta, and I (for both
        components) before combining. Off by default; only meaningful if the
        combined quantities are used in a relative/standardized context
        rather than as physical polarization degree/angle.

    Attributes
    ----------
    p_tot : np.ndarray
        Combined polarization degree (%).
    theta_tot : np.ndarray
        Combined polarization position angle (deg).
    """

    def __init__(self, constant_component, variable_component, is_stokes_params: bool = False, normalize: bool = False):
        self.p_cons, self.theta_cons, self.I_cons = constant_component
        self.p_var, self.theta_var, self.I_var = variable_component

        if not is_stokes_params:
            self.p_cons, self.p_var = np.array(self.p_cons), np.array(self.p_var)
            self.theta_cons, self.theta_var = np.array(self.theta_cons), np.array(self.theta_var)
            self.I_cons, self.I_var = np.array(self.I_cons), np.array(self.I_var)
        else:
            # NOTE: here `p_cons`/`p_var` and `theta_cons`/`theta_var` are
            # actually Stokes Q and U (the parameter names are reused from
            # the non-Stokes case above for interface consistency). This
            # branch is not currently exercised by `run_component_separation.py`
            # (always called with `is_stokes_params=False`); the per-point
            # loop below does not actually index its inputs by `i`; the
            # constants `p_cons`, `theta_cons` are re-appended `len(I_cons)`
            # times unchanged rather than as time series. Kept as in the
            # original implementation, since it isn't exercised elsewhere and
            # the intended per-point behavior isn't clear from the code
            # alone — worth revisiting before relying on this branch.
            p_obs_cons, p_obs_var = [], []
            theta_obs_cons, theta_obs_var = [], []
            for _ in range(len(self.I_cons)):
                p_obs_cons.append(np.sqrt(self.p_cons**2 + self.theta_cons**2) * 100 / self.I_cons)
                p_obs_var.append(np.sqrt(self.p_var**2 + self.theta_var**2) * 100 / self.I_var)
                theta_obs_cons.append(np.rad2deg(np.arctan(self.theta_cons / self.p_cons)) / 2.0)
                theta_obs_var.append(np.rad2deg(np.arctan(self.theta_var / self.p_var)) / 2.0)

            self.p_cons, self.p_var = np.array(p_obs_cons), np.array(p_obs_var)
            self.theta_cons, self.theta_var = np.array(theta_obs_cons), np.array(theta_obs_var)
            self.I_cons, self.I_var = np.array(self.I_cons), np.array(self.I_var)

        if normalize:
            self.p_cons, self.p_var = self.standardize(self.p_cons), self.standardize(self.p_var)
            self.theta_cons, self.theta_var = self.standardize(self.theta_cons), self.standardize(self.theta_var)
            self.I_cons, self.I_var = self.standardize(self.I_cons), self.standardize(self.I_var)

        self.p_tot = self.p_total(self.p_cons, self.p_var, self.theta_cons, self.theta_var, self.I_cons, self.I_var)
        self.theta_tot = self.theta_total(
            self.p_cons, self.p_var, self.theta_cons, self.theta_var, self.I_cons, self.I_var
        )

    def standardize(self, values: np.ndarray) -> np.ndarray:
        """Z-score standardize an array to zero mean and unit standard deviation."""
        return (values - np.mean(values)) / np.std(values)

    def p_total(self, p_cons, p_var, theta_cons, theta_var, I_cons, I_var):
        """Combined polarization degree (%) of the two components.

        Standard intensity-weighted vector-addition formula for two
        partially polarized light components.
        """
        return np.sqrt(
            ((p_cons * I_cons) ** 2 + (p_var * I_var) ** 2
             + 2 * p_cons * I_cons * p_var * I_var * np.cos(2 * np.deg2rad(theta_cons - theta_var)))
            / ((I_cons + I_var) ** 2)
        )

    def theta_total(self, p_cons, p_var, theta_cons, theta_var, I_cons, I_var):
        """Combined polarization position angle (deg) of the two components."""
        return (
            np.rad2deg(
                np.arctan(
                    (p_cons * I_cons * np.cos(2 * np.deg2rad(theta_cons)) + p_var * I_var * np.cos(2 * np.deg2rad(theta_var)))
                    / (p_cons * I_cons * np.sin(2 * np.deg2rad(theta_cons)) + p_var * I_var * np.sin(2 * np.deg2rad(theta_var)))
                )
            )
            / 2.0
            + 90
        )

    def p_total_err(self, p_cons, p_var, theta_cons, theta_var, I_cons, I_var,
                     sigma_p_cons, sigma_p_var, sigma_theta_cons, sigma_theta_var, sigma_I_cons, sigma_I_var):
        """Propagated 1-sigma uncertainty on `p_total`, via standard error propagation.

        Parameters mirror `p_total`, with a `sigma_*` uncertainty supplied
        for each. `sigma_theta_cons`/`sigma_theta_var` are expected in
        degrees (converted internally, matching `theta_cons`/`theta_var`).
        """
        theta_cons_rad = np.deg2rad(theta_cons)
        theta_var_rad = np.deg2rad(theta_var)
        sigma_theta_cons = np.deg2rad(sigma_theta_cons)
        sigma_theta_var = np.deg2rad(sigma_theta_var)

        f = (
            (p_cons * I_cons) ** 2 + (p_var * I_var) ** 2
            + 2 * p_cons * I_cons * p_var * I_var * np.cos(2 * (theta_cons_rad - theta_var_rad))
        ) / ((I_cons + I_var) ** 2)

        partial_p_cons = (2 * p_cons * I_cons**2 + 2 * I_cons * p_var * I_var * np.cos(2 * (theta_cons_rad - theta_var_rad))) / ((I_cons + I_var) ** 2)
        partial_p_var = (2 * p_var * I_var**2 + 2 * I_var * p_cons * I_cons * np.cos(2 * (theta_cons_rad - theta_var_rad))) / ((I_cons + I_var) ** 2)
        partial_theta_cons = (-4 * p_cons * I_cons * p_var * I_var * np.sin(2 * (theta_cons_rad - theta_var_rad))) / ((I_cons + I_var) ** 2)
        partial_theta_var = (4 * p_cons * I_cons * p_var * I_var * np.sin(2 * (theta_cons_rad - theta_var_rad))) / ((I_cons + I_var) ** 2)
        partial_I_cons = (2 * p_cons**2 * I_cons + 2 * p_cons * p_var * I_var * np.cos(2 * (theta_cons_rad - theta_var_rad)) - f * 2 * I_cons) / ((I_cons + I_var) ** 2)
        partial_I_var = (2 * p_var**2 * I_var + 2 * p_cons * p_var * I_cons * np.cos(2 * (theta_cons_rad - theta_var_rad)) - f * 2 * I_var) / ((I_cons + I_var) ** 2)

        sigma_f = np.sqrt(
            (partial_p_cons * sigma_p_cons) ** 2
            + (partial_p_var * sigma_p_var) ** 2
            + (partial_theta_cons * sigma_theta_cons) ** 2
            + (partial_theta_var * sigma_theta_var) ** 2
            + (partial_I_cons * sigma_I_cons) ** 2
            + (partial_I_var * sigma_I_var) ** 2
        )

        return (1 / (2 * np.sqrt(f))) * sigma_f

    def theta_total_err(self, p_cons, p_var, theta_cons, theta_var, I_cons, I_var,
                         sigma_p_cons, sigma_p_var, sigma_theta_cons, sigma_theta_var, sigma_I_cons, sigma_I_var):
        """Propagated 1-sigma uncertainty on `theta_total` (returned in degrees).

        NOTE: `run_component_separation.py` calls this with `p_cons`/`p_var`
        (and their uncertainties) divided by 100 relative to the values used
        for `p_total_err`, converting the polarization degree from percent to
        a unitless fraction. This matches the original implementation; the
        angle formula's partial derivatives are scale-sensitive to `p`, so
        this conversion affects the resulting uncertainty magnitude.
        """
        theta_cons_rad = np.deg2rad(theta_cons)
        theta_var_rad = np.deg2rad(theta_var)
        sigma_theta_cons = np.deg2rad(sigma_theta_cons)
        sigma_theta_var = np.deg2rad(sigma_theta_var)

        numerator = p_cons * I_cons * np.cos(2 * theta_cons_rad) + p_var * I_var * np.cos(2 * theta_var_rad)
        denominator = p_cons * I_cons * np.sin(2 * theta_cons_rad) + p_var * I_var * np.sin(2 * theta_var_rad)

        partial_p_cons = (I_cons * np.cos(2 * theta_cons_rad) * denominator - I_cons * np.sin(2 * theta_cons_rad) * numerator) / (denominator**2 + numerator**2)
        partial_p_var = (I_var * np.cos(2 * theta_var_rad) * denominator - I_var * np.sin(2 * theta_var_rad) * numerator) / (denominator**2 + numerator**2)
        partial_theta_cons = (-2 * p_cons * I_cons * np.sin(2 * theta_cons_rad) * denominator + 2 * p_cons * I_cons * np.cos(2 * theta_cons_rad) * numerator) / (denominator**2 + numerator**2)
        partial_theta_var = (-2 * p_var * I_var * np.sin(2 * theta_var_rad) * denominator + 2 * p_var * I_var * np.cos(2 * theta_var_rad) * numerator) / (denominator**2 + numerator**2)
        partial_I_cons = (p_cons * np.cos(2 * theta_cons_rad) * denominator - p_cons * np.sin(2 * theta_cons_rad) * numerator) / (denominator**2 + numerator**2)
        partial_I_var = (p_var * np.cos(2 * theta_var_rad) * denominator - p_var * np.sin(2 * theta_var_rad) * numerator) / (denominator**2 + numerator**2)

        sigma_theta_total = 0.5 * np.sqrt(
            (partial_p_cons * sigma_p_cons) ** 2
            + (partial_p_var * sigma_p_var) ** 2
            + (partial_theta_cons * sigma_theta_cons) ** 2
            + (partial_theta_var * sigma_theta_var) ** 2
            + (partial_I_cons * sigma_I_cons) ** 2
            + (partial_I_var * sigma_I_var) ** 2
        )

        return np.rad2deg(sigma_theta_total)
