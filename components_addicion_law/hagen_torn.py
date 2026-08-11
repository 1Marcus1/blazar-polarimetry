"""
Hagen & Thorn (1988)-style variable polarized component extraction.

Separates a blazar's polarized emission, observed over a short (intra-night)
time window, into a constant ("quiescent") component and a variable
component, by linearly regressing the Stokes Q and U flux densities against
the total flux density I. Under the assumption that only the variable
component's *flux* changes on this timescale (while its polarization degree
and angle stay fixed within the window), Q and U vary linearly with I, and
the regression slopes directly give the variable component's polarization
degree and angle (Hagen & Thorn 1988; Skiff et al.; sometimes called the
"Q-U vs I" or "QUI" method).

This is the companion method to `addition_law.py`'s explicit vector-addition
approach: here, the fit is a per-night linear regression, and only the
*variable* component's degree/angle come out directly; the constant
component's flux is estimated separately as the average residual intensity
(see `I_cons` below).
"""

import numpy as np


class HagenTorn:
    """Extract the variable polarized component from Stokes or (p, theta, I) data.

    Parameters
    ----------
    Q, U, I : array-like
        If `is_stokes_params=True`: observed Stokes Q, U (normalized by I)
        and total intensity I.
        If `is_stokes_params=False`: observed polarization degree (%),
        position angle (deg), and total intensity I — note the parameter
        names `Q`/`U` are reused for degree/angle in this case, for
        interface consistency with the Stokes-parameter case.
    Q_err, U_err, I_err : array-like
        Uncertainties on `Q`, `U`, `I` (or on degree/angle/I), same
        convention as above.
    is_stokes_params : bool, default True
        Selects the input convention; see `Q`, `U` above.

    Attributes
    ----------
    p_var, theta_var, I_var : float
        Variable component's polarization degree (%), position angle (deg),
        and (weighted-average) intensity.
    p_var_err, theta_var_err, I_var_err : float
        Uncertainties on the above.
    I_cons, I_cons_err : float
        Constant component's intensity (total average intensity minus the
        variable component's average intensity) and its uncertainty.
    """

    def __init__(self, Q, U, I, Q_err, U_err, I_err, is_stokes_params: bool = True):
        if is_stokes_params:
            self.Q, self.U, self.I = np.array(Q), np.array(U), np.array(I)
            self.Q_err, self.U_err, self.I_err = np.array(Q_err), np.array(U_err), np.array(I_err)
            self.p, self.p_err = self.polarization_degree(self.Q, self.U, self.I, self.Q_err, self.U_err, self.I_err)
            self.theta, self.theta_err = self.polarization_angle(self.Q, self.U, self.Q_err, self.U_err)
        else:
            # Here `Q`/`U` are actually polarization degree (%) / angle (deg);
            # convert to Stokes parameters (in the same flux units as `I`).
            degree, angle = Q, U
            Q_obs, Q_obs_err, U_obs, U_obs_err = [], [], [], []
            for i in range(len(degree)):
                angle_rad = np.deg2rad(2 * angle[i])
                Q_obs.append((degree[i] * I[i]) * np.cos(angle_rad) / 100)
                Q_obs_err.append(
                    np.sqrt(
                        ((Q_err[i] * I[i]) * np.cos(angle_rad) / 100) ** 2
                        + ((degree[i] * I_err[i]) * np.cos(angle_rad) / 100) ** 2
                        + ((2 * U_err[i] * degree[i] * I[i]) * np.sin(angle_rad) / 100) ** 2
                    )
                )
                U_obs.append((degree[i] * I[i]) * np.sin(angle_rad) / 100)
                U_obs_err.append(
                    np.sqrt(
                        ((Q_err[i] * I[i]) * np.sin(angle_rad) / 100) ** 2
                        + ((degree[i] * I_err[i]) * np.sin(angle_rad) / 100) ** 2
                        + ((2 * U_err[i] * degree[i] * I[i]) * np.cos(angle_rad) / 100) ** 2
                    )
                )

            self.Q, self.U, self.I = np.array(Q_obs), np.array(U_obs), np.array(I)
            self.Q_err, self.U_err, self.I_err = np.array(Q_obs_err), np.array(U_obs_err), np.array(I_err)
            self.p, self.p_err = np.array(degree), np.array(Q_err)
            self.theta, self.theta_err = np.array(angle), np.array(U_err)

        self.p_var, self.theta_var, self.I1_var, self.p_var_err, self.theta_var_err, self.I1_var_err = self.variable(
            self.Q, self.U, self.I, self.Q_err, self.U_err, self.I_err
        )
        self.I_var = np.average(self.I1_var, weights=self.I1_var_err)
        self.I_var_err = self.I1_var_err.mean()

        self.I_cons = np.average(self.I, weights=self.I_err) - np.average(self.I1_var, weights=self.I1_var_err)
        self.I_cons_err = self.I_var_err

    def polarization_degree(self, Q, U, I, Q_err, U_err, I_err):
        """Polarization degree (%) and uncertainty from Stokes Q, U, I.

        Note: the propagated uncertainty here does not include a
        contribution from `I_err`, even though it's accepted as a parameter
        — this matches the original implementation.
        """
        p = np.sqrt((Q**2 + U**2) / (I**2))
        p_err = (1 / p) * np.sqrt((Q * Q_err) ** 2 + (U * U_err) ** 2)
        return p * 100, p_err * 100

    def polarization_angle(self, Q, U, Q_err, U_err):
        """Polarization position angle (deg) and uncertainty from Stokes Q, U.

        Uses `arctan(U/Q)` (not `arctan2`), so it does not resolve the
        quadrant on its own — matches the original implementation.
        """
        theta = np.rad2deg(np.arctan(U / Q))
        theta_err = np.rad2deg(0.5 / (1 + (U / Q) ** 2) * np.sqrt((U_err / Q) ** 2 + (U * Q_err / (Q**2)) ** 2))
        return theta, theta_err

    def standardize(self, values: np.ndarray) -> np.ndarray:
        """Z-score standardize an array to zero mean and unit standard deviation."""
        return (values - np.mean(values)) / np.std(values)

    def line(self, params, x):
        """Linear model `y = params[0] * x + params[1]`."""
        return params[0] * x + params[1]

    def variable(self, Q, U, I, Q_err, U_err, I_err):
        """Extract the variable component via weighted linear regression of Q, U against I.

        Fits `Q = slope_QI * I + intercept` and `U = slope_UI * I + intercept`
        (weights `1/Q_err`, `1/U_err`); the variable component's polarization
        degree and angle follow from the two slopes, exactly as they would
        from Stokes parameters.

        Two earlier approaches — orthogonal distance regression (`scipy.odr`)
        accounting for errors in both I and Q/U, and a bootstrap of
        `scipy.optimize.minimize` fits — were tried during development and
        are available in the git history; both gave very similar results to
        the `numpy.polyfit` approach used here, which is simpler and faster.

        Parameters
        ----------
        Q, U, I : np.ndarray
            Stokes Q, U, and total intensity I.
        Q_err, U_err, I_err : np.ndarray
            Uncertainties on `Q`, `U`, `I` (`I_err` is currently unused by
            this implementation, matching the original).

        Returns
        -------
        p_var, theta_var, I_var, p_var_err, theta_var_err, I_var_err : np.ndarray
            Variable component's polarization degree (%), angle (deg), and
            `I * p_var` "polarized intensity" proxy, each with its
            propagated uncertainty. `I_var`/`I_var_err` here are arrays (one
            value per input point, `I * p_var`); see `__init__` for how
            these are reduced to single representative values.
        """
        (slope_qi, _), cov_matrix_qi = np.polyfit(I, Q, 1, cov=True, w=1 / Q_err)
        (slope_ui, _), cov_matrix_ui = np.polyfit(I, U, 1, cov=True, w=1 / U_err)

        slope_qi_err = np.sqrt(cov_matrix_qi[0][0])
        slope_ui_err = np.sqrt(cov_matrix_ui[0][0])

        self.slope_qi, self.slope_ui = slope_qi, slope_ui

        p_var = np.sqrt(slope_qi**2 + slope_ui**2)
        theta_var = np.rad2deg(np.arctan(slope_ui / slope_qi) / 2.0)
        I_var = I * p_var

        p_var_err = (1 / p_var) * np.sqrt((slope_qi * slope_qi_err) ** 2 + (slope_ui * slope_ui_err) ** 2)
        theta_var_err = np.rad2deg(
            0.5 / (1 + (slope_ui / slope_qi) ** 2)
            * np.sqrt((slope_ui_err / slope_qi) ** 2 + (slope_ui * slope_qi_err / (slope_qi**2)) ** 2)
        )
        I_var_err = I * p_var_err

        return p_var * 100, theta_var, I_var, p_var_err * 100, theta_var_err, I_var_err
