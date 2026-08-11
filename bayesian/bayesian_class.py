"""
Bayesian long-term / short-term polarization component separation.

Separates a blazar's observed Stokes Q and U polarization time series into a
time-independent ("long-term") component and a time-variable ("short-term")
component, via a Metropolis-like Markov Chain Monte Carlo (MCMC) search for
the long-term Q and U values that best explain the observed fractional
polarization time series under a smoothness prior.

This is a source-agnostic component-separation approach: unlike
`components_addition_law` (the companion method in this repository), it does
not assume a specific quiescent-jet Stokes vector to subtract, but instead
infers the long-term component directly from the data.
"""

import numpy as np


class BayesianComponentSeparation:
    """Bayesian MCMC separation of a polarization time series into long-/short-term components.

    For `n_chains` independent MCMC chains, searches for the constant
    (long-term) Stokes Q and U values that maximize the posterior probability
    of the observed fractional polarization time series, under a likelihood
    based on the Z-scored fractional polarization residual and a
    random-walk-smoothness prior on the proposed Q/U trajectories. The
    final long-term estimate (and its uncertainty) is the mean (and std)
    of the `n_chains` chains' results.

    Parameters
    ----------
    Q, U, I : np.ndarray
        Observed Stokes Q, U (normalized by total flux) and total intensity
        (here, fractional polarization P = sqrt(Q^2+U^2)/I-like quantity;
        see `run_bayesian_separation.py` for how these are built from
        polarization degree/angle measurements). All the same length.
    n_chains : int
        Number of independent MCMC chains to run; the final long-term
        estimate is the mean across chains, with uncertainty given by their
        spread (std).
    num_samples : int
        Number of accepted MCMC steps to run per chain.
    smoothness_scale : float, default 0.5
        Width `w` of the random-walk-smoothness prior on proposed Q/U
        trajectories (see `PI`).

    Attributes
    ----------
    long_term : list[float]
        Best-fit long-term [Q, U, I] values (mean across chains).
    long_term_std : list[float]
        Uncertainty on `long_term` (std across chains).
    """

    def __init__(self, Q, U, I, n_chains, num_samples, smoothness_scale: float = 0.5):
        self.Q = Q
        self.U = U
        self.I = I

        self.smoothness_scale = smoothness_scale
        self.n_chains = n_chains
        self.num_samples = num_samples

        self.Q_score = self.z_score(self.Q)
        self.U_score = self.z_score(self.U)
        self.I_score = self.z_score(self.I)

        self.x = np.array([self.Q, self.U, self.I])

        chain_results = []
        for chain_index in range(1, self.n_chains + 1):
            print(f"Running MCMC chain {chain_index}/{self.n_chains}")
            initial_state = self._initial_state()
            final_sample, *_ = self.mcmc(initial_state, self.num_samples, self.x)
            chain_results.append(
                np.array([np.mean(final_sample[0]), np.mean(final_sample[1]), np.mean(final_sample[2])])
            )

        chain_results = np.array(chain_results)  # shape (n_chains, 3): columns are Q, U, I
        self.long_term = [chain_results[:, i].mean() for i in range(3)]
        self.long_term_std = [chain_results[:, i].std() for i in range(3)]

    def _initial_state(self) -> np.ndarray:
        """Draw the initial (constant-valued) long-term Q/U proposal for one chain.

        Returns
        -------
        np.ndarray, shape (2, len(I))
            Row 0 is a constant array (repeated scalar) drawn from
            Normal(mean(Q), 1); row 1 is the equivalent for U. Proposals are
            kept as constant-valued arrays (rather than scalars) so that the
            random-walk-smoothness prior `PI`, which operates on consecutive
            differences of an array, can be applied uniformly.
        """
        q0 = np.random.normal(np.mean(self.Q))
        u0 = np.random.normal(np.mean(self.U))
        return np.array([[q0] * len(self.I), [u0] * len(self.I)])

    def z_score(self, values: np.ndarray) -> np.ndarray:
        """Standardize an array to zero mean and unit standard deviation."""
        return (values - np.mean(values)) / np.std(values)

    def log_likelihood_ratio_terms(self, proposal: np.ndarray) -> np.ndarray:
        """Compute the per-point likelihood terms for a proposed long-term Q/U.

        The model's fractional-polarization residual,
        `sqrt((Q - Q_long)^2 + (U - U_long)^2)`, is Z-scored and compared to
        the Z-scored total-intensity proxy `I_score` under a Gaussian
        likelihood with data-driven variance.

        Parameters
        ----------
        proposal : np.ndarray, shape (2, len(I))
            Proposed constant-valued long-term [Q, U].

        Returns
        -------
        np.ndarray
            Per-point Gaussian likelihood values.
        """
        residual_amplitude = np.sqrt(
            np.power(self.x[0] - proposal[0], 2) + np.power(self.x[1] - proposal[1], 2)
        )
        residual_score = self.z_score(residual_amplitude)
        sigma = np.std(residual_score)
        return np.exp(-np.power(self.I_score - residual_score, 2) / (2 * sigma**2)) / np.sqrt(
            2 * np.pi * sigma**2
        )

    def likelihood(self, proposal: np.ndarray, x: np.ndarray) -> float:
        """Total likelihood of a proposed long-term Q/U (product over all data points).

        Parameters
        ----------
        proposal : np.ndarray, shape (2, len(I))
            Proposed constant-valued long-term [Q, U].
        x : np.ndarray
            Unused directly (the likelihood is evaluated against `self.x`);
            kept as a parameter for interface symmetry with `posterior`.

        Returns
        -------
        float
            Product of the per-point likelihood terms.
        """
        return np.prod(self.log_likelihood_ratio_terms(proposal))

    def smoothness_prior(self, trajectory: np.ndarray) -> float:
        """Random-walk-smoothness prior: penalizes large consecutive-point jumps.

        Parameters
        ----------
        trajectory : np.ndarray
            A proposed (constant-valued, in this class) Q or U trajectory.

        Returns
        -------
        float
            Product of per-step Gaussian prior densities on
            `trajectory[i+1] - trajectory[i]`. Since proposals here are
            always constant-valued arrays, every consecutive difference is
            zero and this term is a fixed constant; it becomes non-trivial
            if this class is reused with genuinely time-varying proposals.
        """
        step = trajectory[1:] - trajectory[:-1]
        density = np.exp(-np.power(step, 2) / (2 * self.smoothness_scale**2)) / (
            2 * np.pi * self.smoothness_scale**2
        )
        return np.prod(density)

    def posterior(self, proposal: np.ndarray, x: np.ndarray, normalization: float = 1e-55) -> float:
        """Unnormalized posterior probability of a proposed long-term Q/U.

        Parameters
        ----------
        proposal : np.ndarray, shape (2, len(I))
            Proposed constant-valued long-term [Q, U].
        x : np.ndarray
            Passed through to `likelihood` (see its docstring).
        normalization : float, default 1e-55
            Arbitrary scale factor keeping the posterior values in a
            numerically convenient range for the MCMC acceptance test below
            (the absolute scale does not affect the accept/reject decision,
            which only compares consecutive posterior values).

        Returns
        -------
        float
            `likelihood * smoothness_prior(Q) * smoothness_prior(U) / normalization`.
        """
        return (
            self.likelihood(proposal, self.x)
            * self.smoothness_prior(proposal[0])
            * self.smoothness_prior(proposal[1])
            / normalization
        )

    def mcmc(self, initial_state: np.ndarray, num_samples: int, x: np.ndarray):
        """Run a single Metropolis-like MCMC chain for the long-term Q/U.

        At each step, a new candidate is drawn as
        `Normal(mean(current_Q), 1)` / `Normal(mean(current_U), 1)`
        (again as constant-valued arrays), and accepted if either its
        posterior is at least as large as the current state's, or the
        posterior ratio exceeds a threshold `u` drawn from
        `Uniform(0.92, 1)`. Note this acceptance rule is a variant of the
        standard Metropolis-Hastings criterion (which draws `u` from
        `Uniform(0, 1)`): using `Uniform(0.92, 1)` here makes the chain
        reject most posterior-decreasing proposals unless the decrease is
        very small, so it behaves more like a greedy/simulated-annealing
        search than a chain that samples the full posterior. This was kept
        as in the original implementation rather than "corrected" to a
        standard MH criterion, since it changes the sampler's behavior and
        the intended sampling regime isn't fully clear from the code alone.

        Parameters
        ----------
        initial_state : np.ndarray, shape (2, len(I))
            Starting constant-valued [Q, U] proposal (see `_initial_state`).
        num_samples : int
            Number of *accepted* steps to run before stopping.
        x : np.ndarray
            Passed through to `posterior` (see its docstring).

        Returns
        -------
        final_sample : list[np.ndarray]
            `[Q_final, U_final, I_final]`, where `I_final = sqrt(Q_final^2 + U_final^2)`,
            each a constant-valued array of length `len(I)`.
        posterior_history : list[float]
            Posterior value at each *accepted* step (including the initial state).
        n_accepted : int
            Total number of accepted steps (including re-acceptances counted
            during rejection retries; see note in the loop below — this
            matches the original implementation's bookkeeping).
        snapshots : list
            `[step, Q, U]` recorded every 300 accepted steps (and at the
            first and last step), for diagnostic/trace purposes.
        """
        n_accepted = 0
        chain = [initial_state]
        posterior_history = [self.posterior(initial_state, x)]
        snapshots = []

        rejected_in_a_row = 0
        step = 1
        while step <= num_samples:
            current_q_mean = np.mean(chain[step - 1][0])
            current_u_mean = np.mean(chain[step - 1][1])
            candidate = np.array([[np.random.normal(current_q_mean)] * len(self.I), [np.random.normal(current_u_mean)] * len(self.I)])
            chain.append(candidate)
            posterior_history.append(self.posterior(chain[-1], x))

            acceptance_threshold = np.random.uniform(0.92, 1)

            improved = posterior_history[-1] >= posterior_history[-2]
            annealed_accept = (posterior_history[-1] / posterior_history[-2]) > acceptance_threshold

            if improved or annealed_accept:
                rejected_in_a_row = 0
                n_accepted += 1

                if step % 300 == 0 or step == num_samples or step == 1:
                    snapshots.append([step, chain[-1][0], chain[-1][1]])
                    print(f"posterior: {posterior_history[step]}\titeration: {step}")

                step += 1
            else:
                del chain[-1]
                del posterior_history[-1]
                rejected_in_a_row += 1

                if rejected_in_a_row % 10000 == 0:
                    print(f"consecutive rejections: {rejected_in_a_row}\tsmoothness_scale={self.smoothness_scale}")

        chain = np.array(chain)
        q_final, u_final = chain[-1][0], chain[-1][1]
        i_final = np.sqrt(q_final**2 + u_final**2)
        final_sample = [q_final, u_final, i_final]
        return final_sample, posterior_history, n_accepted, snapshots
