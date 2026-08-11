# Blazar Polarimetry

Two independent methods for separating a blazar's observed optical
polarization into a constant ("quiescent jet") component and a
time-variable component, applied to real optical polarimetric monitoring
data (OJ 287 and PKS 2155-304).

```
blazar-polarimetry/
├── bayesian/
│   ├── bayesian_class.py            # MCMC long-term/short-term separation
│   ├── run_bayesian_separation.py   # Example: OJ 287
│   └── data_OJ287.txt
├── components_addition_law/
│   ├── addition_law.py              # Two-component polarization vector addition
│   ├── hagen_torn.py                # Hagen & Thorn (1988) Q/U-vs-I regression
│   ├── run_component_separation.py  # Example: PKS 2155-304
│   └── data_pks2155304_complete.txt
├── requirements.txt
└── README.md
```

## Overview

Blazar optical polarization is generally understood as a superposition of a
stable, low-polarization "quiescent jet" component and a rapidly variable
component associated with shocks or turbulence in the jet. Separating the
two is useful for tracking the variable component's own behavior (e.g.
rotations of its position angle) independent of the underlying quiescent
polarization. This repository implements two different approaches to that
separation:

**`bayesian/`** — treats the long-term (constant) Stokes Q and U as unknown
parameters and searches for the values that best explain the *variability*
of the observed fractional polarization over the full time series, via a
Metropolis-like MCMC search with a smoothness prior. Does not require
binning the data by night; works directly on the full time series. See
`bayesian_class.py`'s module docstring for the likelihood/prior details.

**`components_addition_law/`** — a two-step, per-night approach:

1. **`hagen_torn.py`**: for each night, linearly regresses the observed
   Stokes Q and U against the total flux I; because only the variable
   component's flux (not its polarization) changes within a night, the
   regression slopes directly give the variable component's polarization
   degree and angle (Hagen & Thorn 1988).
2. **`addition_law.py`** + `run_component_separation.py`: fits the constant
   component's polarization degree and angle (via a bootstrap of SLSQP fits)
   to be the one that, when combined with that night's variable component
   through the standard two-component vector-addition formula, best
   reproduces the night's observed polarization.

Both methods are legitimate, independent ways of tackling the same
separation problem, useful as a cross-check of each other on data where
both can be applied.

## Data format

Both example scripts expect a plain-text time series of degree/angle
photopolarimetry:

**`bayesian/data_OJ287.txt`**: semicolon-delimited, no header, four columns:

```
time ; polarization_degree(%) ; polarization_angle(deg) ; fractional_polarization
```

**`components_addition_law/data_pks2155304_complete.txt`**: whitespace-
delimited, with a header row:

```
#MJD  filter  I  Ierr  p  p_err  theta  theta_err
```

## Usage

```bash
cd bayesian
python run_bayesian_separation.py

cd ../components_addition_law
python run_component_separation.py
```

Edit the configuration block at the top of each script (source name, data
file, number of MCMC chains/bootstrap fits, day-binning boundaries) for a
different source or dataset.


## References

- Hagen-Thorn, V. A. (1988). *Polarization variability of BL Lac objects.*
  Astrophysics, 29, 372.
- Bagnulo, S., et al. (2009). *Stellar Spectropolarimetry with Retarder
  Waveplate and Beam Splitter Devices.* PASP, 121, 993 (general reference for
  the two-component Stokes-parameter addition used in `addition_law.py`).
