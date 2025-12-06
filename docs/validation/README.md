# Engine validation playbook

This folder stores lightweight validation runs that exercise the ensemble MCMC
and nested-sampling engines against public ΛCDM benchmarks. The current
notebook-style test focuses on the Pantheon+SH0ES 2022 supernova sample and the
BOSS DR12 BAO consensus distances. To keep runtime manageable, the validation
slices the first 40 Pantheon+SH0ES entries, drops the full covariance matrix and
relies on the published diagonal distance-modulus errors while using the full
BAO covariance. The trimmed set is still drawn from the public release but runs
quickly enough for developers to compare engines interactively.

## How to run
1. Ensure the managed environment is active (launch via `start.sh` on Unix-like
   systems). The script relies on the repository version of CAMB and emcee
   already shipped in the lockfile.
2. Execute `python docs/validation/lcdm_engine_validation.py` from the
   repository root. The runner disables multiprocessing internally so it can be
   executed safely from notebooks or test harnesses.
3. Review the printed summary to confirm each engine’s posterior means and χ²
   contributions stay within the expected tolerances.

## Reference parameter set
The validation pins both engines to Planck 2018’s base-ΛCDM best-fit parameters
(expressed in the model YAML variable names):

- `H_0 = 67.66`
- `Omega_m0 = 0.3111`
- `Omega_b = 0.04897`
- `Omega_gamma = 5.38e-5`
- `z_rec = 1089.92`
- `Neff = 3.044` (note: most sample models now allow `Neff` to float between 2.5 and
  3.5 so this validation run keeps it at the Planck value while the sampler can
  explore small departures later).

Evaluating the trimmed Pantheon+SH0ES sample and the full BOSS DR12 BAO table at
this point yields these reference χ² contributions:

- Supernovae: **45.30**
- BAO: **7.26**
- Total: **52.56**

CMB data remain disabled in this check because the CAMB-dependent spectrum
calculation would dominate runtime in routine documentation builds.

## Tolerances and discrepancy guidance
Short diagnostic runs introduce Monte Carlo noise, so the validation allows the
following absolute deviations:

- Posterior means: ±4.0 for `H_0`, ±0.04 for `Omega_m0`, ±0.01 for `Omega_b`,
  ±2.0e-5 for `Omega_gamma` and ±20 for `z_rec`.
- χ² contributions: ±6.0 for the supernova slice, ±1.5 for BAO and ±7.0 for the
  combined total.

Exceeding these thresholds signals a meaningful change in either the likelihood
implementation or the engine behaviour. Because the supernova covariance is
approximated by its diagonal for speed, supernova χ² values may sit below the
full-covariance fit; the thresholds above absorb that systematic offset. Document
any excursions beyond the tolerances in this file alongside the observed values
and the commit that introduced them.
