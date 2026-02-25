# Third-Party Licenses

The Copernican Suite relies on the following runtime dependencies. Their
versions and license texts are shipped under [`licenses/`](licenses/) so users
can review the full terms offline.

| Dependency | Version | License |
|------------|---------|---------|
| numpy | 1.26.4 | [BSD-3-Clause][] |
| scipy | 1.12.0 | [BSD-3-Clause][] |
| matplotlib | 3.8.2 | [Matplotlib License][] |
| pandas | 2.2.1 | [BSD-3-Clause][] |
| sympy | 1.13.0 | [BSD-3-Clause][] |
| jsonschema | 4.21.1 | [MIT][] |
| camb | 1.6.3 | [LGPL-3.0-or-later][] |
| colorama | 0.4.6 | [BSD-3-Clause][] |
| build | 1.3.0 | [MIT][] |
| click | 8.3.0 | [BSD-3-Clause][] |
| PyYAML | 6.0.1 | [MIT][] |
| astropy | 6.0.0 | [BSD-3-Clause][] |
| psutil | 5.9.8 | [BSD-3-Clause][] |
| setuptools | 69.5.1 | [MIT][] |
| semver | 3.0.1 | [MIT][] |
| emcee | 3.1.4 | [MIT][] |
| h5netcdf | 1.3.0 | [BSD-3-Clause][] |
| h5py | 3.10.0 | [BSD-3-Clause][] |
| iniconfig | 2.1.0 | [MIT][] |
| xarray | 2023.12.0 | [Apache-2.0][] |
| xarray-einstats | 0.6.0 | [Apache-2.0][] |
| packaging | 24.2 | [Apache-2.0][] |
| typing_extensions | 4.10.0 | [MIT][] |
| tqdm | 4.66.5 | [MIT][] |
| pip | 24.2 | [MIT][] |
| pyproject-hooks | 1.2.0 | [MIT][] |
| arviz | 0.16.1 | [Apache-2.0][] |
| contourpy | 1.2.0 | [BSD-3-Clause][] |
| cycler | 0.12.1 | [BSD-3-Clause][] |
| fonttools | 4.51.0 | [MIT][] |
| kiwisolver | 1.4.5 | [BSD-3-Clause][] |
| pillow | 10.3.0 | [HPND][] |
| pluggy | 1.5.0 | [MIT][] |
| pyparsing | 3.1.1 | [MIT][] |
| pytest | 8.2.2 | [MIT][] |
| wheel | 0.43.0 | [MIT][] |
| python-dateutil | 2.9.0.post0 | [BSD-3-Clause][] |
| six | 1.16.0 | [MIT][] |
| pytz | 2024.1 | [MIT][] |
| tzdata | 2024.1 | [MIT][] |
| mpmath | 1.3.0 | [BSD-3-Clause][] |
| attrs | 23.2.0 | [MIT][] |
| jsonschema-specifications | 2023.12.1 | [MIT][] |
| referencing | 0.34.0 | [MIT][] |
| rpds-py | 0.18.0 | [MIT][] |
| pyerfa | 2.0.1.1 | [BSD-3-Clause][] |
| pythonmonkey | 1.3.0 | [MIT][] |
| tkinterweb | vendored | [MIT][] |
| tkinterweb_tkhtml | vendored | [MIT][] |
| astropy-iers-data | 0.2024.10.28.0.34.7 | [BSD-3-Clause][] |

[BSD-3-Clause]: licenses/BSD-3-Clause.txt
[Matplotlib License]: licenses/Matplotlib.txt
[MIT]: licenses/MIT.txt
[LGPL-3.0-or-later]: licenses/LGPL-3.0-or-later.txt
[Apache-2.0]: licenses/Apache-2.0.txt
[HPND]: licenses/HPND.txt

The optional ``dev`` extra installs ``pip-tools==7.4.1`` for contributors who
regenerate ``requirements.lock`` locally.  Because it is not required to run
the suite, it is excluded from the runtime license table above.

## Dataset licenses

- **Union3 (Union Through UNITY 2000 SNe sample)** —
  [MIT][union3-mit]

### Notes on camb (LGPL-3.0-or-later)

The CAMB library is licensed under the GNU Lesser General Public License
version 3 or any later version. This grants you the right to modify CAMB and
relink the Copernican Suite against your modified version. If you distribute
the Suite with a modified CAMB, you must make the CAMB source available and
retain the original notices. See
[`licenses/LGPL-3.0-or-later.txt`](licenses/LGPL-3.0-or-later.txt) for the
complete terms.

[union3-mit]: licenses/Union3-MIT.txt
