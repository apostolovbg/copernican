# Third-Party Licenses

The Copernican Suite relies on the following runtime dependencies. Their
versions and license texts are shipped under [`licenses/`](licenses/) so
users can review the full terms offline.

| Dependency | Version | License |
|------------|---------|---------|
| numpy | 1.26.4 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| scipy | 1.12.0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| matplotlib | 3.8.2 | [Matplotlib License](licenses/Matplotlib.txt) |
| pandas | 2.2.1 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| sympy | 1.13.0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| jsonschema | 4.21.1 | [MIT](licenses/MIT.txt) |
| camb | 1.6.3 | [LGPL-3.0-or-later](licenses/LGPL-3.0-or-later.txt) |
| colorama | 0.4.6 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| build | 1.3.0 | [MIT](licenses/MIT.txt) |
| click | 8.3.0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| PyYAML | 6.0.1 | [MIT](licenses/MIT.txt) |
| astropy | 6.0.0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| psutil | 5.9.8 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| setuptools | 69.5.1 | [MIT](licenses/MIT.txt) |
| setuptools_scm | 8.0.4 | [MIT](licenses/MIT.txt) |
| emcee | 3.1.4 | [MIT](licenses/MIT.txt) |
| h5netcdf | 1.3.0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| h5py | 3.10.0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| iniconfig | 2.1.0 | [MIT](licenses/MIT.txt) |
| xarray | 2023.12.0 | [Apache-2.0](licenses/Apache-2.0.txt) |
| xarray-einstats | 0.6.0 | [Apache-2.0](licenses/Apache-2.0.txt) |
| packaging | 24.2 | [Apache-2.0](licenses/Apache-2.0.txt) |
| typing_extensions | 4.10.0 | [MIT](licenses/MIT.txt) |
| tqdm | 4.66.5 | [MIT](licenses/MIT.txt) |
| pip | 24.2 | [MIT](licenses/MIT.txt) |
| pyproject-hooks | 1.2.0 | [MIT](licenses/MIT.txt) |
| arviz | 0.16.1 | [Apache-2.0](licenses/Apache-2.0.txt) |
| contourpy | 1.2.0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| cycler | 0.12.1 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| fonttools | 4.51.0 | [MIT](licenses/MIT.txt) |
| kiwisolver | 1.4.5 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| pillow | 10.3.0 | [HPND](licenses/HPND.txt) |
| pluggy | 1.5.0 | [MIT](licenses/MIT.txt) |
| pyparsing | 3.1.1 | [MIT](licenses/MIT.txt) |
| pytest | 8.2.2 | [MIT](licenses/MIT.txt) |
| wheel | 0.43.0 | [MIT](licenses/MIT.txt) |
| python-dateutil | 2.9.0.post0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| six | 1.16.0 | [MIT](licenses/MIT.txt) |
| pytz | 2024.1 | [MIT](licenses/MIT.txt) |
| tzdata | 2024.1 | [MIT](licenses/MIT.txt) |
| mpmath | 1.3.0 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| attrs | 23.2.0 | [MIT](licenses/MIT.txt) |
| jsonschema-specifications | 2023.12.1 | [MIT](licenses/MIT.txt) |
| referencing | 0.34.0 | [MIT](licenses/MIT.txt) |
| rpds-py | 0.18.0 | [MIT](licenses/MIT.txt) |
| pyerfa | 2.0.1.1 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |
| tkinterweb | vendored | [MIT](licenses/MIT.txt) |
| tkinterweb_tkhtml | vendored | [MIT](licenses/MIT.txt) |
| astropy-iers-data | 0.2024.10.28.0.34.7 | [BSD-3-Clause](licenses/BSD-3-Clause.txt) |

The optional ``dev`` extra installs ``pip-tools==7.4.1`` for contributors who
regenerate ``requirements.lock`` locally.  Because it is not required to run
the suite, it is excluded from the runtime license table above.

## Dataset licenses

| Dataset | License |
|---------|---------|
| Union3 (Union Through UNITY 2000 SNe sample) | [MIT](licenses/Union3-MIT.txt) |

### Notes on camb (LGPL-3.0-or-later)

The CAMB library is licensed under the GNU Lesser General Public License
version 3 or any later version. This grants you the right to modify CAMB and
relink the Copernican Suite against your modified version. If you distribute
the Suite with a modified CAMB, you must make the CAMB source available and
retain the original notices. See
[`licenses/LGPL-3.0-or-later.txt`](licenses/LGPL-3.0-or-later.txt) for the
complete terms.
