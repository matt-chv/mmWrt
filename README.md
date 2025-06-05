# mmWrt

minimal raytracing for MIMO FMCW radar systems.

Intended usage:
1. educational & experimental

simple install:
1. git clone
2. pip install .

for install for developpers:
1. git clone
2. pip install .[dev]

## Release Notes and Roadmap

### Released

*v0.0.10:

    * adding distance as a function of time in IF compute - allowing for simulation of target changing bin ranges through single chirp
    * 

v0.0.9:

    * adding ULA, URA for TDM MIMO
    * adding 3D points for TDM MIMO (x,y,z)

v0.0.8:

    * adding TDM MIMO
    * adding DDM MIMO (including Doppler desambiguation)

v0.0.7

    * adding AoA
    * adding sparse array initial example (feature dedicated to Amine L.)

v0.0.6:

    * added micro-doppler
    * added non-regression on .ipynb in docs/ folder

v0.0.5:

    * moved dependancies from requirements to setup.py
    * added extras [dev] for developpers (and documentation and read the docs)
    * moved version checking from setup.py to test_basic.py
    * added readthedocs.yaml

v0.0.4:

    * adding frequency estimator
    * added speed processing
    * added support for radar equation (RCS, distance, ...)
    * antenna gains in azimumth, elevation and freq

v0.0.3: first release 

    * point targets only
    * 1D compute of baseband if signal for scene
    * 1D FFT, CFAR, peak grouping and target position error compute
    * single reflections

### NEXT ()

    * 2D (AoA)
    * 2D FFT: range+velocity, range+AoA
    * 2D peak grouping (by velocity sign)
    * 3D position error compute
    * 3D targets (at least spheres)
    * medium attenuation
    * 3D point clouds (i.e. over multiple CTI)
    * multiple single reflections

Not planned yet but considered:

* reads and loads .bin from DCA1000
  * record BB signals in .bin
  * 3D targets and scene rendering with imaging side by side radar
  * Swerling's scattering

## Example Code

Check on Google Colab the code:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/matt-chv/bdd8b835c5cb7e739bb8b68d00257690/fmcw-radar-101.ipynb)

Or Read the Docs on [![Read the docs](https://read-the-docs-guidelines.readthedocs-hosted.com/_images/logo-wordmark-light.png)](https://mmwrt.readthedocs.io/en/latest/)

## On dependencies

* jupyterlab-myst: used for displaying admonition in jupyter notebook (and thus documentation on read the docs)

## Release process

0. Ensure all the new .ipynb are added in docs/Hands-on.md 

1. run pyroma
(should be 10/10)

> pyroma .

2. run flake8 
runs with darglint settings for docstrings to numpy standard set in the .flake8 file
should yield 0 warnings or errors

> flake8

3. run pytest
should yield 100% pass

> pytest

4. run coverage

> coverage run -m pytest

5. run coverage report
(should be 100%)

> coverage report

6. run tox

> tox

7.run sphinx-api 
`updates the *.rst in docs/ folder`

> sphinx-apidoc -f -o docs mmWrt

8. run sphinx-build
(updates the read_the_docs folder)

> sphinx-build -b html docs build/html

9. release the Release-Candidate to pypi-test
check that __init__.py is "0.0.X-pre.Y" for RC numbering

> python setup.py bdist_wheel

> twine upload -r testpypi dist\*

10. check on Google Colab
(Google Colab requires py3.8 as off 2023-Jan-15)

if testing release-candidate need to add `--pre -U` or will install latest stable version. 

```
!python -m pip install -i https://test.pypi.org/simple/ --pre -U mmWrt
from mmWrt import __version__
print(__version__)
```

11. merge dev branch with main

> git checkout main
> git merge dev_branch_name

12. update the version to final
update  __init__.py to remove the suffix -pre.Y "0.0.X-pre.Y"

13. release on pypi (assumes your pypirc is local to the project)

> twine upload -r pypi --config-file=.\.pypirc dist\*

14. check on colab that pypi package works:

>!python -m pip install mmWrt
from mmWrt import __version__
print(__version__)

15. check updates on read_the_docs

> push to git to trigger readthedocs build:
> git push
> navigate to https://readthedocs.org/projects/mmwrt/builds/
> ensure build is successful

16. (optional) add tag for release

> git tag -a v0.0.X -m "version comment"
> git push origin v0.0.X

17. then check on google colab dev extras instals works

>!python -m pip install mmWrt[dev]
