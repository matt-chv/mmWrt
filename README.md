# mmWrt

minimal raytracing for MIMO FMCW radar systems.

Intended usage:
1. educational

## Release Notes and Roadmap

### Released

v0.1: 
    * point targets only
    * 1D compute of baseband if signal for scene
    * 1D FFT, CFAR, peak grouping and target position error compute
    * single reflections


### NEXT

v0.2:
    * point targets only with RCS
    * 2D (AoA)
    * velocity
    * 2D FFT: range+velocity, range+AoA
    * 2D peak grouping (by velocity sign)
    * 3D position error compute
v0.3:
    * 3D targets (at least spheres)
    * medium attenuation
    * 3D point clouds (i.e. over multiple CTI)
    * multiple single reflections

Not planned yet bu considered:
* reads and loads .bin
* record BB signals in .bin
* 3D targets and scene rendering with imaging side by side radar
* Swerling's scatter

## Example Code

Check on Google Colab the code:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/matt-chv/bdd8b835c5cb7e739bb8b68d00257690/fmcw-radar-101.ipynb)


## Release process

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

7.run sphinx-api 
`updates the *.rst in docs/ folder`

> sphinx-apidoc -f -o docs mmWrt

8. run spinx-build 
(updates the read_the_docs folder)

> sphinx-build -b html docs read_the_docs

9. release to pypi-test

> python setup.py bdist_wheel
> twine upload -r testpypi dist\*

10. check on read_the_docs

11. check on Google Colab
(Google Colab requires py3.8 as off 2023-Jan-15)

12. release on pypi
> twine upload -r pypi dist\*



