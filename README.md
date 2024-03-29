# mmWrt

minimal raytracing for MIMO FMCW radar systems.

Intended usage:
1. educational

## Release Notes and Roadmap

### Released

v0.0.3: first release 

    * point targets only
    * 1D compute of baseband if signal for scene
    * 1D FFT, CFAR, peak grouping and target position error compute
    * single reflections

v0.0.4:

    * adding frequency estimator
    * added speed processing
    * added support for radar equation (RCS, distance, ...)
    * antenna gains in azimumth, elevation and freq

v0.0.5:

    * moved dependancies from requirements to setup.py
    * added extras [dev] for developpers (and documentation and read the docs)
    * moved version checking from setup.py to test_basic.py
    * added readthedocs.yaml

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

* reads and loads .bin
  * record BB signals in .bin
  * 3D targets and scene rendering with imaging side by side radar
  * Swerling's scattering

## Example Code

Check on Google Colab the code:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/matt-chv/bdd8b835c5cb7e739bb8b68d00257690/fmcw-radar-101.ipynb)

Or Read the Docs on [![Read the docs](https://read-the-docs-guidelines.readthedocs-hosted.com/_images/logo-wordmark-light.png)](https://mmwrt.readthedocs.io/en/latest/Intro.html)

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

8. run sphinx-build
(updates the read_the_docs folder)

> sphinx-build -b html docs build/html

9. release to pypi-test

> python setup.py bdist_wheel

> twine upload -r testpypi dist\*

10. check updates on read_the_docs

> push to git to trigger readthedocs build:
> git push
> navigate to https://readthedocs.org/projects/mmwrt/builds/
> ensure build is successful

11. check on Google Colab
(Google Colab requires py3.8 as off 2023-Jan-15)

11.a. if testing release-candidate need to spell out or will install latest stable version

```
!python -m pip install -i https://test.pypi.org/simple/ mmWrt==0.0.5rc3
from mmWrt import __version__
print(__version__)
```

11.b seems extras cannot be imported from versions, so `pip install mmWrt=0.0.5rc3[dev]` or `pip install mmWrt==0.0.5[dev]` does not work. Need to upgrade to full version to test dev.

```
!python -m pip install -i https://test.pypi.org/simple/ mmWrt[dev]
from mmWrt import __version__
print(__version__)
```

12. release on pypi
> twine upload -r pypi dist\*

13. check on colab that pypi package works:

>!python -m pip install mmWrt
from mmWrt import __version__
print(__version__)

13.b. then check dev extras install works

>!python -m pip install mmWrt[dev]


