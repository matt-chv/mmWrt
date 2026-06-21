# mmWrt

[![DOI](https://zenodo.org/badge/583878586.svg)](https://doi.org/10.5281/zenodo.17449388)


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

v0.0.12:
    * move to the new artefacts (away from setup.py)
    - TODO: planning for S-FMCW add a chirp ramp start delay (when the TX ramp starts vs chirp start)
            renaming chirp_end_time to chirp_ramp_end_time
            adding chirp_ramp_start_time=0
            adding check that chirp_ramp_start_time + chirp_ramp_end_time < chirp_period
            adding adc_sampling_start_time=0
            add ditherring support
    * define a chirp properties (chirp_slope, chirp_start_frequency, chirp_end_time) on chirp_idx
    - TODO: add a ADC start delay (and check that it does not end after ramp-end or warning)
    - TODO: add pcl (x,y,z,v,A,noise)

v0.0.11:
    * pre.1: WiP (adding interfer radars)
        - ~~TODO: run all unit test pass : 100% (except performances)~~
        - ~~renaming all variables to be explicit in .py~~
            ~~adc_samples_per_chirp -> adc_sample_count~~
            ~~max_adc_sample_rate -> adc_sample_rate_max~~
            ~~max_adc_buffer_size -> adc_sample_count_max~~
            ~~ n_adc -> adc_sample_count ~~
            ~~ number_adc_samples -> adc_sample_count ~~
            ~~chirps_count -> chirp_count~~
            ~~t_inter_chirp -> chirp_period~~
            ~~t_inter_frame -> frame_period~~
            ~~total_number_frames -> frame_count~~
            ~~frames_count -> frame_count~~
            ~~target -> scatterer~~
        - ~~FIXME: complex signal chain seems broken~~
        - ~~range TDM~~
        - ~~2D with URA~~
        - ~~3D compute~~
        - ~~FIXME: need to split the TX_Freq per transmitter radar~~
        - ~~refactor rt_points, but kept the interface untouched (still returns a dict, adc values still a tensor with )~~
        - ~~removed warnings in pytest~~
        - ~~add pcl from (x,y,z)~~ 
        - ~~TODO: add test_interference~~
        - ~~TODO: add test_rsp_interference_mitigation: zero, ramp, IFFT~~
        - ~~TODO: add range DDM and disambiguation~~
        - ~~run pytest --ignore=tests\test_nb.py --ignore=tests\test_docs.py~~
        - done 60 test 100% pass
        - uploaded on https://test.pypi.org/project/mmwrt/0.0.11rc1/

    * pre.2
        - TODO
            - renaming:
                - tx_frequencies -> vco_frequency
                - align pre.1 renaming convention with .ipynb
            - run pytest --ignore=tests\test_nb.py --ignore=tests\test_docs.py
            - if all ok -> release


*v0.0.10:

    * adding distance as a function of time in IF compute - allowing for simulation of target changing bin ranges through single chirp
    * 0.0.10-pre.6: fixes #3 (numpy deprecating complex_ replacing with complex128)
    * -pre.7: 

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

## LOGS

- [ ] 4, refactor for interfers

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

for intermediate releases, excluding some tests can be done with 

> pytest --ignore=tests\test_docs.py --ignore=tests\test_nb.py


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
