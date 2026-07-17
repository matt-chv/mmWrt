""" ensuring the .ipynb in the documentation are still running
for every release of the code
v0.0.14: 19
"""
from os.path import abspath, join, pardir
from os import walk
from pathlib import Path
import pytest
import sys

import nbformat
from nbclient import NotebookClient

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)
dpi = abspath(join(__file__, pardir, pardir, "docs"))

from mmWrt import __version__  # noqa: E402

@pytest.fixture(params=sorted(Path(dpi).glob("*.ipynb")), ids=lambda p: p.name)
def notebook_path(request):
    return request.param

@pytest.mark.skipif(__version__.find("-rc") >= 0, reason="only 4 full release")
@pytest.mark.skipif(__version__.find(".dev") >= 0, reason="only 4 full release")
def test_ipynb(notebook_path):
    print("???????", notebook_path)
    with open(notebook_path) as f:
        print("#####", notebook_path.name)
        to_ignore = ["Chirp_Solver.ipynb",
                              "Resolution.ipynb", 
                              # this one is broken - needs fixing
                              "Precision.ipynb",
                              "FMCW-Radar-107_Micro-Doppler.ipynb",
                              # this one is just for illustration, not for testing
                              "FMCW-Radar-109_Radar_Cube.ipynb",
                              "FMCW-Radar-201_Maths.ipynb",
                              # dont run this one, it breaks pip installs
                              "FMCW-Radar-205_mmWave_Atmospheric_Absorption.ipynb"]
        to_ignore = [Path(r) for r in to_ignore]
        # runs = ["Angle_of_Arrival.ipynb"]
        """runs = ["Angle_of_Arrival.ipynb","FMCW-Radar-100_Radar_Imaging.ipynb",
                "FMCW-Radar-101_Intro.ipynb", "FMCW-Radar-102_CFAR.ipynb",
                "FMCW-Radar-102_CFAR.ipynb", "FMCW-Radar-103_AoA.ipynb",
                "FMCW-Radar-104_Freq_est.ipynb",
                "FMCW-Radar-105_Vibration.ipynb",
                "FMCW-Radar-106_Phased_Arrays.ipynb",
                "FMCW-Radar-108_Antennas.ipynb",
                "FMCW-Radar-202_CFAR_Pd.ipynb",
                "FMCW-Radar-203_SNR.ipynb",
                "FMCW-Radar-204-Stitching.ipynb",
                "FMCW-Radar-206_Reflections.ipynb",
                "Grouping.ipynb",
                "High-Speed.ipynb",
                "Intro_nb.ipynb",
                "MIMO_DDM.ipynb",
                "MIMO_TDM.ipynb",]
        runs = [runs[0]]
        runs = [Path(r) for r in runs]"""

        if notebook_path.name in [r.name for r in to_ignore]:
            pytest.skip(f"Skipping1 {notebook_path.name} as it is in the ignored list")
        # if notebook_path.name not in [r.name for r in runs]:
        #    pytest.skip(f"Skipping2 {notebook_path.name} as it is not in the runs list")
        nb = nbformat.read(f, as_version=4)
        client = NotebookClient(nb, kernel_name='python3')
        try:
            _ = client.execute()
        except Exception as ex:
            print(f"!!!!!!!!!!{notebook_path.name}")
            raise Exception(ex)


def tbd_ipynb(dp=None):
    """ this is to ensure the jupyter notebooks are still running without error
    for every code release

    Parameters
    ----------
    dp: String
        folder path with jupyter notebook to be tested
    Raises
    ------
    Exception
        if the notebook execution failsW
    """
    if dp is None:
        dp = abspath(join(__file__, pardir, pardir, "docs"))
    for root, dirs, files in walk(dp):
        for fn in files:
            if root.find(".ipynb_checkpoints") >= 0:
                continue
            if fn.find(".ipynb") >= 0:
                fp = abspath(join(root, fn))
                # to_be_tested = ["Intro_nb.ipynb", "Grouping.ipynb",
                #                "Speed.ipynb", "Precision.ipynb",
                #                "micro_doppler.ipynb"]
                to_ignored = ["Chirp_Solver.ipynb",
                              "Resolution.ipynb", 
                              "FMCW-Radar-107_Micro-Doppler.ipynb",
                              # dont run this one, it breaks pip installs
                              "FMCW-Radar-205_mmWave_Atmospheric_Absorption.ipynb"]
                if fn not in to_ignored:
                    # fp = "C:/git/mmWrt/docs/Intro_nb.ipynb"
                    nb = nbformat.read(fp, as_version=4)
                    client = NotebookClient(nb, kernel_name='python3')
                    # client.on_cell_complete()
                    try:
                        _ = client.execute()
                    except Exception as ex:
                        print(f"!!!!!!!!!!{fn}")
                        raise Exception(ex)
                else:
                    print("skipping", fp)

if __name__ == "__main__":
    test_ipynb(notebook_path)