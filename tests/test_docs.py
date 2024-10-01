from os.path import abspath, join, pardir
from os import walk
import pytest
import sys

import nbformat
from nbclient import NotebookClient

dp = abspath(join(__file__, pardir, pardir))
sys.path.insert(0, dp)

from mmWrt import __version__  # noqa: E402


@pytest.mark.skipif(__version__.find("-rc") >= 0, reason="only 4 full release")
def test_ipynb(dp=None):
    """ this is to ensure the jupyter notebooks are still running without error
    for every code release

    Parameters
    ----------
    dp: String
        folder path with jupyter notebook to be tested
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
                to_ignored = ["Resolution.ipynb"]
                if fn not in to_ignored:
                    # fp = "C:/git/mmWrt/docs/Intro_nb.ipynb"
                    nb = nbformat.read(fp, as_version=4)
                    client = NotebookClient(nb, kernel_name='python3')
                    # client.on_cell_complete()
                    _ = client.execute()
                else:
                    print("skipping", fp)
