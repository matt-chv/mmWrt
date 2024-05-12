from os.path import abspath, join, pardir
from os import walk

import nbformat
from nbclient import NotebookClient


def test_ipynb(dp=None):
    if dp is None:
        dp = abspath(join(__file__, pardir, pardir, "docs"))
    for root, dirs, files in walk(dp):
        for fn in files:
            if root.find(".ipynb_checkpoints") >= 0:
                continue
            if fn.find(".ipynb") >= 0:
                fp = abspath(join(root, fn))
                to_be_tested = ["Intro_nb.ipynb", "Grouping.ipynb",
                                "Speed.ipynb", "Precision.ipynb",
                                "micro_doppler.ipynb"]
                if fn in to_be_tested:
                    # fp = "C:/git/mmWrt/docs/Intro_nb.ipynb"
                    nb = nbformat.read(fp, as_version=4)
                    client = NotebookClient(nb, kernel_name='python3')
                    # client.on_cell_complete()
                    _ = client.execute()
                else:
                    print("skipping", fp)


if __name__ == "__main__":
    dpt = abspath(join(__file__, pardir, pardir, "docs"))
    test_ipynb(dp=dpt)
