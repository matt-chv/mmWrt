from os.path import abspath, join
from os import walk

import nbformat
from nbclient import NotebookClient

def test_ipynb(dp):
    for root, dirs, files in walk(dp):
        for fn in files:
            if fn.find(".ipynb") >= 0:
                fp = abspath(join(root, fn))
                to_be_tested = ["Intro_nb.ipynb"]
                to_be_tested = ["Grouping.ipynb"]
                if fn in to_be_tested:
                    # fp = "C:/git/mmWrt/docs/Intro_nb.ipynb"
                    nb = nbformat.read(fp, as_version=4)
                    client = NotebookClient(nb, kernel_name='python3')
                    # client.on_cell_complete()
                    nbr = client.execute()
                else:
                    print(fp)

if __name__ == "__main__":
    test_ipynb(dp="C:/git/mmWrt/docs")