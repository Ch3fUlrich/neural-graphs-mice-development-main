import nbformat
from nbclient import NotebookClient
import glob
from pathlib import Path

for nb_path in sorted(glob.glob('notebooks/*.ipynb')):
    print(f'Running {nb_path}...')
    try:
        nb = nbformat.read(nb_path, as_version=4)
        client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': 'notebooks/'}})
        client.execute()
        nbformat.write(nb, nb_path)
        print(f'Success: {nb_path}')
    except Exception as e:
        print(f'Error in {nb_path}: {e}')
