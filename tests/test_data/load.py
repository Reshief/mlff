import importlib.resources as importlib_resources
from pathlib import Path


def load_data(filename):
    p_filename = Path(filename)
    if p_filename.suffix == '.npz':
        import numpy as np
        ref = importlib_resources.files(__name__).joinpath(filename)
        # The np.load object keeps seeking the file pointer, hence we need to keep it open
        # with ref.open('rb') as fp:
        #     return np.load(fp)
        return np.load(ref.open('rb'))
    else:
        from ase.io import iread
        ref = importlib_resources.files(__name__).joinpath(filename)
        # The iread iterator keeps seeking the file pointer, hence we need to keep it open
        # with importlib_resources.as_file(ref) as f:
        #     return iread(f, ':')
        return iread(ref.open('rb'), ':')
