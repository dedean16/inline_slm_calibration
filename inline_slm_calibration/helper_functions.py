"""
Various helper functions.
"""
# External (3rd party)
import numpy as np
import h5py
import git
import matplotlib.pyplot as plt


def add_dict_as_hdf5group(name: str, dic: dict, hdf: h5py.File | h5py.Group):
    """
    Add python dictionary as group to HDF5 file. Supports nested dictionaries.

    Args:
        name: Name of the group
        dic: Dictionary to add.
        hdf: HDF5 file or group to add to.
    """
    subgroup = hdf.create_group(name)
    for key, value in dic.items():
        if isinstance(value, dict):
            add_dict_as_hdf5group(name=key, dic=value, hdf=subgroup)
        else:
            subgroup.create_dataset(key, data=value)


def add_dict_sequence_as_hdf5_groups(name: str, seq: list | tuple, hdf: h5py.File | h5py.Group):
    """
    Add list or tuple of dictionaries as group to HDF5 file. The list/tuple items will be stored as subgroups,
    with the index numbers as keys. Supports nested dictionaries. The list/tuple may also contain items of other type,
    as long as they are supported by h5py.

    Args:
        name: Name of the list or tuple group.
        seq: List or tuple of dictionaries to add.
        hdf: HDF5 file or group to add to.
    """
    subgroup = hdf.create_group(name)
    for n, item in enumerate(seq):
        if isinstance(item, dict):
            add_dict_as_hdf5group(name=f'{n}', dic=item, hdf=subgroup)
        else:
            subgroup.create_dataset(name=f'{n}', data=item)


def get_dict_from_hdf5(group: h5py.Group) -> dict:
    """
    Retrieve a hdf5 file or group as a dictionary. Supports nested dictionaries. Can be used on the main group as well
    to get a dictionary of the whole hdf5 structure.

    Args:
        group: hdf5 group.

    Returns: the group as dictionary
    """
    dic = {}
    for key in group.keys():
        if group[key].__class__.__name__ == 'Group':
            dic[key] = get_dict_from_hdf5(group[key])
        else:
            dic[key] = group[key][()]
    return dic


def gitinfo() -> dict:
    """
    Return a dict with info about the current git commit and repository.
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    working_dir = repo.working_dir
    diff = repo.git.diff()
    commit_timestamp = repo.head.object.committed_datetime.timestamp()
    git_info = {'sha': sha, 'working_dir': working_dir, 'diff': diff, 'commit_timestamp': commit_timestamp}
    return git_info


def fit_quadratic(x, y):
    """Fit quadratic function to data, using least-squares fit."""
    # Flatten
    xf = x.flatten()
    yf = y.flatten()

    # Create the design matrix for quadratic fitting
    X = np.vstack([xf ** 2, xf, np.ones_like(xf)]).T

    # Perform linear regression using numpy's least squares method
    coefficients, _, _, _ = np.linalg.lstsq(X, yf, rcond=None)
    a, b, c = coefficients
    return a, b, c
