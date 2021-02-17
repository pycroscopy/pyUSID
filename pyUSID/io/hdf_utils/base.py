# -*- coding: utf-8 -*-
"""
Simple yet handy HDF5 utilities, independent of the  USID model

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import h5py
from sidpy.hdf import hdf_utils as hut

from ...__version__ import version as py_usid_version

if sys.version_info.major == 3:
    unicode = str


def print_tree(parent, rel_paths=False, main_dsets_only=False):
    """
    Simple function to recursively print the contents of an hdf5 group

    Parameters
    ----------
    parent : :class:`h5py.Group`
        HDF5 (sub-)tree to print
    rel_paths : bool, optional. Default = False
        True - prints the relative paths for all elements.
        False - prints a tree-like structure with only the element names
    main_dsets_only : bool, optional. default=False
        True - prints only groups and Main datasets
        False - prints all dataset and group objects
    """
    # TODO: Leverage copy in sidpy.hdf.hdf_utils
    if not isinstance(parent, (h5py.File, h5py.Group)):
        raise TypeError('Provided object is not a h5py.File or h5py.Group '
                        'object')

    def __print(name, obj):
        show = True
        if main_dsets_only:
            show = False
            from .simple import check_if_main
            if check_if_main(obj) or isinstance(obj, h5py.Group):
                show = True
        if not show:
            return

        if rel_paths:
            print(name)
        else:
            levels = name.count('/')
            curr_name = name[name.rfind('/') + 1:]

            print(levels * '  ' + '├ ' + curr_name)
            if isinstance(obj, h5py.Group):
                print((levels + 1) * '  ' + len(curr_name) * '-')

    print(parent.name)
    parent.visititems(__print)


def get_h5_obj_refs(obj_names, h5_refs):
    """
    Given a list of H5 references and a list of names,
    this method returns H5 objects corresponding to the names

    Parameters
    ----------
    obj_names : string or List of strings
        names of target h5py objects
    h5_refs : H5 object reference or List of H5 object references
        list containing the target reference

    Returns
    -------
    found_objects : List of HDF5 dataset references
        Corresponding references

    """
    from ..usi_data import USIDataset

    found_objects = []
    for h5_object in hut.get_h5_obj_refs(obj_names, h5_refs):
        try:
            found_objects.append(USIDataset(h5_object))
        except TypeError:
            found_objects.append(h5_object)

    return found_objects


def write_book_keeping_attrs(h5_obj):
    """
    Writes basic book-keeping and posterity related attributes to groups
    created in pyUSID such as machine id, pyUSID version, timestamp.

    Parameters
    ----------
    h5_obj : :class:`h5py.Dataset`, :class:`h5py.Group`, or :class:`h5py.File`
        Object to which basic book-keeping attributes need to be written

    """
    hut.write_book_keeping_attrs(h5_obj)
    hut.write_simple_attrs(h5_obj, {'pyUSID_version': py_usid_version})
