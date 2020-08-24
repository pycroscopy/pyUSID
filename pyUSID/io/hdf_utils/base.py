# -*- coding: utf-8 -*-
"""
Simple yet handy HDF5 utilities, independent of the  USID model

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import sys
from warnings import warn
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


def get_auxiliary_datasets(h5_object, aux_dset_name=None):
    """
    Returns auxiliary dataset objects associated with some DataSet through its attributes.
    Note - region references will be ignored.

    Parameters
    ----------
    h5_object : :class:`h5py.Dataset`, :class:`h5py.Group` or :class:`h5py.File`
        Dataset object reference.
    aux_dset_name : str or :class:`list` of str, optional. Default = all
        Name of auxiliary :class:`h5py.Dataset` objects to return.

    Returns
    -------
    list of :class:`h5py.Reference` of auxiliary :class:`h5py.Dataset` objects.

    """
    warn('pyUSID.io.hdf_utils.get_auxiliary_datasets has been moved to '
         'sidpy.hdf.hdf_utils.get_auxiliary_datasets. This copy in pyUSID will'
         'be removed in future release. Please update your import statements')
    return hut.get_auxiliary_datasets(h5_object, aux_dset_name=aux_dset_name)


def get_attr(h5_object, attr_name):
    """
    Returns the attribute from the h5py object

    Parameters
    ----------
    h5_object : :class:`h5py.Dataset`, :class:`h5py.Group` or :class:`h5py.File`
        object whose attribute is desired
    attr_name : str
        Name of the attribute of interest

    Returns
    -------
    att_val : object
        value of attribute, in certain cases (byte strings or list of byte strings) reformatted to readily usable forms

    """
    warn('pyUSID.io.hdf_utils.get_attr has been moved to '
         'sidpy.hdf.hdf_utils.get_attr. This copy in pyUSID will'
         'be removed in future release. Please update your import statements')
    return hut.get_attr(h5_object, attr_name)


def get_attributes(h5_object, attr_names=None, strict=False):
    """
    Returns attribute associated with some DataSet.

    Parameters
    ----------
    h5_object : :class:`h5py.Dataset`
        Dataset object reference.
    attr_names : str or :class:`list` of str, optional. Default = all
        Name of attribute object to return.
    strict : bool, optional. Default = False
        If True - raises a KeyError if desired keys are not found.
        Else, raises warning instead.
        This is especially useful when attempting to read attributes with
        invalid names such as spaces on either sides of text.

    Returns
    -------
    att_dict : dict
        Dictionary containing (name,value) pairs of attributes

    """
    warn('pyUSID.io.hdf_utils.get_attributes has been moved to '
         'sidpy.hdf.hdf_utils.get_attributes. This copy in pyUSID will'
         'be removed in future release. Please update your import statements')
    return hut.get_attributes(h5_object, attr_names=attr_names, strict=strict)


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


def validate_h5_objs_in_same_h5_file(h5_src, h5_other):
    """
    Checks if the provided objects are in the same HDF5 file.
    If not, it throws a ValueError

    Parameters
    ----------
    h5_src : h5py.Dataset, h5py.File, or h5py.Group object
        First object to compare
    h5_other : h5py.Dataset, h5py.File, or h5py.Group object
        Second object to compare
    """
    warn('pyUSID.io.hdf_utils.validate_h5_objs_in_same_h5_file has been moved to '
         'sidpy.hdf.hdf_utils.validate_h5_objs_in_same_h5_file. This copy in pyUSID will'
         'be removed in future release. Please update your import statements')
    return hut.validate_h5_objs_in_same_h5_file(h5_src, h5_other)


def link_h5_objects_as_attrs(src, h5_objects):
    """
    Creates Dataset attributes that contain references to other Dataset Objects.

    Parameters
    -----------
    src : Reference to h5.object
        Reference to the the object to which attributes will be added
    h5_objects : list of references to h5.objects
        objects whose references that can be accessed from src.attrs

    Returns
    --------
    None

    """
    warn('pyUSID.io.hdf_utils.link_h5_objects_as_attrs has been moved to '
         'sidpy.hdf.hdf_utils.link_h5_objects_as_attrs. This copy in pyUSID will'
         'be removed in future release. Please update your import statements')
    return hut.link_h5_objects_as_attrs(src, h5_objects)


def link_h5_obj_as_alias(h5_main, h5_ancillary, alias_name):
    """
    Creates Dataset attributes that contain references to other Dataset Objects.
    This function is useful when the reference attribute must have a reserved name.
    Such as linking 'SHO_Indices' as 'Spectroscopic_Indices'

    Parameters
    ------------
    h5_main : h5py.Dataset
        Reference to the the object to which attributes will be added
    h5_ancillary : h5py.Dataset
        object whose reference that can be accessed from src.attrs
    alias_name : String
        Alias / alternate name for trg

    """
    warn('pyUSID.io.hdf_utils.link_h5_obj_as_alias has been moved to '
         'sidpy.hdf.hdf_utils.link_h5_obj_as_alias. This copy in pyUSID will'
         'be removed in future release. Please update your import statements')
    return hut.link_h5_obj_as_alias(h5_main, h5_ancillary, alias_name)


def is_editable_h5(h5_obj):
    """
    Returns True if the file containing the provided h5 object is in w or r+ modes

    Parameters
    ----------
    h5_obj : h5py.File, h5py.Group, or h5py.Dataset object
        h5py object

    Returns
    -------
    mode : bool
        True if the file containing the provided h5 object is in w or r+ modes

    """
    warn('pyUSID.io.hdf_utils.is_editable_h5 has been moved to '
         'sidpy.hdf.hdf_utils.is_editable_h5. This copy in pyUSID will'
         'be removed in future release. Please update your import statements')
    return hut.is_editable_h5(h5_obj)


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


def write_simple_attrs(h5_obj, attrs, obj_type='', verbose=False):
    """
    Writes attributes to a h5py object

    Parameters
    ----------
    h5_obj : :class:`h5py.File`, :class:`h5py.Group`, or h5py.Dataset object
        h5py object to which the attributes will be written to
    attrs : dict
        Dictionary containing the attributes as key-value pairs
    obj_type : str or unicode, optional. Default = ''
        type of h5py.obj. Examples include 'group', 'file', 'dataset
    verbose : bool, optional. Default=False
        Whether or not to print debugging statements

    """
    warn('pyUSID.io.hdf_utils.write_simple_attrs has been moved to '
         'sidpy.hdf.hdf_utils.write_simple_attrs. This copy in pyUSID will'
         'be removed in future release. Please update your import statements')
    return hut.write_simple_attrs(h5_obj, attrs, obj_type=obj_type,
                                  verbose=verbose)
