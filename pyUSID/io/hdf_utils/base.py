# -*- coding: utf-8 -*-
"""
Simple yet handy HDF5 utilities, independent of the  USID model

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import socket
import sys
from platform import platform

import h5py
import numpy as np

from ...__version__ import version as py_usid_version
from ..io_utils import get_time_stamp
from ..write_utils import clean_string_att
from ..dtype_utils import validate_single_string_arg, validate_list_of_strings

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
    if not isinstance(h5_object, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('h5_object should be a h5py.Dataset, h5py.Group or h5py.File object')

    if aux_dset_name is None:
        aux_dset_name = h5_object.attrs.keys()
    else:
        aux_dset_name = validate_list_of_strings(aux_dset_name, 'aux_dset_name')

    data_list = list()
    curr_name = None
    try:
        h5_file = h5_object.file
        for curr_name in aux_dset_name:
            h5_ref = h5_object.attrs[curr_name]
            if isinstance(h5_ref, h5py.Reference) and isinstance(h5_file[h5_ref], h5py.Dataset) and not \
                    isinstance(h5_ref, h5py.RegionReference):
                data_list.append(h5_file[h5_ref])
    except KeyError:
        raise KeyError('%s is not an attribute of %s' % (str(curr_name), h5_object.name))

    return data_list


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
    if not isinstance(h5_object, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('h5_object should be a h5py.Dataset, h5py.Group or h5py.File object')

    attr_name = validate_single_string_arg(attr_name, 'attr_name')

    if attr_name not in h5_object.attrs.keys():
        raise KeyError("'{}' is not an attribute in '{}'".format(attr_name, h5_object.name))

    att_val = h5_object.attrs.get(attr_name)
    if isinstance(att_val, np.bytes_) or isinstance(att_val, bytes):
        att_val = att_val.decode('utf-8')

    elif type(att_val) == np.ndarray:
        if sys.version_info.major == 3:
            if att_val.dtype.type in [np.bytes_, np.object_]:
                att_val = np.array([str(x, 'utf-8') for x in att_val])

    return att_val


def get_attributes(h5_object, attr_names=None):
    """
    Returns attribute associated with some DataSet.

    Parameters
    ----------
    h5_object : :class:`h5py.Dataset`
        Dataset object reference.
    attr_names : str or :class:`list` of str, optional. Default = all
        Name of attribute object to return.

    Returns
    -------
    att_dict : dict
        Dictionary containing (name,value) pairs of attributes

    """
    if not isinstance(h5_object, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('h5_object should be a h5py.Dataset, h5py.Group or h5py.File object')

    if attr_names is None:
        attr_names = h5_object.attrs.keys()
    else:
        attr_names = validate_list_of_strings(attr_names, 'attr_names')

    att_dict = {}

    for attr in attr_names:
        try:
            att_dict[attr] = get_attr(h5_object, attr)
        except KeyError:
            raise KeyError('%s is not an attribute of %s' % (str(attr), h5_object.name))

    return att_dict


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

    obj_names = validate_list_of_strings(obj_names, 'attr_names')

    if isinstance(h5_refs, (h5py.File, h5py.Group, h5py.Dataset)):
        h5_refs = [h5_refs]
    if not isinstance(h5_refs, (list, tuple)):
        raise TypeError('h5_refs should be a / list of h5py.Dataset, h5py.Group or h5py.File object(s)')

    found_objects = []
    for target_name in obj_names:
        for h5_object in h5_refs:
            if not isinstance(h5_object, (h5py.File, h5py.Group, h5py.Dataset)):
                continue
            if h5_object.name.split('/')[-1] == target_name:
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
    if not isinstance(h5_src, (h5py.Dataset, h5py.File, h5py.Group)):
        raise TypeError('h5_src should either be a h5py Dataset, File, or '
                        'Group')
    if not isinstance(h5_other, (h5py.Dataset, h5py.File, h5py.Group)):
        raise TypeError('h5_other should either be a h5py Dataset, File, or'
                        ' Group')
    if h5_src.file != h5_other.file:
        raise ValueError('Cannot link h5 objects across files. '
                         '{} is present in file: {}, while {} is in file :'
                         '{}'.format(h5_src.name, h5_src.file, h5_other.name,
                                     h5_other.file))


def __link_h5_obj(h5_src, h5_other, alias=None):
    validate_h5_objs_in_same_h5_file(h5_src, h5_other)
    if alias is None:
        alias = h5_other.name.split('/')[-1]
    h5_src.attrs[alias] = h5_other.ref


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
    if not isinstance(src, (h5py.Dataset, h5py.File, h5py.Group)):
        raise TypeError('src should either be a h5py Dataset, File, or Group')
    if isinstance(h5_objects, (h5py.Dataset, h5py.Group)):
        h5_objects = [h5_objects]

    for itm in h5_objects:
        if not isinstance(itm, (h5py.Dataset, h5py.Group)):
            raise TypeError('h5_objects should only contain h5py. Dataset and Group objects')
        __link_h5_obj(src, itm)


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
    if not isinstance(h5_main, (h5py.Dataset, h5py.File, h5py.Group)):
        raise TypeError('h5_main should either be a h5py Dataset, File, or Group')
    if not isinstance(h5_ancillary, (h5py.Dataset, h5py.Group)):
        raise TypeError('h5_ancillary should be a h5py. Dataset or Group object')
    alias_name = validate_single_string_arg(alias_name, 'alias_name')

    __link_h5_obj(h5_main, h5_ancillary, alias=alias_name)


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
    if not isinstance(h5_obj, (h5py.File, h5py.Group, h5py.Dataset)):
        raise TypeError('h5_obj should be a h5py File, Group or Dataset object but is instead of type '
                        '{}t'.format(type(h5_obj)))
    try:
        file_handle = h5_obj.file
    except RuntimeError:
        raise ValueError('Encountered a RuntimeError possibly due to a closed file')
    # file handle is actually an open hdf file

    if file_handle.mode == 'r':
        return False
    return True


def write_book_keeping_attrs(h5_obj):
    """
    Writes basic book-keeping and posterity related attributes to groups created in pyUSID such as machine id,
    pyUSID version, timestamp.

    Parameters
    ----------
    h5_obj : :class:`h5py.Dataset`, :class:`h5py.Group`, or :class:`h5py.File`
        Object to which basic book-keeping attributes need to be written

    """
    if not isinstance(h5_obj, (h5py.Group, h5py.File, h5py.Dataset)):
        raise TypeError('h5_obj should be a h5py.Group, h5py.File, or h5py.Dataset object')
    write_simple_attrs(h5_obj, {'machine_id': socket.getfqdn(),
                                'timestamp': get_time_stamp(),
                                'pyUSID_version': py_usid_version,
                                'platform': platform()},
                       verbose=False)


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
    if not isinstance(attrs, dict):
        raise TypeError('attrs should be a dictionary but is instead of type '
                        '{}'.format(type(attrs)))
    if not isinstance(h5_obj, (h5py.File, h5py.Group, h5py.Dataset)):
        raise TypeError('h5_obj should be a h5py File, Group or Dataset object but is instead of type '
                        '{}t'.format(type(h5_obj)))

    for key, val in attrs.items():
        if val is None:
            continue
        if verbose:
            print('Writing attribute: {} with value: {}'.format(key, val))
        clean_val = clean_string_att(val)
        if verbose:
            print('Attribute cleaned into: {}'.format(clean_val))
        h5_obj.attrs[key] = clean_val
    if verbose:
        print('Wrote all (simple) attributes to {}: {}\n'.format(obj_type, h5_obj.name.split('/')[-1]))