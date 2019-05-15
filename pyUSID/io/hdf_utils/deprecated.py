from warnings import warn
from ..reg_ref import __all__ as func_names

__all__ = func_names


def __raise_move_warning(new_func_address):
    warn('This function has been moved to {}. This alias will be removed in a '
         'future version. Please update your import statements accordingly'
         '.'.format(new_func_address), FutureWarning)


def get_region(h5_dset, reg_ref_name):
    """
    Gets the region in a dataset specified by a region reference

    Parameters
    ----------
    h5_dset : h5py.Dataset
        Dataset containing the region reference
    reg_ref_name : str / unicode
        Name of the region reference

    Returns
    -------
    value : np.ndarray
        Data specified by the region reference. Note that a squeeze is applied by default.

    Notes
    -----
    Please use pyUSID.io.reg_ref.get_region() instead
    """
    __raise_move_warning('pyUSID.io.reg_ref.get_region()')
    from ..reg_ref import get_region as orig_func
    return orig_func(h5_dset, reg_ref_name)


def get_indices_for_region_ref(h5_main, ref, return_method='slices'):
    """
    Given an hdf5 region reference and the dataset it refers to,
    return an array of indices within that dataset that
    correspond to the reference.

    Parameters
    ----------
    h5_main : HDF5 Dataset
        dataset that the reference can be returned from
    ref : HDF5 Region Reference
        Region reference object
    return_method : {'slices', 'corners', 'points'}
        slices : the reference is return as pairs of slices

        corners : the reference is returned as pairs of corners representing
        the starting and ending indices of each block

        points : the reference is returns as a list of tuples of points

    Returns
    -------
    ref_inds : Numpy Array
        array of indices in the source dataset that ref accesses

    Notes
    -----
    Please use pyUSID.io.reg_ref.get_indices_for_region_ref() instead
    """
    __raise_move_warning('pyUSID.io.reg_ref.get_indices_for_region_ref()')
    from ..reg_ref import get_indices_for_region_ref as orig_func
    return orig_func(h5_main, ref, return_method=return_method)


def clean_reg_ref(h5_dset, reg_ref_tuple, verbose=False):
    """
    Makes sure that the provided instructions for a region reference are indeed valid
    This method has become necessary since h5py allows the writing of region references larger than the maxshape

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references will be added as attributes
    reg_ref_tuple : list / tuple
        The slicing information formatted using tuples of slice objects.
    verbose : Boolean (Optional. Default = False)
        Whether or not to print status messages

    Returns
    -------
    new_reg_refs : tuple
        Instructions for the corrected region reference

    Notes
    -----
    Please use pyUSID.io.reg_ref.clean_reg_ref() instead in the future
    """
    __raise_move_warning('pyUSID.io.reg_ref.clean_reg_ref()')
    from ..reg_ref import clean_reg_ref as orig_func
    return orig_func(h5_dset, reg_ref_tuple, verbose=verbose)


def attempt_reg_ref_build(h5_dset, dim_names, verbose=False):
    """

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references need to be added as attributes
    dim_names : list or tuple
        List of the names of the region references (typically names of dimensions)
    verbose : bool, optional. Default=False
        Whether or not to print debugging statements

    Returns
    -------
    labels_dict : dict
        The slicing information must be formatted using tuples of slice objects.
        For example {'region_1':(slice(None, None), slice (0,1))}

    Notes
    -----
    Please use pyUSID.io.reg_ref.attempt_reg_ref_build() instead in the future
    """
    __raise_move_warning('pyUSID.io.reg_ref.attempt_reg_ref_build()')
    from ..reg_ref import attempt_reg_ref_build as orig_func
    return orig_func(h5_dset, dim_names, verbose=verbose)


def copy_reg_ref_reduced_dim(h5_source, h5_target, h5_source_inds, h5_target_inds, key):
    """
    Copies a region reference from one dataset to another taking into account that a dimension
    has been lost from source to target

    Parameters
    ----------
    h5_source : HDF5 Dataset
            source dataset for region reference copy
    h5_target : HDF5 Dataset
            target dataset for region reference copy
    h5_source_inds : HDF5 Dataset
            indices of each dimension of the h5_source dataset
    h5_target_inds : HDF5 Dataset
            indices of each dimension of the h5_target dataset
    key : String
            Name of attribute in h5_source that contains
            the Region Reference to copy

    Returns
    -------
    ref_inds : Nx2x2 array of unsigned integers
            Array containing pairs of points that define
            the corners of each hyperslab in the region
            reference

    Notes
    -----
    Please use pyUSID.io.reg_ref.copy_reg_ref_reduced_dim() instead in the future
    """
    __raise_move_warning('pyUSID.io.reg_ref.copy_reg_ref_reduced_dim()')
    from ..reg_ref import copy_reg_ref_reduced_dim as orig_func
    return orig_func(h5_source, h5_target, h5_source_inds, h5_target_inds, key)


def create_region_reference(h5_main, ref_inds):
    """
    Create a region reference in the destination dataset using an iterable of pairs of indices
    representing the start and end points of a hyperslab block

    Parameters
    ----------
    h5_main : HDF5 dataset
        dataset the region will be created in
    ref_inds : Iterable
        index pairs, [start indices, final indices] for each block in the
        hyperslab

    Returns
    -------
    new_ref : HDF5 Region reference
        reference in `h5_main` for the blocks of points defined by `ref_inds`

    Notes
    -----
    Please use pyUSID.io.reg_ref.create_region_reference() instead in the future
    """
    __raise_move_warning('pyUSID.io.reg_ref.create_region_reference()')
    from ..reg_ref import create_region_reference as orig_func
    return orig_func(h5_main, ref_inds)


def simple_region_ref_copy(h5_source, h5_target, key):
    """
    Copies a region reference from one dataset to another
    without alteration

    Parameters
    ----------
    h5_source : HDF5 Dataset
            source dataset for region reference copy
    h5_target : HDF5 Dataset
            target dataset for region reference copy
    key : String
            Name of attribute in h5_source that contains
            the Region Reference to copy

    Returns
    -------
    ref_inds : Nx2x2 array of unsigned integers
            Array containing pairs of points that define
            the corners of each hyperslab in the region
            reference

    Notes
    -----
    Please use pyUSID.io.reg_ref.simple_region_ref_copy() instead in the future
    """
    __raise_move_warning('pyUSID.io.reg_ref.simple_region_ref_copy()')
    from ..reg_ref import simple_region_ref_copy as orig_func
    return orig_func(h5_source, h5_target, key)


def write_region_references(h5_dset, reg_ref_dict, add_labels_attr=True,
                            verbose=False):
    """
    Creates attributes of a h5py.Dataset that refer to regions in the dataset

    Parameters
    ----------
    h5_dset : h5.Dataset instance
        Dataset to which region references will be added as attributes
    reg_ref_dict : dict
        The slicing information must be formatted using tuples of slice objects.
        For example {'region_1':(slice(None, None), slice (0,1))}
    add_labels_attr : bool, optional, default = True
        Whether or not to write an attribute named 'labels' with the
    verbose : Boolean (Optional. Default = False)
        Whether or not to print status messages

    Notes
    -----
    Please use pyUSID.io.reg_ref.write_region_references() instead in the future
    """
    __raise_move_warning('pyUSID.io.reg_ref.write_region_references()')
    from ..reg_ref import write_region_references as orig_func
    return orig_func(h5_dset, reg_ref_dict, add_labels_attr=add_labels_attr,
                     verbose=verbose)