"""
:class:`~pyUSID.io.image.ImageTranslator` class that translates conventional 2D images to USID HDF5 files

Created on Feb 9, 2016

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os
import sys
import h5py
import numpy as np
from PIL import Image

from .numpy_translator import ArrayTranslator
from .write_utils import Dimension
from .dtype_utils import contains_integers
from .hdf_utils import write_simple_attrs

if sys.version_info.major == 3:
    unicode = str
else:
    FileExistsError = ValueError
    FileNotFoundError = ValueError


class ImageTranslator(ArrayTranslator):
    """
    Translates data from an image file to an HDF5 file
    """

    def __init__(self, *args, **kwargs):
        super(ImageTranslator, self).__init__(*args, **kwargs)

    @staticmethod
    def _parse_file_path(image_path, h5_path=None):
        """
        Returns a list of all files in the directory given by path

        Parameters
        ---------------
        image_path : str
            absolute path to the image file
        h5_path : str, optional
            absolute path to the desired output HDF5 file. If nothing is provided, a valid file path will be provided

        Returns
        ----------
        image_path : str
            Absolute file path to the image
        h5_path : str
            absolute path to the desired output HDF5 file.
        """
        if not isinstance(image_path, (str, unicode)):
            raise TypeError("'image_path' argument for ImageTranslator should be a str or unicode")
        if not os.path.exists(os.path.abspath(image_path)):
            raise FileNotFoundError('Specified image does not exist.')
        else:
            image_path = os.path.abspath(image_path)

        if h5_path is not None:
            if not isinstance(h5_path, (str, unicode)):
                raise TypeError("'h5_path' argument for ImageTranslator should be a str or unicode (if provided)")
            # NOT checking the extension of the file path for simplicity
        else:
            base_name, _ = os.path.splitext(image_path)
            h5_name = base_name + '.h5'
            h5_path = os.path.join(image_path, h5_name)

        if os.path.exists(os.path.abspath(h5_path)):
            raise FileExistsError("ImageTranslator: There is already a valid (output HDF5) file at:\n{}\n"
                                  "Please consider providing an alternate path or deleting the "
                                  "specified file".format(h5_path))

        return image_path, h5_path

    def translate(self, image_path, h5_path=None, bin_factor=None, interp_func=Image.BICUBIC, normalize=False,
                  **image_args):
        """
        Translates the image in the provided file into a USID HDF5 file

        Parameters
        ----------------
        image_path : str
            Absolute path to folder holding the image files
        h5_path : str, optional
            Absolute path to where the HDF5 file should be located.
            Default is None
        bin_factor : uint or array-like of uint, optional
            Down-sampling factor for each dimension.  Default is None.
            If specifying different binning for each dimension, please specify as (height binning, width binning)
        interp_func : int, optional. Default = :attr:`PIL.Image.BICUBIC`
            How the image will be interpolated to provide the down-sampled or binned image.
            For more information see instructions for the `resample` argument for :meth:`PIL.Image.resize`
        normalize : boolean, optional. Default = False
            Should the raw image be normalized between the values of 0 and 1
        image_args : dict
            Arguments to be passed to read_image.  Arguments depend on the type of image.

        Returns
        ----------
        h5_main : h5py.Dataset
            HDF5 Dataset object that contains the flattened images

        """
        image_path, h5_path = self._parse_file_path(image_path, h5_path=h5_path)

        image = read_image(image_path, **image_args)
        image_parms = dict()
        usize, vsize = image.shape[:2]

        '''
        Check if a bin_factor is given.  Set up binning objects if it is.
        '''
        if bin_factor is not None:
            if isinstance(bin_factor, (list, tuple)):
                if not contains_integers(bin_factor, min_val=1):
                    raise TypeError('bin_factor should contain positive whole integers')
                if len(bin_factor) == 2:
                    bin_factor = tuple(bin_factor)
                else:
                    raise ValueError('Input parameter `bin_factor` must be a length 2 array-like or an integer.\n' +
                                     '{} was given.'.format(bin_factor))

            elif isinstance(bin_factor, int):
                bin_factor = (bin_factor, bin_factor)
            else:
                raise TypeError('bin_factor should either be an integer or an iterable of positive integers')

            if np.min(bin_factor) < 0:
                raise ValueError('bin_factor must consist of positive factors')

            if interp_func not in [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]:
                raise ValueError("'interp_func' argument for ImageTranslator.translate must be one of "
                                 "PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.LANCZOS")

            image_parms.update({'image_binning_size': bin_factor, 'image_PIL_resample_mode': interp_func})
            usize = int(usize / bin_factor[0])
            vsize = int(vsize / bin_factor[1])

            # Unfortunately, we need to make a round-trip through PIL for the interpolation. Not possible with numpy
            img_obj = Image.fromarray(image)
            img_obj = img_obj.resize((vsize, usize), resample=interp_func)
            image = np.asarray(img_obj)

        # Working around occasional "cannot modify read-only array" error
        image = image.copy()

        '''
        Normalize Raw Image
        '''
        if normalize:
            image -= np.min(image)
            image = image / np.float32(np.max(image))

        image_parms.update({'normalized': normalize,
                            'image_min': np.min(image), 'image_max': np.max(image)})

        """
        Enable the line below if there is a need make the image "look" the right side up. This would be manipulation
        # of the original data. Therefore it remains commented
        """
        # image = np.flipud(image)

        '''
        Ready to write to h5
        '''

        pos_dims = [Dimension('Y', 'a.u.', np.arange(usize)), Dimension('X', 'a.u.', np.arange(vsize))]
        spec_dims = Dimension('arb', 'a.u.', 1)

        # Need to transpose to for correct reshaping
        image = image.transpose()

        h5_path = super(ImageTranslator, self).translate(h5_path, 'Raw_Data', image.reshape((-1, 1)),
                                                         'Intensity', 'a.u.', pos_dims, spec_dims,
                                                         translator_name='ImageTranslator', parm_dict=image_parms)

        with h5py.File(h5_path, mode='r+') as h5_f:

            # For legacy reasons:
            write_simple_attrs(h5_f, {'data_type': 'ImageData'})

        return h5_path


def read_image(image_path, as_grayscale=True, as_numpy_array=True, *args, **kwargs):
    """
    Read the image file at `image_path` into a numpy array either via numpy (.txt) or via pillow (.jpg, .tif, etc.)

    Parameters
    ----------
    image_path : str
        Path to the image file
    as_grayscale : bool, optional. Default = True
        Whether or not to read the image as a grayscale image
    as_numpy_array : bool, optional. Default = True
        If set to True, the image is read into a numpy array. If not, it is returned as a pillow Image

    Returns
    -------
    image : :class:`numpy.ndarray` or :class:`PIL.Image.Image`
        if `as_numpy_array` is set to True - Array containing the image from the file `image_path`.
        If `as_numpy_array` is set to False - PIL.Image object containing the image within the file - `image_path`.
    """
    ext = os.path.splitext(image_path)[-1]
    if ext in ['.txt', '.csv']:
        if ext == '.csv' and 'delimiter' not in kwargs.keys():
            kwargs['delimiter'] = ','
        img_data = np.loadtxt(image_path, *args, **kwargs)
        if as_numpy_array:
            return img_data
        else:
            img_obj = Image.fromarray(img_data)
            img_obj = img_obj.convert(mode="L")
            return img_obj
    else:
        img_obj = Image.open(image_path)
        if as_grayscale:
            img_obj = img_obj.convert(mode="L", **kwargs)

        if as_numpy_array:
            # Open the image as a numpy array
            return np.asarray(img_obj)
        return img_obj
