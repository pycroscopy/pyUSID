"""
Tools to read, write data in h5USID files

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    hdf_utils
    image
    array_translator
    usi_data
    dimension
    translator
    anc_build_utils

"""
from sidpy.sid.translator import Translator
from . import usi_data
from . import array_translator
from . import hdf_utils
from . import anc_build_utils
from . import dimension

from .usi_data import USIDataset
from .array_translator import ArrayTranslator
from .image import ImageTranslator
from .dimension import DimType, Dimension

# For legacy reasons
write_utils = anc_build_utils
numpy_translator = array_translator
NumpyTranslator = ArrayTranslator

__all__ = ['USIDataset', 'hdf_utils', 'write_utils', 'Dimension', 'DimType',
           'ImageTranslator', 'ArrayTranslator', 'Translator',
           'anc_build_utils',
           'write_utils', 'numpy_translator', 'NumpyTranslator']
