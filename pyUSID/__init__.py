"""
The pyUSID package.

Submodules
----------

.. autosummary::
    :toctree: _autosummary

"""
from . import io
from .io import *
from . import processing
from .processing import *
from .__version__ import version as __version__

__all__ = ['__version__']
__all__ += io.__all__
__all__ += processing.__all__
