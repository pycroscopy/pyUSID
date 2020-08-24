"""
Tools for static and interactive visualization of USID main datasets and scientific imaging and spectroscopy data

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    plot_utils
    jupyter_utils

"""
from warnings import warn

warn('Please use sidpy.viz.plot_utils instead of pyUSID.viz.plot_utils. '
     'pyUSID.plot_utils will be removed in a future release of pyUSID',
     FutureWarning)

from sidpy.viz import plot_utils, jupyter_utils

__all__ = ['plot_utils', 'jupyter_utils']
