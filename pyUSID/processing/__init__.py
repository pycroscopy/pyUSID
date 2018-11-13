"""
Formalizing data processing on USID datasets using parallel computing tools

Submodules
----------

.. autosummary::
    :toctree: _autosummary

"""

from .process import Process
from .comp_utils import parallel_compute
from . import comp_utils

__all__ = ['Process', 'parallel_compute', 'comp_utils']
