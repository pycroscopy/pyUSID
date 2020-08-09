"""
Formalizing data processing on USID datasets using parallel computing tools

Submodules
----------

.. autosummary::
    :toctree: _autosummary

"""

from .process import Process
from sidpy.proc import comp_utils
from sidpy.proc.comp_utils import parallel_compute

__all__ = ['Process', 'parallel_compute', 'comp_utils']
