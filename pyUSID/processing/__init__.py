"""
Formalizing data processing on USID datasets using parallel computing tools

Submodules
----------

.. autosummary::
    :toctree: _autosummary

"""

from .process import Process, parallel_compute

__all__ = ['parallel_compute', 'Process']
