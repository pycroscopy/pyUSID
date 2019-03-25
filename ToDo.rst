.. contents::

Goals
-------

* Add requirements.txt
* Fix problems with Travis CI
* Add `coverage calculation <https://coveralls.io>`_
* Extend ``Process`` class to work with multiple GPUs using `cupy <https://cupy.chainer.org>`_
* Consider Dask based backend for ``Process``
* Relax restrictions with regards to expecting region references
* ``USIDataset`` - Do slicing on ND dataset if available by flattening to 2D and then slicing. Work entirely on Dask instead of numpy (in-memory only)
* Examples within docstrings for popular functions where possible - not a high priority due to the complexity of functions, need for data, availability of cookbooks
* file dialog for Jupyter not working on Mac OS
* Revisit and address as many pending TODOs as possible
* Technical poster for USID, pyUSID
* Itk for visualization - https://github.com/InsightSoftwareConsortium/itk-jupyter-widgets

Long-term
~~~~~~~~~
* Look into versioneer
* A sister package with the base labview subvis that enable writing pycroscopy compatible hdf5 files. The actual acquisition can be ignored.
* Intelligent method (using timing) to ensure that process and Fitter compute over small chunks and write to file periodically. Alternatively expose number of positions to user and provide intelligent guess by default
* function for saving sub-tree to new h5 file
* Windows compatible function for deleting sub-tree
* Profile code to see where things are slow

Back-burner
~~~~~~~~~~~~
* Cloud deployment

  * Container installation
  * Check out HDF5Cloud
  * AWS cloud cluster
* ``Pydap.client``: wrapper of ``opendap`` – accessing data remotely and remote execution of notebooks - https://github.com/caseyjlaw/jupyter-notebooks/blob/master/vlite_hdfits_opendap_demo.ipynb
* Alternate visualization packages - http://lightning-viz.org