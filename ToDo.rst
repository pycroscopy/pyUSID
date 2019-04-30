.. contents::

Near-term Goals
---------------

* Add requirements.txt
* Simplify ``write_main_dataset()``:

  * accept the N-dimensional array itself. No flattening required by user.
  * Minimize the requirements like quantity and units. Things will look ugly with a lot of unknowns. Is this relaxation worth it.
* Start swapping out hard requirements for numpy to accept Dask arrays as well or by default.
* Simplify documentation or add new series to focus on important functions rather than going over the many developer functions that users will not care about
* Integrate pyUSID with more scientific domain packages to improve adoption. Eg - HyperSpy
* New optional attribute attached to Main dataset - ``type``: ``AFM scan``.
* Simplify ``USIDataset``:

  * This or a new class could extend ``Dask.Array`` instead of ``h5py.Dataset``. Users would not need to look at the 2D form of data.
    A reference to the HDF5 datasets would still be maintained
  * Alternatively, ``USIDataset.data`` could be the N-dimensional form in Dask.
    However, note that this array can be manipulated. Will the changes be reflected into the HDF5 dataset?
* Extend USIDataset to a scientific data type such as an ``PFMDataset`` and add domain specific capabilities.
* ``USIDataset`` - Do slicing on ND dataset if available by flattening to 2D and then slicing.
* Consider Dask based backend for ``Process``
* Fix problems with Travis CI
* Extend ``Process`` class to work with multiple GPUs using `cupy <https://cupy.chainer.org>`_
* Relax restrictions with regards to expecting region references
* Examples within docstrings for popular functions where possible - not a high priority due to the complexity of functions, need for data, availability of cookbooks
* file dialog for Jupyter not working on Mac OS
* Revisit and address as many pending TODOs as possible
* Technical poster for USID, pyUSID

Long-term
~~~~~~~~~
* Itk for visualization - https://github.com/InsightSoftwareConsortium/itk-jupyter-widgets
* Look into versioneer
* A sister package with the base labview subvis that enable writing pycroscopy compatible hdf5 files. The actual acquisition can be ignored.
* Intelligent method (using timing) to ensure that process and Fitter compute over small chunks and write to file periodically. Alternatively expose number of positions to user and provide intelligent guess by default
* function for saving sub-tree to new h5 file
* Windows compatible function for deleting sub-tree
* Profile code to see where things are slow

Back-burner
~~~~~~~~~~~~
* Cloud deployment

  * Container installation - not all that challenging given that pyUSID is pure python
  * Check out HDF5Cloud
  * AWS cloud cluster
* ``Pydap.client``: wrapper of ``opendap`` – accessing data remotely and remote execution of notebooks - https://github.com/caseyjlaw/jupyter-notebooks/blob/master/vlite_hdfits_opendap_demo.ipynb
* Alternate visualization packages - http://lightning-viz.org