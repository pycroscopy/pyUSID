.. contents::

Goals
-------

Immediate
~~~~~~~~~
* Solicit brand names:

  * Data model / structure (does NOT include storage in HDF5):

    * Keywords: Open, universal, flexible, Spectroscopy, Imaging, MultiDimensional, Tensor, Model / Data Format
    * We want the name to be easy to understand and descriptive - this data model allows one to represent ANY measurement data regardless of dimensionality, instrument, modality, etc.
    * @rama - Not Another Data Format - NADF
    * @rama - Spectral Data Format - SDF
    * @ssomnath - Open MutiDimensional Imaging and Spectroscopy Data - MISD / OMDISD
    * @ssomnath - Universal Spectral Data - USD
    * @ssomnath - Universal Spectroscopy and Imaging Data - USID
    * @ssomnath - Open HyperSpectral Data - OSD / OHSD
  * File Format - h5<Data Model Acronym / name>
  * package: py<Data Model Acronym / name>

* deploy pyUSID

  * Add requirements.txt
  * Add `coverage calculation <https://coveralls.io>`_

Short-term - by Jul 1 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* New package for facilitating **scalable ensemble runs**:

  * Explore scalability, simplicity, portability of various solutions:

    * `MPI4py <https://github.com/pycroscopy/mpiUSID>`_
    * Dask
    * pyspark
    * ipyparallel...
  * Deploy on CADES SHPC Condo
  * Use stand-alone Signal filtering, Bayesian Inference (GIV) as examples
  * Modify the ``Process`` class or ``parallel_compute()`` to extend the functionality
  * Minimize changes to children classes of ``Process``

Medium-term - by Aug 1 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* region reference related functionality

  * Move functions to separate file outside ``hdf_utils``
  * Relax restrictions with regards to expecting region references
* ``USIDataset`` - Do slicing on ND dataset if available by flattening to 2D and then slicing
* Move ability to export data as csv into USIDataset
* Examples within docstrings for popular functions where possible - not a high priority due to the complexity of functions, need for data, availability of cookbooks
* file dialog for Jupyter not working on Mac OS
* Revisit and address as many pending TODOs as possible
* Technical poster for USID, pyUSID
* ``USIDataset.slice()`` and ``get_n_dim_form()`` should return ``Xarray`` objects
* Itk for visualization - https://github.com/InsightSoftwareConsortium/itk-jupyter-widgets

Long-term
~~~~~~~~~~
* Think about implementing costly algorithms in a deep learning framework like ``TensorFlow`` / ``PyTorch`` to use GPUs. Test with full Bayesian / simple Bayesian (anything computationally expensive)
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
* Look into ``Adios`` (William G; Norbert is affiliated with ADWG)
* ``Pydap.client``: wrapper of ``opendap`` – accessing data remotely and remote execution of notebooks - https://github.com/caseyjlaw/jupyter-notebooks/blob/master/vlite_hdfits_opendap_demo.ipynb
* Alternate visualization packages - http://lightning-viz.org

Scaling to HPC
--------------
We have two kinds of large computational jobs:

#. Embarrassingly parallel analysis / processing. Can be scaled using:

    * MPI - upgraded Process class `already available <https://github.com/pycroscopy/distUSID/tree/pure_mpi>`_ and in the process of rolling changes into pyUSID
    * Dask - An inplace replacement of multiprocessing will work on laptops and clusters. More elegant and easier to write and maintain compared to MPI at the cost of efficiency

      * `simple dask netcdf example <http://matthewrocklin.com/blog/work/2016/02/26/dask-distributed-part-3>`_
    * pySpark?

#. Machine learning and Statistics

    * Continue using a single FAT node for these jobs

      * Advantages:

        * No optimization required
        * Continue using the same scikit learn packages
      * Disadvantage - Is not optimized for HPC

    * OR use pbdR / write pbdPy (wrappers around pbdR)

      * Advantages:

        * Already optimized / mature project
        * In-house project (good support)
      * Disadvantages:

        * Dependant on pbdR for implementing new algorithms


