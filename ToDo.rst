.. contents::

Goals
-------

Immediate
~~~~~~~~~
* Solicit brand names:

  * Data model (does NOT include storage in HDF5):

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

* Rename USIDataset to USIDataset
* deploy pyUSID

  * continuous integration
  * Add requirements.txt
  * documentation, website, etc.
  * Add `coverage calculation <https://coveralls.io>`_
  * pypi and conda installers

Short-term - by Jul 1 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* region reference related functionality

  * Move functions to separate file outside ``hdf_utils``
  * Relax restrictions with regards to expecting region references
* ``USIDataset`` - Do slicing on ND dataset if available by flattening to 2D and then slicing
* Chris - ``Image Processing`` must be a subclass of ``Process`` and implement resuming of computation and checking for old (both already handled quite well in Process itself) - here only because it is used and requested frequently + should not be difficult to restructure.
* Move ability to export data as csv into USIDataset
* Examples within docs for popular functions where possible
* file dialog for Jupyter not working on Mac OS
* Revisit and address as many pending TODOs as possible

Medium-term - by Aug 1 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Technical poster for USID, pyUSID
* Explore Azure Notebooks for live tutorials
* ``USIDataset.slice()`` and ``get_n_dim_form()`` should return ``Xarray`` objects
* Itk for visualization - https://github.com/InsightSoftwareConsortium/itk-jupyter-widgets
* New package for facilitating **scalable ensemble runs**:

  * Compare scalability, simplicity, portability of various solutions:
    
    * MPI4py
    * Dask (Matthew Rocklin)
    * pyspark
    * ipyparallel... 
  * Deploy on CADES SHPC Condo, Eos, Rhea (CPU partition).
  * Use stand-alone GIV or SHO Fitting as an example
  * Develop some generalized class equivalent to / close to ``Process``

Long-term
~~~~~~~~~~
* Rewrite ``Process`` to use ``Dask`` and ignore ``parallel_compute()`` - try on SHO guess
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
* Look into ``Adios`` i(William G; Norbert is affiliated with ADWG)
* ``Pydap.client``: wrapper of ``opendap`` – accessing data remotely and remote execution of notebooks - https://github.com/caseyjlaw/jupyter-notebooks/blob/master/vlite_hdfits_opendap_demo.ipynb
* Alternate visualization packages - http://lightning-viz.org

Scaling to HPC
--------------
We have two kinds of large computational jobs and one kind of large I/O job:

* I/O - reading and writing large amounts of data:

  * MPI clearly works with very high performance parallel read and write
  * Dask also works but performance is a question. Look at NERSC (Matthew Rocklin et al.)
  * Spark / HDFS requires investigation - Apparently does not work well with HDF5 files

* Computation:

  1. Machine learning and Statistics

    * Use custom algorithms developed for BEAM - NO one is willing to salvage code

      * Advantage - Optimized (and tested) for various HPC environments
      * Disadvantages:

        * Need to integrate non-python code
        * We only have a handful of these. NOT future compatible

    * OR continue using a single FAT node for these jobs

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

  2. Embarrasingly parallel analysis / processing. Can be scaled using:

    * Dask - An inplace replacement of multiprocessing will work on laptops and clusters. More elegant and easier to write and maintain compared to MPI at the cost of efficiency

      * simple dask netcdf example: http://matthewrocklin.com/blog/work/2016/02/26/dask-distributed-part-3
    * MPI - Need alternatives to Optimize / Process classes - Best efficiency but a pain to implement
    * Spark?
    * ipyParallel?
