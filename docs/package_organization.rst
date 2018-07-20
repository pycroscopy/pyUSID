Package Organization
====================
Sub-packages
------------
The package structure is simple, with 3 main modules:

1. ``io``: utilities that simplify the storage and accessing of data stored in h5USID files
2. ``processing``: utilities and classes that support the piecewise (parallel) processing of arbitrarily large datasets
3. ``viz``: plotting utilities and jupyter widgets that simplify common scientific visualization problems

``io``
~~~~~~~
* ``hdf_utils`` - Utilities for greatly simplifying reading and writing to h5USID files.
* ``write_utils`` - Utilities that assist in writing to HDF5 files
* ``dtype_utils`` - Utilities for data transformation (to and from real-valued, complex-valued, and compound-valued data
  arrays) and validation operations
* ``io_utils`` - Utilities for simplifying common computational, communication, and string formatting operations
* ``USIDataset`` - extends h5py.Dataset. We expect that users will use this class at every opportunity in order to
  simplify common operations on datasets such as interactive visualization in jupyter notebooks, slicing by the
  dataset's N-dimensional form and simplified access to supporting information about the dataset.
* ``Translator`` - An abstract class that provides the blueprint for other Translator classes to extract data and
  meta-data from other raw-data files and write them into h5USID files
* ``ImageTranslator`` - Translates data in common image file formats such as .png and .tiff to a h5USID file
* ``NumpyTranslator`` - A generic translator that simplifies writing of a dataset in memory into a h5USID file

``processing``
~~~~~~~~~~~~~~
* ``Process`` - Modularizes, formalizes, and simplifies robust data processing
* ``parallel_compute()`` - Highly simplified one-line call to perform parallel computing on a data array

``viz``
~~~~~~~
* ``plot_utils`` - utilities to simplify common scientific tasks such as plotting a set of curves within the same or
  separate plots, plotting a set of 2D images in a grid, custom color-bars, etc.
* ``jupyter_utils`` - utilities to enable interactive visualization on generic 4D datasets within jupyter notebooks

Branches
--------
* ``master`` : Stable code based off which the pip installer works. Recommended for most people.
* ``dev`` : Experimental code with new features that will be made available in ``master`` periodically after thorough
  testing. By its very definition, this branch is recommended only for developers.