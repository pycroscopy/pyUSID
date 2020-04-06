"""
===============================================================================
01. pyUSID in 10 minutes
===============================================================================

**Suhas Somnath**

4/10/2020

**This document serves as a quick primer to the essential components of
pyUSID**

Recommended pre-requisite reading
---------------------------------
* `Universal Spectroscopic and Imaging Data (USID) model
  </../../../USID/usid_model.html>`_
* `Crash course on HDF5 and h5py <./plot_h5py.html>`_

**This document is under construction**

* first make some dummy data using Numpy or Dask - let it have 3-4 dimensions. Use examples in the [hdf_utils cookbook](https://pycroscopy.github.io/pyUSID/auto_examples/intermediate/plot_hdf_utils_write.html)
* ``hdf_utils.reshape_from_n_dims()`` -> convert main 4D data to 2D
* Construct ``write_utils.Dimension`` objects. Again take inspiration from above example
* Use ``plot_utils.plot_map()`` and one other function here just to point out that such handy functions exist
* Use ``ArrayTranslator`` to write to file
* Show how one would open an existing HDF5 file using ``h5py.File``
* ``hdf_utils.print_tree(h5_file)`` to show all existing contents. Maybe set to ``main_only`` to simplify
* Demonstrate creation of USIDataset via: ``USIDataset(main dataset here)``
* ``print(usi_dataset_object)``
* ``usi_dataset_object.get_n_dim_form()``
* ``usi_dataset_object.slice()`` -> Maybe use ``plot_utils`` here to show results.
* ``usi_dataset_object.visualize()`` -> slice to 2D for simpler / static visualization but point out that interactive widgets are available for > 2D visualization
* ``usi_dataset_object.get_unit_values()``
* Make an indexed group for results using ``hdf_utils.create_indexed_group``. Point out that ``hdf_utils.create_results_group()`` does something similar
* ``hdf_utils.write_simple_attrs()`` to group.
* Show ``hdf_utils.get_attributes()`` and ``hdf_utils.get_attr()`` from this group
* Illustrate how one might create results based on processing performed on the original data
using ``hdf_utils.write_main_data``
* ``hdf_utils.check_if_main(results_dataset)`` to make sure that it is main. No surprises here
* ``hdf_utils.get_all_main(h5_file)`` in the end to show that you will get more than 1 dataset
"""
