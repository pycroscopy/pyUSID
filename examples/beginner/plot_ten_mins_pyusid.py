"""
===============================================================================
01. pyUSID in 10 minutes
===============================================================================

**Suhas Somnath**

**Rajiv Giridharagopal (University of Washington)**

4/24/2020

**This document serves as a quick primer to the essential components of
pyUSID**


Recommended pre-requisite reading
---------------------------------
* `Universal Spectroscopic and Imaging Data (USID) model
  </../../../USID/usid_model.html>`_
* `Crash course on HDF5 and h5py <./plot_h5py.html>`_
"""

import sidpy
import pyUSID as usid
import numpy as np
import h5py

########################################################################################################################
# Converting a 3D Matrix to USID
# ===================
# 
# As an example of how to use pyUSID to reformat data for analysis, let's take
# an image with some time-series at each pixel
# Let's imagine we have a 10 x 10 array that measures the height. But we're applying
# a voltage to oscillate the height at 2 Hz for 1 second, with 10 Hz sampling and a
# slight phase shift and amplitude shift at each pixel.
#
# In other words, a 10 x 10 x 100 array

# Create some data
data = np.zeros((10,10,100))
phase = np.reshape(np.linspace(-np.pi, np.pi, 100), [10,10])
amp = np.reshape(np.linspace(1, 5, 100), [10,10])
for x in np.arange(0,100):
    for r in np.arange(data.shape[0]):
        for c in np.arange(data.shape[1]):
            data[r,c,x] = amp[r][c] * np.sin(2*np.pi * 2*x/100 + phase[r][c]) 

# To visualize a 3D stack, there's a handy function in sidpy
_ = sidpy.plot_utils.plot_map_stack(data, num_comps=4, fig_mult=(2,10), pad_mult=(0.01,.4));

########################################################################################################################
# Instead of being 3D, we need it to be (10x10, 100) in 2D for USID. 
# First, let's define the position dimensions, which are 10 x 10 nm
pos = usid.hdf_utils.build_ind_val_matrices([np.arange(0,10e-9, 1e-9), np.arange(0,10e-9, 1e-9)], 
                                             is_spectral=False)

# pos[0] are the indices (0,1,2...), pos[1] are the values (0,1e-9, 2e-9...)
# Second, let's define he spectral dimensions, which is our 1 s long waveform
spec = usid.hdf_utils.build_ind_val_matrices([np.arange(0,1,0.01)], is_spectral=True)

# Finally, we make our USID-compatible dataset, which is now (100, 100)
data_reshape, _ = usid.hdf_utils.reshape_from_n_dims(data, pos[0], spec[0])

########################################################################################################################
# Creating a USID Dataset 
# ===================
#
# Now that we have a USID-compatible dataset, we're almost there
# Let's actually create the USID dataset

# Use the USID Dimension to define the dimensions directly, including the units
pos_dims = [usid.write_utils.Dimension('Rows', 'm', np.arange(0,10e-9, 1e-9)),
            usid.write_utils.Dimension('Cols', 'm', np.arange(0,10e-9, 1e-9))]

spec_dims = [usid.write_utils.Dimension('Time', 's', np.arange(0,1,0.01))]

# Use the ArrayTranslator to create our file
# Let's define our HDF55 file. This is the name of a file we'll write into
h5_path = 'cookbook_data.h5'

# Now let's use the ArrayTranslator to write our data to an HDF5 file
tran = usid.io.numpy_translator.ArrayTranslator()
tran.translate(h5_path, 'data', data_reshape, 'Height', 'm',
               pos_dims, spec_dims)

########################################################################################################################
# That's it! We're done. It's just a few lines of code to convert a random matrix
# of data into a USID format. We just need to know a little about our data.
# Now that our cookbook_data is full of delicious data, let's crack it open.
#
# We can open the file by using the h5py command
h5_file = h5py.File(h5_path, mode='r+')

########################################################################################################################
# ``print_tree`` shows all the contents in this HDF5 file
sidpy.hdf_utils.print_tree(h5_file, rel_paths=True)

########################################################################################################################
# Our data are in the ``Raw_Data`` dataset. How do we extract our data?
# First let's print all the Main datasets

print(usid.hdf_utils.get_all_main(h5_file))

########################################################################################################################
# This is a list of all the Main datasets. In this case there's only a single one. 
# So we can access our data in two ways:

# Option 1:
# Access a specific dataset in the list of Main datasets
data_usid = usid.hdf_utils.get_all_main(h5_file)[0]

# Option 2:
# We could highlight the path in the tree and access it directly
data_usid = h5_file['Measurement_000/Channel_000/Raw_Data']

########################################################################################################################
# Note that reading the dataset in this manual manner only gives us the
# standard ``h5py.Dataset``:

print(data_usid)

########################################################################################################################
# Lastly, let's upgrade from ``h5py.Dataset`` to a ``USIDataset``

data_usid = usid.USIDataset(data_usid)
print(data_usid) 

########################################################################################################################
# Exploring a USID Dataset
# ===================
#
# There are lots of advantages to this kind of dataset

# Let's take a look via the handy visualize option
# What is the signal vs time at pixel (5,3)
_, _ = data_usid.visualize(slice_dict={'Rows': 5, 'Cols': 3})

# What about the image at 0.3 seconds?
_, _ = data_usid.visualize(slice_dict={'Time': 30})

########################################################################################################################
# To access our data in the HDF5 dataset directly, we can use the ``[()]`` shortcut
two_dim_form = data_usid[()]
print(type(two_dim_form))
print(two_dim_form.shape)

########################################################################################################################
# Given that we are working on 3D dataset, we want reshape the flattened data
# present in the HDF5 dataset back to the original form of (10,10,100)

n_dim_form = data_usid.get_n_dim_form()
print(type(n_dim_form))
print(n_dim_form.shape)

########################################################################################################################
# What are the properties of our data?
print('Rows = \n{}'.format(data_usid.get_pos_values('Rows')))
print('Cols = \n{}'.format(data_usid.get_pos_values('Cols')))
print('Time = \n{}'.format(data_usid.get_spec_values('Time')))

########################################################################################################################
# Attributes of the data when it was written
print(sidpy.hdf_utils.get_attributes(data_usid))

########################################################################################################################
# To get the path of the Main dataset within the HDF5 file
print(data_usid.name)

########################################################################################################################
# And to get the parent folder of this Dataset, you use
print(data_usid.parent.name)

########################################################################################################################
# Adding some new data
# ====================
#
# Let's say we process our data using some method and want to save that process.
# For the sake of argument, we'll just make a matrix that's the  magnitude^2

data_proc = np.array(data_usid[()]**2)

########################################################################################################################
# Let's create a new group within this file to store our results
result_group = usid.hdf_utils.create_indexed_group(h5_file[data_usid.parent.name], 'Magnitude')
print(result_group)

########################################################################################################################
# The "indexed" part means it appends 000, 001, etc if we do this many times
# Many built-in pyUSID and pycroscopy command do this so we don't overwrite old
# results. Because of the power of HDF5 we can go back to old processing and see!
#
# There's an analogous command create_results_group if you'd like
#
# Anyway let's print our tree out for good measure
sidpy.hdf_utils.print_tree(h5_file, rel_paths=True)

########################################################################################################################
# Now we want to add our data. But we want to add some attributes as well to describe
# what we've done to our data. Attributes are a dictionary, so let's create one.
attrs = {'Method': 'Magnitude_Squared', 'units': 'm^2'}
sidpy.hdf_utils.write_simple_attrs(result_group, attrs)

########################################################################################################################
# Now, let's write a new main dataset
data_result = usid.hdf_utils.write_main_dataset(result_group, 
                                                data_proc, 
                                                'Data Squared', 
                                                'Height Squared', 'm^2', 
                                                pos_dims, 
                                                spec_dims)

########################################################################################################################
# This populates our new folder with the new data! Let's look for the Magnitude folder in our tree:
sidpy.hdf_utils.print_tree(h5_file, rel_paths=True)

########################################################################################################################
# If we get all the Main datasets, we see a new dataset pop up in our list.
print(usid.hdf_utils.get_all_main(h5_file))

########################################################################################################################
# And to verify the attributes for the data:
print(sidpy.hdf_utils.get_attributes(data_result))

########################################################################################################################
# ... and the data_group:
print(sidpy.hdf_utils.get_attributes(data_result.parent))

########################################################################################################################
# Lastly, to verify this dataset is a Main dataset (with position and spectral dimensions)
print(sidpy.hdf_utils.get_attributes(data_result))
