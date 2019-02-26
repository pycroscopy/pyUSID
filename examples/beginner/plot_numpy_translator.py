"""
======================================================================================
03.a NumpyTranslator for easy USID HDF5 output
======================================================================================

**Suhas Somnath**

2/25/2019

This document illustrates the ``pyUSID.NumpyTranslator`` which greatly simplifies the process of writing data in
numpy arrays into **Universal Spectroscopy and Imaging Data (USID)** formatted HDF5 files (referred to as a **h5USID**
files)


Introduction
------------
In most scientific disciplines, commercial instruments tend to write the data and metadata out into proprietary file
formats that significantly impede access to the data and metadata, thwart sharing of data and correlation of data from
multiple instruments, and complicate long-term archival, among other things. One of the data wrangling steps in science
is the extraction of the data and metadata out of the proprietary file formats and writing the information into files
that are easier to access, share, etc. The overwhelming part of this data wrangling effort is in investigating how to
extract the data and metadata into memory. Once the information is accessible in the computer memory, such as in the
form of numpy arrays, scientists have a wide variety of tools to write the data out into files.

Simpler data such as images or single spectra can easily be written into plain text files. Simple or complex / large /
multidimensional data can certainly be stored as numpy data files. However, there are significant drawbacks to writing
data into non-standardized structures or file formats. First, while the structure of the data and metadata may be
intuitive for the original author of the data, that may not be the case for another researcher. Furthermore, such
formatting may change from a day-to-day basis. As a consequence, it becomes challenging to develop code that can accept
such data whose format keeps changing.

One solution to these challenges is to write the data out into standardized files such as ``h5USID`` files.
The USID model aims to make data access, storage, curation, etc. simply by storing the data along with all
relevant parameters in a single file (HDF5 for now).

The process of copying data from the original format to **h5USID** files is called
**Translation** and the classes available in pyUSID and children packages such as pycroscopy that perform these
operation are called **Translators**.

This document will show how one can write data present as numpy arrays can be written out easily into h5USID files
using the NumpyTranslator. This topic has been divided into two parts, the first part (this document) will assume that
we have managed to read the proprietary file format and extract all the data and metadata into memory. The
`second part <./plot_translation_extraction.html>`_ will go over the complementary portions that have been ignored in
this example.

In this document, we will focus only on writing the data to h5USID.
We will be working with a ``Band Excitation Piezoresponse Force Microscopy (BE-PFM)`` imaging dataset
acquired from an atomic force microscope. In this dataset, a spectra was collected for each position in a two
# dimensional grid of spatial locations, thereby resulting in a 3D dataset.

Recommended pre-requisite reading
---------------------------------

Before proceeding with this example, we recommend reading the previous documents to learn more about:

* `Universal Spectroscopic and Imaging Data (USID) model </../../../USID/usid_model.html>`_
* The `USIDataset <./plot_usi_dataset.html>`_ class

.. tip::
    You can download and run this document as a Jupyter notebook using the link at the bottom of this page.

Import all necessary packages
-----------------------------
There are a few setup procedures that need to be followed before any code is written. In this step, we simply load a
few python packages that will be necessary in the later steps.
"""

# Ensure python 3 compatibility:
from __future__ import division, print_function, absolute_import, unicode_literals

# The package for accessing files in directories, etc.:
import os
import subprocess
import sys

# Warning package in case something goes wrong
from warnings import warn

# The mathematical computation package:
import numpy as np

# The package used for creating and manipulating HDF5 files:
import h5py

# Packages for plotting:
import matplotlib.pyplot as plt


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


# Package for downloading online files:
try:
    # This package is not part of anaconda and may need to be installed.
    import wget
except ImportError:
    warn('wget not found.  Will install with pip.')
    import pip
    install(wget)
    import wget

# Finally import pyUSID:
try:
    import pyUSID as usid
except ImportError:
    warn('pyUSID not found.  Will install with pip.')
    import pip
    install('pyUSID')
    import pyUSID as usid

####################################################################################
# Procure the data
# ================
# Here, we will download an existing dataset from the pyUSID GitHub repository and extract the data.
# It is not important that you understand how we got this data.

h5_in_path = 'temp.h5'
url = 'https://raw.githubusercontent.com/pycroscopy/pyUSID/master/data/BELine_0004.h5'
if os.path.exists(h5_in_path):
    os.remove(h5_in_path)
_ = wget.download(url, h5_in_path, bar=None)
h5_file_in = h5py.File(h5_in_path, mode='r')
parm_dict = usid.hdf_utils.get_attributes(h5_file_in['Measurement_000/'])
h5_main_in = usid.USIDataset(h5_file_in['Measurement_000/Channel_000/Raw_Data'])
main_data = h5_main_in.get_n_dim_form()
freq_vec = h5_main_in.get_spec_values('Frequency')
x_vec = h5_main_in.get_pos_values('X')
y_vec = h5_main_in.get_pos_values('Y')


def get_qty_units(descriptor):
    temp = descriptor.split(' ')
    return temp[-1].replace('(','').replace(')',''), ' '.join(temp[:-1])


main_units, main_qty = get_qty_units(h5_main_in.data_descriptor)
freq_units, freq_qty = get_qty_units(h5_main_in.spec_dim_descriptors[-1])
x_units, x_qty = get_qty_units(h5_main_in.pos_dim_descriptors[0])
y_units, y_qty = get_qty_units(h5_main_in.pos_dim_descriptors[1])

####################################################################################
# Understanding the data
# ======================
# Lets take a look at some basic information about the dataset we are working with:

print('Data of shape: {} and data-type: {} containing quantity: {} and of units {}'
      '.'.format(main_data.shape, main_data.dtype, main_units, main_qty))
print('Position dimension 0: {}: of size: {},  and units: {}'
      '.'.format(x_qty, x_vec.shape, x_units))
print('Position dimension 1: {}: of size: {},  and units: {}'
      '.'.format(y_qty, y_vec.shape, y_units))
print('Spectroscopic dimension 0: {}: of size: {},  and units: {}'
      '.'.format(freq_qty, freq_vec.shape, freq_units))

####################################################################################
# We even have access to some important parameters that are relevant to provide full context to this measurement:

for key, val in parm_dict.items():
    print('{} : {}'.format(key, val))

####################################################################################
# Here is a visualization of the spectra at a few locations:

fig, axes = usid.plot_utils.plot_complex_spectra(h5_main_in[()], num_comps=6, amp_units='V',
                                                 subtitle_prefix='Position', evenly_spaced=True,
                                                 x_label=h5_main_in.spec_dim_descriptors[0])

####################################################################################
# Here is a visualization of spatial maps at different frequencies (spectroscopic dimension)

fig, axes = usid.plot_utils.plot_map_stack(np.abs(main_data), reverse_dims=True, pad_mult=(0.15, 0.15),
                                           title='Spatial maps at different frequencies', stdevs=2,
                                           color_bar_mode='single', num_ticks=3, x_vec=x_vec, y_vec=y_vec,
                                           evenly_spaced=True, fig_mult=(3, 3), title_yoffset=0.95)
for axis, freq_ind in zip(axes, np.linspace(0, len(freq_vec), 9, endpoint=False, dtype=np.uint)):
    axis.set_title('Frequency = {} kHz'.format(np.rint(freq_vec[freq_ind] * 1E-3)))

h5_file_in.close()


####################################################################################
# 1. NumpyTranslator as a quick file writer
# ======================================
# Though not intended to be used in this manner, the ``NumpyTranslator`` can be used in scripts to quickly write out
# data into a HDF5 file. The benefit over simply saving data using
# `numpy.save() <https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html>`_ is that the data will be
# written in a way that it will be accessible by the `pyUSID.USIDataset <./plot_usi_dataset.html>`_ class that offers
# several handy capabilities. Such usage of the ``NumpyTranslator`` offers minimal benefits over using the
# `pyUSID.hdf_utils.write_main_data() <../intermediate/plot_hdf_utils_write.html#write-main-dataset>`_ function,
# upon which it is based.
#
# a. Preparing Ancillary information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Before the ``NumpyTranslator`` can be used, we need to define the ancillary datasets using simple
# `pyUSID.Dimension <../intermediate/plot_write_utils.html#dimension>`_ objects. These ``Dimension`` objects are simply
# descriptors of dimensions and take the name of the quantity, physical units, and the values over which the dimension
# was varied. Both, the `Position` and `Spectroscopic` dimensions need to defined using ``Dimension`` objects and the
# ``Dimension`` objects should be arranged from fastest varying to slowest varying dimensions.

pos_dims = [usid.Dimension(x_qty, x_units, x_vec),
            usid.Dimension(y_qty, y_units, y_vec)]
spec_dims = usid.Dimension(freq_qty, freq_units, freq_vec)

####################################################################################
# b. Reshaping the data
# ~~~~~~~~~~~~~~~~~~~~~
# Recall that ``Main`` datasets in USID are two dimensional in shape where all position dimensions (``X``, and ``Y`` in
# this case) are collapsed along the first axis and the spectroscopic dimensions (``Frequency`` in this case) are
# collapsed along the second axis. Currently, the data we have has the original 3-dimensional shape. We need to collapse
# the first two axes into a single axis.

print('Shape of original data: {}'.format(main_data.shape))
main_data_2d = main_data.reshape(-1, main_data.shape[-1])
print('Shape of flattened 2D data: {}'.format(main_data_2d.shape))

####################################################################################
# c. Using NumpyTranslator.translate()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We are now ready to use the ``NumpyTranslator!``
# The NumpyTranslator simplifies the creation of h5USID files. It handles the HDF5 file creation,
# HDF5 dataset creation and writing, creation of ancillary HDF5 datasets, group creation, writing parameters, linking
# ancillary datasets to the main dataset etc. With a single call to the ``translate()`` function of the
# ``NumpyTranslator``, we complete the translation process:

h5_path_out_1 = 'BE_external.h5'
main_dataset_name = 'Raw_Data'

tran = usid.NumpyTranslator()
h5_path_out_1 = tran.translate(h5_path_out_1, main_dataset_name, main_data_2d,  main_units, main_qty, pos_dims, spec_dims,
                               parm_dict=parm_dict)

####################################################################################
# Verifying the new H5 file
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Let us open up the new USID formatted HDF5 file that the ``NumpyTranslator`` created to verify that all the data and
# metadata was correctly captured.
with h5py.File(h5_path_out_1, mode='r') as h5_file:
    print('Contents of the file:')
    usid.hdf_utils.print_tree(h5_file)
    print('Parameters embedded in the file:')
    print('--------------------------------')
    for key, val in usid.hdf_utils.get_attributes(h5_file['Measurement_000']).items():
        print('{} : {}'.format(key, val))
    h5_main = usid.hdf_utils.get_all_main(h5_file)[-1]
    print('--------------------------------')
    print('Comprehensive information about the Main dataset:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(h5_main)
    print('--------------------------------')
    fig, axes = h5_main.visualize(slice_dict={'Frequency': 60})
    fig.suptitle('Spatial maps at Frequency = %2.1f kHz ' % (h5_main.get_spec_values('Frequency')[60] * 1E-3),
                 y=1.1)
    fig, axes = h5_main.visualize(slice_dict={'X': 65, 'Y': 95})
    fig.suptitle('Spectra at X = %2.1f um an Y = %2.1f um' % (h5_main.get_pos_values('X')[65] * 1E+6,
                                                            h5_main.get_pos_values('Y')[95] * 1E+6),
                 y=1.1)

####################################################################################
# 2. Extending the NumpyTranslator
# ================================
# What we have done above is essentially, write real measurement data and metadata into a standardized USID HDF5 file.
# As is evident above, the process of writing to the HDF5 file is rather simple because of the ``NumpyTranslator``.
# However, the above code is part of a script that is susceptible to edits. Minor changes in the naming / formatting of
# certain strings, reshaping of the datasets can very quickly break analysis or visualization code later on.
# Encapsulating the data reading and writing process into a formal ``Translator`` class also makes it easier for others
# to use it and write data into the same consistent format. In fact, upon writing the class, proprietary data files
# can be translated using just two lines as we will see below.
# Therefore, we recommend extending the ``NumpyTranslator`` class, when possible, instead of using it independently
# like a function.
#
# Defining the class
# ~~~~~~~~~~~~~~~~~~
# Writing a python class that extends the ``NumpyTranslator`` class is far less intimidating than it sounds. The code
# that goes into the class is virtually identical to what has been written above. In fact the code that will be written
# below is very similar to real ``Translator`` classes found in our sister package - `pycroscopy`. Again, for the sake
# of brevity, we will be skipping the sections that deal with extracting the metadata and data from the proprietary file
# and concentrate on the file writing aspects of the ``Translator`` class


class ExampleTranslator(usid.NumpyTranslator):
    """
    The above definition of the class states that our ExampleTranslator inherits all the capabilities and
    behaviors of the NumpyTranslator class and builds on top of it
    """

    def translate(self, input_file_path):
        """
        Extracts the data and metadata out of proprietary formatted files and writes it into a SID formatted HDF5 file

        Parameters
        ----------
        input_file_path : str
            Path to the input data file containing all the information

        Returns
        -------
        h5_path_out_2 : str
            Path to the USID HDF5 output file
        """

        """
        Opening and extracting all necessary data and metadata here
        This section is heavily dependent on the proprietary file format
        """
        h5_file_in = h5py.File(input_file_path, mode='r')
        parm_dict = usid.hdf_utils.get_attributes(h5_file_in['Measurement_000/'])
        h5_main_in = usid.USIDataset(h5_file_in['Measurement_000/Channel_000/Raw_Data'])
        main_data = h5_main_in.get_n_dim_form()
        freq_vec = h5_main_in.get_spec_values('Frequency')
        x_vec = h5_main_in.get_pos_values('X')
        y_vec = h5_main_in.get_pos_values('Y')

        def get_qty_units(descriptor):
            temp = descriptor.split(' ')
            return temp[-1], ' '.join(temp[:-1])

        main_units, main_qty = get_qty_units(h5_main_in.data_descriptor)
        freq_units, freq_qty = get_qty_units(h5_main_in.spec_dim_descriptors[-1])
        x_units, x_qty = get_qty_units(h5_main_in.pos_dim_descriptors[0])
        y_units, y_qty = get_qty_units(h5_main_in.pos_dim_descriptors[1])
        h5_file_in.close()

        """
        We begin the USID / pyUSID specific portions here
        -------------------------------------------------
        a. Expressing the Position and Spectroscopic Dimensions using pyUSID.Dimension objects
        """
        pos_dims = [usid.Dimension(x_qty, x_units, x_vec),
                    usid.Dimension(y_qty, y_units, y_vec)]
        spec_dims = usid.Dimension(freq_qty, freq_units, freq_vec)

        """
        b. Reshaping the main data as a 2D matrix:
        """
        main_data_2d = main_data.reshape(-1, main_data.shape[-1])

        """
        c. Call the translate() function of the base NumpyTranslator class        
        """
        h5_path_out_1 = 'BE_from_class.h5'
        main_dataset_name = 'Raw_Data'

        h5_path_out_1 = super(ExampleTranslator, self).translate(h5_path_out_1, main_dataset_name, main_data_2d,
                                                                main_units, main_qty, pos_dims, spec_dims,
                                                                parm_dict=parm_dict)

        return h5_path_out_1

####################################################################################
# As you could tell by now, the code in this class is virtually identical to the code in the first section of the
# document. Perhaps the biggest difference between the two codes comes in section ``c`.
# In the first section, we had instantiated the ``NumpyTranslator`` and called its ``translate()`` command:
#
# .. code-block:: python
#
#     tran = usid.NumpyTranslator()
#     h5_path_out_1 = tran.translate(...)
#
# In the case of the ``ExampleTranslator`` class above, we define the class itself as an extension / child class of the
# ``NumpyTranslator`` in this line:
#
# .. code-block:: python
#
#     class ExampleTranslator(usid.NumpyTranslator):
#
# This means that our ``ExampleTranslator`` class inherits all the capabilities (including our favorite -
# ``translate()`` function) and behaviors of the ``NumpyTranslator`` class and builds on top of it. This is why we don't
# need to instantiate the ``NumpyTranslator`` in section ``c``. All we are doing in our ``translate()`` function is
# adding the intelligence that is relevant to our specific scientific example and piggybacking on the many capabilities
# of the ``NumpyTranslator`` class for the actual file writing. This piggybacking is visible in the last line:
#
# .. code-block:: python
#
#     h5_path_out_1 = super(ExampleTranslator, self).translate()
#
# Essentially, we are asking ``NumpyTranslator.translate()`` to take over and do the rest.
#
# Using this ExampleTranslator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# What we did above is provide a template for what should happen when someone provides an input file. We have not really
# tried it out yet. The lines below will illustrate how easy it becomes to perform `translations` from now on:

# instantiate the class first:
my_tran = ExampleTranslator()
h5_path_out_2 = my_tran.translate(h5_in_path)

####################################################################################
# Verifying the output again
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
with h5py.File(h5_path_out_1, mode='r') as h5_file:
    print('Contents of the file:')
    usid.hdf_utils.print_tree(h5_file)
    print('Parameters embedded in the file:')
    print('--------------------------------')
    for key, val in usid.hdf_utils.get_attributes(h5_file['Measurement_000']).items():
        print('{} : {}'.format(key, val))
    h5_main = usid.hdf_utils.get_all_main(h5_file)[-1]
    print('--------------------------------')
    print('Comprehensive information about the Main dataset:')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(h5_main)
    print('--------------------------------')
    fig, axes = h5_main.visualize(slice_dict={'Frequency': 60})
    fig.suptitle('Spatial maps at Frequency = %2.1f kHz ' % (h5_main.get_spec_values('Frequency')[60] * 1E-3),
                 y=1.1)
    fig, axes = h5_main.visualize(slice_dict={'X': 65, 'Y': 95})
    fig.suptitle('Spectra at X = %2.1f um an Y = %2.1f um' % (h5_main.get_pos_values('X')[65] * 1E+6,
                                                            h5_main.get_pos_values('Y')[95] * 1E+6),
                 y=1.1)

####################################################################################
# Limits of the NumpyTranslator
# =============================
# The ``NumpyTranslator`` is perfect when one is dealing with a **single** USID `Main` dataset. However, if the
# proprietary file contained multiple such 3D hyperspectral images, one would need to use the lower-level functions that
# power the ``NumpyTranslator``. pyUSID offers
# `several functions <./intermediate/plot_hdf_utils_write.html#write-main-dataset>`_ that make it easy to handle such
# more involved translations.
#
# More information
# ================
# As mentioned earlier, we have a complementary `example document <./plot_translation_extraction.html>`_ that goes over
# the sections that was ignored in this document - specifically the process of extracting the metadata and data from the
# proprietary file. The reader is highly encouraged to read that document as well.
#
# Our sister class - pycroscopy, has several
# `translators <https://github.com/pycroscopy/pycroscopy/tree/master/pycroscopy/io/translators>`_ that translate popular
# file formats generated by nanoscale imaging instruments. Some of the translators extend the ``NumpyTranslator`` like
# we did above, while others use the low-level functions in ``pyUSID.hdf_utils``.
#
# Cleaning up
# ~~~~~~~~~~~
# Remove both the original and translated files:

os.remove(h5_path_out_2)
os.remove(h5_path_out_1)
os.remove(h5_in_path)
