======
pyUSID
======

**Python framework for storing and processing scientific data formatted according to the USID model**

* The `Universal Spectroscopic and Imaging Data (USID) model <../USID/index.html>`_:

  * facilitates the representation of any spectroscopic or imaging data regardless of its origin, modality, size, or dimensionality.
  * enables the development of instrument- and modality- agnostic data processing and analysis algorithms.
  * is just a definition or a blueprint rather than something tangible and readily usable.
* pyUSID is a `python <http://www.python.org/>`_ package that currently provides two pieces of functionality:

  #. **io**: Primarily, it enables the storage and access of USID in **hierarchical data format** `(HDF5) <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`_ files (referred to as h5USID files) using python
  #. **processing**: It provides a framework for formulating scientific problems into computational problems.
     See `pycroscopy <../pycroscopy/about.html>`_ - a sister project that uses pyUSID for analysis of microscopy data.

* pyUSID uses a data-centric approach wherein the raw data collected from the instrument, results from analysis
  and processing routines are all written to the same h5USID file for traceability, reproducibility, and provenance.
* Just as scipy uses numpy underneath, scientific packages like **pycroscopy** use **pyUSID** for all file-handling, data processing, and generating plots for journal publications
* pyUSID uses popular packages such as numpy, h5py, joblib, matplotlib, etc. for most of the storage, computation, and visualization.
* For more information, please see our recent `Arxiv <https://arxiv.org/abs/1903.09515>`_ **paper**
* See a high-level overview of pyUSID in this `presentation <https://github.com/pycroscopy/pyUSID/blob/master/docs/USID_pyUSID_pycroscopy.pdf>`_
* Jump to our `GitHub project <https://github.com/pycroscopy/pyUSID>`_
* While pyUSID was originally a part of `pycroscopy <../pycroscopy/about.html>`_ up to 2017,
  it has since been serving as an independent, science-agnostic data handling package.
  pyUSID was born so that it can integrate with other existing mature packages in any domain. If you are interested in integrating our data model with your existing package, please `get in touch <./contact.html>`_ with us.
