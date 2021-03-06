======
pyUSID
======

**Python framework for storing, visualizing, and processing spectroscopy, imaging or any observational / experimental data**

What?
------
* The `Universal Spectroscopic and Imaging Data (USID) model <../USID/index.html>`_:

  * facilitates the representation of any spectroscopic or imaging data regardless of its origin, modality, size, or dimensionality.
  * enables the development of instrument- and modality- agnostic data processing and analysis algorithms.
  * is just a definition or a blueprint rather than something tangible and readily usable.
* pyUSID is a `python <http://www.python.org/>`_ package that currently provides three pieces of functionality:

  #. **io**: Primarily, it enables the storage and access of USID in **hierarchical data format** `(HDF5) <http://extremecomputingtraining.anl.gov/files/2015/03/HDF5-Intro-aug7-130.pdf>`_ files (referred to as h5USID files) using python
  #. **viz**: It has handy tools for visualizing USID and general scientific data
  #. **processing**: It provides a framework for formulating scientific problems into computational problems.
     See `pycroscopy <../pycroscopy/about.html>`_ - a sister project that uses pyUSID for analysis of microscopy data.

* pyUSID uses a data-centric approach wherein the raw data collected from the instrument, results from analysis
  and processing routines are all written to the same h5USID file for traceability, reproducibility, and provenance.
* Just as scipy uses numpy underneath, scientific packages like **pycroscopy** use **pyUSID** for all file-handling, data processing, and generating plots for journal publications
* pyUSID uses popular packages such as numpy, h5py, joblib, matplotlib, etc. for most of the storage, computation, and visualization.
* For more information, please see our recent `Arxiv <https://arxiv.org/abs/1903.09515>`_ **paper**
* See a high-level overview of pyUSID in this `presentation <https://github.com/pycroscopy/pyUSID/blob/master/docs/USID_pyUSID_pycroscopy.pdf>`_
* Jump to our `GitHub project <https://github.com/pycroscopy/pyUSID>`_

Why?
-----
As we see it, there are a few opportunities in scientific imaging (that surely apply to several other scientific domains):

**1. Growing data sizes**
  * Cannot use desktop computers for analysis
  * *Need: High performance computing, storage resources and compatible, scalable file structures*

**2. Increasing data complexity**
  * Sophisticated imaging and spectroscopy modes resulting in 5,6,7... dimensional data
  * *Need: Robust software and generalized data formatting*

**3. Multiple file formats**
  * Different formats from each instrument. Proprietary in most cases
  * Incompatible for correlation
  * *Need: Open, instrument-independent data format*

**4. Expensive analysis software**
  * Software supplied with instruments often insufficient / incapable of custom analysis routines
  * Commercial software (Eg: Matlab, Origin..) are often prohibitively expensive.
  * *Need: Free, powerful, open source, user-friendly software*

**5. Closed science**
  * Analysis software and data not shared
  * No guarantees of reproducibility or traceability
  * *Need: open source data structures, file formats, centralized code and data repositories*

Who?
-----
* This project begun largely as an effort by scientists and engineers at the **I**\nstitute for **F**\unctional **I**\maging of **M**\aterials (`IFIM <https://ifim.ornl.gov>`_) to standardize data representation, storage and processing for a very large variety of imaging and spectroscopy instruments.
* It is now being developed and maintained by `Suhas Somnath <https://github.com/ssomnath>`_ of the **A**\dvanced **D**\ata & **W**\orkflows **G**\roup (ADWG) at the **O**\ak Ridge National Laboratory **L**\eadership **C**\omputing **F**\acility (`OLCF <https://www.olcf.ornl.gov>`_) and `Chris R. Smith <https://github.com/CompPhysChris>`_ of IFIM.
  Please visit our `credits and acknowledgements <./credits.html>`_ page for more information.
* While pyUSID was originally a part of pycroscopy up to 2017, it is now serves as an independent, science-agnostic data handling package. pyUSID was born so that it can integrate with other existing mature packages in any domain. If you are interested in integrating our data model with your existing package, please `get in touch <./contact.html>`_ with us.
