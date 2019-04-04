Getting Started
---------------
* Follow `these instructions <./install.html>`_ to install pyUSID
* We have compiled a list of `handy tutorials <./external_guides.html>`_ on basic / prerequisite topics such as programming in python, hdf5 handling, etc.
* See our `examples <./auto_examples/index.html>`_ to get started on using and writing your own pyUSID functions.

  * Please see this `pyUSID tutorial for beginners <https://github.com/pycroscopy/pyUSID_Tutorial>`_ based on the examples on this project.
* Details regarding the definition, implementation, and guidelines for Universal Spectroscopy and Imaging Data (USID) and its implementation in HDF5 (h5USID) are available in `this document <./data_format.html>`_.
* If you are interested in contributing your code to pyUSID, please look at our `guidelines <./contribution_guidelines.html>`_
* We also have a handy document for converting your `matlab code to python <./matlab_to_python.html>`_.
* If you need detailed documentation on what is where and why, all our classes, functions, etc., please visit our `API <./api.html>`_
* For a concise change-log, please see the `release history <https://github.com/pycroscopy/pyUSID/releases>`_.
* Please `get in touch <./contact.html>`_ if you would like to use USID and pyUSID for other new or mature scientific packages.
* Have questions? See our `FAQ <./faq.html>`_ to see if we have already answered them.
* Need more information? Please see our `Arxiv <https://arxiv.org/abs/1903.09515>`_ paper.
* Need help or need to get in touch with us? See our `contact <./contact.html>`_ information.

Guide for python novices
~~~~~~~~~~~~~~~~~~~~~~~~
For the python novices by a python novice - **Nick Mostovych, Brown University**

#. Learn about the `philosophy, purpose, etc. of pyUSID <./about.html>`_.
#. Get an idea of the different resources available by reading the `getting started <./getting_started.html>`_ section
#. Watch the video on `installing Anaconda <https://www.youtube.com/watch?v=YJC6ldI3hWk>`_ from the `Tutorials on Basics <./external_guides.html>`_ page
#. Follow instructions on the `installation <./install.html>`_ page to install Anaconda.
#. Watch the `video tutorial <https://www.youtube.com/watch?v=HW29067qVWk>`_ from the ``Jupyter Notebooks`` section in `the Tutorials on Basics <./external_guides.html>`_ page
#. Read the whole `Tutorial on Basics page <./external_guides.html>`_. Do NOT proceed unless you are familiar with basic python programming and usage.
#. Read `the document on the USID model and h5USID <./data_format.html>`_. This is very important and highlights the advantages of using USID. New users should not jump to the examples until they have a good understanding of the data format.
#. Depending on your needs, go through the `recommended sequence of tutorials and examples <./auto_examples/index.html>`_

Tips and pitfalls
~~~~~~~~~~~~~~~~~
For the python novices by a python novice - **Nick Mostovych, Brown University**

* Documentation and examples on this website are for the **latest** version of pyUSID. If something does not work as shown on this website,
  chances are that you may be using an older version of pyUSID. Follow the instructions to `update pyUSID to the latest version <./install.html#updating-pyUSID>`_
* pyUSID has excellent documentation (+ examples too) for all functions. If you are ever confused with the usage of a
  function or class, you can get help in numerous ways:

  * If you are using jupyter notebooks, just hit the ``Shift+Tab`` keys after typing the name of your function.
    See `this quick video <https://www.youtube.com/watch?v=TgqMK1SG7XI>`_ for a demo.
    E.g. - type ``px.USIDataset(``. Hit ``Shift+Tab`` twice or four times. You should be able to see the documentation for the
    class / function to learn how to supply inputs / extract outputs
  * Use the search function and reference the source code in the `API section <./api.html>`_ for detailed comments.
    Most detailed questions are answered there.
* Use the `USIDataset <./auto_examples/beginner/plot_usi_dataset.html>`_ everywhere possible to simplify your tasks.
* Many functions in pyUSID have a ``verbose`` keyword argument that can be set to ``True`` to get detailed print logs of intermediate steps in the function.
  This is **very** handy for debugging code

If there are tips or pitfalls you would like to add to this list, please `get in touch to us <./contact.html>`_
