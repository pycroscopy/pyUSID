GPU Array Computing
===================
:Authors: Emily Costa
:Created on: 08/07/2019

The following are lessons learned during the exploration of implementing CuPy instead of numpy for GPU computing:

newaxis
-------
**Dimensions**: CuPy does not have a ``newaxis`` function unlike NumPy.
Instead of using new axis to add an additional dimension, you need to use ``cupy.expand_dims()``.
Also, note that CuPy does not lose a dimension during operations with vectors unlike numpy.
So, adding another dimension is often unnecessary as there are no singular dimensions in CuPy.
All vectors are converted into row vectors in numpy after being operated on,
which can be dealt with by adding a new axis and converting back into a column vector for further matrix operations.

The following is an example of how numpy's neawaxis function and how to use cupy's expand_dims in its place:
  
numpy.newaxis
~~~~~~~~~~~~~
  
Import all necessary modules

.. code:: python

  In [1]: import numpy as np

1D array:

.. code:: python

   In [2]: arr = np.arange(5)
  
   In [3]: arr.shape

.. code:: none
  
   Out[3]: (5,)

Make the 1D array becomes a row vector when an axis is inserted along 1st dimension

.. code:: python

  In [4]: row_vec = arr[np.newaxis, :]
  
  In [5]: row_vec.shape

.. code:: none

  Out[5]: (1, 5)

Make the 1D array becomes a column vector when an axis is inserted along 1st dimension

.. code:: python

  In [6]: col_vec = arr[:, np.newaxis]
  
  In [7]: col_vec.shape

.. code:: none

  Out[7]: (5, 1)
  

cupy.expand_dims
~~~~~~~~~~~~~~~~
Import all necessary modules

.. code:: python
  
  In [1]: import cupy as cp

1D array

.. code:: python
  
  In [2]: cp_arr = cp.arange(5)
  
  In [3]: cp_arr.shape

.. code:: none
  
  Out[3]: (5,)

Make the 1D array becomes a row vector when an axis is inserted along 1st dimension

.. code:: python
  
  In [4]: cp_row_vec = cp.expand_dims(cp_arr, axis=0)
  
  In [5]: cp_row_vec.shape

.. code:: none
  
  Out[5]: (1, 5)

Make the 1D array becomes a column vector when an axis is inserted along 1st dimension

.. code:: python
  
  In [6]: cp_col_vec = cp.expand_dims(cp_arr, axis=1)
  
  In [7]: cp_col_vec.shape

.. code:: none
  
  Out[7]: (5, 1)

append
------
CuPy does not have an ``append()`` function unlike NumPy.
As a reminder - the ``append`` function in the NumPy appends values to the end of an array.

The following is an example of numpy's ``append`` function and how to use cupy's ``concatenate`` instead:

numpy.append
~~~~~~~~~~~~

.. code:: python

  In [1]: x = np.array([1,2,3]) 
  
  In [2]: y = [4,5,6] 
  
  In [3]: xy = np.append(x, y)
  
  In [4]: xy

.. code:: none
  
  Out[4]: array([1,2,3,4,5,6])
  
cupy.concatenate
~~~~~~~~~~~~~~~~

.. code:: python

  In [1]: cp_x = cp.array([1,2,3]) 
  
  In [2]: cp_y = cp.array([4,5,6])
  
  In [3]: cp_xy = cp.concatenate([cp_x,cp_y], axis=0)
  
  In [4]: cp_xy

.. code:: none
  
  Out[4]: [1 2 3 4 5 6]
