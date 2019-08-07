**GPU Computing Using CuPy**
========================

Enabling GPU computing, by implementing CuPy in the Bayesian Inference package of pycroscopy, was `completed. 
The following are lessons learned during the exploration of using CuPy:

* **Dimensions** CuPy does not have a newaxis function, like NumPy does. Instead of using new axis to add an additional dimension, you need to use cupy.expand_dims(). Also, note that cupy does not lose a dimension during operations with vectors, like numpy, so adding another dimension is often unnecessary as there are no singular dimensions in cupy.
The following is an example of how numpy's neawaxis function and how to use cupy's expand_dims in its place:

  # Necesaary modules
  In [1]: import numpy as np

  # 1D array
  In [2]: arr = np.arange(5)
  In [3]: arr.shape
  Out[3]: (5,)

  # make the 1D array becomes a row vector when an axis is inserted along 1st dimension
  In [4]: row_vec = arr[np.newaxis, :]
  In [5]: row_vec.shape
  Out[5]: (1, 5)

  # make the 1D array becomes a column vector when an axis is inserted along 1st dimension
  In [6]: col_vec = arr[:, np.newaxis]
  In [7]: col_vec.shape
  Out[7]: (5, 1)

* **Append** CuPy does not have an append function, as numpy does. The append function in the numpy appends values to the end of an array. 
The following is an example of numpy's append function and how to use cupy's concatonate instead:
.. code:: python
  # input
  x = np.arrray([1,2,3]) 
  y = [4,5,6] 
  xy = np.append(x, y)
  # output
  array([1,2,3,4,5,6])
  
