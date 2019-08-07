**GPU Computing Using CuPy**
========================

Enabling GPU computing, by implementing CuPy in the Bayesian Inference package of pycroscopy, was `completed. 
The following are lessons learned during the exploration of using CuPy:

* **Dimensions** CuPy does not have a newaxis function, like NumPy does. Instead of using new axis to add an additional dimension, you need to use cupy.expand_dims(). Also, note that cupy does not lose a dimension during operations with vectors, like numpy, so adding another dimension is often unnecessary as there are no singular dimensions in cupy.
* **Append** CuPy does not have an append function, as numpy does. The append function in the numpy appends values to the end of an array. 
The following is an example of numpy's append function:

.. code:: python
  [in]
  x = np.arrray([1,2,3]) 
  y = [4,5,6] 
  xy = np.append(x, y)
  [out]
  array([1,2,3,4,5,6])
