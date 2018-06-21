from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
from scipy.signal import find_peaks_cwt


def find_all_peaks(vector, width_bounds, num_steps=20, **kwargs):
    """
    This is the function that will be mapped by multiprocess. This is a wrapper around the scipy function.
    It uses a parameter - wavelet_widths that is configured outside this function.

    Parameters
    ----------
    vector : 1D numpy array
        Feature vector containing peaks
    width_bounds : tuple / list / iterable
        Min and max for the size of the window
    num_steps : uint, (optional). Default = 20
        Number of different peak widths to search

    Returns
    -------
    peak_indices : list
        List of indices of peaks within the prescribed peak widths
    """
    # The below numpy array is used to configure the returned function wpeaks
    wavelet_widths = np.linspace(width_bounds[0], width_bounds[1], num_steps)

    peak_indices = find_peaks_cwt(np.abs(vector), wavelet_widths, **kwargs)

    return peak_indices
