import sys
from enum import Enum
from warnings import warn

import numpy as np
from sidpy import Dimension as SIDimension

if sys.version_info.major == 3:
    unicode = str


class DimType(Enum):
    DEFAULT = 0
    INCOMPLETE = 1
    DEPENDENT = 2

    @staticmethod
    def __check_other_type(other):
        if not isinstance(other, DimType):
            raise TypeError('Provided object not of type DimType')

    def __lt__(self, other):
        self.__check_other_type(other)
        return self.value < other.value

    def __gt__(self, other):
        self.__check_other_type(other)
        return self.value > other.value

    def __eq__(self, other):
        self.__check_other_type(other)
        return self.value == other.value


class Dimension(SIDimension):
    """
    ..autoclass::Dimension
    """

    def __new__(cls, name, units, values, quantity='generic',
                dimension_type='unknown', mode=DimType.DEFAULT):
        """
        Simple object that describes a dimension in a dataset by its name, units, and values

        Parameters
        ----------
        name : str or unicode
            Name of the dimension. For example 'Bias'
        units : str or unicode
            Units for this dimension. For example: 'V'
        values : array-like or int
            Values over which this dimension was varied. A linearly increasing set of values will be generated if an
            integer is provided instead of an array.
        mode : Enum, Optional. Default = DimType.DEFAULT
            How the parameter associated with the dimension was varied.
            DimType.DEFAULT - data was recorded for all combinations of values in this dimension against **all** other
            dimensions. This is typically the case.
            DimType.INCOMPLETE - Data not present for all combinations of values in this dimension and all other
                dimensions. Examples include spiral scans, sparse sampling, aborted measurements
            DimType.DEPENDENT - Values in this dimension were varied as a function of another (independent) dimension.
        quantity : str or unicode
            Physical quantity such as Length
        dimension_type : str or sidpy.DimensionTypes
            Type of dimension. such as spectral, spatial, etc.
        """
        if isinstance(values, int):
            if values < 1:
                raise ValueError('values must be a whole number. {} provided'
                                 ''.format(values))
        self = SIDimension.__new__(cls, values, name=name, quantity=quantity,
                                   units=units, dimension_type=dimension_type)
        self.mode = mode
        return self

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if not isinstance(value, DimType):
            raise TypeError('mode must be of type pyUSID.DimType. Provided '
                            'object was of type: {}'.format(type(value)))
        self._mode = value

    @property
    def units(self):
        return self._units

    # pyUSID allows empty values for units unlike sid.Dimension
    @units.setter
    def units(self, value):
        if not isinstance(value, (str, unicode)):
            raise TypeError('units should be a string')
        self._units = value.strip()

    def __repr__(self):
        return '{}: {} ({}) mode:{} : {}' \
               ''.format(self.name, self.quantity, self.units, self.mode,
                         self.values)

    def __str__(self):
        return '{}: {} ({}) mode:{} : {}' \
               ''.format(self.name, self.quantity, self.units, self.mode,
                         self.values)

    def __eq__(self, other):
        # Since __eq__ has not been implemented in sidpy.Dimension:
        if not isinstance(other, Dimension):
            raise TypeError('Cannot compare against object type: {}'
                            ''.format(type(other)))
        if self._name != other._name:
            return False
        if self._quantity != other._quantity:
            return False
        if self.mode != other._mode:
            return False
        if self._units != other._units:
            return False
        if len(self) != len(other):
            return False
        return np.allclose(self, other)


def validate_dimensions(dimensions, dim_type='Position'):
    """
    Checks if the provided object is an iterable with pyUSID.Dimension objects.
    If it is not full of Dimension objects, Exceptions are raised.

    Parameters
    ----------
    dimensions : iterable or pyUSID.Dimension
        Iterable containing pyUSID.Dimension objects
    dim_type : str, Optional. Default = "Position"
        Type of Dimensions in the iterable. Set to "Spectroscopic" if not Position dimensions.
        This string is only used for more descriptive Exceptions

    Returns
    -------
    list
        List containing pyUSID.Dimension objects
    """
    if isinstance(dimensions, Dimension):
        dimensions = [dimensions]
    if isinstance(dimensions, np.ndarray):
        if dimensions.ndim > 1:
            dimensions = dimensions.ravel()
            warn(dim_type + ' dimensions should be specified by a 1D array-like. Raveled this numpy array for now')
    if not isinstance(dimensions, (list, np.ndarray, tuple)):
        raise TypeError(dim_type + ' dimensions should be array-like of Dimension objects')
    if not np.all([isinstance(x, Dimension) for x in dimensions]):
        raise TypeError(dim_type + ' dimensions should be a sequence of Dimension objects')
    return dimensions
