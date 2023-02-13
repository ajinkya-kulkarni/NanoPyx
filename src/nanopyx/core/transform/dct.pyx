# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport cos, sin, pi, sqrt

import numpy as np
cimport numpy as np

from cython.parallel import prange

# calculate the DCT-II of a 2D array
cdef class DCT_II:
    """
    Calculate the DCT-II of a 2D array

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    """
    cdef double[:, :] _dct_matrix

    def __init__(self, int rows, int cols):
        """
        :param rows: The number of rows in the image
        :param cols: The number of columns in the image
        """

        if not self._is_power_of_two(rows):
            raise ValueError("The number of rows must be a power of two")

        if rows != cols:
            raise ValueError("The number of rows and columns must be equal")

        self._dct_matrix = np.zeros((rows, cols), dtype=np.float64)

        cdef int n, m

        with nogil:
            for n in prange(rows):
                for m in range(cols):
                    if n == 0:
                        self._dct_matrix[n, m] = sqrt(1.0 / rows)
                    else:
                        self._dct_matrix[n, m] = sqrt(2.0 / rows) * cos((pi * (2 * m + 1) * n) / (2 * rows))

    cdef double[:, :] _dct(self, double[:, :] image) nogil:
        cdef int n, m, k, l
        cdef double sum
        cdef int rows = image.shape[0]
        cdef int cols = image.shape[1]

        cdef double[:, :] image_dct
        with gil:
            image_dct = np.zeros((rows, cols), dtype=np.float64)

        for n in range(rows):
            for m in range(cols):
                sum = 0.0
                for k in range(rows):
                    for l in range(cols):
                        sum += image[k, l] * self._dct_matrix[n, k] * self._dct_matrix[m, l]
                image_dct[n, m] = sum

        return image_dct

    def dct(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(self._dct(image))

    cdef double[:, :] _idct(self, double[:, :] image_dct) nogil:
        cdef int n, m, k, l
        cdef double sum
        cdef int rows = image_dct.shape[0]
        cdef int cols = image_dct.shape[1]

        cdef double[:, :] image_idct
        with gil:
            image_idct = np.zeros((rows, cols), dtype=np.float64)

        for n in range(rows):
            for m in range(cols):
                sum = 0.0
                for k in range(rows):
                    for l in range(cols):
                        sum += image_dct[k, l] * self._dct_matrix[k, n] * self._dct_matrix[l, m]
                image_idct[n, m] = sum

        return image_idct

    def idct(self, image_dct: np.ndarray) -> np.ndarray:
        return np.asarray(self._idct(image_dct))
