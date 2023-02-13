# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport cos, sin, pi

import numpy as np
cimport numpy as np

# a reimplementation of https://github.com/menzel/LiMo/blob/6e9d2cc50299bbf192a8d1c2298e4e3387830058/src/ij/process/FHT.java
cdef class FHT:

    cdef float[:] _C
    cdef float[:] _S
    cdef int[:] _bitrev
    cdef float[:] tempArr
    cdef int maxN

    cdef np.ndarray image
    cdef float[:,:] _image
    cdef int rows, cols

    def __init__(self, image: np.ndarray):
        _image = image
        rows = image.shape[0]
        cols = image.shape[1]

    cdef bint _powerOf2Size(self):
        cdef int i = 2
        while i < self.rows:
            i *= 2
        return i == self.rows and self.rows == self.cols

    cdef void _initializeTables(self, int maxN) nogil:
        self._makeSinCosTables(maxN)
        self._makeBitReverseTable(maxN)
        with gil:
            self.tempArr = np.zeros(maxN, dtype=np.float32)

    cdef void _makeSinCosTables(self, int maxN) nogil:
        cdef int n = maxN // 4
        with gil:
            self._C = np.empty(n, dtype=np.float32)
            self._S = np.empty(n, dtype=np.float32)

        cdef double theta = 0.0
        cdef double dTheta = 2.0 * pi / maxN
        cdef int i
        for i in range(n):
            self._C[i] = cos(theta)
            self._S[i] = sin(theta)
            theta += dTheta

    cdef void _makeBitReverseTable(self, int maxN) nogil:
        with gil:
            self._bitrev = np.empty(maxN, dtype=np.int32)

        cdef int i
        cdef int nLog2 = self._log2(maxN)
        for i in range(maxN):
            self._bitrev[i] = self._bitRevX(i, nLog2)

    cdef int _log2(self, int x) nogil:
        cdef int count = 15
        while not self._btst(x, count):
            count -= 1
        return count

    cdef bint _btst(self, int x, int bit) nogil:
        return x & (1 << bit) != 0

    cdef int _bitRevX(self, int x, int bitlen) nogil:
        cdef int temp = 0
        cdef int i
        for i in range(bitlen + 1):
            if x & (1 << i) != 0:
                temp |= (1 << (bitlen - i - 1))
        return temp & 0x0000ffff

    cdef void _transposeR(self, float[:] x, int maxN):
        cdef int r, c
        cdef float rTemp

        for r in range(maxN):
            for c in range(r, maxN):
                if r != c:
                    rTemp = x[r*maxN + c]
                    x[r*maxN + c] = x[c*maxN + r]
                    x[c*maxN + r] = rTemp
