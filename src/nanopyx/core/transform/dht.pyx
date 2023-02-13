# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport cos, sin, pi

import numpy as np
cimport numpy as np


cdef void _dht2d_core(double[:,:] image, double[:,:] image_ht, int sign) nogil:
    """
    Compute the discrete Hartley transform of a 2D array.
    """
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]

    cdef int j, i, k, l
    cdef double s, t

    cdef double[:,:] g, g_ht

    with gil:
        g = np.empty_like(image)
        g_ht = np.empty_like(image)

    # Transform the rows
    for j in range(h):
        for i in range(w):
            s = 0
            for k in range(w):
                s += image[j, k] * cos(pi * (2 * k + 1) * i / (2 * w))
            g[j, i] = s

    # Transform the columns
    for j in range(h):
        for i in range(w):
            t = 0
            for l in range(h):
                t += g[l, i] * cos(pi * (2 * l + 1) * j / (2 * h))
            g_ht[j, i] = t

    # Scale the result
    if sign == 1:
        for j in range(h):
            for i in range(w):
                image_ht[j, i] = 2 * g_ht[j, i] / (w * h)
    else:
        for j in range(h):
            for i in range(w):
                image_ht[j, i] = g_ht[j, i] / 2


def dht2d(image: np.ndarray):
    """
    Compute the discrete Hartley transform of a 2D array.
    :param f: The input array.
    :return: The discrete Hartley transform of the input array as a 2D double32 array.
    """
    cdef double[:,:] _image = image.astype(np.double)
    return _dht2d(_image, 1)


cdef double[:,:] _dht2d(double[:,:] image, int sign) nogil:
    cdef double[:,:] image_ht
    with gil:
        image_ht = np.empty_like(image)
    _dht2d_core(image, image_ht, sign)
    return image_ht


def idht2d(image_ht):
    """
    Compute the inverse discrete Hartley transform of a 2D array.
    :param image_ht: The input array.
    :return: The inverse discrete Hartley transform of the input array.
    """
    return _idht2d(image_ht)


cdef double[:,:] _idht2d(double[:,:] image_ht) nogil:
    cdef double[:,:] image
    with gil:
        image = np.empty_like(image_ht)
    _dht2d_core(image_ht, image, -1)
    return image


def test():
    # Create a test image
    f = np.zeros((8, 8), dtype=np.double)
    f[2, 3] = 1
    f[4, 5] = 1

    # Compute the DHT
    f_hat = _dht2d(f, 1)

    # Compute the inverse DHT
    f2 = _idht2d(f_hat)

    # Check that the result is the same as the original
    assert np.allclose(f, f2)


def magnify(image: np.ndarray, magnification: int, bint enforce_same_value = 0):
    """
    Compute the discrete Hartley transform of a 2D array.
    :param image: The input array.
    :param magnification: The magnification factor.
    :param enforce_same_value: If True, the value of the original image is enforced at the center of each magnified pixel.
    """
    cdef double[:,:] _image = image.astype(np.double)
    return _magnify(_image, magnification, enforce_same_value)


cdef double[:,:] _magnify(double[:,:] image, int magnification, bint enforce_same_value) nogil:
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef int w2 = w * magnification
    cdef int h2 = h * magnification
    cdef int i, j
    cdef double[:,:] image_ht, image_ht2, imageM

    # Compute the DHT
    image_ht = _dht2d(image, 1)

    # Zero-pad the DHT
    with gil:
        image_ht2 = np.zeros((h2, w2), dtype=np.double)

    for j in range(h):
        for i in range(w):
            image_ht2[j, i] = image_ht[j, i]

    # Compute the inverse DHT
    imageM = _idht2d(image_ht2)

    # imageM *= (w * h) / (w2 * h2)

    # Correct intensity values
    if enforce_same_value:
        imageM[::magnification, ::magnification] = image

    return imageM
