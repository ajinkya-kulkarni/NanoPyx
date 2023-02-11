# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True

from libc.math cimport cos, pi

import numpy as np
cimport numpy as np


cdef void _dht2d_core(float[:,:] f, float[:,:] f_hat, int sign) nogil:
    """
    Compute the discrete Hartley transform of a 2D array.
    """
    cdef int w = f.shape[1]
    cdef int h = f.shape[0]

    cdef int j, i, k, l
    cdef float s, t

    cdef float[:,:] g, g_hat

    with gil:
        g = np.empty_like(f)
        g_hat = np.empty_like(f)

    # Transform the rows
    for j in range(h):
        for i in range(w):
            s = 0
            for k in range(w):
                s += f[j, k] * cos(pi * (2 * k + 1) * i / (2 * w))
            g[j, i] = s

    # Transform the columns
    for j in range(h):
        for i in range(w):
            t = 0
            for l in range(h):
                t += g[l, i] * cos(pi * (2 * l + 1) * j / (2 * h))
            g_hat[j, i] = t

    # Scale the result
    if sign == 1:
        for j in range(h):
            for i in range(w):
                f_hat[j, i] = 2 * g_hat[j, i] / (w * h)
    else:
        for j in range(h):
            for i in range(w):
                f_hat[j, i] = g_hat[j, i] / 2

def dht2d(image: np.ndarray):
    """
    Compute the discrete Hartley transform of a 2D array.
    :param f: The input array.
    :return: The discrete Hartley transform of the input array as a 2D float32 array.
    """
    cdef float[:,:] _image = image.view(np.float32)
    return _dht2d(_image, 1)

cdef float[:,:] _dht2d(float[:,:] f, int sign) nogil:
    cdef float[:,:] image_ht
    with gil:
        image_ht = np.empty_like(f)
    _dht2d_core(f, image_ht, sign)
    return image_ht

def idht2d(f_hat):
    """
    Compute the inverse discrete Hartley transform of a 2D array.
    :param f_hat: The input array.
    :return: The inverse discrete Hartley transform of the input array.
    """
    return _idht2d(f_hat)

cdef float[:,:] _idht2d(float[:,:] image_ht) nogil:
    cdef float[:,:] image
    with gil:
        image = np.empty_like(image_ht)
    _dht2d_core(image_ht, image, -1)
    return image

def test():
    # Create a test image
    f = np.zeros((8, 8), dtype=np.float32)
    f[2, 3] = 1
    f[4, 5] = 1

    # Compute the DHT
    f_hat = _dht2d(f, 1)

    # Compute the inverse DHT
    f2 = _idht2d(f_hat)

    # Check that the result is the same as the original
    assert np.allclose(f, f2)

def dht2d_zoom(image: np.ndarray, magnification: int):
    """
    Compute the discrete Hartley transform of a 2D array.
    :param f: The input array.
    :return: The discrete Hartley transform of the input array as a 2D float32 array.
    """
    cdef float[:,:] _image = image.view(np.float32)
    return _dht2d_zoom(_image, magnification)

cdef float[:,:] _dht2d_zoom(float[:,:] image, int magnification):
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef int w2 = w * magnification
    cdef int h2 = h * magnification
    cdef int i, j
    cdef float[:,:] image_ht, image_ht2, image2

    # Compute the DHT
    image_ht = _dht2d(image, 1)

    # Zero-pad the DHT
    image_ht2 = np.zeros((h2, w2), dtype=np.float32)
    for j in range(h):
        for i in range(w):
            image_ht2[j, i] = image_ht[j, i]

    # Compute the inverse DHT
    image2 = _idht2d(image_ht2)

    return image2
