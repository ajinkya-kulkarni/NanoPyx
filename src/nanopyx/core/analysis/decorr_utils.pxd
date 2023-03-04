# Code below is autogenerated by pyx2pxd - https://github.com/HenriquesLab/pyx2pxd

cdef double[:, :, :] _normalizeFFT(double[:, :] fft_real, double[:, :] fft_imag)
cdef double[:, :] _apodize_edges(float[:, :] img)
cdef double _linmap(float val, float valmin, float valmax, float mapmin, float mapmax)
cdef double[:, :] _get_mask(int w, float r2)
cdef double _get_corr_coef_norm(double[:, :] fft_real, double[:, :] fft_imag, double[:, :] mask)
cdef double[:] _get_max(float[:] arr, int x1, int x2)
cdef double[:] _get_min(float[:] arr, int x1, int x2)
cdef _get_best_score(float[:] kc, float[:] a)
cdef _get_max_score(float[:] kc, float[:] a)
