import warnings
import numpy as np

try:
    from numba import njit, prange

except ImportError:
    # raise a warning that numba is not installed
    # and that the njit functions will not be used
    # and that the pure python functions will be used instead

    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            warnings.warn(
                f"Numba is not installed. Using pure python for {func.__name__}"
            )
            return func

        return wrapper

def _interpolate(image, row, col, rows, cols):
    r = int(row)
    c = int(col)
    if r<0 or r>=rows or c<0 or c>=cols:
        return 0
    else:
        return image[r,c]
    
@njit(cache=True)
def _njit_interpolate(image, row, col, rows, cols):
    r = int(row)
    c = int(col)
    if r<0 or r>=rows or c<0 or c>=cols:
        return 0
    else:
        return image[r,c]


def shift_magnify(
    image: np.ndarray,
    shift_row: np.ndarray,
    shift_col: np.ndarray,
    magnification_row: float,
    magnification_col: float,
) -> np.ndarray:
    """
    Shift and magnify using nearest neighbor interpolation.
    :param image: 3D numpy array to interpolate with size (nFrames, nRow, nCol)
    :param shift_row: 1D array with size (nFrames) with values to shift the rows 
    :param shift_col: 1D array with size (nFrames) with values to shift the cols
    :param magnification_row: float magnification factor for the rows
    :param magnification_col: float magnification factor for the cols
    :return: 3D float32 numpy array with the result
    """

    nFrames = image.shape[0]
    rows = image.shape[1]
    cols = image.shape[2]
    rowsM = int(rows * magnification_row)
    colsM = int(cols * magnification_col)

    image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
    for f in range(nFrames):
        for j in range(colsM):
            col = j / magnification_col - shift_col[f]
            for i in range(rowsM):
                row = i / magnification_row - shift_row[f]
                image_out[f, i, j] = _interpolate(image[f,:,:],row,col,rows,cols)

    return image_out


@njit(cache=True, parallel=True)
def njit_shift_magnify(    
    image: np.ndarray,
    shift_row: np.ndarray,
    shift_col: np.ndarray,
    magnification_row: float,
    magnification_col: float,
) -> np.ndarray:
    """
    Shift and magnify using nearest neighbor interpolation.
    :param image: 3D numpy array to interpolate with size (nFrames, nRow, nCol)
    :param shift_row: 1D array with size (nFrames) with values to shift the rows 
    :param shift_col: 1D array with size (nFrames) with values to shift the cols
    :param magnification_row: float magnification factor for the rows
    :param magnification_col: float magnification factor for the cols
    :return: 3D float32 numpy array with the result
    """

    nFrames = image.shape[0]
    rows = image.shape[1]
    cols = image.shape[2]
    rowsM = int(rows * magnification_row)
    colsM = int(cols * magnification_col)

    image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
    for f in range(nFrames):
        for j in prange(colsM):
            col = j / magnification_col - shift_col[f]
            for i in range(rowsM):
                row = i / magnification_row - shift_row[f]
                image_out[f, i, j] = _njit_interpolate(image[f,:,:],row,col,rows,cols)

    return image_out