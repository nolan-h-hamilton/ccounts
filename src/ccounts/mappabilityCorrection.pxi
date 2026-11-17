from libc.stdint cimport intptr_t, uint8_t

cdef extern from "native/fenwickMedian.h":
    int ccCorrectMappabilityCounts(const int64_t *, const uint8_t *,
                                   const intptr_t *, size_t, const int64_t *,
                                   size_t, size_t, int64_t *) noexcept nogil


def musicCorrectMappabilityCounts(const int64_t[::1] signal not None,
                                  const uint8_t[::1] donors not None,
                                  Py_ssize_t radius):
    cdef Py_ssize_t nBins = signal.shape[0]
    cdef Py_ssize_t nValues
    cdef int status
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] countValues, result
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] valueIndices

    if nBins == 0 or donors.shape[0] != nBins:
        raise InputError(kind="array")
    if radius < 0 or radius >= nBins:
        raise InputError()
    if radius == 0:
        return np.asarray(signal).copy()

    # [8, 2, 8] --> values [2, 8], indices [1, 0, 1]
    # track how many donors have each value without sorting every window
    countValues, valueIndices = np.unique(np.asarray(signal), return_inverse=True)
    valueIndices = np.ascontiguousarray(valueIndices, dtype=np.intp)
    nValues = countValues.shape[0]
    result = np.empty(nBins, dtype=np.int64)
    with nogil:
        status = ccCorrectMappabilityCounts(
            &signal[0], &donors[0], <const intptr_t *>valueIndices.data, nBins,
            <const int64_t *>countValues.data, nValues, radius,
            <int64_t *>result.data,
        )
    if status != 0:
        raise AllocationError()
    return result
