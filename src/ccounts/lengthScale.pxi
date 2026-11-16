from libc.math cimport lgamma
from libc.stdint cimport intptr_t

cdef extern from "native/fenwickMedian.h":
    int ccMedianProfile(const intptr_t *, size_t, const double *, size_t,
                        size_t, double *) noexcept nogil


def musicBinomialLogTail(cnp.int64_t[::1] countsO, cnp.int64_t[::1] countsE):
    r"""Compute log tail of binomial with p=0.5

    Used for MUSIC to get log P-values for enrichment over background when
    trimming candidate regions.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(countsO.shape[0])
    cdef Py_ssize_t index
    cdef long long n, first, count
    cdef double term, total
    if countsO.shape[0] != countsE.shape[0]:
        raise InputError(kind="array")

    for index in range(countsO.shape[0]):
        if countsE[index] < 0 or countsO[index] < countsE[index]:
            raise CountError()
        if countsO[index] > INT64_MAX - countsE[index]:
            raise NumericalError()
        if countsO[index] == 0:
            result[index] = 0.0
            continue

        n = countsO[index] + countsE[index]
        first = countsO[index]
        term = 1.0
        total = 1.0
        count = first
        while count < n:
            # update term for next k
            term *= <double>(n - count) / (count + 1)
            total += term
            if term < total * 1e-15:
                break
            count += 1

        # log identities
        #   log(tail) = log(p_O) + log(total)
        #   log(p_O) = log C(n, O) - n*log(2)
        #   log C(n, O) = lgamma(n+1) - lgamma(O+1) - lgamma(n-O+1)
        result[index] = (lgamma(n + 1.0) - lgamma(first + 1.0)
            - lgamma(n - first + 1.0) - n * log(2.0) + log(total))
    return result


def musicMedianProfile(cnp.intp_t[::1] valueIndices, double[::1] values, int windowBins):
    cdef Py_ssize_t nBins = valueIndices.shape[0]
    cdef Py_ssize_t nValues = values.shape[0]
    cdef Py_ssize_t index
    cdef int status
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result
    if nBins == 0 or nValues == 0 or windowBins <= 0 or windowBins % 2 == 0:
        raise InputError()
    for index in range(nBins):
        if valueIndices[index] < 0 or valueIndices[index] >= nValues:
            raise InputError(kind="array")

    result = np.empty(nBins)
    with nogil:
        status = ccMedianProfile(<const intptr_t *>&valueIndices[0], nBins,
            &values[0], nValues, windowBins, <double *>result.data)
    if status != 0:
        raise AllocationError()
    return result
