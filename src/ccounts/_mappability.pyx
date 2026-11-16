# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

from ._checks cimport _checkIntervalBuffers
from .exceptions import InputError

cnp.import_array()


def _intervalPrefix(cnp.ndarray positions not None,
                    cnp.ndarray starts not None,
                    cnp.ndarray ends not None,
                    cnp.ndarray totals not None,
                    cnp.ndarray moments=None):
    r"""Get counts of bases in mappability-runs to the left of each query.

    Use a cumulative total of run lengths up to run i, and add the partial run for the query.

    ```plaintext
    runs:    [====] ... [===] ... [==Q===]   Q = query
    lengths:    4         3          5

    totals = [0, 4, 7, 12]
    count at query = totals[2] + len('==') = 9
    ```

    """
    cdef cnp.ndarray result
    cdef Py_ssize_t queryCount = cnp.PyArray_SIZE(positions)
    cdef Py_ssize_t runCount
    cdef Py_ssize_t queryIndex, lower, upper, middle
    cdef const int64_t *queryData
    cdef const int64_t *startData
    cdef const int64_t *endData
    cdef const int64_t *totalData
    cdef const int64_t *momentData = NULL
    cdef int64_t *resultData
    cdef int64_t chromosomeEnd, query, queryEnd, runStart, basesBeforeQuery
    cdef int64_t count, positionSum

    positions = cnp.PyArray_GETCONTIGUOUS(positions)
    runCount = _checkIntervalBuffers(positions, starts, ends, totals, moments)
    if moments is not None:
        momentData = <const int64_t *>cnp.PyArray_DATA(moments)

    result = np.empty_like(positions, order="C", subok=False)
    queryData = <const int64_t *>cnp.PyArray_DATA(positions)
    startData = <const int64_t *>cnp.PyArray_DATA(starts)
    endData = <const int64_t *>cnp.PyArray_DATA(ends)
    totalData = <const int64_t *>cnp.PyArray_DATA(totals)
    resultData = <int64_t *>cnp.PyArray_DATA(result)
    chromosomeEnd = startData[runCount]

    with nogil:
        for queryIndex in range(queryCount):
            query = queryData[queryIndex]
            lower = 0
            upper = runCount
            while lower < upper:
                middle = lower + (upper - lower) // 2
                if endData[middle] <= query:
                    lower = middle + 1
                else:
                    upper = middle

            runStart = startData[lower]
            queryEnd = query
            if queryEnd > chromosomeEnd:
                queryEnd = chromosomeEnd
            basesBeforeQuery = queryEnd - runStart
            if basesBeforeQuery < 0:
                basesBeforeQuery = 0
            count = totalData[lower] + basesBeforeQuery
            if momentData == NULL:
                resultData[queryIndex] = count
            else:
                positionSum = (
                    momentData[lower]
                    + basesBeforeQuery * (2 * runStart + basesBeforeQuery - 1) // 2
                )
                resultData[queryIndex] = query * count - positionSum
    return result
