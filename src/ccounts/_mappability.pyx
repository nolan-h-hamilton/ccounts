# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t
from libc.string cimport memcpy, memset

from ._checks cimport _checkIntervalBuffers
from .exceptions import InputError

cdef struct _RunTable:
    const int64_t *starts
    const int64_t *ends
    const int64_t *totals
    const int64_t *moments
    Py_ssize_t count
    int64_t chromosomeEnd


cdef struct _ScoreCursor:
    Py_ssize_t index
    Py_ssize_t shiftedIndex
    int64_t position
    int64_t prefix


cnp.import_array()


cdef _RunTable _runTable(cnp.ndarray starts, cnp.ndarray ends, cnp.ndarray totals,
                        cnp.ndarray moments, cnp.ndarray positions=None) except *:
    r"""Wrap a chromosome's run bounds, running base counts, and moments"""
    cdef _RunTable runs
    runs.count = _checkIntervalBuffers(positions, starts, ends, totals, moments)
    runs.starts = <const int64_t *>cnp.PyArray_DATA(starts)
    runs.ends = <const int64_t *>cnp.PyArray_DATA(ends)
    runs.totals = <const int64_t *>cnp.PyArray_DATA(totals)
    runs.moments = NULL
    if moments is not None:
        runs.moments = <const int64_t *>cnp.PyArray_DATA(moments)
    runs.chromosomeEnd = runs.starts[runs.count]
    return runs


cdef inline Py_ssize_t _searchRun(const int64_t *ends, int64_t query,
                                  Py_ssize_t lower, Py_ssize_t upper) noexcept nogil:
    cdef Py_ssize_t middle
    while lower < upper:
        middle = lower + (upper - lower) // 2
        if ends[middle] <= query:
            lower = middle + 1
        else:
            upper = middle
    return lower


cdef inline int64_t _prefixAt(const _RunTable *runs, int64_t query,
                             Py_ssize_t index) noexcept nogil:
    cdef int64_t start = runs.starts[index]
    cdef int64_t width = max(0, min(query, runs.chromosomeEnd) - start)
    cdef int64_t count = runs.totals[index] + width
    if runs.moments == NULL:
        return count
    return query * count - (runs.moments[index] + width * (2 * start + width - 1) // 2)


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
    cdef Py_ssize_t queryIndex, index
    cdef const int64_t *queryData
    cdef int64_t *resultData
    cdef int64_t query
    cdef _RunTable runs

    positions = cnp.PyArray_GETCONTIGUOUS(positions)
    runs = _runTable(starts, ends, totals, moments, positions)
    result = np.empty_like(positions, order="C", subok=False)
    queryData = <const int64_t *>cnp.PyArray_DATA(positions)
    resultData = <int64_t *>cnp.PyArray_DATA(result)

    with nogil:
        for queryIndex in range(queryCount):
            query = queryData[queryIndex]
            index = _searchRun(runs.ends, query, 0, runs.count)
            resultData[queryIndex] = _prefixAt(&runs, query, index)
    return result


cdef inline Py_ssize_t _seekRun(const _RunTable *runs, int64_t query,
                                Py_ssize_t index, Py_ssize_t first) noexcept nogil:
    if first and (first == runs.count or query < runs.ends[first]):
        return first

    # Advance to the next run for nearby queries and binary search longer jumps.
    if index >= first and (not index or query >= runs.ends[index - 1]):
        if index == runs.count or query < runs.ends[index]:
            return index
        if index + 1 == runs.count or query < runs.ends[index + 1]:
            return index + 1
    return _searchRun(runs.ends, query, first, runs.count)


cdef inline void _seekPrefix(const _RunTable *runs, int64_t position, int64_t k,
                             _ScoreCursor *cursor, const _ScoreCursor *lower) noexcept nogil:
    cdef Py_ssize_t first
    if position != cursor.position:
        cursor.position = position
        first = 0 if lower == NULL else lower.index
        cursor.index = _seekRun(runs, position, cursor.index, first)
        cursor.prefix = _prefixAt(runs, position, cursor.index)
        if runs.moments != NULL:
            first = 0 if lower == NULL else lower.shiftedIndex
            cursor.shiftedIndex = _seekRun(runs, position - k, cursor.shiftedIndex, first)
            cursor.prefix -= _prefixAt(runs, position - k, cursor.shiftedIndex)


def _scoreIntervals(cnp.ndarray queryStarts not None, cnp.ndarray queryEnds not None,
                    cnp.ndarray starts not None, cnp.ndarray ends not None,
                    cnp.ndarray totals not None, int64_t startShift=0,
                    cnp.ndarray moments=None, int64_t k=0):
    cdef _RunTable runs = _runTable(starts, ends, totals, moments)
    cdef _ScoreCursor left, right
    cdef cnp.ndarray result
    cdef const char *startData = cnp.PyArray_BYTES(queryStarts)
    cdef const char *endData = cnp.PyArray_BYTES(queryEnds)
    cdef const cnp.npy_intp *shape = cnp.PyArray_DIMS(queryStarts)
    cdef const cnp.npy_intp *startStrides = cnp.PyArray_STRIDES(queryStarts)
    cdef const cnp.npy_intp *endStrides = cnp.PyArray_STRIDES(queryEnds)
    cdef cnp.npy_intp coordinates[cnp.NPY_MAXDIMS]
    cdef Py_ssize_t count = cnp.PyArray_SIZE(queryStarts)
    cdef Py_ssize_t rowLength = count, rowCount = 1, startOffset = 0, endOffset = 0
    cdef Py_ssize_t startStride = sizeof(int64_t), endStride = sizeof(int64_t)
    cdef Py_ssize_t row, column
    cdef int axis, dimensions = cnp.PyArray_NDIM(queryStarts), outer = 0
    cdef int64_t start, end, width
    cdef double *output

    if (
        queryStarts.dtype != np.dtype(np.int64)
        or queryEnds.dtype != np.dtype(np.int64)
        or not cnp.PyArray_SAMESHAPE(queryStarts, queryEnds)
        or startShift < 0 or (moments is not None and k <= 0)
    ):
        raise InputError(kind="array")

    result = np.empty_like(queryStarts, dtype=np.float64, order="C", subok=False)
    if not count:
        return result
    output = <double *>cnp.PyArray_DATA(result)
    memset(&left, 0, sizeof(left))
    memset(&right, 0, sizeof(right))

    if not (cnp.PyArray_ISCONTIGUOUS(queryStarts) and cnp.PyArray_ISCONTIGUOUS(queryEnds)):
        outer = dimensions - 1
        rowLength = shape[outer]
        rowCount = count // rowLength
        startStride, endStride = startStrides[outer], endStrides[outer]
        memset(coordinates, 0, sizeof(coordinates))

    with nogil:
        for row in range(rowCount):
            for column in range(rowLength):
                memcpy(&start, startData + startOffset + column * startStride, sizeof(start))
                memcpy(&end, endData + endOffset + column * endStride, sizeof(end))
                width = end - start + startShift
                start -= startShift
                if start == right.position:
                    left = right
                else:
                    _seekPrefix(&runs, start, k, &left, NULL)
                _seekPrefix(&runs, end, k, &right, &left)
                if runs.moments != NULL:
                    width *= k
                output[0] = <double>(right.prefix - left.prefix) / width
                output += 1

            for axis in range(outer - 1, -1, -1):
                coordinates[axis] += 1
                if coordinates[axis] < shape[axis]:
                    startOffset += startStrides[axis]
                    endOffset += endStrides[axis]
                    break
                coordinates[axis] = 0
                startOffset -= (shape[axis] - 1) * startStrides[axis]
                endOffset -= (shape[axis] - 1) * endStrides[axis]

    return result if dimensions else result[()]
