cimport numpy as cnp
from libc.math cimport isfinite
from libc.stdint cimport uintptr_t

from .exceptions import InputError, NumericalError


cdef inline bint _isContiguousBuffer(object array) noexcept:
    return cnp.PyArray_Check(array) and cnp.PyArray_ISCARRAY_RO(<cnp.ndarray>array)


cdef inline Py_ssize_t _checkIntervalBuffers(cnp.ndarray positions,
    cnp.ndarray starts, cnp.ndarray ends, cnp.ndarray totals,
    cnp.ndarray moments) except -1:
    cdef cnp.ndarray array
    cdef Py_ssize_t runCount = cnp.PyArray_SIZE(ends)
    for array in (positions, starts, ends, totals, moments):
        if array is None:
            continue
        if cnp.PyArray_TYPE(array) != cnp.NPY_INT64 or not _isContiguousBuffer(array):
            raise InputError(kind="array")
    if (
        starts.ndim != 1
        or ends.ndim != 1
        or totals.ndim != 1
        or starts.size != runCount + 1
        or totals.size != runCount + 1
    ):
        raise InputError(kind="array")
    if moments is not None and (moments.ndim != 1 or moments.size != runCount + 1):
        raise InputError(kind="array")
    return runCount


cdef inline Py_ssize_t _checkLogValues(object values, object reference,
    double pseudocount, object out) except -1:
    cdef cnp.ndarray valuesArray, referenceArray, outArray
    cdef Py_ssize_t count
    cdef size_t byteCount
    cdef int typeNumber
    cdef uintptr_t valuesAddress, outAddress, referenceAddress
    if not _isContiguousBuffer(values):
        raise InputError(kind="array")
    valuesArray = <cnp.ndarray>values
    typeNumber = cnp.PyArray_TYPE(valuesArray)
    if (
        valuesArray.ndim not in (1, 2)
        or typeNumber not in (cnp.NPY_FLOAT32, cnp.NPY_FLOAT64)
    ):
        raise InputError(kind="array")
    if not _isContiguousBuffer(out):
        raise InputError(kind="output")
    outArray = <cnp.ndarray>out
    if (
        not cnp.PyArray_SAMESHAPE(outArray, valuesArray)
        or cnp.PyArray_TYPE(outArray) != typeNumber
        or not cnp.PyArray_ISWRITEABLE(outArray)
    ):
        raise InputError(kind="output")
    if not isfinite(pseudocount) or pseudocount <= 0:
        raise NumericalError(kind="logDomain")
    if reference is not None:
        if not _isContiguousBuffer(reference):
            raise InputError(kind="array")
        referenceArray = <cnp.ndarray>reference
        if (
            not cnp.PyArray_SAMESHAPE(referenceArray, valuesArray)
            or cnp.PyArray_TYPE(referenceArray) != typeNumber
        ):
            raise InputError(kind="array")
    count = cnp.PyArray_SIZE(valuesArray)
    byteCount = cnp.PyArray_NBYTES(valuesArray)
    valuesAddress = <uintptr_t>cnp.PyArray_DATA(valuesArray)
    outAddress = <uintptr_t>cnp.PyArray_DATA(outArray)
    if byteCount and valuesAddress != outAddress and (
        valuesAddress < outAddress + byteCount and outAddress < valuesAddress + byteCount
    ):
        raise InputError(kind="overlap")
    if reference is not None:
        referenceAddress = <uintptr_t>cnp.PyArray_DATA(referenceArray)
        if byteCount and referenceAddress != outAddress and (
            referenceAddress < outAddress + byteCount and outAddress < referenceAddress + byteCount
        ):
            raise InputError(kind="overlap")
    return count
