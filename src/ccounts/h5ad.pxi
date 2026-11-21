from libc.math cimport isfinite, trunc, truncf, truncl


ctypedef fused csrIndex_t:
    cnp.int32_t
    cnp.int64_t


ctypedef fused csrValue_t:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.float32_t
    cnp.float64_t
    cnp.longdouble_t


cdef inline bint _validCSRValue(csrValue_t value) noexcept nogil:
    if csrValue_t is cnp.float32_t:
        return value >= 0 and isfinite(value) and value == truncf(value)
    elif csrValue_t is cnp.float64_t:
        return value >= 0 and isfinite(value) and value == trunc(value)
    elif csrValue_t is cnp.longdouble_t:
        return value >= 0 and isfinite(value) and value == truncl(value)
    else:
        return value >= 0


def _aggregateH5ADCSR(const csrIndex_t[::1] columns not None,
                      const csrValue_t[::1] values not None,
                      const int64_t[::1] starts not None,
                      const int64_t[::1] stops not None,
                      int64_t offset, int64_t left, int64_t right,
                      double[::1] result not None, double[::1] correction,
                      bint countColumns):
    cdef Py_ssize_t row, index, start, stop, size = values.shape[0]
    cdef int64_t column
    cdef csrValue_t value
    cdef int status = 0
    cdef double high, term, combined, error
    cdef uint64_t integerValue

    if (columns.shape[0] != size or starts.shape[0] != stops.shape[0]
            or offset < 0 or left < 0 or right <= left
            or result.shape[0] != (right - left if countColumns else 1)
            or (correction is not None and correction.shape[0] != result.shape[0])):
        raise InputError(kind="array")

    with nogil:
        for row in range(starts.shape[0]):
            if (starts[row] < offset or stops[row] < starts[row]
                    or stops[row] - offset > size):
                status = 1
                break
            start = <Py_ssize_t>(starts[row] - offset)
            stop = <Py_ssize_t>(stops[row] - offset)
            for index in range(start, stop):
                column = columns[index]
                if column < left or column >= right:
                    continue
                value = values[index]
                if not _validCSRValue(value):
                    status = 2
                    break
                column = column - left if countColumns else 0
                high = result[column]
                term = <double>value
                combined = high + term
                result[column] = combined
                # Nonnegative integer sums below 2**53 are exact in float64.
                if combined >= 9007199254740992.0:
                    if correction is None:
                        with gil:
                            correction = np.zeros(result.shape[0], dtype=np.float64)
                    if high >= term:
                        error = (high - combined) + term
                    else:
                        error = (term - combined) + high
                    if csrValue_t is cnp.int64_t or csrValue_t is cnp.uint64_t:
                        integerValue = <uint64_t>value
                        error += (<double>(integerValue & <uint64_t>0xffffffff) -
                                  (term - <double>((integerValue >> 32) << 32)))
                    elif csrValue_t is cnp.longdouble_t:
                        error += <double>(value - <long double>term)
                    correction[column] += error
            if status:
                break

    if status == 1:
        raise InputError(kind="array")
    if status == 2:
        raise CountError()
    return correction
