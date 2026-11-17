from libc.math cimport lgamma
from libc.stdint cimport intptr_t
from scipy.special.cython_special cimport betainc

cdef extern from "native/fenwickMedian.h":
    int ccMedianProfile(const intptr_t *, size_t, const double *, size_t,
                        size_t, double *) noexcept nogil


cdef inline double _musicUpperLogTail(int64_t observed, int64_t expected) noexcept nogil:
    cdef int64_t n = observed + expected
    cdef int64_t count = observed
    cdef double term = 1.0, total = 1.0

    if observed == 0:
        return 0.0

    # add each larger count's probability relative to P(X=observed)
    # keeping P(X=observed) in log form avoids rounding tiny tails to zero
    while count < n:
        term *= <double>(n - count) / (count + 1)
        total += term
        if term < total * 1e-15:
            break
        count += 1

    return (lgamma(n + 1.0) - lgamma(observed + 1.0)
        - lgamma(n - observed + 1.0) - n * log(2.0) + log(total))


cdef inline double _musicLogTail(int64_t observed, int64_t expected) noexcept nogil:
    if observed >= expected:
        return _musicUpperLogTail(observed, expected)
    if observed == 0:
        return 0.0
    # below background, the tail is >= 1/2, so taking its log is safe
    return log(betainc(<double>observed, <double>(expected + 1), 0.5))


def musicBinomialLogTail(cnp.int64_t[::1] countsO, cnp.int64_t[::1] countsE):
    r"""Compute log tail of binomial with p=0.5

    Used for MUSIC to get log P-values for enrichment over background when
    trimming candidate regions.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] result = np.empty(countsO.shape[0])
    cdef Py_ssize_t index
    if countsO.shape[0] != countsE.shape[0]:
        raise InputError(kind="array")

    for index in range(countsO.shape[0]):
        if countsE[index] < 0 or countsO[index] < countsE[index]:
            raise CountError()
        if countsO[index] > INT64_MAX - countsE[index]:
            raise NumericalError()
        result[index] = _musicUpperLogTail(countsO[index], countsE[index])
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


def musicCandidateRegions(const int64_t[::1] signal not None,
                          const int64_t[::1] filterSignal not None,
                          const double[::1] smooth not None,
                          const intptr_t[::1] boundaries not None,
                          const double[::1] thresholds not None,
                          const int64_t[::1] countsPrefix not None,
                          const double[::1] backgroundPrefix not None,
                          Py_ssize_t width, Py_ssize_t binSizeBP,
                          double smoothingRatio, Py_ssize_t pValueWindowBP,
                          double significanceLevel, dict cache,
                          intptr_t[::1] largest not None,
                          Py_ssize_t supportLevel):
    cdef Py_ssize_t nBins = signal.shape[0]
    cdef Py_ssize_t boundaryCount = boundaries.shape[0]
    cdef Py_ssize_t boundaryIndex, index, left, last, right, regionLeft, regionRight
    cdef Py_ssize_t halfWidth = width // 2
    cdef Py_ssize_t windowLeft, windowRight, added = 0
    cdef Py_ssize_t head = 0, tail = 0, bestIndex
    cdef Py_ssize_t intervalCount = 0, coverageCount = 0
    cdef int64_t rawMaximum, observed, expected, running = 0
    cdef double smoothMaximum, score, bestScore, factor, probability
    cdef tuple key
    cdef object region
    cdef cnp.ndarray[cnp.intp_t, ndim=1, mode="c"] maximumIndices
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] covered

    if (
        nBins == 0
        or filterSignal.shape[0] != nBins
        or smooth.shape[0] != nBins
        or thresholds.shape[0] != nBins
        or largest.shape[0] != nBins
        or countsPrefix.shape[0] != nBins + 1
        or backgroundPrefix.shape[0] != nBins + 1
        or boundaryCount == 0
    ):
        raise InputError(kind="array")
    if (
        width <= 0 or width % 2 == 0 or binSizeBP <= 0
        or pValueWindowBP <= 0 or supportLevel <= 0
    ):
        raise InputError()
    if boundaries[0] != 0 or boundaries[boundaryCount - 1] != nBins - 1:
        raise InputError(kind="array")

    maximumIndices = np.empty(nBins, dtype=np.intp)
    covered = np.zeros(nBins + 1, dtype=np.int64)

    for boundaryIndex in range(boundaryCount - 1):
        left = boundaries[boundaryIndex]
        last = boundaries[boundaryIndex + 1]
        if left < 0 or last <= left or last >= nBins:
            raise InputError(kind="array")
        right = last + 1

        windowLeft = left - halfWidth
        if windowLeft < 0:
            windowLeft = 0
        if halfWidth >= nBins - right:
            windowRight = nBins
        else:
            windowRight = right + halfWidth

        # counts [9, 4, 7] need only [9, 7] as possible maxima
        # the 4 is smaller than 7 and lies further left
        with nogil:
            while added < windowRight:
                while (
                    head < tail
                    and filterSignal[maximumIndices[tail - 1]] <= filterSignal[added]
                ):
                    tail -= 1
                maximumIndices[tail] = added
                tail += 1
                added += 1

            # drop bins left of the window, leaving its largest count at the front
            while head < tail and maximumIndices[head] < windowLeft:
                head += 1
            rawMaximum = filterSignal[maximumIndices[head]]
            smoothMaximum = smooth[left]
            for index in range(left + 1, right):
                if smooth[index] > smoothMaximum:
                    smoothMaximum = smooth[index]

        if smoothMaximum <= 0 or rawMaximum > smoothingRatio * smoothMaximum:
            continue

        # same boundaries at different widths give the same trimmed interval
        key = (left, right)
        if key not in cache:
            regionLeft = -1
            regionRight = -1
            # keep everything from the first bin above threshold to the last
            with nogil:
                for index in range(left, right):
                    if signal[index] > thresholds[index]:
                        if regionLeft < 0:
                            regionLeft = index
                        regionRight = index + 1
            if regionLeft < 0:
                cache[key] = None
            else:
                with nogil:
                    # hold the right edge fixed and pick the smallest P-value
                    bestIndex = regionLeft
                    observed = countsPrefix[regionRight] - countsPrefix[regionLeft]
                    expected = <int64_t>(
                        backgroundPrefix[regionRight] - backgroundPrefix[regionLeft]
                    )
                    bestScore = _musicLogTail(observed, expected)
                    for index in range(regionLeft + 1, regionRight):
                        observed = countsPrefix[regionRight] - countsPrefix[index]
                        expected = <int64_t>(
                            backgroundPrefix[regionRight] - backgroundPrefix[index]
                        )
                        score = _musicLogTail(observed, expected)
                        # keep the first edge if the P-values tie
                        if score < bestScore:
                            bestScore = score
                            bestIndex = index
                    regionLeft = bestIndex

                    # score the right edges with the chosen left edge fixed
                    bestIndex = regionLeft + 1
                    observed = countsPrefix[bestIndex] - countsPrefix[regionLeft]
                    expected = <int64_t>(
                        backgroundPrefix[bestIndex] - backgroundPrefix[regionLeft]
                    )
                    bestScore = _musicLogTail(observed, expected)
                    for index in range(regionLeft + 2, regionRight + 1):
                        observed = countsPrefix[index] - countsPrefix[regionLeft]
                        expected = <int64_t>(
                            backgroundPrefix[index] - backgroundPrefix[regionLeft]
                        )
                        score = _musicLogTail(observed, expected)
                        if score < bestScore:
                            bestScore = score
                            bestIndex = index
                    regionRight = bestIndex

                    # rescale long intervals to pValueWindowBP bases for this test
                    factor = <double>pValueWindowBP / (
                        (regionRight - regionLeft) * binSizeBP
                    )
                    if factor > 1.0:
                        factor = 1.0
                    observed = <int64_t>(
                        (countsPrefix[regionRight] - countsPrefix[regionLeft]) * factor
                    )
                    expected = <int64_t>(
                        (backgroundPrefix[regionRight] - backgroundPrefix[regionLeft])
                        * factor
                    )
                    if factor < 1.0 and expected < 1:
                        expected = 1
                    probability = 1.0
                    if observed > 0:
                        # same binomial P-value, without a Python stats call
                        probability = betainc(
                            <double>observed, <double>(expected + 1), 0.5
                        )
                cache[key] = (regionLeft, regionRight, probability)

        region = cache[key]
        if region is None or region[2] > significanceLevel:
            continue
        regionLeft = region[0]
        regionRight = region[1]
        if regionLeft < 0 or regionRight > nBins or regionLeft >= regionRight:
            raise InputError(kind="array")
        # +1 at the left edge and -1 at the right mark an accepted interval
        covered[regionLeft] += 1
        covered[regionRight] -= 1
        intervalCount += 1

    # a positive running total means an accepted interval covers this bin
    with nogil:
        for index in range(nBins):
            running += covered[index]
            if running > 0:
                largest[index] = supportLevel
                coverageCount += 1
    return intervalCount, coverageCount
