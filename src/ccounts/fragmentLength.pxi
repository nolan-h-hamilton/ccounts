from libc.math cimport sqrt, NAN
from libc.stdint cimport INT64_MAX
from libc.string cimport memset


cdef Py_ssize_t _curve(const int64_t *forward, const int64_t *forwardCount,
    Py_ssize_t forwardSize, const int64_t *reverse, const int64_t *reverseCount,
    Py_ssize_t reverseSize, int64_t dimension, const int64_t *candidates,
    Py_ssize_t candidateCount, int64_t stepBP, double *values,
    Py_ssize_t *firstValid, Py_ssize_t *lastValid, double *minimum) noexcept nogil:
    r"""Compute correlations of forward-strand and reverse-strand counts at candidate shifts/offsets.

    This is used for estimating (pseudo)-fragment lengths in single-end sequencing data.
    For a shift s, compute Pearson correlation between F[x] and R[x + s]. Return
    the index of the shift/offset where fwd/backward correlation is greatest.
    """

    cdef Py_ssize_t i, j, k, reverseStart = 0, forwardEnd = forwardSize
    cdef Py_ssize_t lowerReverse = 0, best = -1
    cdef int64_t distance, shift, overlap
    cdef double sumF = 0, sumR = 0, squareF = 0, squareR = 0
    cdef double weight, varianceF, varianceR, score

    # s = candidates[k], n = dimension - s
    # values[k] = sum_{x=0}^{n-1} F[x] * R[x + s]
    memset(values, 0, candidateCount * sizeof(double))
    for i in range(forwardSize):
        if forwardCount[i] == 0:
            continue
        # skip reverse positions too small for the first candidate.
        while lowerReverse < reverseSize and reverse[lowerReverse] - forward[i] < candidates[0]:
            lowerReverse += 1

        j = lowerReverse
        while j < reverseSize:
            distance = reverse[j] - forward[i]
            if distance > candidates[candidateCount - 1]:
                break
            # distance = candidates[0] + k * stepBP
            if reverseCount[j] and (distance - candidates[0]) % stepBP == 0:
                k = (distance - candidates[0]) // stepBP
                values[k] += <double>forwardCount[i] * reverseCount[j]
            j += 1

    # compute sums and sum of squares for full forward and reverse blocks.
    for i in range(forwardSize):
        weight = forwardCount[i]
        sumF += weight
        squareF += weight * weight

    for j in range(reverseSize):
        weight = reverseCount[j]
        sumR += weight
        squareR += weight * weight

    firstValid[0] = -1
    lastValid[0] = -1
    minimum[0] = 0

    for k in range(candidateCount):
        # F[0:dimension - shift] pairs to R[shift:dimension]
        shift = candidates[k]
        overlap = dimension - shift
        if overlap <= 1:
            values[k] = NAN
            continue

        # resize/update forward block as shift increases
        while forwardEnd > 0 and forward[forwardEnd - 1] >= overlap:
            forwardEnd -= 1
            weight = forwardCount[forwardEnd]
            sumF -= weight
            squareF -= weight * weight

        # resize/update reverse block as shift increases
        while reverseStart < reverseSize and reverse[reverseStart] < shift:
            weight = reverseCount[reverseStart]
            sumR -= weight
            squareR -= weight * weight
            reverseStart += 1

        varianceF = squareF - sumF * sumF / overlap
        varianceR = squareR - sumR * sumR / overlap

        if varianceF <= 0 or varianceR <= 0:
            values[k] = NAN
            continue

        # this gives the Pearson correlation
        score = (values[k] - sumF * sumR / overlap) / sqrt(varianceF * varianceR)
        values[k] = score
        if firstValid[0] < 0:
            firstValid[0] = k
            minimum[0] = score
        elif score < minimum[0]:
            minimum[0] = score
        lastValid[0] = k
        # pick the smaller when tied
        if best < 0 or score > values[best]:
            best = k
    return best


def estimateBlock(cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] forward not None,
    cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] reverse not None,
    int64_t widthBP, *, int64_t minLengthBP=100, int64_t maxLengthBP=1000,
    int64_t stepBP=1):
    """Estimate fragment length from forward/reverse positions in one block.

    Positions are sorted and relative to the block. Repeated positions count
    as separate reads. Return the shift with the greatest correlation, along
    with the curve and flags. Return None if no shift has a valid correlation.
    """
    cdef Py_ssize_t nF = forward.size, nR = reverse.size
    cdef Py_ssize_t i, uniqueF = 0, uniqueR = 0
    cdef Py_ssize_t best, firstValid, lastValid
    cdef int64_t dimension, estimate
    cdef double minimum
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] positionsF, positionsR, countsF, countsR
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] candidates
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] point

    if nF < 1 or nR < 1:
        raise EstimateError(kind="insufficient")
    if widthBP <= 0 or widthBP == INT64_MAX:
        raise InputError(kind="option", context="blockWidthBP")
    if minLengthBP < 1 or maxLengthBP < minLengthBP or maxLengthBP == INT64_MAX or stepBP < 1:
        raise InputError()

    # 0 <= forward < widthBP, 0 < reverse <= widthBP, both sorted
    for i in range(nF):
        if forward[i] < 0 or forward[i] >= widthBP or (i and forward[i] < forward[i - 1]):
            raise InputError(kind="region")
    for i in range(nR):
        if reverse[i] <= 0 or reverse[i] > widthBP or (i and reverse[i] < reverse[i - 1]):
            raise InputError(kind="region")

    # widthBP + 1 coordinates, including endpoints 0 and widthBP
    dimension = widthBP + 1
    positionsF = np.empty(nF, dtype=np.int64)
    positionsR = np.empty(nR, dtype=np.int64)
    countsF = np.zeros(nF, dtype=np.int64)
    countsR = np.zeros(nR, dtype=np.int64)
    candidates = np.arange(minLengthBP, maxLengthBP + 1, stepBP, dtype=np.int64)
    point = np.empty(candidates.size, dtype=np.float64)

    with nogil:
        # fill the input arrays for _curve()
        for i in range(nF):
            if i == 0 or forward[i] != forward[i - 1]:
                positionsF[uniqueF] = forward[i]
                uniqueF += 1
            countsF[uniqueF - 1] += 1

        for i in range(nR):
            if i == 0 or reverse[i] != reverse[i - 1]:
                positionsR[uniqueR] = reverse[i]
                uniqueR += 1
            countsR[uniqueR - 1] += 1

        # returns the best candidate index, or -1 if none
        best = _curve(<int64_t *>positionsF.data, <int64_t *>countsF.data, uniqueF,
            <int64_t *>positionsR.data, <int64_t *>countsR.data, uniqueR, dimension,
            <int64_t *>candidates.data, candidates.size, stepBP, <double *>point.data,
            &firstValid, &lastValid, &minimum)

    # case: no shift with a defined correlation
    if best < 0:
        return None
    estimate = candidates[best]
    peakCorrelation = float(point[best])

    # peak at either end of the valid part of the curve
    boundaryMaximum = best == firstValid or best == lastValid
    # flatMaximum means all valid correlations are tied
    return {"estimateBP": int(estimate), "pointCurve": point,
        "candidateLengthsBP": candidates, "peakCorrelation": peakCorrelation,
        "boundaryMaximum": bool(boundaryMaximum), "flatMaximum": peakCorrelation == minimum,
        "nonpositiveMaximum": peakCorrelation <= 0,
        "nF": int(nF), "nR": int(nR), "uniqueF": int(uniqueF), "uniqueR": int(uniqueR)}
