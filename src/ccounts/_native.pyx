# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
from .exceptions import (
    AllocationError,
    CountError,
    EstimateError,
    EstimateWarning,
    InputError,
    NativeError,
    NumericalError,
    OperationError,
    OutputExistsError,
    SourceError,
)
from pathlib import Path
from time import perf_counter
cimport numpy as cnp
from libc.stdint cimport int64_t, uint64_t, uint16_t, uint32_t, uintptr_t
from libc.math cimport log, log1p, fabs, fmax, isfinite
from libc.stdlib cimport free
from libc.string cimport memcpy

# numeric codes passed to countsBackend
cdef enum:
    sourceBAM = 0
    sourceFragments = 1
    sourceBedGraph = 2

    modeCoverage = 0
    modeConservedFractionalOverlap = 1
    modeCenter = 2
    modeFivePrime = 3
    modeFFP = 4
    modeBinOverlap = 5

    inputReads = 0
    inputFragments = 1
    inputRead1 = 2

cdef enum:
    pairedInspectSampleSize = 1000
    defaultMaxTemplateLengthBP = 1000

# log2(x) = log(x) * log2E
cdef const double log2E = 1.4426950408889634074
cdef const double log1pRatioLimit = 0.5
cdef const double safeDifferenceMagnitude = 8.988465674311578e307
cdef const double minNormalDouble = 2.2250738585072014e-308

cnp.import_array()

include "fragmentLength.pxi"
include "pairedFragmentLength.pxi"
include "lengthScale.pxi"

cdef extern from "native/countsBackend.h":
    ctypedef struct ccSource:
        pass
    ctypedef struct ccOptions:
        int countMode
        int bamInputMode
        uint16_t flagExclude
        int minMappingQuality
        int64_t minTemplateLengthBP
        int64_t maxTemplateLengthBP
        int64_t shiftForward5p
        int64_t shiftReverse5p
        int64_t extendFrom5pBP
        int fragmentsUseReadSupport
    ctypedef struct ccTotals:
        uint64_t scannedRecords
        double eventMass
        double spanBP
    int ccOpen(const char *, int, const char *, const char *, int, ccSource **, const char **) nogil
    void ccClose(ccSource *) noexcept nogil
    int ccChromosomeCount(ccSource *) nogil
    const char *ccChromosomeName(ccSource *, int) nogil
    int64_t ccChromosomeLength(ccSource *, int) nogil
    int ccCount(ccSource *, const char *, int64_t, int64_t, int64_t, int64_t,
                const ccOptions *, double *, double *, size_t, const char **) nogil
    int ccRegionTotals(ccSource *, const char *, int64_t, int64_t, int64_t,
                       const ccOptions *, ccTotals *, const char **) nogil
    int ccInspect(ccSource *, const ccOptions *, int, int *, const char **) nogil
    int ccReadLength(ccSource *, uint16_t, uint32_t *, const char **) nogil
    int ccSourceRange(ccSource *, const char *, uint64_t, uint16_t, uint64_t *, uint64_t *, const char **) nogil
    int ccChromRange(ccSource *, const char *, int64_t, const ccOptions *, int64_t *, int64_t *, const char **) nogil
    ctypedef struct ccEndpointCounts:
        uint64_t queriedRecords
        uint64_t excludedFlags
        uint64_t excludedMappingQuality
        uint64_t excludedSelection
        uint64_t outsideEndpointOwnership
        uint64_t preDedupForwardRecords
        uint64_t preDedupReverseRecords
        uint64_t removedForwardCoordinateDuplicates
        uint64_t removedReverseCoordinateDuplicates
        uint64_t retainedForwardEndpoints
        uint64_t retainedReverseEndpoints
    int ccBlockEndpoints(ccSource *, const char *, int64_t, int64_t, const ccOptions *,
                         int64_t **, int64_t **, size_t *, size_t *, ccEndpointCounts *, const char **) nogil
    int ccBuildIndex(const char *, int, const char *, int, const char **) nogil

cdef int _kind(str kind) except -1:
    if kind == "BAM": return sourceBAM
    if kind == "FRAGMENTS": return sourceFragments
    if kind == "BEDGRAPH": return sourceBedGraph
    raise SourceError()

cdef ccOptions _options(object options) except *:
    cdef ccOptions out
    cdef str mode = options.countMode
    cdef str bamMode = options.bamInputMode
    if mode == "coverage": out.countMode = modeCoverage
    elif mode == "conservedFractionalOverlap": out.countMode = modeConservedFractionalOverlap
    elif mode == "center": out.countMode = modeCenter
    elif mode == "fiveprime": out.countMode = modeFivePrime
    elif mode == "ffp": out.countMode = modeFFP
    elif mode == "binOverlap": out.countMode = modeBinOverlap
    else: raise InputError(kind="option", context="countMode")
    if bamMode in ("reads", "auto"): out.bamInputMode = inputReads
    elif bamMode == "fragments": out.bamInputMode = inputFragments
    elif bamMode == "read1": out.bamInputMode = inputRead1
    else: raise InputError(kind="option", context="bamInputMode")
    out.flagExclude = options.flagExclude
    out.minMappingQuality = options.minMappingQuality
    out.minTemplateLengthBP = options.minTemplateLengthBP
    out.maxTemplateLengthBP = 0 if options.maxTemplateLengthBP is None else options.maxTemplateLengthBP
    out.shiftForward5p = options.shiftForward5p
    out.shiftReverse5p = options.shiftReverse5p
    out.extendFrom5pBP = 0 if options.extendFrom5pBP is None else options.extendFrom5pBP
    out.fragmentsUseReadSupport = options.fragmentsUseReadSupport
    return out

cdef void _raise(int status, const char *error) except *:
    if status != 0:
        if error != NULL:
            raise NativeError((<bytes>error).decode("utf-8"))
        raise OperationError()

cdef class NativeSource:
    cdef ccSource *_source
    cdef dict _sizes
    cdef str _barcodeAllowListFile
    cdef str _barcodeTag
    cdef int _threads
    cdef public tuple chromosomeNames
    cdef public dict chromosomeSizes
    cdef public str path
    cdef public str sourceKind

    def __cinit__(self):
        self._source = NULL

    def __init__(self, path, sourceKind, chromosomeSizes, *, barcodeAllowListFile="", barcodeTag="CB", threads=1):
        cdef bytes pathBytes = str(path).encode("utf-8")
        cdef bytes allowBytes = str(barcodeAllowListFile).encode("utf-8")
        cdef bytes tagBytes = barcodeTag.encode("utf-8")
        cdef const char *pathPointer = pathBytes
        cdef const char *allowPointer = allowBytes
        cdef const char *tagPointer = tagBytes
        cdef int kind = _kind(sourceKind)
        cdef int threadCount = threads
        cdef int status, i, n
        cdef const char *error = NULL
        cdef const char *name
        cdef int64_t length
        if threadCount < 1: raise InputError(kind="integer", context="threads")
        self.path = str(path)
        self._barcodeAllowListFile = str(barcodeAllowListFile)
        self._barcodeTag = str(barcodeTag)
        self._threads = threadCount
        self.sourceKind = sourceKind
        self._sizes = dict(chromosomeSizes)
        with nogil:
            status = ccOpen(pathPointer, kind, allowPointer, tagPointer, threadCount, &self._source, &error)
        _raise(status, error)
        self.chromosomeSizes = {}
        n = ccChromosomeCount(self._source)
        names = []
        for i in range(n):
            name = ccChromosomeName(self._source, i)
            chromosome = (<bytes>name).decode("utf-8")
            names.append(chromosome)
            length = ccChromosomeLength(self._source, i)
            if length >= 0:
                self.chromosomeSizes[chromosome] = int(length)
                if chromosome in self._sizes and self._sizes[chromosome] != length:
                    self.close()
                    raise InputError(kind="grid")
        self.chromosomeNames = tuple(names)

    def __dealloc__(self):
        if self._source != NULL:
            ccClose(self._source)
            self._source = NULL

    cdef void _checkOpen(self) except *:
        if self._source == NULL: raise SourceError(kind="closed")

    def close(self):
        if self._source != NULL:
            with nogil:
                ccClose(self._source)
            self._source = NULL

    def countRegion(self, chromosome, int64_t start, int64_t end, int64_t binSizeBP, options, returnRawNoiseMass=False):
        cdef bytes name = chromosome.encode("utf-8")
        cdef const char *namePointer = name
        cdef int64_t length
        cdef Py_ssize_t count
        cdef ccOptions parsed = _options(options)
        cdef int status
        cdef const char *error = NULL
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] values
        cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] mass
        cdef double *massPointer = NULL
        self._checkOpen()
        length = self._sizes[chromosome]
        if start < 0 or end <= start or end > length or binSizeBP <= 0 or start % binSizeBP or (end != length and end % binSizeBP):
            raise InputError(kind="region")
        count = (end - start + binSizeBP - 1) // binSizeBP
        values = np.zeros(count, dtype=np.float64)
        if returnRawNoiseMass:
            mass = np.zeros(count, dtype=np.float64)
            massPointer = <double *>mass.data
        with nogil:
            status = ccCount(self._source, namePointer, length, start, end, binSizeBP, &parsed,
                             <double *>values.data, massPointer, count, &error)
        _raise(status, error)
        return values, mass if returnRawNoiseMass else None

    def totalsRegion(self, chromosome, int64_t start, int64_t end, options):
        cdef bytes name = chromosome.encode("utf-8")
        cdef const char *namePointer = name
        cdef int64_t length
        cdef ccOptions parsed = _options(options)
        cdef ccTotals totals
        cdef int status
        cdef const char *error = NULL
        self._checkOpen()
        length = self._sizes[chromosome]
        if start < 0 or end <= start or end > length: raise InputError(kind="region")
        with nogil:
            status = ccRegionTotals(self._source, namePointer, length, start, end, &parsed, &totals, &error)
        _raise(status, error)
        return {"scannedRecords": int(totals.scannedRecords), "eventMass": totals.eventMass, "spanBP": totals.spanBP}

    def pairedEnd(self, options):
        cdef ccOptions parsed = _options(options)
        cdef int paired, status
        cdef const char *error = NULL
        self._checkOpen()
        with nogil:
            status = ccInspect(self._source, &parsed, pairedInspectSampleSize, &paired, &error)
        _raise(status, error)
        return bool(paired)

    def readLength(self, options):
        cdef uint16_t flagExclude = options.flagExclude
        cdef uint32_t length
        cdef int status
        cdef const char *error = NULL
        self._checkOpen()
        with nogil:
            status = ccReadLength(self._source, flagExclude, &length, &error)
        _raise(status, error)
        return int(length)

    def sourceRange(self, chromosome, uint16_t flagExclude):
        cdef bytes name = chromosome.encode("utf-8")
        cdef const char *namePointer = name
        cdef uint64_t length = self._sizes[chromosome]
        cdef uint64_t start, end
        cdef int status
        cdef const char *error = NULL
        self._checkOpen()
        with nogil:
            status = ccSourceRange(self._source, namePointer, length, flagExclude, &start, &end, &error)
        _raise(status, error)
        return int(start), int(end)

    def chromRange(self, chromosome, options):
        cdef ccOptions parsed = _options(options)
        cdef bytes name = chromosome.encode("utf-8")
        cdef const char *namePointer = name
        cdef int64_t length = self._sizes[chromosome]
        cdef int64_t start, end
        cdef int status
        cdef const char *error = NULL
        self._checkOpen()
        with nogil:
            status = ccChromRange(self._source, namePointer, length, &parsed, &start, &end, &error)
        _raise(status, error)
        return int(start), int(end)

    def blockEndpoints(self, chromosome, int64_t start, int64_t end, options):
        cdef bytes name = chromosome.encode("utf-8")
        cdef const char *namePointer = name
        cdef ccOptions parsed = _options(options)
        cdef ccEndpointCounts counts
        cdef int status
        cdef const char *error = NULL
        cdef int64_t *forward = NULL
        cdef int64_t *reverse = NULL
        cdef size_t nF = 0, nR = 0
        cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] forwardArray, reverseArray
        self._checkOpen()
        if chromosome not in self.chromosomeSizes or start < 0 or end <= start or end > self.chromosomeSizes[chromosome]:
            raise InputError(kind="region")
        with nogil:
            status = ccBlockEndpoints(self._source, namePointer, start, end, &parsed,
                                       &forward, &reverse, &nF, &nR, &counts, &error)
        try:
            _raise(status, error)
            forwardArray = np.empty(nF, dtype=np.int64)
            reverseArray = np.empty(nR, dtype=np.int64)
            with nogil:
                if nF:
                    memcpy(<void *>forwardArray.data, forward, nF * sizeof(int64_t))
                if nR:
                    memcpy(<void *>reverseArray.data, reverse, nR * sizeof(int64_t))
        finally:
            free(forward)
            free(reverse)
        return forwardArray, reverseArray, {
            "queriedRecords": int(counts.queriedRecords),
            "excludedFlags": int(counts.excludedFlags),
            "excludedMappingQuality": int(counts.excludedMappingQuality),
            "excludedSelection": int(counts.excludedSelection),
            "outsideEndpointOwnership": int(counts.outsideEndpointOwnership),
            "preDedupForwardRecords": int(counts.preDedupForwardRecords),
            "preDedupReverseRecords": int(counts.preDedupReverseRecords),
            "removedForwardCoordinateDuplicates": int(counts.removedForwardCoordinateDuplicates),
            "removedReverseCoordinateDuplicates": int(counts.removedReverseCoordinateDuplicates),
            "retainedForwardEndpoints": int(counts.retainedForwardEndpoints),
            "retainedReverseEndpoints": int(counts.retainedReverseEndpoints)}

    def estimateFragmentLength(self, options, fragmentLengthOptions):
        self._checkOpen()
        if self.sourceKind != "BAM":
            raise SourceError()
        if self.pairedEnd(options):
            started = perf_counter()
            result = _pairedFragmentLength(
                self.path, self._threads, options.flagExclude,
                defaultMaxTemplateLengthBP if options.maxTemplateLengthBP is None else options.maxTemplateLengthBP,
                self._barcodeAllowListFile, self._barcodeTag)
            result.update({"method": "pairedTemplateMedian", "CILowerBP": None,
                           "CIUpperBP": None, "bootstrapStdBP": None, "warnings": [],
                           "runtimeSeconds": perf_counter() - started})
            if result["status"] == "insufficientData":
                result["warnings"].append(EstimateWarning.variants["fewFragments"])
            return result
        from .fragmentLength import _estimateSingleEnd
        return _estimateSingleEnd(self, options, fragmentLengthOptions)


def checkIndex(path, sourceKind, *, threads=1):
    source = NativeSource(path, sourceKind, {}, threads=threads)
    source.close()
    return True


def indexSource(path, sourceKind, *, targetPath=None, threads=1):
    cdef bytes pathBytes = str(path).encode("utf-8")
    cdef bytes targetBytes
    cdef const char *pathPointer = pathBytes
    cdef const char *targetPointer
    cdef int kind = _kind(sourceKind)
    cdef int threadCount = threads
    cdef int status
    cdef const char *error = NULL
    if threadCount < 1: raise InputError(kind="integer", context="threads")
    if kind == sourceBedGraph and (targetPath is None or str(targetPath) == str(path)):
        raise InputError(kind="option", context="targetPath")
    if kind == sourceBedGraph and Path(targetPath).exists():
        raise OutputExistsError()
    targetBytes = b"" if targetPath is None else str(targetPath).encode("utf-8")
    targetPointer = targetBytes
    with nogil:
        status = ccBuildIndex(pathPointer, kind, targetPointer, threadCount, &error)
    _raise(status, error)
    return str(path) if kind != sourceBedGraph else str(targetPath)

ctypedef fused real_t:
    cnp.float32_t
    cnp.float64_t

cdef inline double _logSum(double x, double p) noexcept nogil:
    cdef double high, low, total, removed, residual
    if x > 0:
        high = fmax(x, p)
        low = x if p > x else p
        return (log(high) + log1p(low / high)) * log2E
    total = p + x
    removed = total - p
    residual = (p - (total - removed)) + (x - removed)
    return (log(total) + log1p(residual / total)) * log2E

cdef void _logLoop(const real_t *values, const real_t *reference, real_t *out,
                    Py_ssize_t count, double pseudocount) noexcept nogil:
    cdef Py_ssize_t i
    cdef double x, r, scale, scaledX, scaledR, delta, ratio
    for i in range(count):
        x = values[i]
        if reference == NULL:
            out[i] = <real_t>_logSum(x, pseudocount)
        else:
            r = reference[i]
            scale = fmax(fmax(fabs(x), fabs(r)), pseudocount)
            scaledR = (r + pseudocount) / scale if r < 0 else r / scale + pseudocount / scale
            delta = x / scale - r / scale
            if scaledR > 0 and fabs(delta) <= log1pRatioLimit * scaledR:
                if (x >= 0 and r >= 0) or (x <= 0 and r <= 0) or (fabs(x) <= safeDifferenceMagnitude and fabs(r) <= safeDifferenceMagnitude):
                    delta = (x - r) / scale
                ratio = delta / scaledR
                out[i] = <real_t>(log1p(ratio) * log2E)
            else:
                scaledX = (x + pseudocount) / scale if x < 0 else x / scale + pseudocount / scale
                if scaledX >= minNormalDouble and scaledR >= minNormalDouble:
                    out[i] = <real_t>(log(scaledX / scaledR) * log2E)
                else:
                    out[i] = <real_t>(_logSum(x, pseudocount) - _logSum(r, pseudocount))


cdef inline Py_ssize_t _checkLogValues(object values, object reference,
    double pseudocount, object out) except -1:
    cdef Py_ssize_t count
    cdef size_t byteCount
    cdef uintptr_t valuesAddress, outAddress, referenceAddress
    if (
        not isinstance(values, np.ndarray) or values.ndim not in (1, 2)
        or values.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not values.flags.c_contiguous
    ):
        raise InputError(kind="array")
    if (
        not isinstance(out, np.ndarray) or out.shape != values.shape
        or out.dtype != values.dtype or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise InputError(kind="output")
    if not isfinite(pseudocount) or pseudocount <= 0:
        raise NumericalError(kind="logDomain")
    if reference is not None and (
        not isinstance(reference, np.ndarray) or reference.shape != values.shape
        or reference.dtype != values.dtype or not reference.flags.c_contiguous
    ):
        raise InputError(kind="array")
    count = (<cnp.ndarray>values).size
    byteCount = count * (sizeof(cnp.float32_t) if values.dtype == np.float32 else sizeof(cnp.float64_t))
    valuesAddress = <uintptr_t>(<cnp.ndarray>values).data
    outAddress = <uintptr_t>(<cnp.ndarray>out).data
    if byteCount and valuesAddress != outAddress and (
        valuesAddress < outAddress + byteCount and outAddress < valuesAddress + byteCount
    ):
        raise InputError(kind="overlap")
    if reference is not None:
        referenceAddress = <uintptr_t>(<cnp.ndarray>reference).data
        if byteCount and referenceAddress != outAddress and (
            referenceAddress < outAddress + byteCount and outAddress < referenceAddress + byteCount
        ):
            raise InputError(kind="overlap")
    return count


def logValues(values, reference, double pseudocount, out):
    cdef cnp.ndarray valuesArray
    cdef cnp.ndarray outArray
    cdef cnp.ndarray referenceArray
    cdef Py_ssize_t count = _checkLogValues(values, reference, pseudocount, out)
    cdef const cnp.float32_t *reference32 = NULL
    cdef const cnp.float64_t *reference64 = NULL
    valuesArray = values
    outArray = out
    if reference is not None:
        referenceArray = reference
        if values.dtype == np.float32: reference32 = <const cnp.float32_t *>referenceArray.data
        else: reference64 = <const cnp.float64_t *>referenceArray.data
    if values.dtype == np.float32:
        with nogil:
            _logLoop(<const cnp.float32_t *>valuesArray.data, reference32, <cnp.float32_t *>outArray.data, count, pseudocount)
    else:
        with nogil:
            _logLoop(<const cnp.float64_t *>valuesArray.data, reference64, <cnp.float64_t *>outArray.data, count, pseudocount)
    return out
