from dataclasses import asdict, dataclass
import gzip
from importlib.resources import files
import json
import logging
from time import perf_counter

import numpy as np

from .checks import checkedInteger, checkedReal, checkCountValues
from .exceptions import (
    CountError,
    DependencyError,
    InputError,
    NumericalError,
    SourceError,
)
from .genomes import assemblyBlacklistRanges, assemblyChromosomeSizes


@dataclass(frozen=True)
class LengthScaleOptions:
    r"""Settings for MUSIC-based length scale estimation.

    Attributes
    ----------
    binSizeBP : int or None
        Number of base pairs comprising each genomic bin
    blockCount : int
        Maximum number of genomic blocks sampled by the Reader. Ignored if specific regions are given.
    blockWidthBP : int
        Genomic block width in bp
    scaleStartBP : int or None
        Smallest nominal filter width in bp, with a minimum of three bins.
        None uses three bins.
    scaleEndBP : int or None
        Nominal upper filter width in bp, capped at `blockWidthBP`.
        None uses `blockWidthBP`.
    scaleFactor : float
        Defines the increment in filter width size. Multiplies the previous width.
    smoothingRatio : float
        Reject a candidate if the unsmoothed peak in its surrounding filter
        window is more than this many times the smoothed peak inside the candidate.
    significanceLevel : float
        Tail probability used for 'Poisson trimming' and the final binomial test.
    pValueWindowBP : int
        Reference width in bp for downscaling counts in longer candidates.
    backgroundWindowBP : int
        Window width in bp for computing the mean null or local 'background' count
    confidenceLevel : float
        Nominal confidence level for IJ normal intervals.
    randomSeed : int
        Seed for genomic block sampling.
    returnProfiles : bool
        Include the largest accepted filter width for each sample, block,
        and bin, with zero for unsupported bins.
    """

    binSizeBP: int | None = None
    blockCount: int = 1024
    blockWidthBP: int = 500000

    scaleStartBP: int | None = None
    scaleEndBP: int | None = None
    scaleFactor: float = 1.5

    smoothingRatio: float = 4.0
    significanceLevel: float = 0.05
    pValueWindowBP: int = 1500
    backgroundWindowBP: int | None = None

    confidenceLevel: float = 0.95
    randomSeed: int = 42
    returnProfiles: bool = False
    filterType: str = "median"

    def __post_init__(self):
        # check fields after construction
        for name in (
            "blockCount",
            "blockWidthBP",
            "pValueWindowBP",
        ):
            object.__setattr__(self, name, checkedInteger(getattr(self, name), name, 1))

        object.__setattr__(
            self,
            "scaleEndBP",
            self.blockWidthBP
            if self.scaleEndBP is None
            else checkedInteger(self.scaleEndBP, "scaleEndBP", 1),
        )

        object.__setattr__(
            self,
            "backgroundWindowBP",
            (
                min(5_000_000, max(50_000, self.blockWidthBP // 10))
                if self.backgroundWindowBP is None
                else checkedInteger(self.backgroundWindowBP, "backgroundWindowBP", 1)
            ),
        )

        for name in ("binSizeBP", "scaleStartBP"):
            if getattr(self, name) is not None:
                object.__setattr__(
                    self, name, checkedInteger(getattr(self, name), name, 1)
                )

        object.__setattr__(
            self,
            "randomSeed",
            checkedInteger(self.randomSeed, "randomSeed", 0, 2**64 - 1),
        )

        for name in (
            "scaleFactor",
            "smoothingRatio",
            "significanceLevel",
            "confidenceLevel",
        ):
            object.__setattr__(self, name, checkedReal(getattr(self, name), name))

        if self.scaleFactor <= 1 or self.smoothingRatio < 1:
            raise InputError()
        if not 0 < self.significanceLevel < 1 or not 0 < self.confidenceLevel < 1:
            raise InputError()
        if self.scaleStartBP is not None and self.scaleStartBP > self.scaleEndBP:
            raise InputError()
        if self.binSizeBP is not None and (
            self.blockWidthBP % self.binSizeBP
            or self.backgroundWindowBP < self.binSizeBP
        ):
            raise InputError(kind="grid")

        if type(self.returnProfiles) is not bool:
            raise InputError(kind="option", context="returnProfiles")
        if type(self.filterType) is not str or self.filterType not in (
            "median",
            "mean",
        ):
            raise InputError(kind="option", context="filterType")


def _lengthScaleExclusions(chromosomeSizes, excludeRanges):
    matches = [
        assembly
        for assembly, sizes in assemblyChromosomeSizes.items()
        if all(
            sizes.get(chromosome) == length
            for chromosome, length in chromosomeSizes.items()
        )
    ]
    assembly = matches[0] if len(matches) == 1 else None
    if assembly is None and excludeRanges is None:
        raise InputError(kind="grid", context="assembly")

    exclusions = {}
    if assembly is not None:
        blacklist = assemblyBlacklistRanges.get(assembly, {})
        exclusions = {
            chromosome: list(blacklist.get(chromosome, ()))
            for chromosome in chromosomeSizes
        }

        resource = files("ccounts") / "data" / "gaps" / f"{assembly}.bed.gz"
        with resource.open("rb") as raw, gzip.open(raw, "rt") as handle:
            for line in handle:
                chromosome, start, end = line.split()
                if chromosome in chromosomeSizes:
                    exclusions[chromosome].append((int(start), int(end)))

    if excludeRanges is not None:
        for chromosome, intervals in excludeRanges.items():
            if chromosome not in chromosomeSizes:
                raise SourceError(kind="missing", context=chromosome)

            checked = []
            for start, end in intervals:
                start = checkedInteger(start, "exclusion startBP")
                end = checkedInteger(
                    end, "exclusion endBP", 1, chromosomeSizes[chromosome]
                )
                if start >= end or (checked and start < checked[-1][1]):
                    raise InputError(kind="region")
                checked.append((start, end))

            exclusions.setdefault(chromosome, []).extend(checked)

    merged = {}
    for chromosome, intervals in exclusions.items():
        combined = []
        for start, end in sorted(intervals):
            if combined and start <= combined[-1][1]:
                combined[-1] = (combined[-1][0], max(combined[-1][1], end))
            else:
                combined.append((start, end))

        if combined:
            merged[chromosome] = combined

    return assembly, merged


def _lengthScaleRequest(blockCount, binSizeBP, sampleIDs, options, regions):
    blockCount = checkedInteger(blockCount, "blockCount", 1)
    binSizeBP = checkedInteger(binSizeBP, "binSizeBP", 1)
    if options.binSizeBP is not None and options.binSizeBP != binSizeBP:
        raise InputError(kind="grid")
    if options.blockWidthBP % binSizeBP or options.backgroundWindowBP < binSizeBP:
        raise InputError(kind="grid")

    sampleIDs = tuple(sampleIDs)
    if (
        not sampleIDs
        or any(not isinstance(value, str) or not value for value in sampleIDs)
        or len(set(sampleIDs)) != len(sampleIDs)
    ):
        raise InputError(kind="identifiers")

    if regions is not None:
        if len(regions) != blockCount:
            raise InputError(kind="region")

        checked = []
        for chromosome, start, end in regions:
            start = checkedInteger(start, "startBP")
            end = checkedInteger(end, "endBP", 1)
            if (
                not isinstance(chromosome, str)
                or not chromosome
                or start % binSizeBP
                or end - start != options.blockWidthBP
            ):
                raise InputError(kind="region")
            checked.append([chromosome, start, end])

        regions = checked

    return blockCount, binSizeBP, sampleIDs, regions


def _lengthScaleCountBlock(block, sampleCount, binCount):
    raw = np.asarray(block)
    if raw.shape != (sampleCount, binCount):
        raise InputError(kind="array")
    checkCountValues(raw)
    if np.any(raw.sum(axis=1, dtype=np.float64) > 2**52):
        raise CountError()

    return np.ascontiguousarray(raw, dtype=np.int64)


def correctMappabilityCounts(counts, donorBins, binSizeBP, *, windowBP=2000):
    r"""Raise low counts using nearby bins with reliable read mapping.

    Impute from nearby donor bins to account for mappability artifacts / sparsity.
    Adapted directly from MUSIC.

    References
    ----------
    Harmanci, Rozowsky and Gerstein (2014), MUSIC.
    https://doi.org/10.1186/s13059-014-0474-3
    """
    from . import _native

    values = np.asarray(counts)
    donors = np.asarray(donorBins)
    if (
        values.ndim != 3
        or min(values.shape) < 1
        or donors.shape != values.shape
        or donors.dtype != np.dtype(bool)
    ):
        raise InputError(kind="array")

    binSizeBP = checkedInteger(binSizeBP, "binSizeBP", 1)
    windowBP = checkedInteger(windowBP, "windowBP", 1)
    checkCountValues(values)
    if np.any(values.sum(axis=2, dtype=np.float64) > 2**52):
        raise CountError()

    result = values.copy() if values.dtype.kind in "iu" else values.astype(np.int64)
    radius = min(windowBP // (2 * binSizeBP), values.shape[2] - 1)
    if radius == 0:
        return result

    for block in range(values.shape[0]):
        for sample in range(values.shape[1]):
            signal = np.ascontiguousarray(values[block, sample], dtype=np.int64)
            eligible = donors[block, sample]

            if radius == len(signal) - 1:
                # every bin uses the same donors, so find their middle count once
                selected = signal[eligible]
                if len(selected):
                    median = np.partition(selected, len(selected) // 2)[
                        len(selected) // 2
                    ]
                    result[block, sample] = np.maximum(signal, median)
                continue

            result[block, sample] = _native.musicCorrectMappabilityCounts(
                signal, np.ascontiguousarray(eligible, dtype=np.uint8), radius
            )

    return result


def _summaryValue(values, weights, summaryFunction):
    # A custom summary sees probabilities over the supplied values
    result = summaryFunction(values.copy(), weights=weights / weights.sum())
    if np.ndim(result) != 0:
        raise InputError(kind="scalar", context="summaryFunction")
    return checkedReal(result, "summaryFunction")


def _histogramIJ(histograms, widths, summaryFunction):
    blockCount, sampleCount = histograms.shape[:2]
    supported = histograms[:, :, 1:]
    blockBins = supported.sum(axis=2)
    supportedBlocks = np.count_nonzero(blockBins, axis=0)

    pooled = supported.sum(axis=0)
    totals = pooled.sum(axis=1)
    defined = supportedBlocks > 0
    values = np.zeros(sampleCount)
    influences = np.zeros((blockCount, sampleCount))

    if summaryFunction is np.mean:
        widths = widths.astype(float)
        values[defined] = (pooled[defined] @ widths) / totals[defined]
        blockSums = supported @ widths

        # A block changes both the width sum and the number of supported bins
        influences[:, defined] = (
            blockSums[:, defined] - blockBins[:, defined] * values[defined]
        ) / (totals[defined] / blockCount)
    else:
        step = np.cbrt(np.finfo(float).eps)
        for sample in np.flatnonzero(defined):
            weights = pooled[sample].astype(float)
            values[sample] = _summaryValue(widths, weights, summaryFunction)
            if supportedBlocks[sample] < 2:
                continue

            for block in range(blockCount):
                direction = supported[block, sample] - weights / blockCount
                upper = _summaryValue(
                    widths, weights + step * direction, summaryFunction
                )
                lower = _summaryValue(
                    widths, weights - step * direction, summaryFunction
                )
                influences[block, sample] = (upper - lower) / (2 * step)

            # Center the derivatives because block weights have a fixed total.
            influences[:, sample] -= influences[:, sample].mean()
            influences[:, sample] *= blockCount

        if not np.isfinite(influences).all():
            raise NumericalError()

    return values, influences, supportedBlocks


def _intervalFields(point, influences, supportedBlocks, criticalValue):
    result = dict(
        SEBP=None,
        CILowerBP=None,
        CIUpperBP=None,
        blockInfluenceBP=None,
    )

    if point is None:
        result["CIStatus"] = "noSupport"
    elif supportedBlocks < 2:
        result["CIStatus"] = "insufficientBlocks"
    else:
        standardError = float(np.linalg.norm(influences / len(influences)))
        lower, upper = (
            point - criticalValue * standardError,
            point + criticalValue * standardError,
        )
        if not all(np.isfinite(value) for value in (standardError, lower, upper)):
            raise NumericalError()

        result["CIStatus"] = "ok"
        result["SEBP"] = standardError
        result["blockInfluenceBP"] = influences.tolist()
        result["CILowerBP"], result["CIUpperBP"] = lower, upper

    return result


def _blockSupport(signal, windows, binSizeBP, options, functions, smoothingSignal=None):
    medianProfile, candidateRegions, findPeaks, poisson = functions
    nBins = len(signal)
    largest = np.zeros(nBins, dtype=np.intp)
    intervalCounts = np.zeros(len(windows), dtype=np.int64)
    coverageCounts = np.zeros(len(windows), dtype=np.int64)
    if signal.sum() == 0:
        return largest, intervalCounts, coverageCounts

    filterSignal = signal if smoothingSignal is None else smoothingSignal
    if options.filterType == "median":
        countValues, indices = np.unique(filterSignal, return_inverse=True)
        countValues = countValues.astype(float)
        indices = np.ascontiguousarray(indices, dtype=np.intp)
    else:
        positions = np.arange(nBins)

    means = np.empty(nBins)
    backgroundBins = max(1, round(options.backgroundWindowBP / binSizeBP))
    for start in range(0, nBins, backgroundBins):
        stop = min(nBins, start + backgroundBins)
        means[start:stop] = signal[start:stop].mean()

    thresholds = poisson.ppf(1 - options.significanceLevel, means)
    countsOPrefix = np.r_[np.int64(0), np.cumsum(signal)]
    if options.filterType == "mean":
        smoothingPrefix = (
            countsOPrefix
            if smoothingSignal is None
            else np.r_[np.int64(0), np.cumsum(smoothingSignal)]
        )
    backgroundPrefix = np.r_[0.0, np.cumsum(means)]
    cache = {}

    for scale, width in enumerate(windows):
        if options.filterType == "median":
            smooth = medianProfile(indices, countValues, int(width))
        else:
            windowStarts = np.maximum(0, positions - width // 2)
            windowEnds = np.minimum(nBins, positions + width // 2 + 1)
            smooth = (smoothingPrefix[windowEnds] - smoothingPrefix[windowStarts]) / (
                windowEnds - windowStarts
            )

        # peaks in -smooth are troughs that bound the candidate regions
        minima, _ = findPeaks(-smooth, plateau_size=True)
        boundaries = np.unique(np.r_[0, minima, nBins - 1]).astype(np.intp, copy=False)
        intervalCounts[scale], coverageCounts[scale] = candidateRegions(
            signal,
            filterSignal,
            smooth,
            boundaries,
            thresholds,
            countsOPrefix,
            backgroundPrefix,
            int(width),
            binSizeBP,
            options.smoothingRatio,
            options.pValueWindowBP,
            options.significanceLevel,
            cache,
            largest,
            scale + 1,
        )

    return largest, intervalCounts, coverageCounts


def _estimateLengthScale(
    blocks,
    blockCount,
    binSizeBP,
    sampleIDs,
    options,
    regions=None,
    sourceInfo=None,
    sampling=None,
    smoothingCounts=None,
    summaryFunction=np.mean,
):
    started = perf_counter()
    if not callable(summaryFunction):
        raise InputError(kind="option", context="summaryFunction")

    try:
        import scipy

        # FWHM-based peak heuristic
        from scipy.signal import find_peaks
        from scipy.stats import norm, poisson
    except ImportError as error:
        raise DependencyError() from error
    from . import _native, __version__

    blockCount, binSizeBP, sampleIDs, regions = _lengthScaleRequest(
        blockCount, binSizeBP, sampleIDs, options, regions
    )
    nSamples = len(sampleIDs)
    nBins = options.blockWidthBP // binSizeBP
    criticalValue = float(norm.isf((1 - options.confidenceLevel) / 2))
    if smoothingCounts is not None:
        smoothingCounts = np.asarray(smoothingCounts)
        if smoothingCounts.shape != (blockCount, nSamples, nBins):
            raise InputError(kind="array")

    scaleStart = max(
        3 * binSizeBP,
        0 if options.scaleStartBP is None else options.scaleStartBP,
    )
    maximumWidth = min(options.scaleEndBP, options.blockWidthBP)
    if scaleStart > maximumWidth:
        raise InputError()

    nominalCount = (
        int(np.floor(np.log(maximumWidth / scaleStart) / np.log(options.scaleFactor)))
        + 1
    )
    if nominalCount > 10000:
        raise InputError(kind="grid")

    nominal = scaleStart * options.scaleFactor ** np.arange(nominalCount)
    # Round to odd bin counts for the filter
    windows = np.unique(np.rint(nominal / binSizeBP).astype(np.intp) | 1)
    widths = windows * binSizeBP
    retained = (widths >= scaleStart) & (widths <= maximumWidth)
    windows, widths = windows[retained], widths[retained]
    if not len(windows):
        raise InputError(kind="grid")

    settings = dict(
        asdict(options),
        summaryFunction="mean" if summaryFunction is np.mean else "custom",
        CIMethod="IJNormal",
        binSizeBP=binSizeBP,
        scaleStartBP=scaleStart,
        sampledBlockCount=blockCount,
        sampleIDs=list(sampleIDs),
        regions=regions,
        filterWidthsBP=widths.tolist(),
        sourceInfo=sourceInfo,
        softwareVersions=dict(
            ccounts=__version__, numpy=np.__version__, scipy=scipy.__version__
        ),
    )

    if sampling is not None:
        settings["sampling"] = sampling
    if smoothingCounts is not None:
        settings["smoothingCountsProvided"] = True

    logging.getLogger("ccounts.lengthScale").info(
        "Length-scale configuration %s",
        json.dumps(settings, allow_nan=False, sort_keys=True),
    )

    histograms = np.zeros((blockCount, nSamples, len(widths) + 1), dtype=np.int64)
    intervals = np.zeros((nSamples, len(widths)), dtype=np.int64)
    coverage = np.zeros_like(intervals)
    totals = [0] * nSamples
    profiles = [[] for _ in sampleIDs]
    processed = 0

    functions = (
        _native.musicMedianProfile,
        _native.musicCandidateRegions,
        find_peaks,
        poisson,
    )
    detectorSeconds = 0.0

    for blockIndex, block in enumerate(blocks):
        if blockIndex >= blockCount:
            raise InputError(kind="array")

        values = _lengthScaleCountBlock(block, nSamples, nBins)
        smoothingValues = (
            None
            if smoothingCounts is None
            else _lengthScaleCountBlock(smoothingCounts[blockIndex], nSamples, nBins)
        )

        stamp = perf_counter()
        for sample, signal in enumerate(values):
            largest, intervalCounts, coverageCounts = _blockSupport(
                signal,
                windows,
                binSizeBP,
                options,
                functions,
                None if smoothingValues is None else smoothingValues[sample],
            )

            # Keep the distribution of accepted widths for each genomic block.
            histograms[blockIndex, sample] = np.bincount(
                largest, minlength=len(widths) + 1
            )
            intervals[sample] += intervalCounts
            coverage[sample] += coverageCounts
            totals[sample] += int(signal.sum())
            if options.returnProfiles:
                profiles[sample].append(np.r_[0, widths][largest].tolist())

        detectorSeconds += perf_counter() - stamp
        processed += 1

    if processed != blockCount:
        raise InputError(kind="array")

    summed = histograms.sum(axis=0)
    summaries, influences, supportedBlocks = _histogramIJ(
        histograms, widths, summaryFunction
    )
    defined = supportedBlocks > 0

    sampleResults = []
    for sample, sampleID in enumerate(sampleIDs):
        histogram = summed[sample]
        supported = int(histogram[1:].sum())
        point = float(summaries[sample]) if defined[sample] else None

        result = dict(
            sampleID=sampleID,
            scaleBP=point,
            status="estimated" if defined[sample] else "noSignificantSSERs",
            totalEvents=totals[sample],
            analyzedBP=blockCount * options.blockWidthBP,
            coveredBP=supported * binSizeBP,
            coveredFraction=supported / (blockCount * nBins),
            supportedBinCount=supported,
            supportedBlockCount=int(supportedBlocks[sample]),
            histogramCounts=histogram.tolist(),
            blockHistogramCounts=histograms[:, sample].tolist(),
            SSERCountByScale=intervals[sample].tolist(),
            SSERCoverageBPByScale=(coverage[sample] * binSizeBP).tolist(),
            lowerGridLimited=bool(point == widths[0]),
            upperGridLimited=bool(point == widths[-1]),
            upperGridFraction=float(histogram[-1] / supported) if supported else 0.0,
        )
        result.update(
            _intervalFields(
                point, influences[:, sample], supportedBlocks[sample], criticalValue
            )
        )

        if options.returnProfiles:
            result["largestSupportBP"] = profiles[sample]
        sampleResults.append(result)

    selected = np.flatnonzero(defined)
    point = None
    aggregateInfluence = np.zeros(blockCount)
    aggregateSupport = int(supportedBlocks[selected].min()) if len(selected) else 0

    if len(selected) == 1:
        sample = int(selected[0])
        point = float(summaries[sample])
        aggregateInfluence = influences[:, sample]
    elif len(selected) > 1:
        values = summaries[selected]
        if summaryFunction is np.mean:
            point = float(values.mean())
            # Shared genomic blocks also retain covariance between samples
            aggregateInfluence = influences[:, selected].mean(axis=1)
        else:
            weights = np.ones(len(selected))
            point = _summaryValue(values, weights, summaryFunction)

            if aggregateSupport >= 2:
                gradient = np.empty(len(selected))
                for sample in range(len(selected)):
                    step = np.cbrt(np.finfo(float).eps) * max(abs(values[sample]), 1)
                    upper = values.copy()
                    lower = values.copy()
                    upper[sample] += step
                    lower[sample] -= step
                    gradient[sample] = (
                        _summaryValue(upper, weights, summaryFunction)
                        - _summaryValue(lower, weights, summaryFunction)
                    ) / (2 * step)

                aggregateInfluence = influences[:, selected] @ gradient
                aggregateInfluence -= aggregateInfluence.mean()

    result = dict(
        scaleBP=point,
        status="estimated" if point is not None else "noSignificantSSERs",
        sampleCount=nSamples,
        definedSampleCount=len(selected),
        pointDefinedSampleIDs=[sampleIDs[index] for index in selected],
        sampleResults=sampleResults,
        settings=settings,
        lowerGridLimited=bool(point == widths[0]),
        upperGridLimited=bool(point == widths[-1]),
        detectorSeconds=detectorSeconds,
        runtimeSeconds=perf_counter() - started,
    )
    result.update(
        _intervalFields(point, aggregateInfluence, aggregateSupport, criticalValue)
    )

    return result


def estimateLengthScaleCounts(
    counts,
    binSizeBP,
    *,
    sampleIDs=None,
    options=None,
    regions=None,
    smoothingCounts=None,
    summaryFunction=np.mean,
):
    r"""Estimate length scale from block counts"""

    values = np.asarray(counts)
    if values.ndim != 3 or min(values.shape) < 1:
        raise InputError(kind="array")

    options = LengthScaleOptions() if options is None else options
    sampleIDs = (
        tuple(str(index) for index in range(values.shape[1]))
        if sampleIDs is None
        else tuple(sampleIDs)
    )

    return _estimateLengthScale(
        iter(values),
        values.shape[0],
        binSizeBP,
        sampleIDs,
        options,
        regions=regions,
        smoothingCounts=smoothingCounts,
        summaryFunction=summaryFunction,
    )
