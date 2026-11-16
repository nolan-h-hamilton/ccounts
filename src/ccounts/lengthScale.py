from dataclasses import asdict, dataclass
import gzip
from importlib.resources import files
import json
import logging
from time import perf_counter

import numpy as np

from .checks import checkedInteger, checkedReal, checkCountValues
from .exceptions import CountError, DependencyError, InputError, SourceError
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
        Smallest nominal filter width in bp. None uses the resolved bin size.
    scaleEndBP : int
        Nominal upper filter width in bp. Overridden by `blockWidthBP` if smaller
    scaleFactor : float
        Defines the increment in filter width size. Multiplies the previous width.
    smoothingRatio : float
        Ratio of filter width to the smoothing window.
    significanceLevel : float
        Tail probability used for 'Poisson trimming' and the final binomial test.
    pValueWindowBP : int
        Reference width in bp for downscaling counts in longer candidates.
    backgroundWindowBP : int
        Window width in bp for computing the mean null or local 'background' count
    bootstrapReplicates : int
        Number of genomic block bootstrap draws. At least two.
    confidenceLevel : float
        Nominal confidence level for percentile bootstrap intervals.
    randomSeed : int
        Seed for block sampling and bootstrap draws.
    returnProfiles : bool
        Include the largest accepted filter width for each sample, block,
        and bin, with zero for unsupported bins.
    """

    binSizeBP: int | None = None
    blockCount: int = 128
    blockWidthBP: int = 500000
    scaleStartBP: int | None = None
    scaleEndBP: int = 125_000
    scaleFactor: float = 1.5
    smoothingRatio: float = 4.0
    significanceLevel: float = 0.05
    pValueWindowBP: int = 1500
    backgroundWindowBP: int | None = None
    bootstrapReplicates: int = 1000
    confidenceLevel: float = 0.95
    randomSeed: int = 42
    returnProfiles: bool = False
    filterType: str = "median"

    def __post_init__(self):
        # check fields after construction
        for name in (
            "blockCount",
            "blockWidthBP",
            "scaleEndBP",
            "pValueWindowBP",
        ):
            object.__setattr__(self, name, checkedInteger(getattr(self, name), name, 1))
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
            "bootstrapReplicates",
            checkedInteger(self.bootstrapReplicates, "bootstrapReplicates", 2),
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
        grouped = {}
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
            grouped.setdefault(chromosome, []).append((start, end))
        for intervals in grouped.values():
            intervals.sort()
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise InputError(kind="region")
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
    sentinel = np.iinfo(np.int64).max
    positions = np.arange(values.shape[2])
    for block in range(values.shape[0]):
        for sample in range(values.shape[1]):
            signal = np.asarray(values[block, sample], dtype=np.int64)
            eligible = donors[block, sample]
            if radius == len(signal) - 1:
                selected = signal[eligible]
                if len(selected):
                    median = np.partition(selected, len(selected) // 2)[
                        len(selected) // 2
                    ]
                    result[block, sample] = np.maximum(signal, median)
                continue
            padded = np.pad(
                np.where(eligible, signal, sentinel),
                radius,
                constant_values=sentinel,
            )
            windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
            donorCounts = np.count_nonzero(windows != sentinel, axis=1)
            ordered = np.sort(windows, axis=1)
            medians = ordered[positions, donorCounts // 2]
            supported = donorCounts > 0
            result[block, sample, supported] = np.maximum(
                signal[supported], medians[supported]
            )
    return result


def _histogramQuantiles(histograms, widths, probabilities):
    r"""Get linear quantiles from counts only, rather than storing full arrays
    of repeated values"""
    histograms = np.atleast_2d(histograms)
    cumulative = np.cumsum(histograms[:, 1:], axis=1)
    totals = cumulative[:, -1]
    defined = totals > 0
    values = np.zeros((len(histograms), len(probabilities)))

    for column, probability in enumerate(probabilities):
        # compute positions in the hist/cdf
        positions = probability * (totals[defined] - 1)
        lower = np.floor(positions).astype(np.int64)
        upper = np.ceil(positions).astype(np.int64)
        fraction = positions - lower
        left = widths[np.sum(cumulative[defined] <= lower[:, None], axis=1)].astype(
            float
        )
        right = widths[np.sum(cumulative[defined] <= upper[:, None], axis=1)].astype(
            float
        )

        # linear interpolation between left and right
        difference = right - left
        values[defined, column] = np.where(
            fraction < 0.5,
            left + difference * fraction,
            right - difference * (1 - fraction),
        )

    return values, defined


def _intervalFields(point, draws, blockCount, options, widths):
    retained = np.asarray(
        [value for value in draws if value is not None], dtype=np.float64
    )

    result = dict(
        CILowerBP=None,
        CIUpperBP=None,
        conditionalCILowerBP=None,
        conditionalCIUpperBP=None,
        definedBootstrapFraction=len(retained) / options.bootstrapReplicates,
        bootstrapScalesBP=draws,
        bootstrapLowerGridCount=int(np.count_nonzero(retained == widths[0])),
        bootstrapUpperGridCount=int(np.count_nonzero(retained == widths[-1])),
    )

    # fractions of extreme values
    result["bootstrapLowerGridFraction"] = (
        result["bootstrapLowerGridCount"] / options.bootstrapReplicates
    )
    result["bootstrapUpperGridFraction"] = (
        result["bootstrapUpperGridCount"] / options.bootstrapReplicates
    )

    if point is None:
        result["CIStatus"] = "noSupport"
    elif blockCount < 2:
        result["CIStatus"] = "insufficientBlocks"
    elif len(retained) < options.bootstrapReplicates:
        result["CIStatus"] = "incompleteResamples"
        if len(retained):
            result["conditionalCILowerBP"], result["conditionalCIUpperBP"] = (
                np.quantile(
                    retained,
                    (
                        (1 - options.confidenceLevel) / 2,
                        (1 + options.confidenceLevel) / 2,
                    ),
                ).tolist()
            )
    else:
        result["CIStatus"] = "ok"
        result["CILowerBP"], result["CIUpperBP"] = np.quantile(
            retained,
            ((1 - options.confidenceLevel) / 2, (1 + options.confidenceLevel) / 2),
        ).tolist()

    return result


def _blockSupport(signal, windows, binSizeBP, options, functions, smoothingSignal=None):
    medianProfile, nativeLogTail, findPeaks, poisson, binomial = functions
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

    def logTails(countsO, expected):
        result = np.empty(len(countsO))
        high = countsO >= expected
        result[high] = nativeLogTail(
            np.ascontiguousarray(countsO[high]), np.ascontiguousarray(expected[high])
        )
        result[~high] = binomial.logsf(
            countsO[~high] - 1, countsO[~high] + expected[~high], 0.5
        )
        return result

    for scale, width in enumerate(windows):
        if options.filterType == "median":
            smooth = medianProfile(indices, countValues, int(width))
        else:
            windowStarts = np.maximum(0, positions - width // 2)
            windowEnds = np.minimum(nBins, positions + width // 2 + 1)
            smooth = (smoothingPrefix[windowEnds] - smoothingPrefix[windowStarts]) / (
                windowEnds - windowStarts
            )
        minima, _ = findPeaks(-smooth, plateau_size=True)
        boundaries = np.unique(np.r_[0, minima, nBins - 1])
        covered = np.zeros(nBins + 1, dtype=np.int64)
        for left, last in zip(boundaries[:-1], boundaries[1:], strict=True):
            left, right = int(left), int(last) + 1
            rawMaximum = filterSignal[
                max(0, left - width // 2) : min(nBins, right + width // 2)
            ].max()
            smoothMaximum = smooth[left:right].max()
            if (
                smoothMaximum <= 0
                or rawMaximum > options.smoothingRatio * smoothMaximum
            ):
                continue
            key = (left, right)
            if key not in cache:
                above = np.flatnonzero(signal[left:right] > thresholds[left:right])
                if not len(above):
                    cache[key] = None
                else:
                    regionLeft, regionRight = (
                        left + int(above[0]),
                        left + int(above[-1]) + 1,
                    )
                    starts = np.arange(regionLeft, regionRight)
                    countsO = countsOPrefix[regionRight] - countsOPrefix[starts]
                    expected = (
                        backgroundPrefix[regionRight] - backgroundPrefix[starts]
                    ).astype(np.int64)
                    regionLeft = int(starts[np.argmin(logTails(countsO, expected))])

                    ends = np.arange(regionLeft + 1, regionRight + 1)
                    countsO = countsOPrefix[ends] - countsOPrefix[regionLeft]
                    expected = (
                        backgroundPrefix[ends] - backgroundPrefix[regionLeft]
                    ).astype(np.int64)
                    regionRight = int(ends[np.argmin(logTails(countsO, expected))])

                    factor = min(
                        1.0,
                        options.pValueWindowBP
                        / ((regionRight - regionLeft) * binSizeBP),
                    )
                    countsO = int(
                        (countsOPrefix[regionRight] - countsOPrefix[regionLeft])
                        * factor
                    )
                    expected = int(
                        (backgroundPrefix[regionRight] - backgroundPrefix[regionLeft])
                        * factor
                    )
                    if factor < 1:
                        expected = max(1, expected)
                    probability = (
                        1.0
                        if countsO == 0
                        else float(binomial.sf(countsO - 1, countsO + expected, 0.5))
                    )
                    cache[key] = (regionLeft, regionRight, probability)

            region = cache[key]
            if region is None or region[2] > options.significanceLevel:
                continue
            covered[region[0]] += 1
            covered[region[1]] -= 1
            intervalCounts[scale] += 1

        supportedBins = np.cumsum(covered[:-1]) > 0
        coverageCounts[scale] = np.count_nonzero(supportedBins)
        largest[supportedBins] = scale + 1

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
):
    started = perf_counter()
    try:
        import scipy

        # FWHM-based peak heuristic
        from scipy.signal import find_peaks
        from scipy.stats import binom, poisson
    except ImportError as error:
        raise DependencyError() from error
    from . import _native, __version__

    blockCount, binSizeBP, sampleIDs, regions = _lengthScaleRequest(
        blockCount, binSizeBP, sampleIDs, options, regions
    )
    nSamples = len(sampleIDs)
    nBins = options.blockWidthBP // binSizeBP
    if smoothingCounts is not None:
        smoothingCounts = np.asarray(smoothingCounts)
        if smoothingCounts.shape != (blockCount, nSamples, nBins):
            raise InputError(kind="array")

    scaleStart = max(
        3 * binSizeBP,
        0 if options.scaleStartBP is None else options.scaleStartBP,
    )
    maximumWidth = min(125_000, options.scaleEndBP, options.blockWidthBP)
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
        _native.musicBinomialLogTail,
        find_peaks,
        poisson,
        binom,
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
            # Store width counts so bootstrap draws don't need full bin arrays.
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

    generator = np.random.default_rng(options.randomSeed)
    # Use the same block draws for every sample.
    draws = generator.integers(
        0, blockCount, size=(options.bootstrapReplicates, blockCount)
    )
    multiplicities = np.zeros((options.bootstrapReplicates, blockCount), dtype=np.int64)
    np.add.at(
        multiplicities,
        (np.repeat(np.arange(options.bootstrapReplicates), blockCount), draws.ravel()),
        1,
    )

    summed = histograms.sum(axis=0)
    quantiles, defined = _histogramQuantiles(summed, widths, (0.5, 0.75, 0.9))
    sampleResults = []
    sampleDraws = []
    for sample, sampleID in enumerate(sampleIDs):
        histogram = summed[sample]
        supported = int(histogram[1:].sum())
        repeated, repeatedDefined = _histogramQuantiles(
            multiplicities @ histograms[:, sample], widths, (0.9,)
        )
        bootstrap = [
            float(value) if valid else None
            for value, valid in zip(repeated[:, 0], repeatedDefined, strict=True)
        ]
        sampleDraws.append(bootstrap)
        point = float(quantiles[sample, 2]) if defined[sample] else None

        result = dict(
            sampleID=sampleID,
            scaleBP=point,
            q50BP=float(quantiles[sample, 0]) if defined[sample] else None,
            q75BP=float(quantiles[sample, 1]) if defined[sample] else None,
            q90BP=point,
            status="estimated" if defined[sample] else "noSignificantSSERs",
            totalEvents=totals[sample],
            analyzedBP=blockCount * options.blockWidthBP,
            coveredBP=supported * binSizeBP,
            coveredFraction=supported / (blockCount * nBins),
            supportedBinCount=supported,
            supportedBlockCount=int(
                np.count_nonzero(histograms[:, sample, 1:].sum(axis=1))
            ),
            histogramCounts=histogram.tolist(),
            blockHistogramCounts=histograms[:, sample].tolist(),
            SSERCountByScale=intervals[sample].tolist(),
            SSERCoverageBPByScale=(coverage[sample] * binSizeBP).tolist(),
            lowerGridLimited=bool(point == widths[0]),
            upperGridLimited=bool(point == widths[-1]),
            upperGridFraction=float(histogram[-1] / supported) if supported else 0.0,
        )
        result.update(_intervalFields(point, bootstrap, blockCount, options, widths))
        if options.returnProfiles:
            result["largestSupportBP"] = profiles[sample]
        sampleResults.append(result)

    # keep the same set of samples in every bootstrap median
    selected = np.flatnonzero(defined)
    point = float(np.median(quantiles[selected, 2])) if len(selected) else None
    aggregateDraws = [None] * options.bootstrapReplicates
    if len(selected):
        for repetition in range(options.bootstrapReplicates):
            values = [sampleDraws[sample][repetition] for sample in selected]
            if all(value is not None for value in values):
                aggregateDraws[repetition] = float(np.median(values))

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
    result.update(_intervalFields(point, aggregateDraws, blockCount, options, widths))
    return result


def estimateLengthScaleCounts(
    counts,
    binSizeBP,
    *,
    sampleIDs=None,
    options=None,
    regions=None,
    smoothingCounts=None,
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
    )
