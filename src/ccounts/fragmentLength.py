from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .checks import checkedInteger
from .exceptions import EstimateError, EstimateWarning, InputError, NumericalError


@dataclass(frozen=True)
class FragmentLengthOptions:
    blockCount: int = 512
    blockWidthBP: int = 1000000
    minLengthBP: int | None = None
    maxLengthBP: int = 1000
    candidateStepBP: int = 1
    bootstrapReplicates: int = 1000
    randomSeed: int = 42
    smoothingWidthBP: int = 11

    def __post_init__(self):
        for field in (
            "blockCount",
            "blockWidthBP",
            "maxLengthBP",
            "candidateStepBP",
            "smoothingWidthBP",
        ):
            object.__setattr__(
                self, field, checkedInteger(getattr(self, field), field, 1)
            )
        if self.smoothingWidthBP % 2 == 0:
            raise InputError(kind="option", context="smoothingWidthBP")
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
        if self.minLengthBP is not None:
            object.__setattr__(
                self, "minLengthBP", checkedInteger(self.minLengthBP, "minLengthBP", 1)
            )
        if (
            self.minLengthBP is not None and self.maxLengthBP < self.minLengthBP
        ) or self.maxLengthBP >= self.blockWidthBP:
            raise InputError()


def _fragmentPoolInputs(curves, readCounts, candidateLengthsBP):
    curves = np.asarray(curves, dtype=np.float64)
    readCounts = np.asarray(readCounts, dtype=np.float64)
    candidates = np.asarray(candidateLengthsBP, dtype=np.int64)
    if curves.ndim != 2 or min(curves.shape) < 1:
        raise InputError(kind="array")
    if readCounts.shape != (curves.shape[0],) or candidates.shape != (curves.shape[1],):
        raise InputError(kind="array")
    if (
        not np.all(np.isfinite(readCounts))
        or np.any(readCounts < 0)
        or np.any(np.isinf(curves))
    ):
        raise InputError(kind="values")

    defined = np.isfinite(curves)
    if np.any(defined[readCounts == 0]):
        raise InputError(kind="values")
    return curves, readCounts, candidates, defined


def _smoothCurves(curves, candidates, widthBP):
    if widthBP == 1:
        return curves

    defined = np.isfinite(curves)
    padding = [(0, 0)] * (curves.ndim - 1) + [(1, 0)]
    sums = np.pad(np.cumsum(np.where(defined, curves, 0.0), axis=-1), padding)
    counts = np.pad(np.cumsum(defined, axis=-1), padding)
    left = np.searchsorted(candidates, candidates - widthBP // 2, side="left")
    right = np.searchsorted(candidates, candidates + widthBP // 2, side="right")
    denominator = counts[..., right] - counts[..., left]
    return np.divide(
        sums[..., right] - sums[..., left],
        denominator,
        out=np.full_like(curves, np.nan),
        where=denominator > 0,
    )


def _poolCurves(
    curves,
    readCounts,
    candidateLengthsBP,
    repetitions=1000,
    randomSeed=42,
    smoothingWidthBP=11,
):
    curves, readCounts, candidates, defined = _fragmentPoolInputs(
        curves, readCounts, candidateLengthsBP
    )

    weightedCurves = np.where(defined, curves, 0.0) * readCounts[:, None]
    availableWeights = defined * readCounts[:, None]
    candidateWeightTotals = availableWeights.sum(axis=0)
    candidateBlockCounts = np.count_nonzero(availableWeights, axis=0)
    valid = candidateWeightTotals > 0
    if not np.any(valid):
        raise EstimateError()

    rawPooledCurve = np.divide(
        weightedCurves.sum(axis=0),
        candidateWeightTotals,
        out=np.full(candidates.size, np.nan),
        where=valid,
    )
    pooledCurve = _smoothCurves(rawPooledCurve, candidates, smoothingWidthBP)
    pointIndex = int(np.nanargmax(pooledCurve))
    window = (
        np.abs(candidates - candidates[pointIndex]) <= smoothingWidthBP // 2
    ) & valid
    normalizedWeights = (
        availableWeights[:, window] / candidateWeightTotals[window]
    ).mean(axis=1)
    valid = np.isfinite(pooledCurve)

    indices = np.random.default_rng(randomSeed).integers(
        0, curves.shape[0], size=(repetitions, curves.shape[0])
    )
    multiplicities = np.zeros((repetitions, curves.shape[0]), dtype=np.int64)
    np.add.at(
        multiplicities,
        (np.repeat(np.arange(repetitions), curves.shape[0]), indices.ravel()),
        1,
    )
    repeatedNumerators = multiplicities @ weightedCurves
    repeatedDenominators = multiplicities @ availableWeights
    repeatedValid = repeatedDenominators > 0
    invalidReplicates = np.flatnonzero(~repeatedValid.any(axis=1))
    if invalidReplicates.size:
        raise EstimateError()

    repeatedCurves = np.divide(
        repeatedNumerators,
        repeatedDenominators,
        out=np.full_like(repeatedNumerators, np.nan),
        where=repeatedValid,
    )
    if not np.all(np.isfinite(repeatedCurves[repeatedValid])):
        raise NumericalError()

    repeatedCurves = _smoothCurves(repeatedCurves, candidates, smoothingWidthBP)
    repeatedValid = np.isfinite(repeatedCurves)
    repeatedIndices = np.nanargmax(repeatedCurves, axis=1)
    bootstrapEstimates = candidates[repeatedIndices]
    bounds = np.quantile(bootstrapEstimates, (0.025, 0.975))
    firstValid = np.argmax(repeatedValid, axis=1)
    lastValid = candidates.size - 1 - np.argmax(repeatedValid[:, ::-1], axis=1)
    repeatedMaxima = repeatedCurves[np.arange(repetitions), repeatedIndices]
    validPointIndices = np.flatnonzero(valid)

    return {
        "estimateBP": int(candidates[pointIndex]),
        "CILowerBP": float(bounds[0]),
        "CIUpperBP": float(bounds[1]),
        "rawPooledCurve": rawPooledCurve,
        "pooledCurve": pooledCurve,
        "bootstrapEstimatesBP": bootstrapEstimates,
        "normalizedWeights": normalizedWeights,
        "candidateWeightTotals": candidateWeightTotals,
        "candidateBlockCounts": candidateBlockCounts,
        "bootstrapBlockIndices": indices,
        "bootstrapBlockMultiplicities": multiplicities,
        "effectiveBlockCount": float(1 / np.dot(normalizedWeights, normalizedWeights)),
        "maximumWeight": float(normalizedWeights.max()),
        "peakCorrelation": float(pooledCurve[pointIndex]),
        "boundaryMaximum": bool(
            pointIndex in (validPointIndices[0], validPointIndices[-1])
        ),
        "flatMaximum": bool(np.ptp(pooledCurve[valid]) == 0),
        "nonpositiveMaximum": bool(pooledCurve[pointIndex] <= 0),
        "bootstrapBoundaryMaximumCount": int(
            np.count_nonzero(
                (repeatedIndices == firstValid) | (repeatedIndices == lastValid)
            )
        ),
        "bootstrapNonpositiveMaximumCount": int(np.count_nonzero(repeatedMaxima <= 0)),
        "bootstrapStdBP": float(bootstrapEstimates.std(ddof=1)),
    }


def _estimateSingleEnd(source, countOptions, options):
    from . import _native

    started = perf_counter()
    readLengthBP = source.readLength(countOptions)
    minimum = max(
        readLengthBP + 1,
        countOptions.minTemplateLengthBP,
        1 if options.minLengthBP is None else options.minLengthBP,
    )
    maximum = options.maxLengthBP
    if countOptions.maxTemplateLengthBP is not None:
        maximum = min(maximum, countOptions.maxTemplateLengthBP)
    if minimum > maximum:
        raise InputError()

    settings = asdict(options)
    settings.update(
        {
            "flagExclude": countOptions.flagExclude,
            "minMappingQuality": countOptions.minMappingQuality,
            "deduplicateEndpoints": True,
            "curveWeighting": "retainedEndpointCount",
            "samplingFrame": "completeDisjointBlocks",
            "intervalProbability": 0.95,
        }
    )

    blockWidth = settings["blockWidthBP"]
    candidates = np.arange(
        minimum, maximum + 1, options.candidateStepBP, dtype=np.int64
    )
    frame = []
    chromosomeFrame = {}
    extractionSeconds = 0.0
    localSeconds = 0.0
    counts = {}
    territoryBP = 0
    excludedShortChromosomeBP = 0
    excludedTerminalBP = 0
    for chromosomeID, (chromosome, length) in enumerate(
        ((name, source.chromosomeSizes[name]) for name in source.chromosomeNames)
    ):
        territoryBP += length
        count = length // blockWidth
        chromosomeFrame[chromosome] = {
            "lengthBP": int(length),
            "fullBlockCount": int(count),
            "sampledBlockCount": 0,
            "usableBlockCount": 0,
        }
        if count == 0:
            excludedShortChromosomeBP += length
        else:
            excludedTerminalBP += length % blockWidth
        for blockIndex in range(count):
            start = blockIndex * blockWidth
            frame.append(
                {
                    "blockID": len(frame),
                    "chromosomeID": chromosomeID,
                    "chromosome": chromosome,
                    "startBP": start,
                    "endBP": start + blockWidth,
                }
            )

    if not frame:
        raise EstimateError(kind="insufficient")

    selectedCount = min(settings["blockCount"], len(frame))
    drawnIDs = np.random.default_rng(options.randomSeed).choice(
        len(frame), size=selectedCount, replace=False
    )
    selectedIDs = np.sort(drawnIDs)
    pointCurves = np.full((selectedCount, candidates.size), np.nan)
    readCounts = np.zeros(selectedCount, dtype=np.int64)
    blockRows = []
    usableRows = []

    for rowIndex, frameIndex in enumerate(selectedIDs):
        block = dict(frame[int(frameIndex)])
        chromosomeFrame[block["chromosome"]]["sampledBlockCount"] += 1
        stamp = perf_counter()
        forward, reverse, blockCounts = source.blockEndpoints(
            block["chromosome"], block["startBP"], block["endBP"], countOptions
        )
        extractionSeconds += perf_counter() - stamp
        block.update(
            {
                "recordCounts": blockCounts,
                "usable": False,
                "failureReason": None,
                "normalizedWeight": 0.0,
            }
        )
        readCounts[rowIndex] = forward.size + reverse.size
        block["readCount"] = int(readCounts[rowIndex])
        for name, count in blockCounts.items():
            counts[name] = counts.get(name, 0) + count
        if not forward.size or not reverse.size:
            block["failureReason"] = "emptyStrand"
            blockRows.append(block)
            continue

        stamp = perf_counter()
        local = _native.estimateBlock(
            forward,
            reverse,
            blockWidth,
            minLengthBP=minimum,
            maxLengthBP=maximum,
            stepBP=options.candidateStepBP,
        )
        localSeconds += perf_counter() - stamp
        if local is None:
            block["failureReason"] = EstimateError.defaultMessage
            blockRows.append(block)
            continue

        pointCurves[rowIndex] = local["pointCurve"]
        block.update(
            {
                "usable": True,
                "estimateBP": int(local["estimateBP"]),
                "peakCorrelation": float(local["peakCorrelation"]),
                "boundaryMaximum": bool(local["boundaryMaximum"]),
                "flatMaximum": bool(local["flatMaximum"]),
                "nonpositiveMaximum": bool(local["nonpositiveMaximum"]),
            }
        )
        blockRows.append(block)
        usableRows.append(rowIndex)
        chromosomeFrame[block["chromosome"]]["usableBlockCount"] += 1

    result = {
        "method": "blockStrandCorrelation",
        "readLengthBP": readLengthBP,
        "effectiveMinLengthBP": minimum,
        "effectiveMaxLengthBP": int(candidates[-1]),
        "settings": settings,
        "accepted": False,
        "error": None,
        "estimateBP": None,
        "CILowerBP": None,
        "CIUpperBP": None,
        "samplingFrameBlockCount": len(frame),
        "sampledBlockCount": selectedCount,
        "usableBlockCount": len(usableRows),
        "unusableBlockCount": selectedCount - len(usableRows),
        "territoryBP": territoryBP,
        "samplingTerritoryBP": len(frame) * blockWidth,
        "sampledTerritoryBP": selectedCount * blockWidth,
        "excludedShortChromosomeBP": excludedShortChromosomeBP,
        "excludedTerminalBP": excludedTerminalBP,
        "chromosomeRepresentation": chromosomeFrame,
        "recordCounts": counts,
        "blocks": blockRows,
        "warnings": [],
        "extractionSeconds": extractionSeconds,
        "localSeconds": localSeconds,
        "poolingSeconds": 0.0,
    }
    arrays = {
        "candidateLengthsBP": candidates,
        "pointCurves": pointCurves,
        "drawnBlockIDs": drawnIDs,
        "selectedBlockIDs": selectedIDs,
        "usableRowIndices": np.asarray(usableRows, dtype=np.int64),
        "blockReadCounts": readCounts,
    }

    stamp = perf_counter()
    pooled = _poolCurves(
        pointCurves,
        readCounts,
        candidates,
        options.bootstrapReplicates,
        options.randomSeed,
        options.smoothingWidthBP,
    )
    for name in (
        "rawPooledCurve",
        "pooledCurve",
        "bootstrapEstimatesBP",
        "normalizedWeights",
        "candidateWeightTotals",
        "candidateBlockCounts",
        "bootstrapBlockIndices",
        "bootstrapBlockMultiplicities",
    ):
        arrays[name] = pooled.pop(name)
    for block, weight in zip(blockRows, arrays["normalizedWeights"]):
        block["normalizedWeight"] = float(weight)

    result.update(pooled)
    result["accepted"] = True
    boundaryCount = sum(blockRows[index]["boundaryMaximum"] for index in usableRows)
    flatCount = sum(blockRows[index]["flatMaximum"] for index in usableRows)
    nonpositiveCount = sum(
        blockRows[index]["nonpositiveMaximum"] for index in usableRows
    )
    if boundaryCount:
        result["boundaryBlockCount"] = boundaryCount
    if flatCount:
        result["flatMaximumBlockCount"] = flatCount
    if nonpositiveCount:
        result["nonpositiveMaximumBlockCount"] = nonpositiveCount

    if (
        result["boundaryMaximum"]
        or result["bootstrapBoundaryMaximumCount"]
        or boundaryCount
    ):
        result["warnings"].append(EstimateWarning.variants["boundary"])
    if result["flatMaximum"] or flatCount:
        result["warnings"].append(EstimateWarning.variants["flat"])
    if (
        result["nonpositiveMaximum"]
        or result["bootstrapNonpositiveMaximumCount"]
        or nonpositiveCount
    ):
        result["warnings"].append(EstimateWarning.variants["nonpositive"])

    result["poolingSeconds"] = perf_counter() - stamp
    if result["unusableBlockCount"]:
        result["warnings"].append(EstimateError.variants["insufficient"])
    result["unavailableBlockReasons"] = {}
    for block in blockRows:
        if not block["usable"]:
            reason = block["failureReason"]
            result["unavailableBlockReasons"][reason] = (
                result["unavailableBlockReasons"].get(reason, 0) + 1
            )

    result["runtimeSeconds"] = perf_counter() - started
    result.update(arrays)
    for value in arrays.values():
        value.flags.writeable = False
    return result
