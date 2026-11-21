from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .checks import checkedInteger, coordinateLimit
from .exceptions import EstimateError, EstimateWarning, InputError
from .genomes import assemblyChromosomeSizes, standardChromosomes


@dataclass(frozen=True)
class FragmentLengthOptions:
    minLengthBP: int = 100
    maxLengthBP: int = 1000

    def __post_init__(self):
        for field in ("minLengthBP", "maxLengthBP"):
            object.__setattr__(
                self,
                field,
                checkedInteger(getattr(self, field), field, 1, coordinateLimit - 1),
            )
        if self.minLengthBP > self.maxLengthBP:
            raise InputError(kind="option", context="minLengthBP")


def _estimateSingleEnd(source, countOptions, options):
    started = perf_counter()
    minimum = max(options.minLengthBP, countOptions.minTemplateLengthBP)
    maximum = options.maxLengthBP
    if countOptions.maxTemplateLengthBP is not None:
        maximum = min(maximum, countOptions.maxTemplateLengthBP)
    if minimum > maximum:
        raise InputError(kind="option", context="Fragment length bounds")

    sizes = source.chromosomeSizes
    assemblyMatches = {
        assembly: sum(
            sizes.get(chromosome) == assemblyChromosomeSizes[assembly][chromosome]
            for chromosome in names
        )
        for assembly, names in standardChromosomes.items()
    }
    bestMatch = max(assemblyMatches.values())
    standard = {
        chromosome
        for assembly, names in standardChromosomes.items()
        if assemblyMatches[assembly] == bestMatch
        for chromosome in names
    }
    chromosomes = tuple(name for name in source.chromosomeNames if name in standard)
    if not chromosomes:
        raise EstimateError(kind="insufficient", context="No standard chromosomes")

    lags = np.arange(maximum + 1, dtype=np.int64)
    curves = np.empty((len(chromosomes), lags.size), dtype=np.float64)
    readCounts = np.empty((len(chromosomes), 2), dtype=np.uint64)
    pooledCurve = np.zeros(lags.size, dtype=np.float64)
    chromosomeEstimates = {}
    totalReads = 0
    for index, chromosome in enumerate(chromosomes):
        curve, forward, reverse = source.chromosomeCorrelations(
            chromosome, countOptions, maximum
        )
        curves[index] = curve
        readCounts[index] = (forward, reverse)
        chromosomeEstimates[chromosome] = None
        if forward and reverse:
            weight = forward + reverse
            pooledCurve += weight * curve
            totalReads += weight
            chromosomeEstimates[chromosome] = minimum + int(curve[minimum:].argmax())

    if not totalReads:
        raise EstimateError(
            kind="insufficient", context="No standard chromosome with both strands"
        )

    pooledCurve /= totalReads
    search = pooledCurve[minimum:]
    estimate = minimum + int(search.argmax())
    peak = float(pooledCurve[estimate])
    warningKeys = []
    if estimate in (minimum, maximum):
        warningKeys.append("boundary")
    if search.min() == peak:
        warningKeys.append("flat")
    if peak <= 0:
        warningKeys.append("nonpositive")

    estimates = [value for value in chromosomeEstimates.values() if value is not None]
    lower, median, upper = np.quantile(estimates, (0.25, 0.5, 0.75))
    for array in (lags, curves, readCounts, pooledCurve):
        array.flags.writeable = False

    return {
        "method": "csawStrandCrossCorrelation",
        "estimateBP": estimate,
        "effectiveMinLengthBP": minimum,
        "effectiveMaxLengthBP": maximum,
        "peakCorrelation": peak,
        "settings": {
            **asdict(options),
            "flagExclude": countOptions.flagExclude,
            "minMappingQuality": countOptions.minMappingQuality,
        },
        "candidateLengthsBP": lags,
        "pooledCurve": pooledCurve,
        "chromosomeNames": chromosomes,
        "chromosomeCurves": curves,
        "chromosomeReadCounts": readCounts,
        "chromosomeEstimatesBP": chromosomeEstimates,
        "chromosomeHeterogeneity": {
            "chromosomeCount": len(estimates),
            "minBP": min(estimates),
            "maxBP": max(estimates),
            "medianBP": float(median),
            "iqrBP": float(upper - lower),
        },
        "warnings": [EstimateWarning.variants[key] for key in warningKeys],
        "runtimeSeconds": perf_counter() - started,
    }
