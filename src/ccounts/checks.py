import math
import operator

import numpy as np

from .exceptions import CountError, InputError, SourceError
from .genomes import assemblyChromosomeSizes

coordinateLimit = 2**31 - 1


def validIdentifier(value):
    return (
        isinstance(value, str)
        and bool(value)
        and "\0" not in value
        and not any(c.isspace() for c in value)
    )


def checkedInteger(value, name, minimum=0, maximum=coordinateLimit):
    if isinstance(value, (bool, np.bool_)):
        raise InputError(kind="integer", context=name)
    result = operator.index(value)
    if result < minimum or result > maximum:
        raise InputError(kind="integer", context=name)
    return int(result)


def checkedReal(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise InputError(kind="scalar", context=name)
    result = float(value)
    if not math.isfinite(result):
        raise InputError(kind="scalar", context=name)
    return result


def selectedSamples(requested, available):
    selected = tuple(available if requested is None else requested)
    if not selected or len(set(selected)) != len(selected):
        raise InputError(kind="identifiers")
    if any(sampleID not in available for sampleID in selected):
        raise SourceError(kind="missing")
    return selected


def checkCountValues(values):
    if values.dtype.kind not in "biuf":
        raise CountError()
    if values.dtype.kind == "f" and (
        not np.isfinite(values).all() or np.any(values != np.rint(values))
    ):
        raise CountError()
    if np.any(values < 0):
        raise CountError()
    return values


def checkOutputArray(out, shape, dtype):
    if (
        out.shape != shape
        or out.dtype != dtype
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise InputError(kind="output")
    return out


def checkedMappabilityReference(assembly, chromosome, referenceSizes):
    if type(assembly) is not str or assembly not in assemblyChromosomeSizes:
        raise SourceError(kind="missing", context=str(assembly))
    sizes = assemblyChromosomeSizes[assembly]
    if type(chromosome) is not str or chromosome not in sizes:
        raise SourceError(kind="missing", context=f"{assembly}/{chromosome}")
    if referenceSizes is not None and dict(referenceSizes) != sizes:
        raise InputError(kind="grid", context="mappability reference")
    return sizes


def checkedMappabilityEdges(runBytes, chromosomeLength, k):
    if runBytes.dtype != np.uint8 or runBytes.ndim != 2 or runBytes.shape[0] != 4:
        raise SourceError(kind="metadata", context="mappability encoding")
    runLengths = np.ascontiguousarray(runBytes.T).view("<u4").reshape(-1)
    if (
        runLengths.size % 2 == 0
        or (runLengths[1:] == 0).any()
        or runLengths[-1] < min(chromosomeLength, k - 1)
    ):
        raise SourceError(kind="metadata", context="mappability runs")
    edges = np.r_[np.int64(0), np.cumsum(runLengths, dtype=np.int64)]
    if edges[-1] != chromosomeLength:
        raise SourceError(kind="metadata", context="mappability runs")
    return edges


def checkedMappabilityIntervals(starts, ends, chromosomeLength, measure):
    if type(measure) is not str or measure not in (
        "multiRead",
        "meanMultiRead",
        "singleRead",
        "uniqueStart",
    ):
        raise InputError(kind="option", context="measure")
    starts, ends = np.broadcast_arrays(np.asarray(starts), np.asarray(ends))
    if starts.dtype.kind not in "iu" or ends.dtype.kind not in "iu":
        raise InputError(kind="integer", context="mappability intervals")
    if (starts < 0).any() or (ends > chromosomeLength).any() or (ends <= starts).any():
        raise InputError(kind="region", context="mappability intervals")
    return starts.astype(np.int64, copy=False), ends.astype(np.int64, copy=False)
