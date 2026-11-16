import math
import operator

import numpy as np

from .exceptions import CountError, InputError, SourceError

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
