import numpy as np

from .exceptions import InputError, NumericalError
from .checks import checkedReal


def logTransform(values, *, reference=None, pseudocount=1.0, out=None):
    r"""Repo-wide standardized log-transform function for values with optional reference and pseudocount.
    """
    from . import _native

    array = np.asarray(values)
    target = np.empty_like(array) if out is None else out
    pseudocount = checkedReal(pseudocount, "pseudocount")
    if pseudocount <= 0:
        raise InputError(kind="scalar", context="pseudocount")
    if (array.dtype.kind == "f" and not np.all(np.isfinite(array))) or float(
        np.min(array, initial=0.0)
    ) <= -pseudocount:
        raise NumericalError(kind="logDomain")

    if reference is not None:
        reference = np.asarray(reference)
        if (
            reference.dtype.kind == "f" and not np.all(np.isfinite(reference))
        ) or float(np.min(reference, initial=0.0)) <= -pseudocount:
            raise NumericalError(kind="logDomain")

    _native.logValues(array, reference, pseudocount, target)
    return target
