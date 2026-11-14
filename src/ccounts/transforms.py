import numpy as np


def arrayOutput(values, out):
    array = np.asarray(values)
    if array.ndim not in (1, 2) or array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Input must be a 1D/2D float32 or float64 array")
    if not array.flags.c_contiguous:
        raise ValueError("Input must be C-contiguous")
    if out is None:
        return array, np.empty_like(array)
    if not isinstance(out, np.ndarray) or out.shape != array.shape or out.dtype != array.dtype:
        raise ValueError("out must match input shape and dtype")
    if not out.flags.c_contiguous or not out.flags.writeable:
        raise ValueError("out must be writable and C-contiguous")
    if np.shares_memory(array, out) and (
        array.ctypes.data != out.ctypes.data or array.strides != out.strides
    ):
        raise ValueError("Partial input/output overlap is unsupported")
    return array, out


def logTransform(values, *, reference=None, pseudocount=1.0, out=None):
    from . import _native

    array, target = arrayOutput(values, out)
    if isinstance(pseudocount, (bool, np.bool_)):
        raise ValueError("pseudocount must be positive")
    pseudocount = float(pseudocount)
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be finite and positive")
    if not np.all(np.isfinite(array)) or float(np.min(array, initial=0.0)) <= -pseudocount:
        raise ValueError("Log input must be finite with values greater than -pseudocount")
    if reference is not None:
        reference = np.asarray(reference)
        if (
            reference.shape != array.shape or reference.dtype != array.dtype
            or not reference.flags.c_contiguous
        ):
            raise ValueError("reference must match input shape, dtype, and layout")
        if np.shares_memory(reference, target) and (
            reference.ctypes.data != target.ctypes.data
            or reference.strides != target.strides
        ):
            raise ValueError("Partial reference/output overlap is unsupported")
        if not np.all(np.isfinite(reference)) or float(np.min(reference, initial=0.0)) <= -pseudocount:
            raise ValueError("Log reference must be finite with values greater than -pseudocount")
    _native.logValues(array, reference, pseudocount, target)
    return target
