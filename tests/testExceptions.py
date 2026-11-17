import pickle

import numpy as np
import pytest

import ccounts as cc
from ccounts import _native


@pytest.mark.parametrize(
    "errorType,builtinType,arguments,message",
    [
        (
            cc.InputError,
            ValueError,
            {"kind": "integer", "context": "binSizeBP"},
            "Invalid integer: binSizeBP",
        ),
        (cc.SelectionError, TypeError, {}, "Invalid cell selection"),
        (
            cc.CountError,
            ValueError,
            {},
            "Counts must be finite nonnegative integers",
        ),
        (
            cc.SourceError,
            ValueError,
            {"kind": "missing", "context": "chr1"},
            "Unknown source or chromosome: chr1",
        ),
        (cc.EstimateError, ValueError, {}, "No usable estimate"),
        (cc.NumericalError, ValueError, {}, "Invalid numerical result"),
        (
            cc.NativeError,
            ValueError,
            {"message": "Invalid input file"},
            "Invalid input file",
        ),
        (cc.OperationError, RuntimeError, {}, "Operation failed"),
        (cc.DependencyError, ImportError, {}, "Missing optional dependency"),
        (cc.OutputExistsError, FileExistsError, {}, "Output already exists"),
        (cc.AllocationError, MemoryError, {}, "Allocation failed"),
        (
            cc.EstimateWarning,
            RuntimeWarning,
            {},
            "Estimate may be unreliable. Inspect result diagnostics.",
        ),
    ],
)
def testExceptionMessagesAndSerialization(errorType, builtinType, arguments, message):
    error = errorType(**arguments)
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(error, builtinType)
    assert type(restored) is errorType
    assert str(error) == str(restored) == message
    assert error.args == restored.args == (message,)


def testPythonAndCythonExceptions(tmp_path):
    options = cc.LengthScaleOptions(blockWidthBP=300)
    with pytest.raises(cc.CountError, match="Counts must be"):
        cc.estimateLengthScaleCounts([[[-1, 0, 0]]], 100, options=options)
    with pytest.raises(cc.CountError, match="Counts must be"):
        _native.musicBinomialLogTail(np.array([0]), np.array([1]))
    with pytest.raises(cc.InputError, match="Invalid output"):
        cc.logTransform(np.ones(2), out=np.empty(1))
    with pytest.raises(cc.NativeError, match="Could not build source index"):
        cc.prepareIndex(tmp_path / "missing.bam", "BAM")
