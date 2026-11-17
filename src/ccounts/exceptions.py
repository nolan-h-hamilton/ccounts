class _Message:
    def __init__(self, message=None, *, kind=None, context=None):
        if message is None:
            message = self.defaultMessage if kind is None else self.variants[kind]
        if context is not None:
            message = f"{message}: {context}"
        super().__init__(message)


class InputError(_Message, ValueError):
    defaultMessage = "Incompatible options"
    variants = {
        "integer": "Invalid integer",
        "scalar": "Invalid numeric value",
        "option": "Invalid option",
        "identifiers": "Invalid identifiers",
        "selection": "Invalid cell selection",
        "array": "Invalid array shape or dtype",
        "values": "Invalid array values",
        "output": "Invalid output shape, dtype, or layout",
        "overlap": "Unsupported array overlap",
        "grid": "Invalid genomic grid",
        "region": "Invalid genomic region",
        "normalization": "Incompatible normalization",
        "factors": "Factors must be finite and positive",
    }


class SelectionError(_Message, TypeError):
    defaultMessage = "Invalid cell selection"


class CountError(_Message, ValueError):
    defaultMessage = "Counts must be finite nonnegative integers"


class SourceError(_Message, ValueError):
    defaultMessage = "Invalid source"
    variants = {
        "metadata": "Invalid cell metadata",
        "file": "Invalid input file",
        "missing": "Unknown source or chromosome",
        "index": "Source index required",
        "closed": "Reader is closed",
        "io": "Input/output operation failed",
    }


class EstimateError(_Message, ValueError):
    defaultMessage = "No usable estimate"
    variants = {"insufficient": "Insufficient data"}


class NumericalError(_Message, ValueError):
    defaultMessage = "Invalid numerical result"
    variants = {"logDomain": "Invalid log domain"}


class NativeError(_Message, ValueError):
    defaultMessage = "Operation failed"


class OperationError(_Message, RuntimeError):
    defaultMessage = NativeError.defaultMessage


class DependencyError(_Message, ImportError):
    defaultMessage = "Missing optional dependency"


class OutputExistsError(_Message, FileExistsError):
    defaultMessage = "Output already exists"


class AllocationError(_Message, MemoryError):
    defaultMessage = "Allocation failed"


class EstimateWarning(_Message, RuntimeWarning):
    defaultMessage = "Estimate may be unreliable. Inspect result diagnostics."
    variants = {
        "boundary": "Estimate reaches a search boundary",
        "flat": "Estimate has a flat maximum",
        "nonpositive": "Estimate has a nonpositive maximum",
        "fewFragments": "Insufficient paired fragments. Using the 147 bp fallback.",
        "fewBlocks": "Fewer eligible genomic blocks than requested",
    }
