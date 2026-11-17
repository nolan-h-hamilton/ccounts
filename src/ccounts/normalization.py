from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .exceptions import EstimateError, InputError, SourceError
from .checks import checkedInteger, checkedReal, checkOutputArray, selectedSamples


def estimateSizeFactors(
    counts, *, centerMedian=True, minRefDist=10, factorBounds=(0.01, 100.0)
):
    values = np.asarray(counts)
    if values.ndim != 2 or not all(values.shape) or values.dtype.kind not in "iuf":
        raise InputError(kind="array")
    if type(centerMedian) is not bool:
        raise InputError(kind="option", context="centerMedian")
    if factorBounds is not None:
        bounds = np.asarray(factorBounds, dtype=object)
        if bounds.shape != (2,):
            raise InputError(kind="option", context="factorBounds")
        lower, upper = (checkedReal(value, "factorBounds") for value in bounds)
        if not 0 < lower <= upper:
            raise InputError(kind="option", context="factorBounds")

    distance = checkedInteger(minRefDist, "minRefDist", 0)
    sampleCount, binCount = values.shape
    eligible = np.ones(binCount, dtype=np.bool_)
    for row in values:
        if (values.dtype.kind == "f" and not np.all(np.isfinite(row))) or np.any(
            row < 0
        ):
            raise InputError(kind="values")
        np.logical_and(eligible, row >= 1.0, out=eligible)

    if distance > 1 and np.all(eligible):
        indices = np.arange(0, binCount, distance, dtype=np.intp)
    else:
        indices = np.flatnonzero(eligible)
        if distance > 1:
            selectedCount = 0
            nextIndex = 0
            for index in memoryview(indices):
                if index >= nextIndex:
                    indices[selectedCount] = index
                    selectedCount += 1
                    nextIndex = index + distance
            indices = indices[:selectedCount]

    required = int(np.ceil(max(10.0, min(500.0, np.sqrt(binCount * 0.5)))))
    if indices.size < required:
        raise EstimateError(kind="insufficient")

    referenceLog = np.zeros(indices.size, dtype=np.float64)
    work = np.empty_like(referenceLog)
    factors = np.empty(sampleCount, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        for row in values:
            np.log(row[indices], out=work, dtype=np.float64)
            referenceLog += work
        referenceLog /= sampleCount
        for sourceIndex, row in enumerate(values):
            np.log(row[indices], out=work, dtype=np.float64)
            work -= referenceLog
            factors[sourceIndex] = np.exp(np.median(work, overwrite_input=True))

    if centerMedian:
        factors /= np.exp(np.median(np.log(factors)))
    np.reciprocal(factors, out=factors)
    if factorBounds is not None:
        # Bounds apply to the multipliers we return.
        np.clip(factors, lower, upper, out=factors)
    return factors


@dataclass(frozen=True)
class LibraryTotals:
    records: dict
    grid: object
    chromosomes: tuple

    def __post_init__(self):
        records = {
            key: MappingProxyType(dict(value)) for key, value in self.records.items()
        }
        object.__setattr__(self, "records", MappingProxyType(records))
        object.__setattr__(self, "chromosomes", tuple(self.chromosomes))

    @property
    def sampleIDs(self):
        return tuple(self.records)

    def forSources(self, sampleIDs):
        sampleIDs = selectedSamples(tuple(sampleIDs), self.records)

        return type(self)(
            {key: self.records[key] for key in sampleIDs}, self.grid, self.chromosomes
        )


class Normalization:
    def __init__(self, sampleIDs, factors, widthRules, units, grid, method, records):
        self.sampleIDs = tuple(sampleIDs)
        factors = np.asarray(factors, dtype=np.float64)
        if (
            not self.sampleIDs
            or len(set(self.sampleIDs)) != len(self.sampleIDs)
            or factors.shape != (len(self.sampleIDs),)
        ):
            raise InputError(kind="factors")
        if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
            raise InputError(kind="factors")

        self.sourceFactors = factors.copy()
        self.sourceFactors.flags.writeable = False
        self.widthRules = tuple(widthRules)
        self.units = None if units is None else tuple(units)
        self.grid = grid
        self.method = method
        self.records = records

    @classmethod
    def fromFactors(cls, sampleIDs, factors):
        sampleIDs = tuple(sampleIDs)
        return cls(
            sampleIDs, factors, (False,) * len(sampleIDs), None, None, "supplied", None
        )

    @classmethod
    def fromTotals(cls, totals, *, method, effectiveGenomeSizeBP=None):
        if method not in ("CPM", "RPKM", "RPGC"):
            raise InputError(kind="option", context="method")

        if method == "RPGC":
            if isinstance(effectiveGenomeSizeBP, dict):
                genomeSizes = effectiveGenomeSizeBP
                if set(genomeSizes) != set(totals.sampleIDs):
                    raise InputError(kind="normalization")
            else:
                genomeSizes = dict.fromkeys(totals.sampleIDs, effectiveGenomeSizeBP)
        else:
            if effectiveGenomeSizeBP is not None:
                raise InputError()
            genomeSizes = {}

        factors, widths, units = [], [], []
        for sampleID, record in totals.records.items():
            unit = record["unit"]
            kind = record["sourceKind"]
            mode = record.get("countMode")
            if kind == "BEDGRAPH":
                raise InputError(kind="normalization")
            eventMass = checkedReal(record["eventMass"], "eventMass")
            if eventMass <= 0:
                raise InputError(kind="scalar", context="eventMass")

            divideByWidth = False
            if method == "CPM":
                factor = 1_000_000.0 / eventMass
            elif method == "RPKM":
                if unit not in ("eventMass", "storedBinCount"):
                    raise InputError(kind="normalization")
                factor = 1_000_000_000.0 / eventMass
                divideByWidth = True
            else:
                size = checkedReal(genomeSizes[sampleID], "effectiveGenomeSizeBP")
                if size <= 0:
                    raise InputError(kind="scalar", context="effectiveGenomeSizeBP")
                if kind == "H5AD":
                    raise InputError(kind="normalization")
                if mode == "coverage":
                    span = checkedReal(record["spanBP"], "spanBP")
                    if span <= 0:
                        raise InputError(kind="scalar", context="spanBP")
                    factor = size / span
                elif mode == "conservedFractionalOverlap":
                    factor = size / eventMass
                    divideByWidth = True
                else:
                    raise InputError(kind="normalization")

            factors.append(factor)
            widths.append(divideByWidth)
            units.append(unit)

        return cls(
            totals.sampleIDs,
            factors,
            widths,
            units,
            totals.grid,
            method,
            totals.records,
        )

    def _checkBlock(self, block):
        rowsByID = {sampleID: index for index, sampleID in enumerate(block.sampleIDs)}
        rowIndices = tuple(rowsByID[sampleID] for sampleID in self.sampleIDs)
        if (
            self.grid is not None
            and self.grid is not block.grid
            and (
                self.grid.binSizeBP != block.grid.binSizeBP
                or self.grid.chromosomeSizes != block.grid.chromosomeSizes
            )
        ):
            raise InputError(kind="normalization")

        if self.units is not None:
            for sampleID, unit, rowIndex in zip(self.sampleIDs, self.units, rowIndices):
                if block.units[rowIndex] != unit:
                    raise InputError(kind="normalization")
                if self.records is not None:
                    expectedMode = self.records[sampleID].get("countMode")
                    if block.countModes[rowIndex] != expectedMode:
                        raise InputError(kind="normalization")

        return rowIndices

    def multipliers(self, block, *, sampleID):
        self._checkBlock(block)
        index = self.sampleIDs.index(sampleID)
        factor = self.sourceFactors[index]
        if not self.widthRules[index]:
            return float(factor)
        return float(factor / block.grid.binSizeBP)

    def apply(self, block, *, out=None):
        rowIndices = self._checkBlock(block)

        shape = (len(self.sampleIDs), block.values.shape[1])
        if out is None:
            target = np.empty(shape, dtype=block.values.dtype)
        else:
            target = checkOutputArray(out, shape, block.values.dtype)

        if np.shares_memory(target, block.values) and (
            target.shape != block.values.shape
            or target.ctypes.data != block.values.ctypes.data
            or self.sampleIDs != block.sampleIDs
        ):
            raise InputError(kind="overlap")

        with np.errstate(over="raise", invalid="raise"):
            for index, rowIndex in enumerate(rowIndices):
                row = block.values[rowIndex]
                if not np.all(np.isfinite(row)):
                    raise InputError(kind="values")
                factor = self.sourceFactors[index]
                if self.widthRules[index]:
                    factor /= block.grid.binSizeBP
                np.multiply(row, factor, out=target[index], dtype=np.float64)
        return target

    def forPair(self, treatmentID, referenceID, *, fixReference=False):
        if self.units is None:
            raise InputError(kind="normalization")
        if type(fixReference) is not bool:
            raise InputError(kind="option", context="fixReference")
        if treatmentID == referenceID:
            raise InputError(kind="identifiers")
        if treatmentID not in self.sampleIDs or referenceID not in self.sampleIDs:
            raise SourceError(kind="missing")

        first, second = self.sampleIDs.index(treatmentID), self.sampleIDs.index(
            referenceID
        )
        if (
            self.units[first] != self.units[second]
            or self.widthRules[first] != self.widthRules[second]
        ):
            raise InputError(kind="normalization")

        a, b = self.sourceFactors[first], self.sourceFactors[second]
        if fixReference:
            factors = (1.0 if a >= b else a / b, 1.0)
        else:
            divisor = max(a, b)
            factors = (a / divisor, b / divisor)

        return type(self)(
            (treatmentID, referenceID),
            factors,
            (False, False),
            (self.units[first], self.units[second]),
            self.grid,
            "paired",
            self.records,
        )
