from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .sources import checkedInteger


def estimateSizeFactors(counts, *, centerMedian=True, minRefDist=10):
    values = np.asarray(counts)
    if values.ndim != 2 or not all(values.shape) or values.dtype.kind not in "iuf":
        raise ValueError("Counts must be a nonempty two-dimensional numeric array")
    if type(centerMedian) is not bool:
        raise ValueError("centerMedian must be boolean")
    distance = checkedInteger(minRefDist, "minRefDist", 0)
    sampleCount, binCount = values.shape
    eligible = np.ones(binCount, dtype=np.bool_)
    for row in values:
        if not np.all(np.isfinite(row)) or np.any(row < 0):
            raise ValueError("Counts must be finite and nonnegative")
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
        raise ValueError(f"Size-factor estimation requires {required} reference bins, found {indices.size}")
    referenceLog = np.zeros(indices.size, dtype=np.float64)
    work = np.empty_like(referenceLog)
    factors = np.empty(sampleCount, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for row in values:
                np.log(row[indices], out=work, dtype=np.float64)
                referenceLog += work
            referenceLog /= sampleCount
            for sourceIndex, row in enumerate(values):
                np.log(row[indices], out=work, dtype=np.float64)
                work -= referenceLog
                factors[sourceIndex] = np.exp(np.median(work, overwrite_input=True))
    except FloatingPointError as error:
        raise ValueError("Size-factor calculation exceeds float64 range") from error
    np.clip(factors, 0.2, 5.0, out=factors)
    if centerMedian:
        factors /= np.exp(np.median(np.log(factors + 1e-8)))
        np.clip(factors, 0.2, 5.0, out=factors)
    np.reciprocal(factors, out=factors)
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
        sampleIDs = tuple(sampleIDs)
        if len(set(sampleIDs)) != len(sampleIDs) or not sampleIDs:
            raise ValueError("Source selection must contain unique identifiers")
        if any(sampleID not in self.records for sampleID in sampleIDs):
            raise ValueError("Unknown source identifier")
        return type(self)({key: self.records[key] for key in sampleIDs},
                          self.grid, self.chromosomes)


class Normalization:
    def __init__(self, sampleIDs, factors, widthRules, units, grid, method, records):
        self.sampleIDs = tuple(sampleIDs)
        factors = np.asarray(factors, dtype=np.float64)
        if (
            not self.sampleIDs or len(set(self.sampleIDs)) != len(self.sampleIDs)
            or factors.shape != (len(self.sampleIDs),)
        ):
            raise ValueError("Factors require one value per unique sampleID")
        if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
            raise ValueError("Normalization factors must be finite and positive")
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
        return cls(sampleIDs, factors, (False,) * len(sampleIDs),
                   None, None, "supplied", None)

    @classmethod
    def fromTotals(cls, totals, *, method, effectiveGenomeSizeBP=None):
        if not isinstance(totals, LibraryTotals):
            raise TypeError("totals must be LibraryTotals")
        if method not in ("CPM", "RPKM", "RPGC"):
            raise ValueError(f"Unsupported normalization method {method!r}")
        if method == "RPGC":
            if isinstance(effectiveGenomeSizeBP, dict):
                genomeSizes = effectiveGenomeSizeBP
                if set(genomeSizes) != set(totals.sampleIDs):
                    raise ValueError("Effective genome sizes must match sampleIDs")
            else:
                genomeSizes = dict.fromkeys(totals.sampleIDs, effectiveGenomeSizeBP)
        else:
            if effectiveGenomeSizeBP is not None:
                raise ValueError("Effective genome size applies to RPGC")
            genomeSizes = {}
        factors, widths, units = [], [], []
        for sampleID, record in totals.records.items():
            unit = record["unit"]
            kind = record["sourceKind"]
            mode = record.get("countMode")
            if kind == "BEDGRAPH":
                raise ValueError("BEDGRAPH requires supplied factors")
            eventMass = float(record["eventMass"])
            if not np.isfinite(eventMass) or eventMass <= 0:
                raise ValueError(f"Positive library total required for {sampleID}")
            divideByWidth = False
            if method == "CPM":
                factor = 1_000_000.0 / eventMass
            elif method == "RPKM":
                if unit not in ("eventMass", "storedBinCount"):
                    raise ValueError("RPKM requires event or stored-bin counts")
                factor = 1_000_000_000.0 / eventMass
                divideByWidth = True
            else:
                size = genomeSizes[sampleID]
                if isinstance(size, (bool, np.bool_)) or size is None:
                    raise ValueError("RPGC requires a positive effective genome size")
                size = float(size)
                if not np.isfinite(size) or size <= 0:
                    raise ValueError("RPGC requires a positive effective genome size")
                if kind == "H5AD":
                    raise ValueError("H5AD requires supplied factors for RPGC")
                if mode == "coverage":
                    span = float(record["spanBP"])
                    if not np.isfinite(span) or span <= 0:
                        raise ValueError("RPGC coverage requires positive retained span")
                    factor = size / span
                elif mode == "conservedFractionalOverlap":
                    factor = size / eventMass
                    divideByWidth = True
                else:
                    raise ValueError("RPGC requires coverage or conserved fractional mass")
            factors.append(factor)
            widths.append(divideByWidth)
            units.append(unit)
        return cls(totals.sampleIDs, factors, widths, units, totals.grid,
                   method, totals.records)

    def _checkBlock(self, block):
        if any(sampleID not in block.sampleIDs for sampleID in self.sampleIDs):
            raise ValueError("Count block lacks a normalization source")
        if self.grid is not None and (
            self.grid.binSizeBP != block.grid.binSizeBP
            or dict(self.grid.chromosomeSizes) != dict(block.grid.chromosomeSizes)
        ):
            raise ValueError("Normalization and count grids differ")
        if self.units is not None:
            for sampleID, unit in zip(self.sampleIDs, self.units):
                rowIndex = block.sampleIDs.index(sampleID)
                if block.units[rowIndex] != unit:
                    raise ValueError("Normalization and count units differ")
                if self.records is not None:
                    expectedMode = self.records[sampleID].get("countMode")
                    if block.countModes[rowIndex] != expectedMode:
                        raise ValueError(
                            f"Counting mode mismatch for {sampleID!r}: "
                            f"expected {expectedMode!r}, got {block.countModes[rowIndex]!r}")

    def multipliers(self, block, *, sampleID):
        self._checkBlock(block)
        if sampleID not in self.sampleIDs:
            raise ValueError(f"Unknown normalization source {sampleID!r}")
        index = self.sampleIDs.index(sampleID)
        factor = self.sourceFactors[index]
        if not self.widthRules[index]:
            return float(factor)
        return float(factor / block.grid.binSizeBP)

    def apply(self, block, *, out=None):
        self._checkBlock(block)
        shape = (len(self.sampleIDs), block.values.shape[1])
        if out is None:
            target = np.empty(shape, dtype=block.values.dtype)
        else:
            if (
                not isinstance(out, np.ndarray) or out.shape != shape
                or out.dtype != block.values.dtype
                or not out.flags.c_contiguous or not out.flags.writeable
            ):
                raise ValueError("out must match normalized shape, dtype, and layout")
            target = out
        if np.shares_memory(target, block.values) and (
            target.shape != block.values.shape
            or target.ctypes.data != block.values.ctypes.data
            or self.sampleIDs != block.sampleIDs
        ):
            raise ValueError("Normalization does not support partial overlap or aliased row reordering")
        try:
            with np.errstate(over="raise", invalid="raise"):
                for index, sampleID in enumerate(self.sampleIDs):
                    row = block.values[block.sampleIDs.index(sampleID)]
                    if not np.all(np.isfinite(row)):
                        raise ValueError("Normalization requires finite input values")
                    np.multiply(row, self.multipliers(block, sampleID=sampleID),
                                out=target[index])
        except FloatingPointError as error:
            raise ValueError("Normalized values exceed the output dtype") from error
        return target

    def forPair(self, treatmentID, referenceID, *, fixReference=False):
        if self.units is None:
            raise ValueError("Pair matching requires totals with known measurement units")
        if type(fixReference) is not bool:
            raise ValueError("fixReference must be boolean")
        if treatmentID == referenceID:
            raise ValueError("A pair requires distinct logical source identifiers")
        if treatmentID not in self.sampleIDs or referenceID not in self.sampleIDs:
            raise ValueError("Unknown pair source")
        first, second = self.sampleIDs.index(treatmentID), self.sampleIDs.index(referenceID)
        if self.units[first] != self.units[second] or self.widthRules[first] != self.widthRules[second]:
            raise ValueError("Pair matching requires equal units and bin-width rules")
        a, b = self.sourceFactors[first], self.sourceFactors[second]
        if fixReference:
            factors = (1.0 if a >= b else a / b, 1.0)
        else:
            divisor = max(a, b)
            factors = (a / divisor, b / divisor)
        return type(self)((treatmentID, referenceID), factors, (False, False),
                          (self.units[first], self.units[second]), self.grid,
                          "paired", self.records)
