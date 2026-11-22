from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from collections.abc import Mapping
from functools import wraps
from threading import RLock
from time import perf_counter
import gzip
import logging
import os
import warnings

import numpy as np

from .sources import (
    GridSpec,
    CountOptions,
    resolveSelection,
    _cellIDs,
)
from .checks import checkedInteger, selectedSamples, checkCountValues, checkOutputArray
from .exceptions import (
    EstimateError,
    EstimateWarning,
    InputError,
    NativeError,
    NumericalError,
    SourceError,
)
from .normalization import LibraryTotals
from .fragmentLength import FragmentLengthOptions
from .lengthScale import (
    LengthScaleOptions,
    _defaultBinSizeBP,
    _estimateLengthScale,
    _lengthScaleExclusions,
)


def _locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


@dataclass(frozen=True)
class CountBlock:
    values: np.ndarray
    sampleIDs: tuple
    grid: GridSpec
    chromosome: str
    startBP: int
    endBP: int
    units: tuple
    countModes: tuple
    rawNoiseMass: np.ndarray | None = None
    noiseAvailable: tuple | None = None

    def __post_init__(self):
        start, end = self.grid.checkRegion(self.chromosome, self.startBP, self.endBP)
        count = (end - start + self.grid.binSizeBP - 1) // self.grid.binSizeBP
        sampleIDs = tuple(self.sampleIDs)
        if not sampleIDs or len(set(sampleIDs)) != len(sampleIDs):
            raise InputError(kind="identifiers")
        if (
            not isinstance(self.values, np.ndarray)
            or self.values.shape != (len(sampleIDs), count)
            or self.values.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not self.values.flags.c_contiguous
        ):
            raise InputError(kind="array")
        if len(self.units) != len(sampleIDs) or len(self.countModes) != len(sampleIDs):
            raise InputError(kind="array")
        if self.rawNoiseMass is not None and (
            self.rawNoiseMass.shape != self.values.shape
            or self.rawNoiseMass.dtype != self.values.dtype
            or not self.rawNoiseMass.flags.c_contiguous
        ):
            raise InputError(kind="array")
        available = self.noiseAvailable
        if available is None:
            available = (self.rawNoiseMass is not None,) * len(sampleIDs)
        if len(available) != len(sampleIDs):
            raise InputError(kind="array")
        if self.rawNoiseMass is None and any(available):
            raise InputError(kind="array")

        object.__setattr__(self, "sampleIDs", sampleIDs)
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "countModes", tuple(self.countModes))
        object.__setattr__(self, "noiseAvailable", tuple(available))


class _H5ADFile:
    def __init__(self, path, grid):
        import h5py
        from anndata.io import read_elem, sparse_dataset

        self.handle = h5py.File(path, "r")
        self.path = path
        self.grid = grid
        self.slices = {}
        self.indptr = None
        try:
            if (
                "var" not in self.handle
                or "obs" not in self.handle
                or "X" not in self.handle
            ):
                raise SourceError(kind="metadata")
            frame = read_elem(self.handle["var"])
            if not {"chrom", "start", "end"}.issubset(frame.columns):
                raise SourceError(kind="metadata")
            chromosomes = np.asarray(frame["chrom"])
            starts = np.asarray(frame["start"])
            ends = np.asarray(frame["end"])
            if (
                not len(frame)
                or starts.dtype.kind not in "iu"
                or ends.dtype.kind not in "iu"
                or np.any(starts < 0)
                or np.any(ends <= starts)
            ):
                raise InputError(kind="grid")

            for chromosome in dict.fromkeys(chromosomes):
                if chromosome not in grid.chromosomeSizes:
                    raise SourceError(kind="missing", context=chromosome)
                indices = np.flatnonzero(chromosomes == chromosome)
                left, right = int(indices[0]), int(indices[-1]) + 1
                if right - left != indices.size:
                    raise InputError(kind="grid")
                length = grid.chromosomeSizes[chromosome]
                expected = np.arange(0, length, grid.binSizeBP, dtype=np.int64)
                if (
                    expected.size != indices.size
                    or not np.array_equal(starts[left:right], expected)
                    or not np.array_equal(
                        ends[left:right], np.minimum(expected + grid.binSizeBP, length)
                    )
                ):
                    raise InputError(kind="grid")
                self.slices[chromosome] = (left, right)

            self.obs = read_elem(self.handle["obs"])
            node = self.handle["X"]
            if isinstance(node, h5py.Dataset):
                self.matrix = node
            else:
                if node.attrs.get("encoding-type") not in ("csr_matrix", "csc_matrix"):
                    raise SourceError(kind="metadata")
                self.matrix = sparse_dataset(node)
            if (
                self.matrix.shape != (len(self.obs), len(frame))
                or self.matrix.dtype.kind not in "iuf"
            ):
                raise InputError(kind="array")

            if getattr(self.matrix, "format", None) == "csr":
                self.data = node["data"]
                self.indices = node["indices"]
                pointers = node["indptr"]
                if (
                    self.data.ndim != 1
                    or self.indices.shape != self.data.shape
                    or self.indices.dtype.kind not in "iu"
                    or pointers.shape != (len(self.obs) + 1,)
                    or pointers.dtype.kind not in "iu"
                ):
                    raise InputError(kind="array")
                self.indptr = np.asarray(pointers, dtype=np.int64)
                if (
                    self.indptr[0] != 0
                    or self.indptr[-1] != len(self.data)
                    or np.any(self.indptr[1:] < self.indptr[:-1])
                ):
                    raise InputError(kind="array")
                self.chunkSize = (
                    min(self.data.chunks[0], self.indices.chunks[0])
                    if self.data.chunks and self.indices.chunks
                    else 1
                )
        except BaseException:
            self.handle.close()
            raise

    def close(self):
        self.handle.close()


class _H5ADSource:
    def __init__(self, source, file, selectedBarcodes):
        self.file = file
        self.slices = file.slices
        self.grid = file.grid
        self.matrix = file.matrix
        obs = file.obs
        descriptor = source.selection if isinstance(source.selection, Mapping) else {}
        metadataPath = descriptor.get("path")
        externalMetadata = (
            metadataPath is not None and Path(metadataPath).resolve() != file.path
        )
        barcodeColumn = None if externalMetadata else descriptor.get("barcodeColumn")
        if barcodeColumn is not None and barcodeColumn not in obs:
            raise SourceError(kind="metadata", context=barcodeColumn)
        barcodes = obs.index if barcodeColumn is None else obs[barcodeColumn]
        selected = np.ones(len(obs), dtype=bool)
        sampleColumn = descriptor.get("sampleColumn")
        if sampleColumn is not None and (
            not externalMetadata or not barcodes.is_unique
        ):
            if sampleColumn not in obs:
                raise SourceError(kind="metadata", context=sampleColumn)
            selected &= np.asarray(
                obs[sampleColumn] == descriptor.get("sampleValue", source.sampleID)
            )
        groups = descriptor.get("groups")
        if groups is not None and metadataPath is None:
            groupColumn = descriptor.get("groupColumn")
            if groupColumn not in obs:
                raise SourceError(kind="metadata")
            selected &= np.asarray(obs[groupColumn].isin(groups))
        if selectedBarcodes is not None:
            selected &= np.asarray(barcodes.isin(selectedBarcodes))
        self.rows = np.flatnonzero(selected)
        self.selectedBarcodes = frozenset(_cellIDs(barcodes[selected]))
        if source.matrixRows == "pseudobulks" and self.rows.size > 1:
            raise InputError(kind="selection")
        self.selectedCellCount = (
            self.rows.size if source.matrixRows == "cells" else None
        )

        if file.indptr is not None:
            starts = file.indptr[self.rows]
            stops = file.indptr[self.rows + 1]
            nonempty = starts < stops
            self.csrStarts, self.csrStops = starts[nonempty], stops[nonempty]
            starts, stops = self.csrStarts, self.csrStops
            # Include a gap only when it can share a storage chunk.
            gaps = (starts[1:] > stops[:-1]) & (
                starts[1:] // file.chunkSize != (stops[:-1] - 1) // file.chunkSize
            )
            self.csrBoundaries = np.r_[0, np.flatnonzero(gaps) + 1, starts.size]

    def _columns(self, chromosome, startBP, endBP):
        if chromosome not in self.slices:
            raise SourceError(kind="missing", context=chromosome)
        base = self.slices[chromosome][0]
        binSize = self.grid.binSizeBP
        return base + startBP // binSize, base + (endBP + binSize - 1) // binSize

    def _csrSlabs(self):
        starts, stops = self.csrStarts, self.csrStops
        slabSize = 2**20
        for first, last in zip(self.csrBoundaries[:-1], self.csrBoundaries[1:]):
            while first < last:
                offset = int(starts[first])
                width = np.searchsorted(stops[first:last], offset + slabSize, side="right")
                end = first + max(1, width)
                limit = int(stops[end - 1])
                for begin in range(offset, limit, slabSize):
                    stop = min(begin + slabSize, limit)
                    values = self.file.data[begin:stop]
                    indices = self.file.indices[begin:stop]
                    if values.dtype.kind == "f" and values.dtype.itemsize == 2:
                        values = values.astype(np.float32)
                    elif not values.dtype.isnative:
                        values = values.astype(values.dtype.newbyteorder("="))
                    if indices.dtype not in (np.dtype("int32"), np.dtype("int64")):
                        indices = indices.astype(np.int64)

                    if begin == offset and stop == limit:
                        slabStarts, slabStops = starts[first:end], stops[first:end]
                    else:
                        # Only a single oversized row can cross this boundary.
                        slabStarts = np.array([begin], dtype=np.int64)
                        slabStops = np.array([stop], dtype=np.int64)
                    yield indices, values, slabStarts, slabStops, begin
                first = end

    def _blocks(self, chromosome, startBP, endBP):
        left, right = self._columns(chromosome, startBP, endBP)
        storage = getattr(self.matrix, "format", None)
        for columnStart in range(left, right, 4096):
            columnEnd = min(columnStart + 4096, right)
            columns = (
                self.matrix[:, columnStart:columnEnd] if storage == "csc" else None
            )
            for rowStart in range(0, self.rows.size, 1024):
                rows = self.rows[rowStart : rowStart + 1024]
                if storage == "csc":
                    block = columns[rows]
                else:
                    block = self.matrix[rows, columnStart:columnEnd]
                yield columnStart - left, block

    @staticmethod
    def _checkedValues(block):
        from scipy import sparse

        values = block.data if sparse.issparse(block) else np.asarray(block)
        return checkCountValues(values)

    def count(self, chromosome, startBP, endBP):
        count = (endBP - startBP + self.grid.binSizeBP - 1) // self.grid.binSizeBP
        result = np.zeros(count, dtype=np.float64)

        if self.file.indptr is not None:
            from ._native import _aggregateH5ADCSR

            left, right = self._columns(chromosome, startBP, endBP)
            correction = None
            for slab in self._csrSlabs():
                correction = _aggregateH5ADCSR(*slab, left, right, result, correction, True)
            if correction is not None:
                result += correction
            return result

        with np.errstate(over="raise", invalid="raise"):
            for offset, block in self._blocks(chromosome, startBP, endBP):
                self._checkedValues(block)
                sums = np.asarray(block.sum(axis=0, dtype=np.float64)).reshape(-1)
                result[offset : offset + block.shape[1]] += sums

        return result

    def libraryTotal(self, chromosomes):
        total = 0.0

        if self.file.indptr is not None:
            from ._native import _aggregateH5ADCSR

            ranges = []
            for left, right in sorted(
                self.slices[chromosome] for chromosome in chromosomes
            ):
                if ranges and ranges[-1][1] == left:
                    ranges[-1] = (ranges[-1][0], right)
                else:
                    ranges.append((left, right))
            result = np.zeros(1, dtype=np.float64)
            correction = None
            for slab in self._csrSlabs():
                for left, right in ranges:
                    correction = _aggregateH5ADCSR(
                        *slab, left, right, result, correction, False
                    )
            total = float(result[0]) + (0.0 if correction is None else correction[0])
        else:
            with np.errstate(over="raise", invalid="raise"):
                for chromosome in chromosomes:
                    for _, block in self._blocks(
                        chromosome, 0, self.grid.chromosomeSizes[chromosome]
                    ):
                        values = self._checkedValues(block)
                        total += float(np.sum(values, dtype=np.float64))
        if not np.isfinite(total):
            raise NumericalError()

        return total


def _cpuBudget():
    try:
        if hasattr(os, "process_cpu_count"):
            budget = os.process_cpu_count()
        elif hasattr(os, "sched_getaffinity"):
            budget = len(os.sched_getaffinity(0))
        else:
            budget = os.cpu_count()
    except OSError:
        budget = None

    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation is None:
        return budget
    try:
        allocated = int(allocation)
    except ValueError:
        return None
    return min(budget, allocated) if budget is not None else allocated


class Reader:
    def __init__(
        self, sources, grid, *, threads=1, dtype="float32", fragmentLengthOptions=None
    ):
        from . import _native

        if fragmentLengthOptions is None:
            fragmentLengthOptions = FragmentLengthOptions()
        self.fragmentLengthOptions = fragmentLengthOptions
        self._lock = RLock()
        self.sources = tuple(sources)
        if not self.sources:
            raise SourceError()
        self._sourcesByID = {source.sampleID: source for source in self.sources}
        self.sampleIDs = tuple(self._sourcesByID)
        if len(self.sampleIDs) != len(self.sources):
            raise InputError(kind="identifiers")
        self.grid = grid
        autoThreads = isinstance(threads, str) and threads == "auto"
        self.threads = 1 if autoThreads else checkedInteger(threads, "threads", 1, 65535)
        self._singleEndThreads = self.threads
        if autoThreads:
            bamCount = sum(source.sourceKind == "BAM" for source in self.sources)
            if bamCount:
                cpusPerBAM = (_cpuBudget() or 0) // bamCount
                self._singleEndThreads = max(1, min(4, cpusPerBAM - 2))
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise InputError(kind="array")

        self._handles = {}
        self._h5adFiles = {}
        self._info = {}
        self._readLengthFlags = {}
        self._sourceRanges = {}
        self._totals = {}
        self._closed = False
        self._selectionDir = TemporaryDirectory(prefix="countsCells_")
        qualityWarning = None

        try:
            for index, source in enumerate(self.sources):
                if source.sourceKind == "H5AD":
                    path = Path(source.path).resolve()
                    file = self._h5adFiles.get(path)
                    if file is None:
                        file = self._h5adFiles[path] = _H5ADFile(path, grid)
                    selected = resolveSelection(source, h5adObs=file.obs)
                else:
                    selected = resolveSelection(source)
                info = {
                    "path": source.path,
                    "sourceKind": source.sourceKind,
                    "selectedBarcodes": selected,
                    "selectedCellCount": None if selected is None else len(selected),
                    "readLength": None,
                    "pairedEnd": False,
                    "fragmentLengthBP": None,
                    "fragmentLengthDiagnostics": None,
                    "chromosomeRanges": {},
                    "inferenceRequested": bool(
                        source.countOptions is not None
                        and source.countOptions.inferFragmentLength
                    ),
                    "inferenceUsed": False,
                    "inferenceStatus": "notRequested",
                }
                if source.sourceKind == "H5AD":
                    handle = _H5ADSource(source, file, selected)
                    info["selectedBarcodes"] = handle.selectedBarcodes
                    info["selectedCellCount"] = handle.selectedCellCount
                    info["chromosomeNames"] = tuple(handle.slices)
                    info["chromosomeSizes"] = {
                        chromosome: grid.chromosomeSizes[chromosome]
                        for chromosome in handle.slices
                    }
                    info["countOptions"] = None
                    info["unit"] = "storedBinCount"
                    info["countMode"] = None
                else:
                    allowPath = ""
                    if selected is not None:
                        allowPath = str(
                            Path(self._selectionDir.name) / f"source{index}.txt"
                        )
                        Path(allowPath).write_text(
                            "".join(f"{barcode}\n" for barcode in sorted(selected))
                        )

                    path = source.path
                    if (
                        source.sourceKind == "BAM"
                        and self._singleEndThreads > self.threads
                    ):
                        path = Path(path).absolute()
                    handle = _native.NativeSource(
                        path,
                        source.sourceKind,
                        dict(grid.chromosomeSizes),
                        barcodeAllowListFile=allowPath,
                        barcodeTag=source.barcodeTag,
                        threads=self.threads,
                    )
                    self._handles[source.sampleID] = handle
                    options = source.countOptions
                    if source.sourceKind == "BAM":
                        paired = bool(handle.pairedEnd(options))
                        mode = options.bamInputMode
                        if mode == "auto":
                            mode = "fragments" if paired else "reads"

                        inspectionFlags = options.flagExclude
                        if selected is not None:
                            inspectionFlags |= (
                                136
                                if mode == "read1"
                                else (128 if mode == "fragments" else 0)
                            )
                        self._readLengthFlags[source.sampleID] = inspectionFlags
                        flags = options.flagExclude | (136 if mode == "read1" else 0)
                        infer = options.inferFragmentLength and mode != "fragments"
                        if options.inferFragmentLength and not infer:
                            info["inferenceStatus"] = "observedTemplate"
                        options = replace(
                            options,
                            bamInputMode=mode,
                            flagExclude=flags,
                            inferFragmentLength=infer,
                        )
                        if options.inferFragmentLength:
                            diagnostics = handle.estimateFragmentLength(
                                options,
                                self.fragmentLengthOptions,
                                self._singleEndThreads,
                            )
                            length = int(diagnostics["estimateBP"])
                            info["fragmentLengthDiagnostics"] = diagnostics
                            if diagnostics["warnings"]:
                                if diagnostics.get("status") == "insufficientData":
                                    qualityWarning = EstimateWarning(kind="fewFragments")
                                elif qualityWarning is None:
                                    qualityWarning = EstimateWarning()
                            if length <= 0:
                                raise EstimateError()
                            options = replace(
                                options,
                                inferFragmentLength=False,
                                extendFrom5pBP=length,
                            )
                            info["fragmentLengthBP"] = length
                            info["inferenceUsed"] = True
                            info["inferenceStatus"] = "inferred"
                        info["pairedEnd"] = paired
                    elif source.sourceKind == "FRAGMENTS":
                        options = replace(options, bamInputMode="reads")
                    else:
                        options = CountOptions(
                            countMode="coverage", bamInputMode="reads"
                        )

                    info["chromosomeNames"] = tuple(handle.chromosomeNames)
                    info["chromosomeSizes"] = dict(handle.chromosomeSizes)
                    info["countOptions"] = options
                    info["countMode"] = options.countMode
                    info["unit"] = (
                        "signalMean"
                        if source.sourceKind == "BEDGRAPH"
                        else (
                            "meanCoverage"
                            if options.countMode == "coverage"
                            else "eventMass"
                        )
                    )

                self._handles[source.sampleID] = handle
                self._info[source.sampleID] = info
            for file in self._h5adFiles.values():
                del file.obs
            if qualityWarning is not None:
                warnings.warn(qualityWarning, stacklevel=2)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        self._checkOpen()
        return self

    def __exit__(self, excType, excValue, traceback):
        self.close()

    def _checkOpen(self):
        if self._closed:
            raise SourceError(kind="closed")

    @_locked
    def close(self):
        if not self._closed:
            for sampleID, handle in self._handles.items():
                if self._sourcesByID[sampleID].sourceKind != "H5AD":
                    handle.close()
            for file in self._h5adFiles.values():
                file.close()
            self._selectionDir.cleanup()
            self._closed = True

    @property
    @_locked
    def sourceInfo(self):
        self._checkOpen()
        return MappingProxyType(
            {key: MappingProxyType(dict(value)) for key, value in self._info.items()}
        )

    @_locked
    def getSourceInfo(self, sampleID, *, inspectLengths=False):
        self._checkOpen()
        if sampleID not in self._info:
            raise SourceError(kind="missing")
        info = self._info[sampleID]
        if (
            inspectLengths
            and info["sourceKind"] == "BAM"
            and info["readLength"] is None
        ):
            options = replace(
                info["countOptions"], flagExclude=self._readLengthFlags[sampleID]
            )
            info["readLength"] = int(self._handles[sampleID].readLength(options))
        return MappingProxyType(dict(info))

    @_locked
    def setReadExtension(self, sampleID, lengthBP):
        self._checkOpen()
        if sampleID not in self._info or self._info[sampleID]["sourceKind"] != "BAM":
            raise SourceError()
        info = self._info[sampleID]
        if info["countOptions"].bamInputMode == "fragments":
            raise InputError()
        length = checkedInteger(lengthBP, "lengthBP", 1)
        info["countOptions"] = replace(
            info["countOptions"], extendFrom5pBP=length, inferFragmentLength=False
        )
        info["fragmentLengthBP"] = length
        info["fragmentLengthDiagnostics"] = None
        info["chromosomeRanges"].clear()
        self._totals.clear()

    @_locked
    def estimateFragmentLength(self, sampleID):
        self._checkOpen()
        if sampleID not in self._info or self._info[sampleID]["sourceKind"] != "BAM":
            raise SourceError()
        info = self._info[sampleID]
        if info["fragmentLengthBP"] is None:
            diagnostics = self._handles[sampleID].estimateFragmentLength(
                info["countOptions"], self.fragmentLengthOptions, self._singleEndThreads
            )
            value = int(diagnostics["estimateBP"])
            info["fragmentLengthDiagnostics"] = diagnostics
            if diagnostics["warnings"]:
                warning = EstimateWarning(
                    kind="fewFragments"
                    if diagnostics.get("status") == "insufficientData"
                    else None
                )
                warnings.warn(warning, stacklevel=2)
            if value <= 0:
                raise EstimateError()
            info["fragmentLengthBP"] = value
        return info["fragmentLengthBP"]

    @_locked
    def estimateLengthScale(
        self,
        *,
        sampleIDs=None,
        options=None,
        regions=None,
        excludeRanges=None,
        bedGraphCounts=False,
        summaryFunction=np.mean,
    ):
        r"""Estimate a length scale for enriched regions

        * Detect enriched regions according to a local vs. background heuristic
          at multiple filter widths.
        * For each enriched bin, record the largest filter width under which
          it qualified as enriched.
        * Return the average of these maximal filter widths.

        Confidence intervals use a fast infinitesimal jackknife (IJ) standard error
        from the genomic blocks and a normal approximation.

        References
        ----------
        Harmanci, Rozowsky and Gerstein (2014), MUSIC.
        https://doi.org/10.1186/s13059-014-0474-3
        """
        self._checkOpen()
        started = perf_counter()
        options = LengthScaleOptions() if options is None else options
        if type(bedGraphCounts) is not bool:
            raise InputError()
        selected = selectedSamples(sampleIDs, self.sampleIDs)

        logger = logging.getLogger("ccounts.lengthScale")
        kinds = [self._info[sampleID]["sourceKind"] for sampleID in selected]
        binned = [
            f"{sampleID} ({kind})"
            for sampleID, kind in zip(selected, kinds)
            if kind not in ("BAM", "FRAGMENTS")
        ]
        gridBinSize = self.grid.binSizeBP
        bedGraphNote = (
            " bedGraph values are read as one count per bin, so %s must match "
            "the bedGraph's interval width."
        )
        if options.binSizeBP is not None:
            binSize = options.binSizeBP
            logger.info(
                "Length scale estimation is using %d bp bins, as set by "
                "LengthScaleOptions.binSizeBP. The Reader's grid "
                "(GridSpec.binSizeBP=%d) isn't used for this estimate.%s",
                binSize,
                gridBinSize,
                (
                    bedGraphNote % "LengthScaleOptions.binSizeBP"
                    if "BEDGRAPH" in kinds
                    else ""
                ),
            )
        elif not binned:
            binSize = _defaultBinSizeBP
            logger.info(
                "Length scale estimation is using %d bp bins, the default when "
                "LengthScaleOptions.binSizeBP is None and every source is BAM or "
                "FRAGMENTS. The Reader's grid (GridSpec.binSizeBP=%d) isn't used "
                "for this estimate.",
                binSize,
                gridBinSize,
            )
        else:
            binSize = gridBinSize
            listed = ", ".join(binned[:5])
            if len(binned) > 5:
                listed += f", and {len(binned) - 5} more"
            notes = ""
            if "H5AD" in kinds:
                notes += " H5AD counts can't be used at any other bin size."
            if "BEDGRAPH" in kinds:
                notes += bedGraphNote % "GridSpec.binSizeBP"
            if binSize == _defaultBinSizeBP:
                logger.info(
                    "Length scale estimation is using the Reader's %d bp bins "
                    "(GridSpec.binSizeBP), because LengthScaleOptions.binSizeBP "
                    "is None and these sources are already binned: %s. That is "
                    "also the bin size it uses when every source is BAM or "
                    "FRAGMENTS.%s",
                    binSize,
                    listed,
                    notes,
                )
            else:
                logger.warning(
                    "Length scale estimation is using the Reader's %d bp bins "
                    "(GridSpec.binSizeBP), not the %d bp it uses when every "
                    "source is BAM or FRAGMENTS, because "
                    "LengthScaleOptions.binSizeBP is None and these sources are "
                    "already binned: %s.%s To confirm this bin size and silence "
                    "this message, set LengthScaleOptions(binSizeBP=%d).",
                    binSize,
                    _defaultBinSizeBP,
                    listed,
                    notes,
                    binSize,
                )
        grid = GridSpec(self.grid.chromosomeSizes, binSize)
        width = options.blockWidthBP - options.blockWidthBP % binSize
        if not width:
            raise InputError(kind="region")

        assembly, exclusions = _lengthScaleExclusions(
            grid.chromosomeSizes, excludeRanges
        )

        explicit = regions is not None
        excluded = 0
        if explicit:
            chosen = []
            moved = 0
            for chromosome, start, end in regions:
                if chromosome not in grid.chromosomeSizes:
                    raise SourceError(kind="missing")
                start = checkedInteger(start, "startBP")
                end = checkedInteger(end, "endBP")
                length = grid.chromosomeSizes[chromosome]
                if end > length or end - start not in (width, options.blockWidthBP):
                    raise InputError(kind="region")
                moved += bool(start % binSize)
                start -= start % binSize
                start, end = grid.checkRegion(chromosome, start, start + width)
                intervals = exclusions.get(chromosome, ())
                position = bisect_left(intervals, (end,))
                if position > 0 and intervals[position - 1][1] > start:
                    raise InputError(kind="region")
                chosen.append((chromosome, start, end))
            eligibleCount = len(chosen)
            if moved:
                logger.info(
                    "Length scale estimation moved %d of %d region starts down "
                    "to the nearest %d bp bin boundary. The regions actually "
                    'used are in the result\'s settings["regions"].',
                    moved,
                    len(chosen),
                    binSize,
                )
        else:
            startRanges = []
            rangeEnds = []
            eligibleCount = 0
            candidateCount = 0
            for chromosome, length in grid.chromosomeSizes.items():
                candidateCount += max(0, (length - width) // binSize + 1)
                spanStart = 0
                for left, right in (*exclusions.get(chromosome, ()), (length, length)):
                    # The whole block has to fit inside this unmasked span.
                    first = (spanStart + binSize - 1) // binSize
                    stop = (left - width) // binSize + 1
                    if stop > first:
                        startRanges.append((chromosome, first))
                        eligibleCount += stop - first
                        rangeEnds.append(eligibleCount)
                    spanStart = right
            excluded = candidateCount - eligibleCount

        if not eligibleCount:
            raise EstimateError(kind="insufficient")
        if not explicit and eligibleCount < options.blockCount:
            warnings.warn(
                EstimateWarning(
                    kind="fewBlocks",
                    context=(
                        f"{eligibleCount} available, {options.blockCount} requested"
                    ),
                ),
                stacklevel=3,
            )

        if not explicit:
            # Rank valid starts even when their blocks overlap.
            ranks = np.sort(
                np.random.default_rng(options.randomSeed).choice(
                    eligibleCount, min(options.blockCount, eligibleCount), replace=False
                )
            )
            rangeEnds = np.asarray(rangeEnds, dtype=np.int64)
            rangeIndices = np.searchsorted(rangeEnds, ranks, side="right")
            chosen = []
            for rank, index in zip(ranks, rangeIndices):
                chromosome, first = startRanges[index]
                offset = int(rank) - (int(rangeEnds[index - 1]) if index else 0)
                start = (first + offset) * binSize
                chosen.append((chromosome, start, start + width))

        sourceInfo = {}
        countOptions = {}
        for sampleID in selected:
            info = self._info[sampleID]
            kind = info["sourceKind"]
            if any(
                chromosome not in info["chromosomeNames"] for chromosome, _, _ in chosen
            ):
                raise SourceError(kind="missing", context=sampleID)
            if kind == "H5AD" and binSize != self.grid.binSizeBP:
                raise InputError(kind="grid")
            if kind == "BEDGRAPH" and not bedGraphCounts:
                raise InputError()

            resolved = info["countOptions"]
            if kind in ("BAM", "FRAGMENTS"):
                resolved = replace(resolved, countMode="center")
            countOptions[sampleID] = resolved
            source = self._sourcesByID[sampleID]
            unit = (
                "storedBinCount"
                if kind == "H5AD"
                else (
                    "declaredBinCount"
                    if kind == "BEDGRAPH"
                    else (
                        "fragmentCenter"
                        if kind == "FRAGMENTS" or resolved.bamInputMode == "fragments"
                        else "readCenter"
                    )
                )
            )

            sourceInfo[sampleID] = {
                "path": info["path"],
                "sourceKind": kind,
                "eventUnit": unit,
                "countOptions": None if resolved is None else asdict(resolved),
                "selection": (
                    {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in source.selection.items()
                    }
                    if isinstance(source.selection, Mapping)
                    else None if source.selection is None else sorted(source.selection)
                ),
                "selectedCellCount": info["selectedCellCount"],
                "matrixRows": source.matrixRows,
                "barcodeTag": source.barcodeTag,
                "fragmentLengthBP": info["fragmentLengthBP"],
                "inferenceUsed": info["inferenceUsed"],
                "inferenceStatus": info["inferenceStatus"],
            }

        batches = self._fragmentBatches(selected, countOptions)

        def blocks():
            for chromosome, start, end in chosen:
                values = np.empty((len(selected), width // binSize), dtype=np.float64)
                rows = self._countRows(
                    selected, batches, (chromosome, start, end), binSize,
                    countOptions=countOptions,
                )
                for row, (counts, _) in enumerate(rows):
                    values[row] = counts
                yield values

        sampling = {
            "explicitRegions": explicit,
            "samplingMethod": (
                "explicitRegions" if explicit else "uniformStartsWithoutReplacement"
            ),
            "eligibleBlockCount": eligibleCount,
            "selectedBlockCount": len(chosen),
            "excludedBlockCount": excluded,
            "terminalBP": sum(
                length if length < width else length % binSize
                for length in grid.chromosomeSizes.values()
            ),
            "exclusionAssembly": assembly,
            "excludeRanges": exclusions,
        }
        result = _estimateLengthScale(
            blocks(),
            len(chosen),
            binSize,
            selected,
            options,
            chosen,
            sourceInfo,
            sampling,
            summaryFunction=summaryFunction,
        )
        result["runtimeSeconds"] = perf_counter() - started
        return result

    @_locked
    def sourceRange(self, sampleID, chromosome, *, flagExclude):
        self._checkOpen()
        if sampleID not in self._info or chromosome not in self.grid.chromosomeSizes:
            raise SourceError(kind="missing")
        flags = checkedInteger(flagExclude, "flagExclude", 0, 65535)
        key = (sampleID, chromosome, flags)
        if key not in self._sourceRanges:
            length = self.grid.chromosomeSizes[chromosome]
            if self._info[sampleID]["sourceKind"] == "H5AD":
                if chromosome not in self._handles[sampleID].slices:
                    raise SourceError(kind="missing", context=chromosome)
                result = (0, length)
            else:
                start, end = self._handles[sampleID].sourceRange(chromosome, flags)
                start = min(max(0, start), length)
                end = min(max(0, end), length)
                if end <= start and (start > 0 or end > 0):
                    start, end = 0, length
                result = (start, end)
            self._sourceRanges[key] = result
        return self._sourceRanges[key]

    @_locked
    def chromRange(self, sampleID, chromosome):
        self._checkOpen()
        if sampleID not in self._info or chromosome not in self.grid.chromosomeSizes:
            raise SourceError(kind="missing")
        info = self._info[sampleID]
        isH5AD = info["sourceKind"] == "H5AD"
        if isH5AD and chromosome not in self._handles[sampleID].slices:
            raise SourceError(kind="missing", context=chromosome)
        if chromosome not in info["chromosomeRanges"]:
            if isH5AD:
                result = (0, self.grid.chromosomeSizes[chromosome])
            else:
                result = tuple(
                    self._handles[sampleID].chromRange(chromosome, info["countOptions"])
                )
            info["chromosomeRanges"][chromosome] = result
        return info["chromosomeRanges"][chromosome]

    def _fragmentBatches(self, samples, countOptions=None):
        groups = {}
        for sampleID in samples:
            source = self._sourcesByID[sampleID]
            if source.sourceKind == "FRAGMENTS":
                groups.setdefault(source.path, []).append(sampleID)

        batches = {}
        for members in groups.values():
            if len(members) < 2:
                continue
            handles = tuple(self._handles[sampleID] for sampleID in members)
            if countOptions is None:
                options = tuple(self._info[name]["countOptions"] for name in members)
            else:
                options = tuple(countOptions[sampleID] for sampleID in members)
            batches.update(dict.fromkeys(members, (members, handles, options)))
        return batches

    def _countRows(
        self, samples, batches, region, binSizeBP,
        *, returnRawNoiseMass=False, countOptions=None,
    ):
        chromosome, start, end = region
        pending = {}
        for sampleID in samples:
            batch = batches.get(sampleID)
            if batch is not None:
                if sampleID not in pending:
                    from ._native import _fragmentsBatch

                    members, handles, options = batch
                    pending.update(
                        zip(
                            members,
                            _fragmentsBatch(
                                handles, options, *region, binSizeBP, returnRawNoiseMass
                            ),
                        )
                    )
                result = pending.pop(sampleID)
                if isinstance(result, NativeError):
                    raise result
                yield result
                continue

            info = self._info[sampleID]
            handle = self._handles[sampleID]
            if info["sourceKind"] == "H5AD":
                yield handle.count(chromosome, start, end), None
            else:
                options = (
                    info["countOptions"] if countOptions is None else countOptions[sampleID]
                )
                hasNoise = returnRawNoiseMass and info["sourceKind"] in ("BAM", "FRAGMENTS")
                yield handle.countRegion(*region, binSizeBP, options, hasNoise)

    @_locked
    def readRegion(
        self,
        chromosome,
        startBP,
        endBP,
        *,
        sampleIDs=None,
        returnRawNoiseMass=False,
        out=None,
    ):
        self._checkOpen()
        start, end = self.grid.checkRegion(chromosome, startBP, endBP)
        if type(returnRawNoiseMass) is not bool:
            raise InputError()
        selected = selectedSamples(sampleIDs, self.sampleIDs)

        count = (end - start + self.grid.binSizeBP - 1) // self.grid.binSizeBP
        shape = (len(selected), count)
        if out is None:
            values = np.empty(shape, dtype=self.dtype)
        else:
            values = checkOutputArray(out, shape, self.dtype)

        noise = np.full(shape, np.nan, dtype=self.dtype) if returnRawNoiseMass else None
        units, modes, available = [], [], []
        limit = np.finfo(self.dtype).max

        rows = self._countRows(
            selected, self._fragmentBatches(selected), (chromosome, start, end),
            self.grid.binSizeBP, returnRawNoiseMass=returnRawNoiseMass,
        )
        for row, (sampleID, (raw, rawNoise)) in enumerate(zip(selected, rows)):
            info = self._info[sampleID]
            hasNoise = returnRawNoiseMass and info["sourceKind"] in ("BAM", "FRAGMENTS")
            maximum = float(np.max(raw, initial=0.0))
            minimum = (
                float(np.min(raw, initial=0.0))
                if info["sourceKind"] == "BEDGRAPH"
                else 0.0
            )
            if not np.isfinite(maximum) or maximum > limit or minimum < -limit:
                raise NumericalError()
            values[row] = raw
            if hasNoise:
                if rawNoise is None:
                    raise NativeError()
                maximumNoise = float(np.max(rawNoise, initial=0.0))
                if not np.isfinite(maximumNoise) or maximumNoise > limit:
                    raise NumericalError()
                noise[row] = rawNoise
            units.append(info["unit"])
            modes.append(info["countMode"])
            available.append(hasNoise)

        return CountBlock(
            values,
            selected,
            self.grid,
            chromosome,
            start,
            end,
            tuple(units),
            tuple(modes),
            noise,
            tuple(available),
        )

    def iterBlocks(
        self,
        chromosomes,
        *,
        blockSizeBins=500_000,
        sampleIDs=None,
        returnRawNoiseMass=False,
    ):
        self._checkOpen()
        count = checkedInteger(blockSizeBins, "blockSizeBins", 1)
        step = count * self.grid.binSizeBP

        for chromosome in chromosomes:
            if chromosome not in self.grid.chromosomeSizes:
                raise SourceError(kind="missing", context=chromosome)
            end = self.grid.chromosomeSizes[chromosome]
            for start in range(0, end, step):
                yield self.readRegion(
                    chromosome,
                    start,
                    min(start + step, end),
                    sampleIDs=sampleIDs,
                    returnRawNoiseMass=returnRawNoiseMass,
                )

    @_locked
    def commonFeatureTerritory(self, *, chromosomes, sampleIDs=None):
        self._checkOpen()
        requested = selectedSamples(sorted(chromosomes), self.grid.chromosomeSizes)
        samples = selectedSamples(sampleIDs, self.sampleIDs)
        missingBySource = {}
        missing = set()
        for sampleID in samples:
            info = self._info[sampleID]
            absent = ()
            # Tabix indexes list chromosomes with records.
            if info["sourceKind"] in ("BAM", "H5AD"):
                available = set(info["chromosomeNames"])
                absent = tuple(
                    chromosome
                    for chromosome in requested
                    if chromosome not in available
                )
            missingBySource[sampleID] = absent
            missing.update(absent)

        return MappingProxyType(
            {
                "requestedChromosomes": requested,
                "usedChromosomes": tuple(
                    chromosome for chromosome in requested if chromosome not in missing
                ),
                "missingChromosomesBySource": MappingProxyType(missingBySource),
            }
        )

    @_locked
    def _regionEndpointTotals(self, regionPath, *, sampleIDs):
        self._checkOpen()
        samples = selectedSamples(sampleIDs, self.sampleIDs)
        for sampleID in samples:
            if self._info[sampleID]["sourceKind"] not in ("BAM", "FRAGMENTS"):
                raise InputError(
                    "Region normalization requires BAM or FRAGMENTS input",
                    context=sampleID,
                )

        path = Path(regionPath)
        sizes = self.grid.chromosomeSizes
        intervals = {}
        opener = gzip.open if path.suffix in (".gz", ".bgz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for lineNumber, line in enumerate(handle, 1):
                fields = line.split(maxsplit=3)
                if not fields or fields[0].startswith("#"):
                    continue
                chromosome = fields[0]
                if chromosome not in sizes and chromosome in ("track", "browser"):
                    continue
                context = f"{path}:{lineNumber}"
                if len(fields) < 3:
                    raise InputError(kind="region", context=context)
                if chromosome not in sizes:
                    raise SourceError(
                        kind="missing", context=f"{context} ({chromosome})"
                    )
                try:
                    start, end = int(fields[1]), int(fields[2])
                except ValueError as error:
                    raise InputError(kind="region", context=context) from error
                if not 0 <= start < end <= sizes[chromosome]:
                    raise InputError(kind="region", context=context)
                intervals.setdefault(chromosome, []).append((start, end))
        if not intervals:
            raise InputError(kind="region", context=path)

        territory = self.commonFeatureTerritory(chromosomes=intervals, sampleIDs=samples)
        for sampleID, missing in territory["missingChromosomesBySource"].items():
            if missing:
                raise SourceError(kind="missing", context=f"{sampleID}/{missing[0]}")

        merged = {}
        for chromosome, ranges in intervals.items():
            union = []
            for start, end in sorted(ranges):
                if union and start <= union[-1][1]:
                    union[-1] = (union[-1][0], max(end, union[-1][1]))
                else:
                    union.append((start, end))
            merged[chromosome] = np.asarray(union, dtype=np.int64)

        records = {}
        for sampleID in samples:
            info = self._info[sampleID]
            source = self._handles[sampleID]
            options = replace(info["countOptions"], countMode="fiveprime")
            mass = sum(
                source.regionEndpointTotal(chromosome, ranges, options)
                for chromosome, ranges in merged.items()
            )
            if mass <= 0:
                raise EstimateError(
                    "No endpoints in normalization regions", context=sampleID
                )
            records[sampleID] = MappingProxyType(
                {
                    "sourceKind": info["sourceKind"],
                    "unit": info["unit"],
                    "countMode": info["countMode"],
                    "regionPath": str(path),
                    "regionEndpointMass": mass,
                }
            )
        return MappingProxyType(records)

    @_locked
    def libraryTotals(self, *, chromosomes, sampleIDs=None):
        self._checkOpen()
        selected = selectedSamples(sorted(chromosomes), self.grid.chromosomeSizes)
        samples = selectedSamples(sampleIDs, self.sampleIDs)
        key = (selected, samples)
        if key in self._totals:
            return self._totals[key]

        for sampleID in samples:
            if self._info[sampleID]["sourceKind"] == "H5AD":
                features = self._handles[sampleID].slices
                for chromosome in selected:
                    if chromosome not in features:
                        raise SourceError(kind="missing", context=chromosome)

        batches = self._fragmentBatches(
            sampleID
            for sampleID in samples
            if self._info[sampleID]["selectedCellCount"] != 0
        )
        pending = {}
        records = {}
        for sampleID in samples:
            source = self._sourcesByID[sampleID]
            info = self._info[sampleID]
            handle = self._handles[sampleID]
            record = {
                "path": source.path,
                "barcodeTag": source.barcodeTag,
                "sourceKind": source.sourceKind,
                "unit": info["unit"],
                "countMode": info["countMode"],
                "eventMass": 0.0,
                "spanBP": 0.0,
                "denominatorUnit": (
                    "storedBinCount" if source.sourceKind == "H5AD" else "emittedEvent"
                ),
                "requestedMethod": "exact",
                "usedMethod": "exact",
                "countOptions": info["countOptions"],
                "chromosomes": selected,
                "selectedCellCount": info["selectedCellCount"],
                "inferenceRequested": info["inferenceRequested"],
                "inferenceUsed": info["inferenceUsed"],
                "inferenceStatus": info["inferenceStatus"],
                "selectedBarcodes": (
                    tuple(sorted(info["selectedBarcodes"]))
                    if source.selection is not None
                    and info["selectedBarcodes"] is not None
                    else None
                ),
                "selection": (
                    {
                        key: (
                            tuple(value)
                            if isinstance(value, (tuple, frozenset))
                            else str(value) if isinstance(value, Path) else value
                        )
                        for key, value in source.selection.items()
                    }
                    if isinstance(source.selection, Mapping)
                    else (
                        tuple(sorted(source.selection))
                        if source.selection is not None
                        else None
                    )
                ),
                "metadata": {"chromosomes": {}},
            }

            if info["selectedCellCount"] == 0:
                record["metadata"]["selectedCellCount"] = 0
            elif source.sourceKind == "BEDGRAPH":
                record["eventMass"] = None
                record["spanBP"] = None
                record["usedMethod"] = "unavailable"
            elif source.sourceKind == "H5AD":
                record["eventMass"] = handle.libraryTotal(selected)
                record["spanBP"] = None
            else:
                batch = batches.get(sampleID)
                if batch is not None and sampleID not in pending:
                    from ._native import _fragmentsBatch

                    members, handles, options = batch
                    pending.update((member, []) for member in members)
                    for chromosome in selected:
                        length = self.grid.chromosomeSizes[chromosome]
                        totals = _fragmentsBatch(handles, options, chromosome, 0, length)
                        for member, evidence in zip(members, totals):
                            pending[member].append(evidence)

                batchTotals = pending.pop(sampleID, None)
                for index, chromosome in enumerate(selected):
                    if batchTotals is None:
                        length = self.grid.chromosomeSizes[chromosome]
                        evidence = handle.totalsRegion(
                            chromosome, 0, length, info["countOptions"]
                        )
                    else:
                        evidence = batchTotals[index]
                        if isinstance(evidence, NativeError):
                            raise evidence
                    record["eventMass"] += evidence["eventMass"]
                    record["spanBP"] += evidence["spanBP"]
                    record["metadata"]["chromosomes"][chromosome] = evidence
            records[sampleID] = record

        result = LibraryTotals(records, self.grid, selected)
        self._totals[key] = result
        return result


def openSources(
    sources, grid, *, threads=1, dtype="float32", fragmentLengthOptions=None
):
    return Reader(
        sources,
        grid,
        threads=threads,
        dtype=dtype,
        fragmentLengthOptions=fragmentLengthOptions,
    )
