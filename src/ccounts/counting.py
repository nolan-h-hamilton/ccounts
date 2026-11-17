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


class _H5ADSource:
    def __init__(self, source, grid, selectedBarcodes):
        import h5py
        from anndata.io import read_elem, sparse_dataset

        self.handle = h5py.File(source.path, "r")
        self.slices = {}
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

            obs = read_elem(self.handle["obs"])
            descriptor = (
                source.selection if isinstance(source.selection, Mapping) else {}
            )
            metadataPath = descriptor.get("path")
            externalMetadata = (
                metadataPath is not None
                and Path(metadataPath).resolve() != Path(source.path).resolve()
            )
            barcodeColumn = (
                None if externalMetadata else descriptor.get("barcodeColumn")
            )
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
            if groups is not None and descriptor.get("path") is None:
                groupColumn = descriptor.get("groupColumn")
                if groupColumn not in obs:
                    raise SourceError(kind="metadata")
                selected &= np.asarray(obs[groupColumn].isin(groups))
            if selectedBarcodes is not None:
                selected &= np.asarray(barcodes.isin(selectedBarcodes))
            self.rows = np.flatnonzero(selected)
            selectedIDs = _cellIDs(barcodes[selected])
            if source.matrixRows == "pseudobulks" and self.rows.size > 1:
                raise InputError(kind="selection")

            node = self.handle["X"]
            if isinstance(node, h5py.Dataset):
                self.matrix = node
            else:
                if node.attrs.get("encoding-type") not in ("csr_matrix", "csc_matrix"):
                    raise SourceError(kind="metadata")
                self.matrix = sparse_dataset(node)
            if (
                self.matrix.shape != (len(obs), len(frame))
                or self.matrix.dtype.kind not in "iuf"
            ):
                raise InputError(kind="array")

            self.selectedBarcodes = frozenset(selectedIDs)
            self.selectedCellCount = (
                self.rows.size if source.matrixRows == "cells" else None
            )
            self.grid = grid
        except BaseException:
            self.handle.close()
            raise

    def close(self):
        self.handle.close()

    def _blocks(self, chromosome, startBP, endBP):
        if chromosome not in self.slices:
            raise SourceError(kind="missing", context=chromosome)
        base = self.slices[chromosome][0]
        left = base + startBP // self.grid.binSizeBP
        right = base + (endBP + self.grid.binSizeBP - 1) // self.grid.binSizeBP
        storage = getattr(self.matrix, "format", None)
        if storage == "csr":
            for rowStart in range(0, self.rows.size, 1024):
                rows = self.rows[rowStart : rowStart + 1024]
                yield 0, self.matrix[rows, left:right]
            return
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

        with np.errstate(over="raise", invalid="raise"):
            for offset, block in self._blocks(chromosome, startBP, endBP):
                self._checkedValues(block)
                sums = np.asarray(block.sum(axis=0, dtype=np.float64)).reshape(-1)
                result[offset : offset + block.shape[1]] += sums

        return result

    def libraryTotal(self, chromosomes):
        total = 0.0

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
        self.sampleIDs = tuple(source.sampleID for source in self.sources)
        if len(set(self.sampleIDs)) != len(self.sampleIDs):
            raise InputError(kind="identifiers")
        self.grid = grid
        self.threads = checkedInteger(threads, "threads", 1, 65535)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise InputError(kind="array")

        self._handles = {}
        self._info = {}
        self._readLengthFlags = {}
        self._sourceRanges = {}
        self._totals = {}
        self._closed = False
        self._selectionDir = TemporaryDirectory(prefix="countsCells_")
        qualityWarning = False

        try:
            for index, source in enumerate(self.sources):
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
                    handle = _H5ADSource(source, grid, selected)
                    info["selectedBarcodes"] = handle.selectedBarcodes
                    info["selectedCellCount"] = handle.selectedCellCount
                    info["chromosomeNames"] = tuple(handle.slices)
                    info["chromosomeSizes"] = dict(grid.chromosomeSizes)
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

                    handle = _native.NativeSource(
                        source.path,
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
                                options, self.fragmentLengthOptions
                            )
                            length = int(diagnostics["estimateBP"])
                            info["fragmentLengthDiagnostics"] = diagnostics
                            if diagnostics["warnings"]:
                                qualityWarning = True
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
            if qualityWarning:
                warnings.warn(EstimateWarning(), stacklevel=2)
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
            for handle in self._handles.values():
                handle.close()
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
                info["countOptions"], self.fragmentLengthOptions
            )
            value = int(diagnostics["estimateBP"])
            info["fragmentLengthDiagnostics"] = diagnostics
            if diagnostics["warnings"]:
                warnings.warn(EstimateWarning(), stacklevel=2)
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
        r"""Estimate a length scale for enriched regions (based on MUSIC).

        * Detect enriched regions according to a local vs. background heuristic
          at multiple filter widths.
        * For each enriched bin, record the largest filter width under which
          it qualified as enriched.
        * Return the average of these maximal filter widths.

        Bootstrap confidence intervals are included with the returned estimate
        for uncertainty quantification. The intervals may be conditional on
        detected enrichment and may understate the full variability.

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

        binSize = (
            self.grid.binSizeBP if options.binSizeBP is None else options.binSizeBP
        )
        grid = GridSpec(self.grid.chromosomeSizes, binSize)
        width = options.blockWidthBP
        if width % binSize:
            raise InputError(kind="region")

        assembly, exclusions = _lengthScaleExclusions(
            grid.chromosomeSizes, excludeRanges
        )

        explicit = regions is not None
        excluded = 0
        if explicit:
            chosen = []
            for chromosome, start, end in regions:
                start, end = grid.checkRegion(chromosome, start, end)
                if end - start != width or end % binSize:
                    raise InputError(kind="region")
                intervals = exclusions.get(chromosome, ())
                position = bisect_left(intervals, (end,))
                if position > 0 and intervals[position - 1][1] > start:
                    raise InputError(kind="region")
                chosen.append((chromosome, start, end))
            eligibleCount = len(chosen)
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
            source = self.sources[self.sampleIDs.index(sampleID)]
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

        def blocks():
            for chromosome, start, end in chosen:
                values = np.empty((len(selected), width // binSize), dtype=np.float64)
                for row, sampleID in enumerate(selected):
                    handle = self._handles[sampleID]
                    if self._info[sampleID]["sourceKind"] == "H5AD":
                        values[row] = handle.count(chromosome, start, end)
                    else:
                        values[row] = handle.countRegion(
                            chromosome,
                            start,
                            end,
                            binSize,
                            countOptions[sampleID],
                            False,
                        )[0]
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
        if chromosome not in info["chromosomeRanges"]:
            if info["sourceKind"] == "H5AD":
                result = (0, self.grid.chromosomeSizes[chromosome])
            else:
                result = tuple(
                    self._handles[sampleID].chromRange(chromosome, info["countOptions"])
                )
            info["chromosomeRanges"][chromosome] = result
        return info["chromosomeRanges"][chromosome]

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

        for row, sampleID in enumerate(selected):
            info = self._info[sampleID]
            handle = self._handles[sampleID]
            hasNoise = returnRawNoiseMass and info["sourceKind"] in ("BAM", "FRAGMENTS")
            if info["sourceKind"] == "H5AD":
                raw = handle.count(chromosome, start, end)
                rawNoise = None
            else:
                raw, rawNoise = handle.countRegion(
                    chromosome,
                    start,
                    end,
                    self.grid.binSizeBP,
                    info["countOptions"],
                    hasNoise,
                )

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
    def libraryTotals(self, *, chromosomes):
        self._checkOpen()
        selected = selectedSamples(sorted(chromosomes), self.grid.chromosomeSizes)
        if selected in self._totals:
            return self._totals[selected]

        records = {}
        for source in self.sources:
            sampleID = source.sampleID
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
                for chromosome in selected:
                    evidence = handle.totalsRegion(
                        chromosome,
                        0,
                        self.grid.chromosomeSizes[chromosome],
                        info["countOptions"],
                    )
                    record["eventMass"] += evidence["eventMass"]
                    record["spanBP"] += evidence["spanBP"]
                    record["metadata"]["chromosomes"][chromosome] = evidence
            records[sampleID] = record

        result = LibraryTotals(records, self.grid, selected)
        self._totals[selected] = result
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
