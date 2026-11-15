from __future__ import annotations

import csv
import gzip
import operator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np


coordinateLimit = 2**31 - 1
countModes = ("coverage", "binOverlap", "conservedFractionalOverlap", "center", "fiveprime", "ffp")
sourceKinds = ("BAM", "FRAGMENTS", "BEDGRAPH", "H5AD")


def checkedInteger(value, name, minimum=0, maximum=coordinateLimit):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return int(result)


def readChromSizes(path):
    result = {}
    with open(path, encoding="utf-8") as handle:
        for lineNumber, line in enumerate(handle, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 2:
                raise ValueError(f"Invalid chromosome sizes row {lineNumber}")
            name = fields[0]
            if name in result:
                raise ValueError(f"Duplicate chromosome {name}")
            try:
                length = int(fields[1])
            except ValueError as error:
                raise ValueError(f"Invalid chromosome length on row {lineNumber}") from error
            result[name] = checkedInteger(length, "chromosome length", 1)
    if not result:
        raise ValueError("Chromosome sizes must not be empty")
    return result


@dataclass(frozen=True)
class GridSpec:
    chromosomeSizes: Mapping
    binSizeBP: int = 50

    def __post_init__(self):
        sizes = {}
        for name, value in self.chromosomeSizes.items():
            if not isinstance(name, str) or not name or "\0" in name or any(c.isspace() for c in name):
                raise ValueError("Chromosome names must be nonempty exact identifiers")
            sizes[name] = checkedInteger(value, "chromosome length", 1)
        if not sizes:
            raise ValueError("Chromosome sizes must not be empty")
        object.__setattr__(self, "chromosomeSizes", MappingProxyType(sizes))
        object.__setattr__(self, "binSizeBP", checkedInteger(self.binSizeBP, "binSizeBP", 1))

    @classmethod
    def fromSizesFile(cls, path, binSizeBP=50):
        return cls(readChromSizes(path), binSizeBP)

    def checkRegion(self, chromosome, startBP, endBP):
        if chromosome not in self.chromosomeSizes:
            raise ValueError(f"Unknown chromosome {chromosome!r}")
        start = checkedInteger(startBP, "startBP")
        end = checkedInteger(endBP, "endBP")
        length = self.chromosomeSizes[chromosome]
        if start >= end or end > length or start % self.binSizeBP:
            raise ValueError("Region must follow the chromosome-origin bin grid")
        if end != length and end % self.binSizeBP:
            raise ValueError("Region end must follow the bin grid or chromosome end")
        return start, end

    def binWidths(self, chromosome, startBP, endBP):
        start, end = self.checkRegion(chromosome, startBP, endBP)
        count = (end - start + self.binSizeBP - 1) // self.binSizeBP
        widths = np.full(count, self.binSizeBP, dtype=np.float64)
        widths[-1] = end - start - (count - 1) * self.binSizeBP
        return widths


@dataclass(frozen=True)
class CountOptions:
    countMode: str = "conservedFractionalOverlap"
    bamInputMode: str = "auto"
    flagExclude: int = 3844
    minMappingQuality: int = 20
    minTemplateLengthBP: int = -1
    maxTemplateLengthBP: int | None = 2500
    shiftForward5p: int = 0
    shiftReverse5p: int = 0
    extendFrom5pBP: int | None = None
    inferFragmentLength: bool = False
    fragmentsUseReadSupport: bool = False

    def __post_init__(self):
        if self.countMode not in countModes:
            raise ValueError(f"Unsupported countMode {self.countMode!r}")
        if self.bamInputMode not in ("auto", "reads", "read1", "fragments"):
            raise ValueError(f"Unsupported bamInputMode {self.bamInputMode!r}")
        for field, maximum in (("flagExclude", 65535), ("minMappingQuality", 255)):
            object.__setattr__(self, field, checkedInteger(getattr(self, field), field, 0, maximum))
        object.__setattr__(self, "minTemplateLengthBP", checkedInteger(
            self.minTemplateLengthBP, "minTemplateLengthBP", -1))
        if self.maxTemplateLengthBP is not None:
            maximum = checkedInteger(self.maxTemplateLengthBP, "maxTemplateLengthBP", 1)
            if maximum < self.minTemplateLengthBP:
                raise ValueError("Template length bounds are reversed")
            object.__setattr__(self, "maxTemplateLengthBP", maximum)
        for field in ("shiftForward5p", "shiftReverse5p"):
            object.__setattr__(self, field, checkedInteger(
                getattr(self, field), field, -coordinateLimit))
        if self.extendFrom5pBP is not None:
            object.__setattr__(self, "extendFrom5pBP", checkedInteger(
                self.extendFrom5pBP, "extendFrom5pBP", 1))
        for field in ("inferFragmentLength", "fragmentsUseReadSupport"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be boolean")
        if self.inferFragmentLength and self.extendFrom5pBP is not None:
            raise ValueError("Extension and inference cannot be combined")
        if self.bamInputMode == "fragments" and (
            self.extendFrom5pBP is not None or self.inferFragmentLength
        ):
            raise ValueError("Template spans do not accept read extension")


def _cellIDs(values):
    if isinstance(values, str):
        raise TypeError("Cell selection must be a sequence of identifiers")
    identifiers = tuple(values)
    if any(
        not isinstance(value, str) or not value or "\0" in value or any(c.isspace() for c in value)
        for value in identifiers
    ):
        raise ValueError("Invalid cell identifier")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate selected cell identifier")
    return identifiers


@dataclass(frozen=True)
class SourceSpec:
    path: str
    sourceKind: str
    sampleID: str
    countOptions: CountOptions | None = None
    selection: Mapping | tuple | frozenset | None = None
    barcodeTag: str = "CB"
    matrixRows: str = "cells"

    def __post_init__(self):
        if self.sourceKind not in sourceKinds:
            raise ValueError(f"Unsupported sourceKind {self.sourceKind!r}")
        if not isinstance(self.sampleID, str) or not self.sampleID:
            raise ValueError("sampleID must be a nonempty string")
        object.__setattr__(self, "path", str(Path(self.path)))
        if "\0" in self.path:
            raise ValueError("Source paths cannot contain a null character")
        if self.countOptions is not None and not isinstance(self.countOptions, CountOptions):
            raise TypeError("countOptions must be CountOptions")
        if self.sourceKind in ("BAM", "FRAGMENTS"):
            options = self.countOptions or CountOptions()
            if self.sourceKind == "BAM" and options.fragmentsUseReadSupport:
                raise ValueError("BAM input does not accept fragmentsUseReadSupport")
            if self.sourceKind == "FRAGMENTS":
                defaults = CountOptions()
                for field in (
                    "bamInputMode", "flagExclude", "minMappingQuality",
                    "minTemplateLengthBP", "maxTemplateLengthBP",
                    "shiftForward5p", "shiftReverse5p", "extendFrom5pBP",
                    "inferFragmentLength",
                ):
                    if getattr(options, field) != getattr(defaults, field):
                        raise ValueError(f"FRAGMENTS input does not accept {field}")
                if options.countMode == "ffp":
                    raise ValueError("FRAGMENTS input has no read1 identity")
            object.__setattr__(self, "countOptions", options)
        elif self.countOptions is not None:
            raise ValueError(f"{self.sourceKind} input does not accept countOptions")
        if self.matrixRows not in ("cells", "pseudobulks"):
            raise ValueError("matrixRows must be cells or pseudobulks")
        if self.sourceKind != "H5AD" and self.matrixRows != "cells":
            raise ValueError("matrixRows applies to H5AD input")
        if (
            not isinstance(self.barcodeTag, str) or len(self.barcodeTag) != 2
            or not self.barcodeTag.isascii() or not self.barcodeTag.isalnum()
        ):
            raise ValueError("barcodeTag must contain two characters")
        if self.sourceKind == "BEDGRAPH" and self.selection is not None:
            raise ValueError("BEDGRAPH input has no cell selection")
        if isinstance(self.selection, Mapping):
            descriptor = dict(self.selection)
            if descriptor.get("groups") is not None:
                if isinstance(descriptor["groups"], str):
                    raise TypeError("groups must be a sequence of exact identifiers")
                descriptor["groups"] = tuple(descriptor["groups"])
            if descriptor.get("barcodes") is not None:
                descriptor["barcodes"] = _cellIDs(descriptor["barcodes"])
            object.__setattr__(self, "selection", MappingProxyType(descriptor))
        elif self.selection is not None:
            object.__setattr__(self, "selection", frozenset(_cellIDs(self.selection)))


def _metadataCatalog(selection):
    path = selection["path"]
    barcodeColumn = selection.get("barcodeColumn")
    sampleColumn = selection.get("sampleColumn")
    groupColumn = selection.get("groupColumn")
    if selection.get("groups") is not None and groupColumn is None:
        raise ValueError("Group selection requires groupColumn")
    catalog = {}
    identities = set()
    with ExitStack() as handles:
        if str(path).endswith(".h5ad"):
            import h5py
            from anndata.io import read_elem
            from pandas import DataFrame

            handle = handles.enter_context(h5py.File(path, "r"))
            if "obs" not in handle:
                raise ValueError("H5AD metadata requires obs")
            frame = read_elem(handle["obs"])
            if not isinstance(frame, DataFrame):
                raise ValueError("H5AD obs must be a cell metadata table")
            fields = tuple(frame.columns)
            if not frame.columns.is_unique:
                raise ValueError("Cell metadata requires unique columns")
            barcodeIndex = 0 if barcodeColumn is None else None
            offset = 1
            width = len(fields) + 1
            rows = frame.itertuples(index=True, name=None)
        else:
            opener = gzip.open if str(path).endswith(".gz") else open
            handle = handles.enter_context(opener(path, "rt", encoding="utf-8", newline=""))
            firstLine = handle.readline()
            handle.seek(0)
            rows = csv.reader(handle, delimiter="\t" if "\t" in firstLine else ",", strict=True)
            try:
                fields = tuple(next(rows))
            except StopIteration as error:
                raise ValueError("Cell metadata requires a column header") from error
            if not fields or len(set(fields)) != len(fields):
                raise ValueError("Cell metadata requires unique columns")
            if barcodeColumn is None:
                barcodeColumn = fields[0]
            barcodeIndex = None
            offset = 0
            width = len(fields)
        for column in (barcodeColumn, sampleColumn, groupColumn):
            if column is not None and column not in fields:
                raise ValueError(f"Missing metadata column {column!r}")
        if barcodeIndex is None:
            barcodeIndex = fields.index(barcodeColumn) + offset
        sampleIndex = None if sampleColumn is None else fields.index(sampleColumn) + offset
        groupIndex = None if groupColumn is None else fields.index(groupColumn) + offset
        for row in rows:
            if not row and offset == 0:
                continue
            if len(row) != width:
                raise ValueError("Cell metadata row does not match its columns")
            barcode = row[barcodeIndex]
            sample = row[sampleIndex] if sampleIndex is not None else None
            group = row[groupIndex] if groupIndex is not None else None
            if not isinstance(barcode, str) or not barcode or "\0" in barcode or any(c.isspace() for c in barcode):
                raise ValueError("Invalid cell identifier")
            if sampleColumn and (not isinstance(sample, str) or not sample or "\0" in sample):
                raise ValueError("Invalid metadata sample identifier")
            if groupColumn and (not isinstance(group, str) or not group or "\0" in group):
                raise ValueError("Invalid metadata group identifier")
            identity = (sample, barcode)
            if identity in identities:
                raise ValueError("Duplicate sample/barcode identity")
            identities.add(identity)
            catalog.setdefault(sample, {}).setdefault(group, set()).add(barcode)
    return catalog


def _groupMapCatalog(path):
    catalog = {}
    identities = set()
    with open(path, encoding="utf-8") as handle:
        for lineNumber, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"Invalid barcode group-map row {lineNumber}")
            barcode, group = fields
            if not barcode or any(c.isspace() for c in barcode) or not group:
                raise ValueError("Invalid barcode/group identifier")
            if barcode in identities:
                raise ValueError("Duplicate barcode in group map")
            identities.add(barcode)
            catalog.setdefault(group, set()).add(barcode)
    return catalog


def _allowCells(path):
    result = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            value = line.rstrip("\r\n")
            if not value or value.startswith("#"):
                continue
            if any(c.isspace() for c in value):
                raise ValueError("Invalid barcode in allow list")
            if value in result:
                raise ValueError("Duplicate barcode in allow list")
            result.add(value)
    return result


def resolveSelection(source):
    selection = source.selection
    if selection is None:
        return None
    if not isinstance(selection, Mapping):
        values = frozenset(selection)
    else:
        allowed = {
            "path", "barcodeColumn", "sampleColumn", "groupColumn",
            "sampleValue", "groups", "barcodes", "barcodeAllowListFile",
            "barcodeGroupMapFile",
        }
        unknown = set(selection) - allowed
        if unknown:
            raise ValueError(f"Unknown cell selection fields {sorted(unknown)}")
        explicit = selection.get("barcodes")
        path = selection.get("path")
        groupMap = selection.get("barcodeGroupMapFile")
        allowPath = selection.get("barcodeAllowListFile")
        if path is not None and groupMap is not None:
            raise ValueError("Cell metadata and a barcode group map cannot be combined")
        if path is None:
            if any(selection.get(field) is not None for field in (
                "barcodeColumn", "sampleColumn", "groupColumn"
            )):
                if source.sourceKind != "H5AD":
                    raise ValueError("Metadata selection requires a path")
                if groupMap is not None:
                    raise ValueError("Cell metadata and a barcode group map cannot be combined")
            values = None if explicit is None else frozenset(explicit)
            if groupMap is not None:
                groups = selection.get("groups")
                values = set()
                for group, barcodes in _groupMapCatalog(groupMap).items():
                    if groups is None or group in groups:
                        values.update(barcodes)
                if explicit is not None:
                    values.intersection_update(explicit)
                values = frozenset(values)
            elif selection.get("groups") is not None and source.sourceKind != "H5AD":
                raise ValueError("Group selection requires metadata or a group map")
        else:
            catalog = _metadataCatalog(selection)
            sampleValue = selection.get("sampleValue", source.sampleID) if selection.get("sampleColumn") else None
            selectedGroups = catalog.get(sampleValue, {})
            groups = selection.get("groups")
            values = set()
            for group, barcodes in selectedGroups.items():
                if groups is None or group in groups:
                    values.update(barcodes)
            if explicit is not None:
                values.intersection_update(explicit)
            values = frozenset(values)
        if allowPath is not None:
            allow = _allowCells(allowPath)
            values = frozenset(allow) if values is None else values.intersection(allow)
    if values is not None and any(
        not isinstance(value, str) or not value or "\0" in value or any(c.isspace() for c in value)
        for value in values
    ):
        raise ValueError("Invalid cell identifier")
    return values


def prepareIndex(path, sourceKind, *, targetPath=None, threads=1):
    from . import _native

    checkedInteger(threads, "threads", 1, 65535)
    if sourceKind not in ("BAM", "FRAGMENTS", "BEDGRAPH"):
        raise ValueError("Index preparation requires BAM, FRAGMENTS, or BEDGRAPH")
    if sourceKind == "BEDGRAPH" and targetPath is None:
        raise ValueError("BEDGRAPH indexing requires a separate targetPath")
    if sourceKind == "BEDGRAPH":
        sourcePath = Path(path).resolve()
        target = Path(targetPath).resolve()
        if sourcePath == target:
            raise ValueError("BEDGRAPH indexing requires a separate targetPath")
        if target.exists():
            raise FileExistsError("BEDGRAPH targetPath already exists")
    if sourceKind != "BEDGRAPH" and targetPath is not None:
        raise ValueError("BAM/FRAGMENTS index preparation uses its standard index path")
    return _native.indexSource(str(path), sourceKind, targetPath=targetPath, threads=threads)


def selectionGroups(metadata, sampleValue=None):
    if not isinstance(metadata, Mapping):
        raise ValueError("Group discovery requires a selection descriptor")
    groupMap = metadata.get("barcodeGroupMapFile")
    if groupMap is not None:
        if metadata.get("path") is not None:
            raise ValueError("Metadata and group-map inputs cannot be combined")
        catalog = _groupMapCatalog(groupMap)
    else:
        if metadata.get("path") is None or metadata.get("groupColumn") is None:
            raise ValueError("Group discovery requires a metadata path and groupColumn")
        if metadata.get("sampleColumn") is not None and sampleValue is None:
            raise ValueError("Group discovery requires an explicit sampleValue")
        sampleKey = sampleValue if metadata.get("sampleColumn") is not None else None
        catalog = _metadataCatalog(metadata).get(sampleKey, {})
    allowPath = metadata.get("barcodeAllowListFile")
    allowed = None if allowPath is None else _allowCells(allowPath)
    explicit = metadata.get("barcodes")
    if explicit is not None:
        explicit = _cellIDs(explicit)
        allowed = set(explicit) if allowed is None else allowed.intersection(explicit)
    return {
        group: frozenset(values if allowed is None else values.intersection(allowed))
        for group, values in catalog.items()
    }
