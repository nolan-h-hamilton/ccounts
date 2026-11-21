from __future__ import annotations

import csv
import gzip
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from .exceptions import InputError, OutputExistsError, SelectionError, SourceError
from .checks import checkedInteger, coordinateLimit, validIdentifier

countModes = (
    "coverage",
    "binOverlap",
    "conservedFractionalOverlap",
    "center",
    "fiveprime",
    "ffp",
)
sourceKinds = ("BAM", "FRAGMENTS", "BEDGRAPH", "H5AD")


def readChromSizes(path):
    result = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 2:
                raise SourceError(kind="file")
            name = fields[0]
            if name in result:
                raise InputError(kind="grid")
            length = int(fields[1])
            result[name] = checkedInteger(length, "chromosome length", 1)

    if not result:
        raise InputError(kind="grid")
    return result


@dataclass(frozen=True)
class GridSpec:
    """Chromosome lengths and a bin grid anchored at each chromosome's start.

    Attributes
    ----------
    chromosomeSizes : Mapping[str, int]
        Chromosome names mapped to lengths in bp. For instance, {"chr1": 248956422}.
    binSizeBP : int
        Genomic bin size in bp. NOTE: Bins at chromosome ends may be shorter.
    """

    chromosomeSizes: Mapping
    binSizeBP: int = 50

    def __post_init__(self):
        sizes = {}
        for name, value in self.chromosomeSizes.items():
            if not validIdentifier(name):
                raise InputError(kind="identifiers")
            sizes[name] = checkedInteger(value, "chromosome length", 1)
        if not sizes:
            raise InputError(kind="grid")

        object.__setattr__(self, "chromosomeSizes", MappingProxyType(sizes))
        object.__setattr__(
            self, "binSizeBP", checkedInteger(self.binSizeBP, "binSizeBP", 1)
        )

    @classmethod
    def fromSizesFile(cls, path, binSizeBP=50):
        return cls(readChromSizes(path), binSizeBP)

    def checkRegion(self, chromosome, startBP, endBP):
        if chromosome not in self.chromosomeSizes:
            raise SourceError(kind="missing")
        start = checkedInteger(startBP, "startBP")
        end = checkedInteger(endBP, "endBP")
        length = self.chromosomeSizes[chromosome]
        if start >= end or end > length or start % self.binSizeBP:
            raise InputError(kind="region")
        if end != length and end % self.binSizeBP:
            raise InputError(kind="region")
        return start, end

    def binWidths(self, chromosome, startBP, endBP):
        start, end = self.checkRegion(chromosome, startBP, endBP)
        count = (end - start + self.binSizeBP - 1) // self.binSizeBP
        widths = np.full(count, self.binSizeBP, dtype=np.float64)
        # adjusts the last bin if shorter than binSizeBP
        widths[-1] = end - start - (count - 1) * self.binSizeBP
        return widths


@dataclass(frozen=True)
class CountOptions:
    """Settings for read filtering, fragment handling, and genomic counting.

    Attributes
    ----------
    countMode : str
        How reads/fragments contribute to bins:

        - "coverage": sum read/fragment overlaps in bp, divided by bin width.
        - "binOverlap": count each read/fragment in every bin it overlaps.
        - "conservedFractionalOverlap" (default): each read/fragment contributes
          overlap bp / its length to each bin that it covers.
        - "center": count each read/fragment in the bin containing its midpoint.
        - "fiveprime": count each read's 5' end, or both ends of a fragment.
        - "ffp": count each BAM read's 5' end. In fragment mode, count only
          the first mate's 5' end. BAM only.

        Example with bamInputMode="fragments", weight 1, and no shifts.
        The first mate is forward, and the fragment spans [5, 25)::

            bp                          0         10        20        30
            bins                        |---------|---------|---------|
            fragment                         [===================)
            read1                            5'--->
            read2                                           <---5'

            countMode                    [0, 10)   [10, 20)  [20, 30)
            coverage                       0.5       1.0       0.5
            binOverlap                      1         1         1
            conservedFractionalOverlap     0.25      0.50      0.25
            center                          0         1         0
            fiveprime                       1         0         1
            ffp                             1         0         0

    bamInputMode : str
        How BAM records are used:

        - "auto" (default): use "fragments" for paired input and "reads" otherwise.
        - "reads": count reads separately, including both mates of a pair.
        - "read1": count only first mates with mapped partners.
        - "fragments": count each proper pair on the same chromosome as one
          fragment, using its template span (`TLEN`).

        Example with countMode="coverage", no shifts, and no read extension.
        Both reads in this proper pair pass the filters::

            bp                          0         10        20        30
            bins                        |---------|---------|---------|
            read1 (forward)                  [--->)
            read2 (reverse)                                 [<---)
            fragment (abs(TLEN)=20)          [===================)

            bamInputMode                 [0, 10)   [10, 20)  [20, 30)
            reads                          0.5        0        0.5
            read1                          0.5        0         0
            fragments                      0.5       1.0       0.5
            auto (paired)                  0.5       1.0       0.5

        For single-end input, "auto" uses "reads".

    flagExclude : int
        SAM flag mask used to exclude matching BAM records. See
        `Broad SAM flags widget <https://broadinstitute.github.io/picard/explain-flags.html>`_.
    minMappingQuality : int
        Minimum accepted BAM mapping quality.
    minTemplateLengthBP : int
        Minimum paired-fragment length in bp. -1 uses the
        median read length.
    maxTemplateLengthBP : int or None
        Maximum paired-fragment length in bp. None removes the limit.
    shiftForward5p : int
        Forward-strand BAM shift in bp, added to genomic coordinates.
    shiftReverse5p : int
        Reverse-strand BAM shift in bp, subtracted from genomic coordinates.
    extendFrom5pBP : int or None
        Read-extension length from the shifted 5' end. None disables extension.
    inferFragmentLength : bool
        Estimate a read-extension length when opening BAM sources in read mode.
    fragmentsUseReadSupport : bool
        Weight fragment-table records by their fifth-column read-support count.
        False assigns each record weight one.
    """

    countMode: str = "conservedFractionalOverlap"
    bamInputMode: str = "auto"
    flagExclude: int = 3844
    minMappingQuality: int = 20
    minTemplateLengthBP: int = 0
    maxTemplateLengthBP: int | None = None
    shiftForward5p: int = 0
    shiftReverse5p: int = 0
    extendFrom5pBP: int | None = None
    inferFragmentLength: bool = False
    fragmentsUseReadSupport: bool = False

    def __post_init__(self):
        if self.countMode not in countModes:
            raise InputError(kind="option", context="countMode")
        if self.bamInputMode not in ("auto", "reads", "read1", "fragments"):
            raise InputError(kind="option", context="bamInputMode")
        for field, maximum in (("flagExclude", 65535), ("minMappingQuality", 255)):
            object.__setattr__(
                self, field, checkedInteger(getattr(self, field), field, 0, maximum)
            )
        object.__setattr__(
            self,
            "minTemplateLengthBP",
            checkedInteger(self.minTemplateLengthBP, "minTemplateLengthBP", -1),
        )
        if self.maxTemplateLengthBP is not None:
            maximum = checkedInteger(self.maxTemplateLengthBP, "maxTemplateLengthBP", 1)
            if maximum < self.minTemplateLengthBP:
                raise InputError()
            object.__setattr__(self, "maxTemplateLengthBP", maximum)
        for field in ("shiftForward5p", "shiftReverse5p"):
            object.__setattr__(
                self,
                field,
                checkedInteger(getattr(self, field), field, -coordinateLimit),
            )
        if self.extendFrom5pBP is not None:
            object.__setattr__(
                self,
                "extendFrom5pBP",
                checkedInteger(self.extendFrom5pBP, "extendFrom5pBP", 1),
            )
        for field in ("inferFragmentLength", "fragmentsUseReadSupport"):
            if type(getattr(self, field)) is not bool:
                raise InputError(kind="option", context=field)
        if self.inferFragmentLength and self.extendFrom5pBP is not None:
            raise InputError()
        if self.bamInputMode == "fragments" and (
            self.extendFrom5pBP is not None or self.inferFragmentLength
        ):
            raise InputError()


def _cellIDs(values):
    if isinstance(values, str):
        raise SelectionError()
    identifiers = tuple(values)
    if any(not validIdentifier(value) for value in identifiers):
        raise InputError(kind="identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise InputError(kind="identifiers")
    return identifiers


@dataclass(frozen=True)
class SourceSpec:
    """Encompasses a specification for a data source (file and options)

    An input file, its sample identity, counting options, and cell selection.

    Attributes
    ----------
    path : str
        Path to the input file.
    sourceKind : str
        File format: "BAM", "FRAGMENTS", "BEDGRAPH", or "H5AD".
    sampleID : str
        Identifier for the source, unique among sources in a Reader.
    countOptions : CountOptions or None
        Counting settings for BAM or FRAGMENTS. None uses default settings.
        Must be None for BEDGRAPH and H5AD.
    selection : Mapping, iterable of str, or None
        Cell barcodes or a metadata selection descriptor.
        None applies no cell filter.
    barcodeTag : str
        BAM tag containing cell barcodes, such as "CB".
    matrixRows : str
        Interpret H5AD rows as "cells" or "pseudobulks".
        A pseudobulk source may select at most one row.
    """

    path: str
    sourceKind: str
    sampleID: str
    countOptions: CountOptions | None = None
    selection: Mapping | tuple | frozenset | None = None
    barcodeTag: str = "CB"
    matrixRows: str = "cells"

    def __post_init__(self):
        if self.sourceKind not in sourceKinds:
            raise InputError(kind="option", context="sourceKind")
        if not isinstance(self.sampleID, str) or not self.sampleID:
            raise InputError(kind="identifiers")
        object.__setattr__(self, "path", str(Path(self.path)))
        if "\0" in self.path:
            raise SourceError()
        if self.sourceKind in ("BAM", "FRAGMENTS"):
            options = CountOptions() if self.countOptions is None else self.countOptions
            if self.sourceKind == "BAM" and options.fragmentsUseReadSupport:
                raise InputError()
            if self.sourceKind == "FRAGMENTS":
                defaults = CountOptions()
                for field in (
                    "bamInputMode",
                    "flagExclude",
                    "minMappingQuality",
                    "minTemplateLengthBP",
                    "maxTemplateLengthBP",
                    "shiftForward5p",
                    "shiftReverse5p",
                    "extendFrom5pBP",
                    "inferFragmentLength",
                ):
                    if getattr(options, field) != getattr(defaults, field):
                        raise InputError(kind="option", context=field)
                if options.countMode == "ffp":
                    raise InputError()
            object.__setattr__(self, "countOptions", options)
        elif self.countOptions is not None:
            raise InputError()
        if self.matrixRows not in ("cells", "pseudobulks"):
            raise InputError(kind="option", context="matrixRows")
        if self.sourceKind != "H5AD" and self.matrixRows != "cells":
            raise InputError()
        if (
            not isinstance(self.barcodeTag, str)
            or len(self.barcodeTag) != 2
            or not self.barcodeTag.isascii()
            or not self.barcodeTag.isalnum()
        ):
            raise InputError(kind="option", context="barcodeTag")
        if self.sourceKind == "BEDGRAPH" and self.selection is not None:
            raise InputError(kind="selection")

        if isinstance(self.selection, Mapping):
            descriptor = dict(self.selection)
            if descriptor.get("groups") is not None:
                if isinstance(descriptor["groups"], str):
                    raise SelectionError()
                descriptor["groups"] = tuple(descriptor["groups"])
            if descriptor.get("barcodes") is not None:
                descriptor["barcodes"] = _cellIDs(descriptor["barcodes"])
            object.__setattr__(self, "selection", MappingProxyType(descriptor))
        elif self.selection is not None:
            object.__setattr__(self, "selection", frozenset(_cellIDs(self.selection)))


def _metadataCatalog(selection, h5adObs=None):
    path = selection["path"]
    barcodeColumn = selection.get("barcodeColumn")
    sampleColumn = selection.get("sampleColumn")
    groupColumn = selection.get("groupColumn")
    if selection.get("groups") is not None and groupColumn is None:
        raise InputError(kind="selection")

    catalog = {}
    identities = set()
    with ExitStack() as handles:
        if str(path).endswith(".h5ad"):
            import h5py
            from anndata.io import read_elem

            if h5adObs is None:
                handle = handles.enter_context(h5py.File(path, "r"))
                if "obs" not in handle:
                    raise SourceError(kind="metadata")
                frame = read_elem(handle["obs"])
            else:
                frame = h5adObs
            fields = tuple(frame.columns)
            if not frame.columns.is_unique:
                raise SourceError(kind="metadata")
            barcodeIndex = 0 if barcodeColumn is None else None
            offset = 1
            width = len(fields) + 1
            rows = frame.itertuples(index=True, name=None)
        else:
            opener = gzip.open if str(path).endswith(".gz") else open
            handle = handles.enter_context(
                opener(path, "rt", encoding="utf-8", newline="")
            )
            firstLine = handle.readline()
            handle.seek(0)
            rows = csv.reader(
                handle, delimiter="\t" if "\t" in firstLine else ",", strict=True
            )
            fields = tuple(next(rows))
            if not fields or len(set(fields)) != len(fields):
                raise SourceError(kind="metadata")
            if barcodeColumn is None:
                barcodeColumn = fields[0]
            barcodeIndex = None
            offset = 0
            width = len(fields)

        for column in (barcodeColumn, sampleColumn, groupColumn):
            if column is not None and column not in fields:
                raise SourceError(kind="metadata")
        if barcodeIndex is None:
            barcodeIndex = fields.index(barcodeColumn) + offset
        sampleIndex = (
            None if sampleColumn is None else fields.index(sampleColumn) + offset
        )
        groupIndex = None if groupColumn is None else fields.index(groupColumn) + offset

        for row in rows:
            if not row and offset == 0:
                continue
            if len(row) != width:
                raise SourceError(kind="metadata")
            barcode = row[barcodeIndex]
            sample = row[sampleIndex] if sampleIndex is not None else None
            group = row[groupIndex] if groupIndex is not None else None
            if not validIdentifier(barcode):
                raise InputError(kind="identifiers")
            if sampleColumn and (
                not isinstance(sample, str) or not sample or "\0" in sample
            ):
                raise SourceError(kind="metadata")
            if groupColumn and (
                not isinstance(group, str) or not group or "\0" in group
            ):
                raise SourceError(kind="metadata")

            identity = (sample, barcode)
            if identity in identities:
                raise SourceError(kind="metadata")
            identities.add(identity)
            catalog.setdefault(sample, {}).setdefault(group, set()).add(barcode)

    return catalog


def _groupMapCatalog(path):
    catalog = {}
    identities = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise SourceError(kind="metadata")
            barcode, group = fields
            if not validIdentifier(barcode) or not group or "\0" in group:
                raise InputError(kind="identifiers")
            if barcode in identities:
                raise InputError(kind="identifiers")

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
            if not validIdentifier(value):
                raise InputError(kind="identifiers")
            if value in result:
                raise InputError(kind="identifiers")

            result.add(value)

    return result


def resolveSelection(source, *, h5adObs=None):
    """Determine which cell barcodes to keep for a source input

    Takes the union of the requested groups, and intersects with explicit
    barcodes and the allow-list if given.
    """
    selection = source.selection
    if selection is None:
        return None
    if not isinstance(selection, Mapping):
        # case: barcodes given directly
        values = frozenset(selection)
    else:
        allowed = {
            "path",
            "barcodeColumn",
            "sampleColumn",
            "groupColumn",
            "sampleValue",
            "groups",
            "barcodes",
            "barcodeAllowListFile",
            "barcodeGroupMapFile",
        }

        # check for unknown/extra fields
        unknown = set(selection) - allowed
        if unknown:
            raise InputError(kind="selection")

        explicit = selection.get("barcodes")
        path = selection.get("path")
        groupMap = selection.get("barcodeGroupMapFile")
        allowPath = selection.get("barcodeAllowListFile")

        # use either a metadata file or a barcode/group map
        if path is not None and groupMap is not None:
            raise InputError()

        if path is None:
            # for H5AD, read these columns from `obs` table if available
            if any(
                selection.get(field) is not None
                for field in ("barcodeColumn", "sampleColumn", "groupColumn")
            ):
                if source.sourceKind != "H5AD":
                    raise InputError(kind="selection")
                if groupMap is not None:
                    raise InputError()

            values = None if explicit is None else frozenset(explicit)
            if groupMap is not None:
                # join the requested groups, or all groups when groups is None
                groups = selection.get("groups")
                values = set()
                for group, barcodes in _groupMapCatalog(groupMap).items():
                    if groups is None or group in groups:
                        values.update(barcodes)
                if explicit is not None:
                    values.intersection_update(explicit)
                values = frozenset(values)
            elif selection.get("groups") is not None and source.sourceKind != "H5AD":
                raise InputError(kind="selection")
        else:
            # catalog[sample][group] = set of barcodes
            sharedObs = (
                h5adObs
                if h5adObs is not None
                and Path(path).resolve() == Path(source.path).resolve()
                else None
            )
            catalog = _metadataCatalog(selection, sharedObs)
            # without sampleColumn, all rows use the None sample key
            sampleValue = (
                selection.get("sampleValue", source.sampleID)
                if selection.get("sampleColumn")
                else None
            )
            selectedGroups = catalog.get(sampleValue, {})
            groups = selection.get("groups")
            values = set()
            # join the requested groups for this sample (None means all groups)
            for group, barcodes in selectedGroups.items():
                if groups is None or group in groups:
                    values.update(barcodes)
            if explicit is not None:
                values.intersection_update(explicit)
            values = frozenset(values)

        if allowPath is not None:
            # use the allow-list alone if values is None, otherwise intersect
            allow = _allowCells(allowPath)
            values = frozenset(allow) if values is None else values.intersection(allow)

    return values


def prepareIndex(path, sourceKind, *, targetPath=None, threads=1):
    from . import _native

    checkedInteger(threads, "threads", 1, 65535)
    if sourceKind not in ("BAM", "FRAGMENTS", "BEDGRAPH"):
        raise SourceError()
    if sourceKind == "BEDGRAPH" and targetPath is None:
        raise InputError()
    if sourceKind == "BEDGRAPH":
        sourcePath = Path(path).resolve()
        target = Path(targetPath).resolve()
        if sourcePath == target:
            raise InputError()
        if target.exists():
            raise OutputExistsError()
    if sourceKind != "BEDGRAPH" and targetPath is not None:
        raise InputError()
    return _native.indexSource(
        str(path), sourceKind, targetPath=targetPath, threads=threads
    )


def selectionGroups(metadata, sampleValue=None):
    """Return group --> frozenset of barcodes, with any sample/barcode filters applied."""
    groupMap = metadata.get("barcodeGroupMapFile")
    if groupMap is not None:
        if metadata.get("path") is not None:
            raise InputError()
        catalog = _groupMapCatalog(groupMap)
    else:
        if metadata.get("path") is None or metadata.get("groupColumn") is None:
            raise InputError(kind="selection")
        if metadata.get("sampleColumn") is not None and sampleValue is None:
            raise InputError(kind="selection")
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
