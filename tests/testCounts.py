from dataclasses import replace
from pathlib import Path
import gzip
import math
from shutil import copyfile
from unittest.mock import patch
import h5py
import numpy as np
import pysam
import pytest
import ccounts as cc
from ccounts import _native


@pytest.mark.parametrize(
    "mode,values,mass",
    [
        ("coverage", [2, 2, 4 / 3], [1.1, 2, 2 / 3]),
        ("binOverlap", [5, 2, 3], [5, 2, 3]),
        (
            "conservedFractionalOverlap",
            [19 / 6, 5 / 3, 7 / 6],
            [217 / 72, 25 / 18, 73 / 72],
        ),
        ("center", [3, 2, 1], [3, 2, 1]),
        ("fiveprime", [8, 0, 4], [14, 0, 6]),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def testFixtureCountsAndNormalization(countSources, mode, values, mass, dtype):
    grid = cc.GridSpec.fromSizesFile(countSources["sizes"], binSizeBP=50)
    sources = [
        cc.SourceSpec(
            countSources[path],
            kind,
            name,
            cc.CountOptions(countMode=mode, **options),
            selection=("A",),
        )
        for path, kind, name, options in (
            ("eventsBAM", "BAM", "bam", {"bamInputMode": "fragments"}),
            ("events", "FRAGMENTS", "fragments", {"fragmentsUseReadSupport": True}),
        )
    ]

    with cc.openSources(sources, grid, dtype=dtype) as reader:
        whole = reader.readRegion("chr1", 0, 115, returnRawNoiseMass=True)

        np.testing.assert_allclose(whole.values, [values] * 2, rtol=2e-7)
        np.testing.assert_allclose(whole.rawNoiseMass, [mass] * 2, rtol=2e-7)
        assert whole.values.dtype == np.dtype(dtype)

        totals = reader.libraryTotals(chromosomes=["chr1"])

        assert reader.libraryTotals(chromosomes=["chr1"]) is totals

        eventTotal = 12 if mode == "fiveprime" else 6

        for record in totals.records.values():
            assert (record["eventMass"], record["spanBP"], record["usedMethod"]) == (
                eventTotal,
                220,
                "exact",
            )

        for width in (1, 2):
            pieces = list(
                reader.iterBlocks(
                    ["chr1"], blockSizeBins=width, returnRawNoiseMass=True
                )
            )

            for field in ("values", "rawNoiseMass"):
                np.testing.assert_array_equal(
                    np.concatenate([getattr(p, field) for p in pieces], 1),
                    getattr(whole, field),
                )

        factors = {
            "CPM": 1e6 / eventTotal,
            "RPKM": None if mode == "coverage" else 1e9 / eventTotal / 50,
            "RPGC": {
                "coverage": 115 / 220,
                "conservedFractionalOverlap": 115 / eventTotal / 50,
            }.get(mode),
        }

        for method, factor in factors.items():
            args = (
                {
                    "effectiveGenomeSizeBP": 115,
                    "effectiveGenomeSizeChromosomes": ("chr1",),
                }
                if method == "RPGC"
                else {}
            )

            if factor is None:
                with pytest.raises(ValueError):
                    cc.Normalization.fromTotals(totals, method=method, **args)
                continue

            norm = cc.Normalization.fromTotals(totals, method=method, **args)

            with patch.object(
                norm, "_checkBlock", wraps=norm._checkBlock
            ) as checkBlock:
                np.testing.assert_allclose(norm.apply(whole), whole.values * factor)
                checkBlock.assert_called_once_with(whole)

            terminal = reader.readRegion("chr1", 100, 115)

            np.testing.assert_allclose(
                norm.apply(terminal), whole.values[:, -1:] * factor
            )

            with pytest.raises(ValueError):
                norm.apply(replace(whole, countModes=("ffp", "ffp")))

        np.testing.assert_array_equal(
            cc.Normalization.fromFactors(whole.sampleIDs, [2, 3]).apply(whole),
            whole.values * np.array([[2], [3]], dtype=dtype),
        )

        distinct = replace(
            whole, values=whole.values * np.array([[1], [2]], dtype=dtype)
        )
        reordered = cc.Normalization.fromFactors(whole.sampleIDs[::-1], [2, 3])
        expected = distinct.values[::-1] * np.array([[2], [3]], dtype=dtype)
        np.testing.assert_array_equal(reordered.apply(distinct), expected)
        with pytest.raises(ValueError, match="overlap"):
            reordered.apply(distinct, out=distinct.values)

        supplied = cc.Normalization.fromFactors(whole.sampleIDs, [2, 3])
        expected = distinct.values * np.array([[2], [3]], dtype=dtype)
        assert supplied.apply(distinct, out=distinct.values) is distinct.values
        np.testing.assert_array_equal(distinct.values, expected)


@pytest.mark.parametrize("compressed,scaleTo", [(False, None), (True, 35.0)])
def testNormalizationFromRegions(countSources, tmp_path, compressed, scaleTo):
    grid = cc.GridSpec.fromSizesFile(countSources["sizes"], binSizeBP=50)
    sources = [
        cc.SourceSpec(
            countSources["eventsBAM"],
            "BAM",
            "bam",
            cc.CountOptions(countMode="coverage", bamInputMode="fragments"),
            selection=("A",),
        ),
        cc.SourceSpec(
            countSources["events"],
            "FRAGMENTS",
            "fragments",
            cc.CountOptions(countMode="center", fragmentsUseReadSupport=True),
            selection=("B",),
        ),
    ]
    regions = (
        "# reference regions\nchr1\t60\t61\tpeak\n\n"
        "chr1\t34\t45\nchr1\t33\t35\nchr1\t104\t105\n"
    )
    if compressed:
        regionPath = tmp_path / "reference.bed.gz"
        with gzip.open(regionPath, "wt", encoding="utf-8") as handle:
            handle.write(regions)
    else:
        regionPath = tmp_path / "reference.bed"
        regionPath.write_text(regions, encoding="utf-8")

    args = {} if scaleTo is None else {"scaleTo": scaleTo}
    with cc.openSources(sources, grid, dtype="float64") as reader:
        block = reader.readRegion("chr1", 0, 115)
        sampleIDs = block.sampleIDs[::-1]
        norm = cc.Normalization.fromRegions(
            reader, regionPath, sampleIDs=sampleIDs, **args
        )

        # BAM contributes at 34 and 104. The selected fragment contributes at 60.
        expected = (10_000.0 if scaleTo is None else scaleTo) / np.array([7.0, 5.0])
        assert norm.method == "ReadsInRegions"
        assert norm.sampleIDs == sampleIDs
        np.testing.assert_array_equal(norm.sourceFactors, expected)
        np.testing.assert_allclose(
            norm.apply(block), block.values[::-1] * expected[:, None]
        )

        empty = tmp_path / "noInsertions.bed"
        empty.write_text("chr1\t61\t62\n", encoding="utf-8")
        with pytest.raises(cc.EstimateError, match="No endpoints.*bam"):
            cc.Normalization.fromRegions(reader, empty)


def testNormalizationFromSeparatedRegions(tmp_path):
    chromosome = "chr1"
    length = 2_600_000
    grid = cc.GridSpec({chromosome: length}, binSizeBP=1)
    bamPath = tmp_path / "separated.bam"
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": chromosome, "LN": length}],
    }
    with pysam.AlignmentFile(bamPath, "wb", header=header) as handle:
        for name, flag, start, mate, span in (
            ("longPair", 99, 50, 799_941, 799_901),
            ("gapRead", 0, 300_000, -1, 0),
            ("longPair", 147, 799_941, 50, -799_901),
            ("reverseRead", 16, 800_041, -1, 0),
            ("distantRead", 0, 2_500_000, -1, 0),
        ):
            record = pysam.AlignedSegment()
            record.query_name = name
            record.query_sequence = "A" * 10
            record.flag = flag
            record.reference_id = 0
            record.reference_start = start
            record.mapping_quality = 60
            record.cigar = [(0, 10)]
            record.next_reference_id = 0 if mate >= 0 else -1
            record.next_reference_start = mate
            record.template_length = span
            handle.write(record)
    pysam.index(str(bamPath))

    fragmentsPath = tmp_path / "separated.tsv.gz"
    raw = tmp_path / "separated.tsv"
    raw.write_text(
        "chr1\t100\t800001\tA\t3\nchr1\t300000\t300010\tA\t11\n"
        "chr1\t800001\t800002\tA\t2\n",
        encoding="utf-8",
    )
    pysam.tabix_compress(str(raw), str(fragmentsPath))
    pysam.tabix_index(str(fragmentsPath), preset="bed")

    regions = ((99, 102), (800_000, 800_002), (2_500_050, 2_500_051))
    bedRegions = (regions[1], (99, 101), (100, 102), regions[1], regions[2])
    regionPath = tmp_path / "separated.bed"
    regionPath.write_text(
        "".join(f"{chromosome}\t{start}\t{end}\n" for start, end in bedRegions),
        encoding="utf-8",
    )

    sources = [
        cc.SourceSpec(
            bamPath,
            "BAM",
            mode,
            cc.CountOptions(
                countMode="coverage",
                bamInputMode=mode,
                shiftForward5p=50,
                shiftReverse5p=50,
            ),
        )
        for mode in ("reads", "fragments")
    ]
    sources.append(
        cc.SourceSpec(
            fragmentsPath,
            "FRAGMENTS",
            "weighted",
            cc.CountOptions(countMode="coverage", fragmentsUseReadSupport=True),
        )
    )
    endpointSources = [
        replace(source, countOptions=replace(source.countOptions, countMode="fiveprime"))
        for source in sources
    ]
    with cc.openSources(endpointSources, grid, dtype="float64") as reader:
        expected = np.column_stack(
            [
                reader.readRegion(chromosome, start, end).values.sum(axis=1)
                for start, end in regions
            ]
        )
    np.testing.assert_array_equal(expected, [[1, 1, 1], [1, 1, 0], [3, 7, 0]])
    with cc.openSources(sources, grid, dtype="float64") as reader:
        for _ in range(2):
            norm = cc.Normalization.fromRegions(reader, regionPath, scaleTo=30)
            masses = [record["regionEndpointMass"] for record in norm.records.values()]
            np.testing.assert_array_equal(masses, expected.sum(axis=1))


def testRPGCTerritory():
    grid = cc.GridSpec({"chr1": 100, "chr2": 200, "chr3": 300})
    record = {
        "unit": "meanCoverage",
        "sourceKind": "BAM",
        "countMode": "coverage",
        "eventMass": 10,
        "spanBP": 100,
    }
    totals = cc.LibraryTotals({"a": record, "b": record}, grid, ("chr1", "chr2"))
    norm = cc.Normalization.fromTotals(
        totals,
        method="RPGC",
        effectiveGenomeSizeBP={"b": 200, "a": 300},
        effectiveGenomeSizeChromosomes=("chr2", "chr1"),
    )
    np.testing.assert_array_equal(norm.sourceFactors, [3, 2])

    for chromosomes in (
        None,
        ("chr1",),
        ("chr1", "chr3"),
        ("chr1", "chr2", "chr2"),
        tuple(grid.chromosomeSizes),
    ):
        with pytest.raises(cc.InputError, match="effectiveGenomeSizeChromosomes"):
            cc.Normalization.fromTotals(
                totals,
                method="RPGC",
                effectiveGenomeSizeBP=600,
                effectiveGenomeSizeChromosomes=chromosomes,
            )

    with pytest.raises(cc.InputError, match="effectiveGenomeSizeBP"):
        cc.Normalization.fromTotals(
            totals, method="RPGC", effectiveGenomeSizeChromosomes=totals.chromosomes
        )

    with pytest.raises(cc.InputError):
        cc.Normalization.fromTotals(
            totals, method="CPM", effectiveGenomeSizeChromosomes=totals.chromosomes
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testNormalizationFactorPrecision(dtype):
    values = np.array([[2**100, -(2**100)], [2**-100, -(2**-100)]], dtype=dtype)
    expected = np.array([[2**-50, -(2**-50)], [2**50, -(2**50)]], dtype=dtype)
    block = cc.CountBlock(
        values,
        ("a", "b"),
        cc.GridSpec({"chr1": 2}, 1),
        "chr1",
        0,
        2,
        ("eventMass", "eventMass"),
        ("center", "center"),
    )
    normalization = cc.Normalization.fromFactors(block.sampleIDs, [2**-150, 2**150])

    for target in (None, np.empty_like(values), values):
        result = normalization.apply(block, out=target)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == values.dtype
        if target is not None:
            assert result is target


@pytest.mark.parametrize("mode", ["coverage", "binOverlap"])
@pytest.mark.parametrize(
    "name,length,expected",
    [
        ("cancellation", 200, [2**54] * 10 + [1] * 5 + [0] * 5),
        ("precision", 115, [float(2**54 + 3)] * 12),
    ],
)
def testLargeFixtureWeights(countSources, mode, name, length, expected):
    source = cc.SourceSpec(
        countSources[name],
        "FRAGMENTS",
        "s",
        cc.CountOptions(countMode=mode, fragmentsUseReadSupport=True),
    )

    with cc.openSources(
        [source], cc.GridSpec({"chr1": length}, 10), dtype="float64"
    ) as reader:
        whole = reader.readRegion("chr1", 0, length, returnRawNoiseMass=True)

        for field in ("values", "rawNoiseMass"):
            np.testing.assert_array_equal(getattr(whole, field), [expected])

            pieces = list(
                reader.iterBlocks(["chr1"], blockSizeBins=3, returnRawNoiseMass=True)
            )

            np.testing.assert_array_equal(
                np.concatenate([getattr(p, field) for p in pieces], 1), [expected]
            )


@pytest.mark.parametrize(
    "path,kind,expected",
    [
        ("eventsBAM", "BAM", (5, 115)),
        ("events", "FRAGMENTS", (5, 115)),
        ("bedGraph", "BEDGRAPH", (10, 105)),
    ],
)
def testSourceRange(countSources, tmp_path, path, kind, expected):
    sourcePath = countSources[path]
    if kind == "BEDGRAPH":
        sourcePath = cc.prepareIndex(
            sourcePath, kind, targetPath=tmp_path / "signal.bg.gz"
        )
    source = cc.SourceSpec(
        sourcePath, kind, "s", selection=None if kind == "BEDGRAPH" else ("B",)
    )

    with cc.openSources(
        [source], cc.GridSpec({"chr1": 115, "chrMissing": 115})
    ) as reader:
        assert reader.sourceRange("s", "chr1", flagExclude=3844) == expected
        if kind == "BAM":
            with pytest.raises(ValueError):
                reader.sourceRange("s", "chrMissing", flagExclude=3844)
        else:
            assert reader.sourceRange("s", "chrMissing", flagExclude=3844) == (0, 0)


@pytest.mark.parametrize("kind", ["BAM", "FRAGMENTS"])
@pytest.mark.parametrize("chromosomeLength", [120, 2000000])
def testSourceRangeNestedIntervals(tmp_path, kind, chromosomeLength):
    if kind == "BAM":
        path = tmp_path / "nested.bam"
        header = {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": chromosomeLength}],
        }
        with pysam.AlignmentFile(path, "wb", header=header) as handle:
            for index, (start, end) in enumerate(((10, 100), (20, 30))):
                record = pysam.AlignedSegment()
                record.query_name = f"read{index}"
                record.query_sequence = "A" * (end - start)
                record.reference_id = 0
                record.reference_start = start
                record.mapping_quality = 60
                record.cigartuples = [(0, end - start)]
                handle.write(record)
        pysam.index(str(path))
    else:
        raw = tmp_path / "nested.tsv"
        raw.write_text("chr1\t10\t100\tA\t1\nchr1\t20\t30\tA\t1\n")
        path = tmp_path / "nested.tsv.gz"
        pysam.tabix_compress(str(raw), str(path))
        pysam.tabix_index(str(path), preset="bed")

    source = cc.SourceSpec(path, kind, "s")
    with cc.openSources([source], cc.GridSpec({"chr1": chromosomeLength}, 10)) as reader:
        assert reader.sourceRange("s", "chr1", flagExclude=3844) == (10, 100)


@pytest.mark.parametrize("warmup", ["partial", "filtered", "reads", "totals"])
def testUnboundedFragmentQueries(tmp_path, warmup):
    sizes = {"chrShort": 4000, "chrLong": 40000, "chrEmpty": 4000}
    path = tmp_path / "fragments.bam"
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in sizes.items()],
    }
    records = []

    for tid, (length, span) in enumerate(((4000, 600), (40000, 8000))):
        for index, (left, width, reverse, mapQ, extraFlags) in enumerate(
            (
                (100, 30, False, 60, 0),
                (0, span, True, 0, 1024),
                (length // 5 - 100, span, False, 0, 1024),
                (length // 2 + 100, span, True, 0, 1024),
                (length - span, span, False, 0, 1024),
            )
        ):
            right = left + width - 10
            mates = (
                ((83, right, left, -width), (163, left, right, width))
                if reverse
                else ((99, left, right, width), (147, right, left, -width))
            )
            for flag, position, mate, templateLength in mates:
                record = pysam.AlignedSegment()
                record.query_name = f"pair{tid}_{index}"
                record.query_sequence = "A" * 10
                record.flag = flag | extraFlags
                record.reference_id = tid
                record.reference_start = position
                record.mapping_quality = mapQ
                record.cigar = [(0, 10)]
                record.next_reference_id = tid
                record.next_reference_start = mate
                record.template_length = templateLength
                records.append(record)

    with pysam.AlignmentFile(path, "wb", header=header) as handle:
        for record in sorted(
            records, key=lambda item: (item.reference_id, item.reference_start)
        ):
            handle.write(record)
    pysam.index(str(path))

    native = _native.NativeSource(path, "BAM", sizes)
    reference = _native.NativeSource(path, "BAM", sizes)
    filtered = cc.CountOptions(
        bamInputMode="fragments", minTemplateLengthBP=20, maxTemplateLengthBP=50
    )

    try:
        if warmup == "partial":
            native.countRegion("chrShort", 2800, 2900, 100, filtered)
        elif warmup == "totals":
            native.totalsRegion("chrShort", 0, 4000, filtered)
        else:
            options = (
                replace(filtered, bamInputMode="reads")
                if warmup == "reads"
                else filtered
            )
            native.countRegion("chrShort", 0, 4000, 100, options)

        # Longer fragments become eligible when the quality, flag, and length filters change.
        for chromosome, length in sizes.items():
            for mode in cc.countModes:
                for maximum, shift in ((None, 0), (None, 150), (None, -150), (50, 0)):
                    options = cc.CountOptions(
                        bamInputMode="fragments",
                        countMode=mode,
                        flagExclude=0,
                        minMappingQuality=0,
                        maxTemplateLengthBP=maximum,
                        shiftForward5p=shift,
                        shiftReverse5p=shift,
                    )
                    expected = reference.countRegion(
                        chromosome, 0, length, 100, options, True
                    )
                    pieces = [
                        native.countRegion(
                            chromosome, start, start + 1000, 100, options, True
                        )
                        for start in range(0, length, 1000)
                    ]

                    for field in (0, 1):
                        np.testing.assert_allclose(
                            np.concatenate([piece[field] for piece in pieces]),
                            expected[field],
                            rtol=1e-13,
                            atol=1e-13,
                        )
    finally:
        native.close()
        reference.close()


def testFixtureIndexesAndSelection(countSources, tmp_path):
    with cc.openSources(
        [
            cc.SourceSpec(
                countSources["eventsBAM"],
                "BAM",
                "s",
                cc.CountOptions(countMode="coverage", bamInputMode="reads"),
            )
        ],
        cc.GridSpec({"chrTotals": 115}),
        dtype="float64",
    ) as reader:
        assert reader.getSourceInfo("s", inspectLengths=True)["readLength"] == 10

        totals = reader.libraryTotals(chromosomes=["chrTotals"])

        assert (totals.records["s"]["eventMass"], totals.records["s"]["spanBP"]) == (
            7,
            26,
        )
        np.testing.assert_allclose(
            reader.readRegion("chrTotals", 0, 115).values @ [50, 50, 15], [26]
        )

    raw = Path(countSources["bedGraph"])
    target = tmp_path / "signal.bg.gz"

    assert cc.prepareIndex(raw, "BEDGRAPH", targetPath=target) == str(target)

    with cc.openSources(
        [cc.SourceSpec(target, "BEDGRAPH", "s")], cc.GridSpec({"chr1": 105})
    ) as reader:
        for start, end, expected in (
            (0, 105, [[1, 0, 3]]),
            (100, 105, [[3]]),
            (0, 50, [[1]]),
        ):
            np.testing.assert_array_equal(
                reader.readRegion("chr1", start, end).values, expected
            )

    with pytest.raises(FileExistsError):
        cc.prepareIndex(raw, "BEDGRAPH", targetPath=target)

    unindexed = tmp_path / "unindexed.bam"
    copyfile(countSources["eventsBAM"], unindexed)

    with pytest.raises(ValueError, match="index"):
        cc.openSources(
            [cc.SourceSpec(unindexed, "BAM", "s")], cc.GridSpec({"chr1": 115})
        )

    metadata = Path(__file__).parent / "data" / "cells.tsv"
    selection = {
        "path": str(metadata),
        "barcodeColumn": "cell",
        "sampleColumn": "sample",
        "groupColumn": "group",
        "groups": ("y",),
    }
    source = cc.SourceSpec(
        countSources["barcodes"],
        "FRAGMENTS",
        "s",
        cc.CountOptions(countMode="center"),
        selection,
    )

    with cc.openSources([source], cc.GridSpec({"chr1": 150})) as reader:
        np.testing.assert_array_equal(
            reader.readRegion("chr1", 0, 150).values, [[0, 1, 0]]
        )
        assert reader.sourceInfo["s"]["selectedCellCount"] == 1


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def testH5ADSelectionAndTotals(storage):
    path = Path(__file__).parent / "data" / f"counts-{storage}.h5ad"
    allow = Path(__file__).parent / "data" / "allow.txt"
    grid = cc.GridSpec({"chr1": 115, "chr2": 115})

    for barcodes, cellCount, expected in (
        (("A", "B"), 1, [3, 1, 2]),
        (("B",), 0, [0, 0, 0]),
    ):
        selection = {
            "barcodeColumn": "cellID",
            "groupColumn": "group",
            "groups": ("g",),
            "barcodes": barcodes,
            "barcodeAllowListFile": str(allow),
        }

        with cc.openSources(
            [cc.SourceSpec(path, "H5AD", "s", selection=selection)],
            grid,
        ) as reader:
            block = reader.readRegion("chr1", 0, 115, returnRawNoiseMass=True)

            np.testing.assert_array_equal(block.values, [expected])
            assert block.noiseAvailable == (False,)
            info = reader.sourceInfo["s"]
            assert info["selectedCellCount"] == cellCount
            assert info["chromosomeNames"] == ("chr1",)
            assert info["chromosomeSizes"] == {"chr1": 115}
            assert reader.grid.chromosomeSizes == {"chr1": 115, "chr2": 115}

            for query, kwargs in (
                (reader.sourceRange, {"flagExclude": 3844}),
                (reader.chromRange, {}),
            ):
                assert query("s", "chr1", **kwargs) == (0, 115)
                with pytest.raises(
                    cc.SourceError, match="Unknown source or chromosome"
                ):
                    query("s", "chr2", **kwargs)

            totals = reader.libraryTotals(chromosomes=["chr1"])
            assert totals.records["s"]["eventMass"] == sum(expected)
            with pytest.raises(cc.SourceError, match="Unknown source or chromosome"):
                reader.libraryTotals(chromosomes=["chr2"])


def testCommonFeatureTerritoryAndSelectedTotals(countSources, tmp_path):
    chr1Path = Path(__file__).parent / "data" / "counts-dense.h5ad"
    chr2Path = tmp_path / "chr2.h5ad"
    copyfile(chr1Path, chr2Path)
    with h5py.File(chr2Path, "r+") as handle:
        handle["var/chrom/categories"][0] = "chr2"
        handle["X"][1] = 0

    sources = [
        cc.SourceSpec(chr1Path, "H5AD", "chr1Matrix"),
        cc.SourceSpec(chr2Path, "H5AD", "chr2Matrix"),
        cc.SourceSpec(
            chr2Path,
            "H5AD",
            "zeroMatrix",
            selection={"barcodeColumn": "cellID", "barcodes": ("B",)},
        ),
        cc.SourceSpec(countSources["eventsBAM"], "BAM", "bam"),
        cc.SourceSpec(countSources["events"], "FRAGMENTS", "fragments"),
    ]
    chromosomes = ("chr2", "chr1")
    sampleIDs = ("fragments", "chr2Matrix", "zeroMatrix")
    with cc.openSources(sources, cc.GridSpec({"chr1": 115, "chr2": 115})) as reader:
        assert reader.commonFeatureTerritory(chromosomes=chromosomes) == {
            "requestedChromosomes": ("chr1", "chr2"),
            "usedChromosomes": (),
            "missingChromosomesBySource": {
                "chr1Matrix": ("chr2",),
                "chr2Matrix": ("chr1",),
                "zeroMatrix": ("chr1",),
                "bam": ("chr2",),
                "fragments": (),
            },
        }
        territory = reader.commonFeatureTerritory(
            chromosomes=chromosomes, sampleIDs=sampleIDs
        )
        assert territory == {
            "requestedChromosomes": ("chr1", "chr2"),
            "usedChromosomes": ("chr2",),
            "missingChromosomesBySource": {
                "fragments": (),
                "chr2Matrix": ("chr1",),
                "zeroMatrix": ("chr1",),
            },
        }

        with pytest.raises(cc.SourceError, match="Unknown source or chromosome"):
            reader.libraryTotals(chromosomes=chromosomes, sampleIDs=sampleIDs)

        used = territory["usedChromosomes"]
        totals = reader.libraryTotals(chromosomes=used, sampleIDs=sampleIDs)
        assert totals.chromosomes == used
        assert totals.sampleIDs == sampleIDs
        assert totals.records["fragments"]["eventMass"] == 0
        assert totals.records["chr2Matrix"]["eventMass"] == 6
        assert totals.records["zeroMatrix"]["selectedCellCount"] == 1
        assert totals.records["zeroMatrix"]["eventMass"] == 0
        assert reader.chromRange("zeroMatrix", "chr2") == (0, 115)
        assert reader.libraryTotals(chromosomes=used, sampleIDs=sampleIDs) is totals
        reversedIDs = sampleIDs[::-1]
        reversedTotals = reader.libraryTotals(chromosomes=used, sampleIDs=reversedIDs)
        assert reversedTotals.sampleIDs == reversedIDs
        with pytest.raises(cc.SourceError, match="Unknown source or chromosome"):
            reader.libraryTotals(chromosomes=used)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testLogDomainAndPrecision(dtype):
    values = np.array([[-1, 0, 1e-20]], dtype=dtype)

    result = cc.logTransform(values, pseudocount=1 + 2**-30)

    np.testing.assert_allclose(
        result[0, :2], [-30, np.log2(1 + 2**-30)], rtol=4 * np.finfo(dtype).eps, atol=0
    )

    reference = np.array([1e20], dtype=dtype)
    nearby = np.nextafter(reference, np.array(np.inf, dtype=dtype))
    expected = np.log1p(
        (nearby.astype(float) - reference) / (reference.astype(float) + 1)
    ) / np.log(2)

    np.testing.assert_allclose(
        cc.logTransform(nearby, reference=reference),
        expected,
        rtol=4 * np.finfo(dtype).eps,
        atol=0,
    )

    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            cc.logTransform(np.array([bad], dtype=dtype))

    storage = np.arange(5, dtype=dtype)

    with pytest.raises(ValueError, match="overlap"):
        cc.logTransform(storage[:-1], out=storage[1:])

    np.testing.assert_allclose(
        cc.logTransform(np.array([-1e-20, 1e-20], dtype=dtype)),
        [-1e-20 / np.log(2), 1e-20 / np.log(2)],
        rtol=4 * np.finfo(dtype).eps,
        atol=0,
    )

    values = np.ones((2, 3), dtype=dtype)
    readonly = np.ones_like(values)
    readonly.flags.writeable = False
    for array, reference, target in (
        (values.astype(np.int32), None, np.empty((2, 3), dtype=np.int32)),
        (values[:, ::2], None, np.empty((2, 2), dtype=dtype)),
        (values, None, np.empty(2, dtype=dtype)),
        (values, None, readonly),
        (values, np.ones(3, dtype=dtype), np.empty_like(values)),
        (storage[:-1], None, storage[1:]),
        (np.ones(4, dtype=dtype), storage[:-1], storage[1:]),
    ):
        with pytest.raises(ValueError):
            _native.logValues(array, reference, 1.0, target)

    unaligned = np.ndarray(
        values.shape, dtype=dtype, buffer=bytearray(values.nbytes + 1), offset=1
    )
    swapped = values.astype(values.dtype.newbyteorder("S"))
    for invalid in (unaligned, swapped):
        for array, reference, target in (
            (invalid, None, np.empty_like(values)),
            (values, invalid, np.empty_like(values)),
            (values, None, invalid),
        ):
            with pytest.raises(cc.InputError):
                _native.logValues(array, reference, 1.0, target)

    target = np.empty_like(values)
    assert _native.logValues(readonly, readonly, 1.0, target) is target
    np.testing.assert_array_equal(target, 0)
    inPlace = np.full_like(values, 3)
    assert _native.logValues(inPlace, readonly, 1.0, inPlace) is inPlace
    np.testing.assert_array_equal(inPlace, 1)


def testFloat64SizeFactorsAndCatalogs():
    ratio = math.sqrt(1 + 2**-24)

    for profile, options, expected in (
        ([4, 16, 64], {"centerMedian": False}, [4, 1, 0.25]),
        ([2**24, 2**24 + 1], {"centerMedian": False}, [ratio, 1 / ratio]),
        ([1, 1000000], {}, [100, 0.01]),
        ([1, 1000000], {"factorBounds": None}, [1000, 0.001]),
        ([1, 4, 64], {"factorBounds": None}, [4, 1, 0.0625]),
        ([1, 4, 64], {"factorBounds": [0.1, 2]}, [2, 1, 0.1]),
        ([1, 4, 64], {"factorBounds": (3, 3)}, [3, 3, 3]),
    ):
        counts = np.tile(np.asarray(profile, dtype=np.float64)[:, None], (1, 100))

        np.testing.assert_allclose(
            cc.estimateSizeFactors(counts, **options), expected, rtol=1e-14
        )

    for bounds in (
        (1,),
        (1, 2, 3),
        (0, 1),
        (2, 1),
        (1, np.inf),
        (np.nan, 1),
        (True, 1),
        "15",
    ):
        with pytest.raises(cc.InputError):
            cc.estimateSizeFactors(counts, factorBounds=bounds)

    for assembly, sizes in cc.assemblyChromosomeSizes.items():
        assert all(name in sizes for name in cc.standardChromosomes[assembly])
        assert all(value > 0 for value in sizes.values())
        with pytest.raises(TypeError):
            sizes[next(iter(sizes))] = 1

    for assembly, chromosomes in cc.assemblyBlacklistRanges.items():
        sizes = cc.assemblyChromosomeSizes[assembly]

        assert all(
            0 <= startBP < endBP <= sizes[chromosome]
            for chromosome, ranges in chromosomes.items()
            for startBP, endBP in ranges
        )

    assert cc.assemblyChromosomeSizes["hs1"]["chrY"] == 62460029
    assert sum(map(len, cc.assemblyBlacklistRanges["sacCer3"].values())) == 409


@pytest.mark.parametrize(
    "fields",
    [
        {"flagExclude": -1},
        {"minMappingQuality": True},
        {"countMode": "midpoint"},
        {"inferFragmentLength": True, "extendFrom5pBP": 100},
    ],
)
def testInvalidCountOptions(fields):
    with pytest.raises(ValueError):
        cc.CountOptions(**fields)
