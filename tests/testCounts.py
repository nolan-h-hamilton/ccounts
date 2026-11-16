from dataclasses import replace
from pathlib import Path
import math
from shutil import copyfile
from unittest.mock import patch
import numpy as np
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
            args = {"effectiveGenomeSizeBP": 115} if method == "RPGC" else {}

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
def testH5ADSelectionAndTotals(tmp_path, storage):
    path = Path(__file__).parent / "data" / f"counts-{storage}.h5ad"
    allow = Path(__file__).parent / "data" / "allow.txt"

    for barcodes, expected in ((("A", "B"), [[3, 1, 2]]), (("B",), [[0, 0, 0]])):
        selection = {
            "barcodeColumn": "cellID",
            "groupColumn": "group",
            "groups": ("g",),
            "barcodes": barcodes,
            "barcodeAllowListFile": str(allow),
        }

        with cc.openSources(
            [cc.SourceSpec(path, "H5AD", "s", selection=selection)],
            cc.GridSpec({"chr1": 115}),
        ) as reader:
            block = reader.readRegion("chr1", 0, 115, returnRawNoiseMass=True)

            np.testing.assert_array_equal(block.values, expected)
            assert block.noiseAvailable == (False,)
            assert reader.libraryTotals(chromosomes=["chr1"]).records["s"][
                "eventMass"
            ] == sum(expected[0])


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
    readonly = np.empty_like(values)
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


def testFloat64SizeFactorsAndCatalogs():
    ratio = math.sqrt(1 + 2**-24)

    for profile, expected in (
        ([4, 16, 64], [4, 1, 0.25]),
        ([2**24, 2**24 + 1], [ratio, 1 / ratio]),
    ):
        counts = np.tile(np.asarray(profile, dtype=np.float64)[:, None], (1, 100))

        np.testing.assert_allclose(
            cc.estimateSizeFactors(counts, centerMedian=False), expected, rtol=1e-14
        )

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
