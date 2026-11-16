from dataclasses import asdict, replace
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

import ccounts as cc
from ccounts import _native
from ccounts.lengthScale import _histogramQuantiles


def testNumericalKernels():
    for signal in (
        np.array([3, 1, 1, 9, 0, 7, 2], dtype=float),
        np.array([5.0]),
        np.array([5.0, 1.0]),
        np.full(4, 2.0),
        np.arange(10, dtype=float)[::-1].copy(),
    ):
        values, indices = np.unique(signal, return_inverse=True)

        for width in (1, 3, 7, 15):
            reference = [
                np.sort(
                    signal[
                        max(0, i - width // 2) : min(len(signal), i + width // 2 + 1)
                    ]
                )[
                    len(
                        signal[
                            max(0, i - width // 2) : min(len(signal), i + width // 2 + 1)
                        ]
                    )
                    // 2
                ]
                for i in range(len(signal))
            ]

            np.testing.assert_array_equal(
                _native.musicMedianProfile(indices, values, width), reference
            )

    countsO, countsE = np.array([0, 1, 4, 2000]), np.array([0, 0, 2, 0])

    tails = _native.musicBinomialLogTail(countsO, countsE)

    np.testing.assert_allclose(
        tails[:3],
        binom.logsf(countsO[:3] - 1, countsO[:3] + countsE[:3], 0.5),
        atol=1e-14,
    )
    assert tails[-1] == -2000 * np.log(2)

    histograms, widths = np.array([[2, 1, 1, 1], [4, 0, 1, 0], [4, 0, 0, 0]]), np.array(
        [100, 300, 900]
    )

    quantiles, defined = _histogramQuantiles(histograms, widths, (0.5, 0.75, 0.9))

    for row in (0, 1):

        np.testing.assert_array_equal(
            quantiles[row],
            np.quantile(np.repeat(widths, histograms[row, 1:]), (0.5, 0.75, 0.9)),
        )
    assert defined.tolist() == [True, True, False]


def testFrozenDetectorAndResampling(caplog):
    counts = np.zeros((4, 5, 200), dtype=np.int64)

    for block in range(4):
        counts[block, :3] = 1
        for start in (20, 120):
            counts[block, 0, start : start + 4 + 2 * block] = 20
            counts[block, 2, start : start + 10 * (block + 2)] = 20
        counts[block, 1] = counts[block, 0]
    counts[0, 4] = counts[0, 0]
    saved = counts.copy()
    options = cc.LengthScaleOptions(
        blockWidthBP=20000, bootstrapReplicates=64, returnProfiles=True
    )

    with caplog.at_level(logging.INFO, logger="ccounts.lengthScale"):
        result = cc.estimateLengthScaleCounts(
            counts,
            100,
            sampleIDs=("narrow", "copy", "wide", "empty", "rare"),
            options=options,
        )

    rows = result["sampleResults"]

    assert [
        (r["q50BP"], r["q75BP"], r["scaleBP"], r["supportedBinCount"]) for r in rows
    ] == [
        (1100, 1700, 1700, 56),
        (1100, 1700, 1700, 56),
        (13100, 19500, 19500, 280),
        (None, None, None, 0),
        (500, 500, 500, 8),
    ]
    assert rows[0]["histogramCounts"] == [744, 0, 0, 8, 0, 28, 20, 0, 0, 0, 0, 0, 0]
    assert rows[2]["histogramCounts"] == [520, 0, 0, 0, 0, 0, 0, 0, 20, 50, 59, 58, 93]

    draws = np.random.default_rng(42).integers(0, 4, size=(64, 4))

    for row in rows:
        profile = np.asarray(row["largestSupportBP"])
        reference = []

        for draw in draws:
            selected = profile[draw].ravel()
            positive = selected[selected > 0]
            reference.append(
                float(np.quantile(positive, 0.9)) if len(positive) else None
            )

        assert row["bootstrapScalesBP"] == reference

        np.testing.assert_array_equal(
            np.sum(row["blockHistogramCounts"], axis=0), row["histogramCounts"]
        )

    assert result["pointDefinedSampleIDs"] == ["narrow", "copy", "wide", "rare"]
    assert (result["scaleBP"], result["definedSampleCount"], result["sampleCount"]) == (
        1700,
        4,
        5,
    )

    reference = [
        (
            None
            if any(rows[i]["bootstrapScalesBP"][r] is None for i in (0, 1, 2, 4))
            else float(
                np.median([rows[i]["bootstrapScalesBP"][r] for i in (0, 1, 2, 4)])
            )
        )
        for r in range(64)
    ]

    assert result["bootstrapScalesBP"] == reference
    assert rows[0]["CIStatus"] == "ok" and rows[2]["upperGridLimited"]
    assert (
        rows[3]["CIStatus"] == "noSupport"
        and rows[3]["bootstrapScalesBP"] == [None] * 64
    )

    for row in (result, rows[4]):
        assert (
            row["CIStatus"] == "incompleteResamples"
            and row["CILowerBP"] is None
            and row["CIUpperBP"] is None
        )
        assert 0 < row["definedBootstrapFraction"] < 1
    assert (rows[4]["conditionalCILowerBP"], rows[4]["conditionalCIUpperBP"]) == (
        500,
        500,
    )

    single = cc.estimateLengthScaleCounts(
        counts[:1, :1], 100, options=replace(options, returnProfiles=False)
    )

    assert single["CIStatus"] == "insufficientBlocks" and single["CILowerBP"] is None
    assert "largestSupportBP" not in single["sampleResults"][0]

    records = [r for r in caplog.records if r.name == "ccounts.lengthScale"]

    assert len(records) == 1
    assert (
        json.loads(records[0].getMessage().split("configuration ", 1)[1])
        == result["settings"]
    )

    json.dumps(result, allow_nan=False)
    np.testing.assert_array_equal(counts, saved)


def testReaderCentersAndOwnership(countSources, monkeypatch):
    sources = [
        cc.SourceSpec(
            countSources[path],
            kind,
            name,
            cc.CountOptions(countMode="coverage", **fields),
            selection=("A",),
        )
        for path, kind, name, fields in (
            ("eventsBAM", "BAM", "bam", {"bamInputMode": "fragments"}),
            ("events", "FRAGMENTS", "fragments", {"fragmentsUseReadSupport": True}),
        )
    ]
    calls = []

    class HandleSpy:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def countRegion(self, *args):
            calls.append(args)
            return self.handle.countRegion(*args)

    options = cc.LengthScaleOptions(blockWidthBP=100, bootstrapReplicates=4)

    with cc.openSources(
        sources, cc.GridSpec({"chr1": 115}, 50), dtype="float64"
    ) as reader:
        initial = reader.readRegion("chr1", 0, 115, returnRawNoiseMass=True)
        saved = {
            key: asdict(value["countOptions"])
            for key, value in reader.sourceInfo.items()
        }

        for sampleID in reader.sampleIDs:
            monkeypatch.setitem(
                reader._handles, sampleID, HandleSpy(reader._handles[sampleID])
            )

        result = reader.estimateLengthScale(
            sampleIDs=("fragments", "bam"), options=options
        )

        assert [r["sampleID"] for r in result["sampleResults"]] == ["fragments", "bam"]
        assert [r["totalEvents"] for r in result["sampleResults"]] == [5, 5]
        assert len(calls) == 2 and all(
            args[4].countMode == "center" and args[5] is False for args in calls
        )
        assert all(
            info["eventUnit"] == "fragmentCenter"
            for info in result["settings"]["sourceInfo"].values()
        )
        assert result["settings"]["sourceInfo"]["fragments"]["selection"] == ["A"]
        assert (
            result["settings"]["regions"] == [["chr1", 0, 100]]
            and result["settings"]["sampling"]["terminalBP"] == 15
        )
        assert {
            key: asdict(value["countOptions"])
            for key, value in reader.sourceInfo.items()
        } == saved

        final = reader.readRegion("chr1", 0, 115, returnRawNoiseMass=True)

        np.testing.assert_array_equal(final.values, initial.values)
        np.testing.assert_array_equal(final.rawNoiseMass, initial.rawNoiseMass)

        callCount = len(calls)
        for fields in (
            {"sampleIDs": ("bam", "bam")},
            {"regions": [("chr1", 0, 100)] * 2},
            {"excludeRanges": {"chr1": [(0, 1)]}},
            {"regions": [("chr1", 0, 100)], "excludeRanges": {"chr1": [(0, 1)]}},
        ):
            with pytest.raises(ValueError):
                reader.estimateLengthScale(options=options, **fields)
        assert len(calls) == callCount

        serialized = json.dumps(result, allow_nan=False)
        assert serialized.count(sources[0].path) == 1
        assert "regions" not in result and "sourceInfo" not in result
        for row in result["sampleResults"]:
            assert (
                len(row["histogramCounts"])
                == len(result["settings"]["filterWidthsBP"]) + 1
            )
            assert not {
                "sourceInfo",
                "filterWidthsBP",
                "histogramWidthsBP",
            }.intersection(row)

    with pytest.raises(ValueError):
        reader.estimateLengthScale(options=options)


@pytest.mark.parametrize("value", [2, 0.5, -1])
def testBedGraphCountSemantics(tmp_path, value):
    path = tmp_path / "counts.bg"
    path.write_text(f"chr1\t0\t50\t{value}\n")

    indexed = cc.prepareIndex(path, "BEDGRAPH", targetPath=tmp_path / "counts.bg.gz")
    options = cc.LengthScaleOptions(blockWidthBP=100, bootstrapReplicates=2)

    with cc.openSources(
        [cc.SourceSpec(indexed, "BEDGRAPH", "s")], cc.GridSpec({"chr1": 100}, 50)
    ) as reader:
        with pytest.raises(ValueError):
            reader.estimateLengthScale(options=options)

        if value == 2:
            result = reader.estimateLengthScale(options=options, bedGraphCounts=True)

            assert result["sampleResults"][0]["totalEvents"] == 2
            assert (
                result["settings"]["sourceInfo"]["s"]["eventUnit"] == "declaredBinCount"
            )
        else:
            with pytest.raises(ValueError, match="Counts"):
                reader.estimateLengthScale(options=options, bedGraphCounts=True)


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def testH5ADCountsAndGrid(storage):
    data = Path(__file__).parent / "data"
    selection = {
        "barcodeColumn": "cellID",
        "groupColumn": "group",
        "groups": ("g",),
        "barcodes": ("A", "B"),
        "barcodeAllowListFile": str(data / "allow.txt"),
    }
    source = cc.SourceSpec(
        data / f"counts-{storage}.h5ad", "H5AD", "s", selection=selection
    )
    options = cc.LengthScaleOptions(blockWidthBP=100, bootstrapReplicates=2)

    with cc.openSources([source], cc.GridSpec({"chr1": 115}, 50)) as reader:
        result = reader.estimateLengthScale(options=options)

        assert result["sampleResults"][0]["totalEvents"] == 4
        assert result["settings"]["sourceInfo"]["s"]["eventUnit"] == "storedBinCount"
        assert (
            result["settings"]["sourceInfo"]["s"]["selectedCellCount"]
            == reader.sourceInfo["s"]["selectedCellCount"]
        )

        with pytest.raises(ValueError, match="grid"):
            reader.estimateLengthScale(options=replace(options, binSizeBP=25))


def testInvalidCountsOptionsAndRegions():
    options = cc.LengthScaleOptions(blockWidthBP=200, bootstrapReplicates=2)

    for counts in (
        np.zeros((1, 2)),
        np.zeros((0, 1, 2)),
        [[[-1, 0]]],
        [[[0.5, 0]]],
        [[[np.nan, 0]]],
        [[[1j, 0]]],
        [[[2**52, 1]]],
    ):
        with pytest.raises(ValueError):
            cc.estimateLengthScaleCounts(counts, 100, options=options)

    for fields in (
        {"bootstrapReplicates": 1},
        {"scaleFactor": 1},
        {"confidenceLevel": 1},
        {"returnProfiles": 1},
        {"binSizeBP": 100, "blockWidthBP": 150},
    ):
        with pytest.raises(ValueError):
            cc.LengthScaleOptions(**fields)

    for fields in (
        {"sampleIDs": ("s", "s")},
        {"regions": [("chr1", 0, 200)] * 2},
        {"regions": [("chr1", 50, 250)]},
        {"options": replace(options, binSizeBP=50)},
        {"options": replace(options, scaleFactor=np.nextafter(1.0, 2.0))},
    ):
        with pytest.raises(ValueError):
            cc.estimateLengthScaleCounts(
                np.zeros((1, 1, 2)), 100, **({"options": options} | fields)
            )
