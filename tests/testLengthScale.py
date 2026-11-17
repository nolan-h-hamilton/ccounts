from dataclasses import asdict, replace
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.stats import binom, norm

import ccounts as cc
from ccounts import _native
from ccounts.lengthScale import _histogramIJ


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
                            max(0, i - width // 2) : min(
                                len(signal), i + width // 2 + 1
                            )
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

    for summaryFunction in (np.mean, np.average):
        summaries, influences, supportedBlocks = _histogramIJ(
            histograms[None, :, :], widths, summaryFunction
        )
        for row in (0, 1):
            assert summaries[row] == pytest.approx(
                np.mean(np.repeat(widths, histograms[row, 1:]))
            )
        assert supportedBlocks.tolist() == [1, 1, 0]
        np.testing.assert_allclose(influences, 0, atol=1e-8)


def weightedRMS(values, *, weights):
    assert np.sum(weights) == pytest.approx(1)
    return np.sqrt(np.average(np.square(values), weights=weights))


@pytest.mark.parametrize(
    "summaryFunction,confidenceLevel", [(np.mean, 0.8), (weightedRMS, 0.95)]
)
def testFrozenDetectorAndIJ(caplog, summaryFunction, confidenceLevel):
    counts = np.zeros((4, 5, 200), dtype=np.int64)

    for block in range(4):
        counts[block, :3] = 1
        for start in (20, 120):
            counts[block, 0, start : start + 4 + 2 * block] = 20
            counts[block, 2, start : start + 10 * (block + 2)] = 20
        counts[block, 1] = counts[block, 0]
    counts[0, 4] = counts[0, 0]
    saved = counts.copy()
    regions = [("chr1", start, start + 20000) for start in (1000, 0, 500, 100)]
    options = cc.LengthScaleOptions(
        blockWidthBP=20000,
        returnProfiles=True,
        confidenceLevel=confidenceLevel,
    )
    assert options.filterType == "median"

    with caplog.at_level(logging.INFO, logger="ccounts.lengthScale"):
        result = cc.estimateLengthScaleCounts(
            counts,
            100,
            sampleIDs=("narrow", "copy", "wide", "empty", "rare"),
            options=options,
            regions=regions,
            **(
                {}
                if summaryFunction is np.mean
                else {"summaryFunction": summaryFunction}
            ),
        )

    rows = result["sampleResults"]

    if summaryFunction is np.mean:
        np.testing.assert_allclose(
            [rows[index]["scaleBP"] for index in (0, 1, 2, 4)],
            [1300, 1300, 3345800 / 280, 700],
        )
    assert rows[3]["scaleBP"] is None
    assert [r["supportedBinCount"] for r in rows] == [56, 56, 280, 0, 8]
    assert all(not {"q50BP", "q75BP", "q90BP"}.intersection(row) for row in rows)
    assert result["settings"]["summaryFunction"] == (
        "mean" if summaryFunction is np.mean else "custom"
    )
    assert result["settings"]["CIMethod"] == "IJNormal"
    assert result["settings"]["regions"] == [list(region) for region in regions]
    assert "bootstrapReplicates" not in result["settings"]
    assert rows[0]["histogramCounts"] == [744, 0, 0, 8, 12, 36, 0, 0, 0, 0, 0, 0]
    assert rows[2]["histogramCounts"] == [520, 0, 0, 0, 0, 0, 0, 20, 30, 33, 93, 104]

    influences = []
    criticalValue = norm.isf((1 - confidenceLevel) / 2)
    for row in rows:
        profile = np.asarray(row["largestSupportBP"])
        supported = profile[profile > 0]
        np.testing.assert_array_equal(
            np.sum(row["blockHistogramCounts"], axis=0), row["histogramCounts"]
        )
        assert not any("bootstrap" in key or "Bootstrap" in key for key in row)
        assert not {"conditionalCILowerBP", "conditionalCIUpperBP"}.intersection(row)
        if not len(supported):
            assert row["CIStatus"] == "noSupport"
            assert row["SEBP"] is row["blockInfluenceBP"] is None
            influences.append(None)
            continue
        point = (
            np.mean(supported)
            if summaryFunction is np.mean
            else np.sqrt(np.mean(np.square(supported)))
        )
        assert row["scaleBP"] == pytest.approx(point)
        blockCounts = np.count_nonzero(profile, axis=1)
        if summaryFunction is np.mean:
            influence = 4 * (profile.sum(axis=1) - point * blockCounts) / len(supported)
        else:
            influence = (
                4
                * (np.square(profile).sum(axis=1) - point**2 * blockCounts)
                / (2 * point * len(supported))
            )
        influences.append(influence)
        if np.count_nonzero(blockCounts) < 2:
            assert row["CIStatus"] == "insufficientBlocks"
            assert row["SEBP"] is row["blockInfluenceBP"] is None
            assert row["CILowerBP"] is row["CIUpperBP"] is None
            continue
        standardError = np.linalg.norm(influence) / 4
        np.testing.assert_allclose(
            row["blockInfluenceBP"], influence, rtol=2e-6, atol=1e-7
        )
        assert row["SEBP"] == pytest.approx(standardError, rel=2e-6)
        np.testing.assert_allclose(
            [row["CILowerBP"], row["CIUpperBP"]],
            [
                row["scaleBP"] - criticalValue * standardError,
                row["scaleBP"] + criticalValue * standardError,
            ],
            rtol=2e-6,
        )

    assert result["pointDefinedSampleIDs"] == ["narrow", "copy", "wide", "rare"]
    points = np.array([rows[index]["scaleBP"] for index in (0, 1, 2, 4)])
    assert result["scaleBP"] == pytest.approx(
        np.mean(points)
        if summaryFunction is np.mean
        else np.sqrt(np.mean(np.square(points)))
    )
    assert (result["definedSampleCount"], result["sampleCount"]) == (4, 5)
    assert result["CIStatus"] == "insufficientBlocks"
    assert result["SEBP"] is result["blockInfluenceBP"] is None
    assert result["CILowerBP"] is result["CIUpperBP"] is None
    assert not any("bootstrap" in key or "Bootstrap" in key for key in result)
    assert rows[0]["CIStatus"] == "ok"
    assert not rows[2]["upperGridLimited"]

    aggregate = cc.estimateLengthScaleCounts(
        counts[:, :4], 100, options=options, summaryFunction=summaryFunction
    )
    if summaryFunction is np.mean:
        aggregatePoint = points[:3].mean()
        aggregateInfluence = np.mean(np.stack(influences[:3]), axis=0)
    else:
        aggregatePoint = np.sqrt(np.mean(np.square(points[:3])))
        gradient = points[:3] / (3 * aggregatePoint)
        aggregateInfluence = np.stack(influences[:3], axis=1) @ gradient
    assert aggregate["scaleBP"] == pytest.approx(aggregatePoint)
    np.testing.assert_allclose(
        aggregate["blockInfluenceBP"], aggregateInfluence, rtol=2e-6, atol=1e-7
    )
    assert aggregate["CIStatus"] == "ok"
    assert aggregate["SEBP"] == pytest.approx(
        np.linalg.norm(aggregateInfluence) / 4, rel=2e-6
    )

    single = cc.estimateLengthScaleCounts(
        counts[:1, :1], 100, options=replace(options, returnProfiles=False)
    )

    assert single["CIStatus"] == "insufficientBlocks" and single["CILowerBP"] is None
    assert single["SEBP"] is single["blockInfluenceBP"] is None
    assert "largestSupportBP" not in single["sampleResults"][0]

    records = [r for r in caplog.records if r.name == "ccounts.lengthScale"]

    assert len(records) == 1
    assert (
        json.loads(records[0].getMessage().split("configuration ", 1)[1])
        == result["settings"]
    )

    json.dumps(result, allow_nan=False)
    np.testing.assert_array_equal(counts, saved)


def testMeanFilterReference(monkeypatch):
    from scipy.signal import find_peaks

    counts = np.zeros((3, 4, 51), dtype=np.int64)
    for block in range(2):
        counts[block, :3] = 1
        counts[block, 0, :5] = 20
        counts[block, 0, 20 : 26 + 2 * block] = 20
        counts[block, 0, -5:] = 20
        counts[block, 1] = counts[block, 0]
        counts[block, 2, 10 : 35 + 5 * block] = 20
    saved = counts.copy()
    options = cc.LengthScaleOptions(
        blockWidthBP=5100,
        returnProfiles=True,
        filterType="mean",
    )
    actualProfiles = []
    referenceProfiles = []

    def capturePeaks(values, **fields):
        actualProfiles.append(-values.copy())
        return find_peaks(values, **fields)

    def meanProfile(valueIndices, values, width):
        signal = values[valueIndices]
        profile = np.array(
            [
                signal[
                    max(0, index - width // 2) : min(
                        len(signal), index + width // 2 + 1
                    )
                ].mean()
                for index in range(len(signal))
            ]
        )
        referenceProfiles.append(profile)
        return profile

    monkeypatch.setattr("scipy.signal.find_peaks", capturePeaks)
    actual = cc.estimateLengthScaleCounts(counts, 100, options=options)
    monkeypatch.setattr("scipy.signal.find_peaks", find_peaks)
    monkeypatch.setattr(_native, "musicMedianProfile", meanProfile)
    reference = cc.estimateLengthScaleCounts(
        counts, 100, options=replace(options, filterType="median")
    )

    assert len(actualProfiles) == 6 * len(actual["settings"]["filterWidthsBP"])
    for profile, expected in zip(actualProfiles, referenceProfiles, strict=True):
        np.testing.assert_array_equal(profile, expected)
    for key in actual:
        if key not in ("settings", "detectorSeconds", "runtimeSeconds"):
            assert actual[key] == reference[key]
    assert actual["settings"] == reference["settings"] | {"filterType": "mean"}
    np.testing.assert_array_equal(counts, saved)
    json.dumps(actual, allow_nan=False)


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

    options = cc.LengthScaleOptions(
        blockWidthBP=100,
        filterType="mean",
        significanceLevel=0.99,
    )

    def halveMean(values, *, weights):
        assert np.sum(weights) == pytest.approx(1)
        return np.average(values, weights=weights) / 2

    with cc.openSources(
        sources, cc.GridSpec({"chr1": 115}, 25), dtype="float64"
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
            sampleIDs=("fragments", "bam"),
            options=options,
            excludeRanges={},
            summaryFunction=halveMean,
        )

        assert [r["sampleID"] for r in result["sampleResults"]] == ["fragments", "bam"]
        assert [r["totalEvents"] for r in result["sampleResults"]] == [5, 5]
        assert [r["scaleBP"] for r in result["sampleResults"]] == [37.5, 37.5]
        assert result["scaleBP"] == 18.75
        assert result["CIStatus"] == "insufficientBlocks"
        assert result["SEBP"] is None
        assert result["settings"]["summaryFunction"] == "custom"
        assert len(calls) == 2 and all(
            args[4].countMode == "center" and args[5] is False for args in calls
        )
        assert all(
            info["eventUnit"] == "fragmentCenter"
            for info in result["settings"]["sourceInfo"].values()
        )
        assert result["settings"]["sourceInfo"]["fragments"]["selection"] == ["A"]
        assert result["settings"]["filterType"] == "mean"
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

        single = reader.estimateLengthScale(
            sampleIDs=("bam",),
            options=options,
            excludeRanges={},
            summaryFunction=halveMean,
        )
        sample = single["sampleResults"][0]
        assert single["scaleBP"] == sample["scaleBP"] == 37.5
        assert single["blockInfluenceBP"] is sample["blockInfluenceBP"] is None
        assert single["SEBP"] is sample["SEBP"] is None

        callCount = len(calls)
        for fields in (
            {"sampleIDs": ("bam", "bam")},
            {"regions": [("chr1", 0, 75)]},
            {"excludeRanges": {"chr1": [(0, 1)]}},
            {"regions": [("chr1", 0, 100)], "excludeRanges": {"chr1": [(0, 1)]}},
        ):
            with pytest.raises(ValueError):
                reader.estimateLengthScale(
                    options=options, **({"excludeRanges": {}} | fields)
                )
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
    options = cc.LengthScaleOptions(blockWidthBP=100)

    with cc.openSources(
        [cc.SourceSpec(indexed, "BEDGRAPH", "s")], cc.GridSpec({"chr1": 100}, 25)
    ) as reader:
        with pytest.raises(ValueError):
            reader.estimateLengthScale(options=options, excludeRanges={})

        if value == 2:
            result = reader.estimateLengthScale(
                options=options, bedGraphCounts=True, excludeRanges={}
            )

            assert result["sampleResults"][0]["totalEvents"] == 4
            assert (
                result["settings"]["sourceInfo"]["s"]["eventUnit"] == "declaredBinCount"
            )
        else:
            with pytest.raises(ValueError, match="Counts"):
                reader.estimateLengthScale(
                    options=options, bedGraphCounts=True, excludeRanges={}
                )


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def testH5ADCountsAndGrid(storage, tmp_path):
    data = Path(__file__).parent / "data"
    selection = {
        "barcodeColumn": "cellID",
        "groupColumn": "group",
        "groups": ("g",),
        "barcodes": ("A", "B"),
        "barcodeAllowListFile": str(data / "allow.txt"),
    }
    path = tmp_path / f"counts-{storage}.h5ad"
    path.write_bytes((data / path.name).read_bytes())
    with h5py.File(path, "r+") as handle:
        handle["var/end"][-1] = 150
    source = cc.SourceSpec(path, "H5AD", "s", selection=selection)
    options = cc.LengthScaleOptions(blockWidthBP=150)

    with cc.openSources([source], cc.GridSpec({"chr1": 150}, 50)) as reader:
        result = reader.estimateLengthScale(options=options, excludeRanges={})

        assert result["sampleResults"][0]["totalEvents"] == 6
        assert result["settings"]["sourceInfo"]["s"]["eventUnit"] == "storedBinCount"
        assert (
            result["settings"]["sourceInfo"]["s"]["selectedCellCount"]
            == reader.sourceInfo["s"]["selectedCellCount"]
        )

        with pytest.raises(ValueError, match="grid"):
            reader.estimateLengthScale(
                options=replace(options, binSizeBP=25), excludeRanges={}
            )


def testReferenceMasksBeforeSampling(countSources, recwarn):
    from ccounts.lengthScale import _lengthScaleExclusions

    assert cc.LengthScaleOptions().backgroundWindowBP == 50000
    options = cc.LengthScaleOptions(filterType="mean", backgroundWindowBP=1000000)
    assert (
        options.binSizeBP,
        options.blockCount,
        options.blockWidthBP,
        options.randomSeed,
    ) == (None, 4096, 500000, 42)

    sizes = {"chr1": cc.assemblyChromosomeSizes["hg38"]["chr1"]}
    assembly, mask = _lengthScaleExclusions(sizes, None)
    assert assembly == "hg38"
    for position in (628903, 131000000):
        assert any(start <= position < end for start, end in mask["chr1"])

    width = options.blockWidthBP
    frame = [
        ("chr1", start, start + width)
        for start in range(0, sizes["chr1"] - width + 1, 25000)
    ]
    extraStart = next(
        start
        for _, start, end in frame
        if not any(start < stop and end > begin for begin, stop in mask["chr1"])
    )
    extras = {"chr1": [(extraStart, extraStart + 1)]}
    source = cc.SourceSpec(countSources["events"], "FRAGMENTS", "s")

    with cc.openSources([source], cc.GridSpec(sizes, 25000)) as reader:
        for supplied, blockCount in (
            (None, 128),
            ({}, 128),
            (extras, 128),
            (None, options.blockCount),
        ):
            sampledOptions = replace(options, blockCount=blockCount)
            resolvedAssembly, exclusions = _lengthScaleExclusions(sizes, supplied)
            assert resolvedAssembly == "hg38"
            eligible = [
                (chromosome, start, end)
                for chromosome, start, end in frame
                if not any(
                    start < stop and end > begin
                    for begin, stop in exclusions[chromosome]
                )
            ]
            indices = np.random.default_rng(42).choice(
                len(eligible), min(blockCount, len(eligible)), replace=False
            )
            expected = [list(eligible[int(index)]) for index in sorted(indices)]

            recwarn.clear()
            result = reader.estimateLengthScale(
                options=sampledOptions, excludeRanges=supplied
            )
            if len(eligible) < blockCount:
                emitted = recwarn.pop(cc.EstimateWarning)
                assert str(emitted.message) == (
                    "Fewer eligible genomic blocks than requested: "
                    f"{len(eligible)} available, {blockCount} requested"
                )
                assert Path(emitted.filename) == Path(__file__)
            assert not recwarn

            assert result["settings"]["regions"] == expected
            sampling = result["settings"]["sampling"]
            assert sampling["exclusionAssembly"] == "hg38"
            assert sampling["excludeRanges"] == exclusions
            assert sampling["eligibleBlockCount"] == len(eligible)
            assert sampling["excludedBlockCount"] == len(frame) - len(eligible)
            assert sampling["terminalBP"] == sizes["chr1"] % 25000
            if supplied is extras:
                assert not any(start <= extraStart < end for _, start, end in expected)

        for start in (500000, 131000000):
            with pytest.raises(cc.InputError, match="region"):
                reader.estimateLengthScale(
                    options=options, regions=[("chr1", start, start + width)]
                )


def testCustomMasksAndHalfOpenRegions(countSources, recwarn):
    from ccounts.lengthScale import _lengthScaleExclusions

    sizes = {"chr1": 115}
    options = cc.LengthScaleOptions(blockCount=3, blockWidthBP=10, filterType="mean")
    exclusions = {"chr1": [(10, 20), (50, 51)]}
    source = cc.SourceSpec(countSources["events"], "FRAGMENTS", "s")
    with cc.openSources([source], cc.GridSpec(sizes, 2)) as reader:
        with pytest.raises(cc.InputError, match="assembly"):
            reader.estimateLengthScale(options=options)

        unmasked = reader.estimateLengthScale(options=options, excludeRanges={})
        assert unmasked["settings"]["sampling"]["eligibleBlockCount"] == 53
        assert unmasked["settings"]["sampling"]["terminalBP"] == 1

        result = reader.estimateLengthScale(options=options, excludeRanges=exclusions)
        assert result["settings"]["sampling"]["exclusionAssembly"] is None
        eligible = [
            ("chr1", start, start + 10)
            for start in range(0, 106, 2)
            if not any(
                start < end and start + 10 > begin
                for begin, end in exclusions["chr1"]
            )
        ]
        indices = np.random.default_rng(42).choice(len(eligible), 3, replace=False)
        assert result["settings"]["regions"] == [
            list(eligible[int(index)]) for index in sorted(indices)
        ]
        assert result["settings"]["sampling"]["excludedBlockCount"] == 14

        for requested in (len(eligible) - 1, len(eligible), len(eligible) + 1):
            recwarn.clear()
            sampled = reader.estimateLengthScale(
                options=replace(options, blockCount=requested),
                excludeRanges=exclusions,
            )
            assert sampled["settings"]["sampledBlockCount"] == min(requested, len(eligible))
            if requested > len(eligible):
                emitted = recwarn.pop(cc.EstimateWarning)
                assert str(emitted.message).endswith(
                    f"{len(eligible)} available, {requested} requested"
                )
            assert not recwarn

        regions = [("chr1", 22, 32), ("chr1", 20, 30)]
        result = reader.estimateLengthScale(
            options=options, regions=regions, excludeRanges=exclusions
        )
        assert result["settings"]["regions"] == [list(region) for region in regions]
        assert not recwarn

        with pytest.raises(cc.InputError, match="region"):
            reader.estimateLengthScale(
                options=options, regions=[("chr1", 10, 20)], excludeRanges=exclusions
            )

    ambiguous = {"chrM": cc.assemblyChromosomeSizes["hg38"]["chrM"]}
    with pytest.raises(cc.InputError, match="assembly"):
        _lengthScaleExclusions(ambiguous, None)
    for grid in (sizes, ambiguous):
        assembly, mask = _lengthScaleExclusions(grid, {})
        assert assembly is None and all(not intervals for intervals in mask.values())


def testPhysicalFilterGrid():
    defaultWidths = [
        300,
        500,
        700,
        1100,
        1500,
        2300,
        3500,
        5100,
        7700,
        11500,
        17300,
        25900,
        38900,
        58500,
        87700,
    ]
    options = cc.LengthScaleOptions()
    assert options.scaleEndBP == 125000

    for binSizeBP, fields, expected in (
        (100, {}, defaultWidths),
        (100, {"scaleStartBP": 1, "scaleEndBP": 500000}, defaultWidths),
        (100, {"scaleStartBP": 301, "scaleEndBP": 2500}, [500, 700, 1100, 1500, 2300]),
        (100, {"scaleStartBP": 500, "scaleEndBP": 2500}, [500, 900, 1100, 1700]),
        (
            2000,
            {"scaleStartBP": 100000, "scaleEndBP": 250000, "scaleFactor": 1.25},
            [102000],
        ),
        (41666, {"blockWidthBP": 416660}, [124998]),
    ):
        configured = replace(options, **fields)
        counts = np.zeros((1, 1, configured.blockWidthBP // binSizeBP), dtype=np.int64)

        result = cc.estimateLengthScaleCounts(counts, binSizeBP, options=configured)

        widths = result["settings"]["filterWidthsBP"]
        lower = max(3 * binSizeBP, configured.scaleStartBP or 0)
        upper = min(125000, configured.scaleEndBP, configured.blockWidthBP)
        assert widths == expected
        assert result["settings"]["scaleStartBP"] == lower
        assert all(
            lower <= width <= upper
            and width % binSizeBP == 0
            and (width // binSizeBP) % 2 == 1
            for width in widths
        )

    for binSizeBP, fields in (
        (100, {"blockWidthBP": 200}),
        (2000, {"scaleStartBP": 125000, "scaleEndBP": 125000}),
        (41667, {"blockWidthBP": 416670}),
    ):
        configured = replace(options, **fields)
        counts = np.zeros((1, 1, configured.blockWidthBP // binSizeBP), dtype=np.int64)

        with pytest.raises(cc.InputError):
            cc.estimateLengthScaleCounts(counts, binSizeBP, options=configured)


def testMappabilityCorrection():
    counts = np.array(
        [
            [[0, 8, 0, 20, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0]],
            [[2, 0, 6, 0, 10, 0, 4], [9, 1, 0, 3, 0, 0, 0]],
        ],
        dtype=np.int64,
    )
    donors = np.array(
        [
            [[False, True, False, True, False, True, False], [True] * 7],
            [[True, False, True, False, True, False, True], [False] * 7],
        ],
        dtype=bool,
    )
    stridedCounts = counts.astype(np.float64)[:, :, ::-1]
    stridedDonors = donors.copy()[:, :, ::-1]
    stridedCounts.setflags(write=False)
    stridedDonors.setflags(write=False)
    for values, eligible in (
        (counts.astype(np.int16), donors),
        ((counts * 2**40).astype(np.uint64), donors),
        (stridedCounts, stridedDonors),
        (counts, donors),
    ):
        saved = values.copy()
        outputDtype = values.dtype if values.dtype.kind in "iu" else np.dtype("int64")
        for windowBP in (100, 200, 400, 2000):
            radius = windowBP // 200
            expected = values.astype(outputDtype, copy=True)
            for block, sample, index in np.ndindex(values.shape):
                left, right = max(0, index - radius), min(7, index + radius + 1)
                selected = np.sort(
                    values[block, sample, left:right][
                        eligible[block, sample, left:right]
                    ]
                )
                if len(selected):
                    expected[block, sample, index] = max(
                        values[block, sample, index], selected[len(selected) // 2]
                    )
            corrected = cc.correctMappabilityCounts(
                values, eligible, 100, windowBP=windowBP
            )
            np.testing.assert_array_equal(corrected, expected)
            np.testing.assert_array_equal(values, saved)
            assert corrected.dtype == outputDtype
            assert not np.shares_memory(corrected, values)

    options = cc.LengthScaleOptions(blockWidthBP=700)
    for filterType in ("median", "mean"):
        configured = replace(options, filterType=filterType)
        plain = cc.estimateLengthScaleCounts(counts, 100, options=configured)
        identity = cc.estimateLengthScaleCounts(
            counts, 100, options=configured, smoothingCounts=counts.copy()
        )
        assert plain["sampleResults"] == identity["sampleResults"]
        correctedResult = cc.estimateLengthScaleCounts(
            counts, 100, options=configured, smoothingCounts=corrected
        )
        assert [
            row["totalEvents"] for row in correctedResult["sampleResults"]
        ] == counts.sum(axis=(0, 2)).tolist()
        assert correctedResult["settings"]["smoothingCountsProvided"] is True

    for invalid in (donors[0], donors.astype(int)):
        with pytest.raises(cc.InputError):
            cc.correctMappabilityCounts(counts, invalid, 100)
    with pytest.raises(cc.CountError):
        cc.correctMappabilityCounts(-counts, donors, 100)
    with pytest.raises(cc.InputError):
        cc.estimateLengthScaleCounts(
            counts, 100, options=options, smoothingCounts=counts[0]
        )


def testInvalidCountsOptionsAndRegions():
    options = cc.LengthScaleOptions(blockWidthBP=600)
    assert options.backgroundWindowBP == 50000

    for width, expected in (
        (2000000, 200000),
        (2000009, 200000),
        (60000000, 5000000),
    ):
        resolved = cc.LengthScaleOptions(blockWidthBP=width)
        assert (type(resolved.backgroundWindowBP), resolved.backgroundWindowBP) == (
            int,
            expected,
        )

    for width in (1, 6000000):
        resolved = cc.LengthScaleOptions(backgroundWindowBP=width)
        assert resolved.backgroundWindowBP == width

    for width, error in (
        (0, cc.InputError),
        (-1, cc.InputError),
        (True, cc.InputError),
        (1.5, TypeError),
    ):
        with pytest.raises(error):
            cc.LengthScaleOptions(backgroundWindowBP=width)

    for counts in (
        np.zeros((1, 2)),
        np.zeros((0, 1, 6)),
        [[[-1, 0, 0, 0, 0, 0]]],
        [[[0.5, 0, 0, 0, 0, 0]]],
        [[[np.nan, 0, 0, 0, 0, 0]]],
        [[[1j, 0, 0, 0, 0, 0]]],
        [[[2**52, 1, 0, 0, 0, 0]]],
    ):
        with pytest.raises(ValueError):
            cc.estimateLengthScaleCounts(counts, 100, options=options)

    for fields in (
        {"scaleFactor": 1},
        {"confidenceLevel": 1},
        {"returnProfiles": 1},
        {"binSizeBP": 100, "blockWidthBP": 150},
        {"filterType": "average"},
        {"filterType": "Mean"},
        {"filterType": " mean "},
        {"filterType": None},
        {"filterType": ["mean"]},
    ):
        with pytest.raises(ValueError):
            cc.LengthScaleOptions(**fields)

    with pytest.raises(TypeError):
        cc.LengthScaleOptions(bootstrapReplicates=2)

    for fields in (
        {"sampleIDs": ("s", "s")},
        {"summaryFunction": None},
        {"summaryFunction": "mean"},
        {"regions": [("chr1", 0, 600)] * 2},
        {"regions": [("chr1", 50, 650)]},
        {"options": replace(options, binSizeBP=50)},
        {"options": replace(options, scaleFactor=np.nextafter(1.0, 2.0))},
    ):
        with pytest.raises(ValueError):
            cc.estimateLengthScaleCounts(
                np.zeros((1, 1, 6)), 100, **({"options": options} | fields)
            )

    supported = np.array([[[0, 50, 50, 0, 0, 0]]] * 2)
    configured = replace(options, significanceLevel=0.99)
    for value in (None, True, [1.0], 1j, np.nan, np.inf):
        with pytest.raises((TypeError, ValueError)):
            cc.estimateLengthScaleCounts(
                supported,
                100,
                options=configured,
                summaryFunction=lambda values, *, weights: value,
            )
    with pytest.raises(TypeError):
        cc.estimateLengthScaleCounts(
            supported, 100, options=configured, summaryFunction=lambda values: 0.0
        )
    result = cc.estimateLengthScaleCounts(
        supported,
        100,
        options=configured,
        summaryFunction=lambda values, *, weights: 0.0,
    )
    assert result["scaleBP"] == 0
    assert result["SEBP"] == 0
    np.testing.assert_array_equal(result["blockInfluenceBP"], [0.0, 0.0])
