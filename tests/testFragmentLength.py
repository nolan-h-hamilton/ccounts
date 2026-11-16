from dataclasses import asdict, replace
from pathlib import Path
import warnings
import numpy as np
import pytest
import ccounts as cc
from ccounts import _native
from ccounts.exceptions import EstimateWarning
from ccounts.fragmentLength import _poolCurves


@pytest.mark.parametrize(
    "width,forward,reverse,minimum,maximum,step",
    [
        (128, [0, 0, 5, 17, 37, 80, 127], [1, 9, 21, 44, 81, 128], 5, 90, 1),
        (17, [0, 0, 2, 4, 8, 13, 16], [1, 3, 5, 10, 17], 1, 22, 3),
        (31, [0, 8, 16, 24], [8, 16, 24], 8, 24, 8),
        (8, list(range(8)), list(range(1, 9)), 1, 4, 1),
    ],
)
def testCorrelation(width, forward, reverse, minimum, maximum, step):
    result = _native.estimateBlock(
        np.array(forward, dtype=np.int64),
        np.array(reverse, dtype=np.int64),
        width,
        minLengthBP=minimum,
        maxLengthBP=maximum,
        stepBP=step,
    )

    candidates = np.arange(minimum, maximum + 1, step)
    forward = np.bincount(forward, minlength=width + 1)
    reverse = np.bincount(reverse, minlength=width + 1)
    expected = np.full(candidates.size, np.nan)

    for index, distance in enumerate(candidates):
        overlap = width + 1 - distance
        if overlap > 1 and np.var(forward[:overlap]) and np.var(reverse[distance:]):
            expected[index] = np.corrcoef(forward[:overlap], reverse[distance:])[0, 1]

    if not np.isfinite(expected).any():
        assert result is None
        return

    np.testing.assert_allclose(
        result["pointCurve"], expected, rtol=1e-12, atol=1e-14, equal_nan=True
    )
    assert result["estimateBP"] == candidates[np.nanargmax(expected)]


@pytest.mark.parametrize(
    "curves,counts",
    [
        (
            [[0.2, np.nan, np.nan], [np.nan, 0.5, np.nan], [0.1, 0.3, np.nan]],
            [10, 20, 30],
        ),
        ([[1, 1, 0], [0.5, 0.5, 0]], [10, 30]),
        ([[-1, -2, -3], [-0.5, -1, -1.5]], [10, 30]),
        ([[0.5, 0.1, 0.2], [np.nan] * 3] + [[0.2, 0.8, 0.1]] * 6, [2, 0] + [400] * 6),
        ([[1, 0, 0], [0, 0, 1]], [10, 10]),
    ],
)
@pytest.mark.parametrize(
    "smoothingWidthBP,candidateStepBP", [(1, 1), (3, 1), (11, 1), (11, 400)]
)
def testWholeBlockUncertainty(curves, counts, smoothingWidthBP, candidateStepBP):
    curves, counts = np.asarray(curves), np.asarray(counts)
    candidates = 100 + np.arange(3) * candidateStepBP

    result = _poolCurves(
        curves,
        counts,
        candidates,
        repetitions=128,
        randomSeed=73,
        smoothingWidthBP=smoothingWidthBP,
    )

    draws = np.random.default_rng(73).integers(len(counts), size=(128, len(counts)))
    expectedPeaks = []

    for indices in [np.arange(len(counts)), *draws]:
        expected = []

        for column in curves.T:
            usable = indices[np.isfinite(column[indices]) & (counts[indices] > 0)]
            expected.append(
                np.average(column[usable], weights=counts[usable])
                if usable.size
                else np.nan
            )

        raw = np.asarray(expected)
        expected = [
            raw[window].mean() if window.any() else np.nan
            for position in candidates
            for window in [
                (np.abs(candidates - position) <= smoothingWidthBP // 2)
                & np.isfinite(raw)
            ]
        ]
        expectedPeaks.append(candidates[np.nanargmax(expected)])

        if len(expectedPeaks) == 1:
            np.testing.assert_allclose(result["rawPooledCurve"], raw, equal_nan=True)
            np.testing.assert_allclose(result["pooledCurve"], expected, equal_nan=True)
            np.testing.assert_allclose(result["normalizedWeights"].sum(), 1)

    assert result["estimateBP"] == expectedPeaks[0]
    np.testing.assert_array_equal(result["bootstrapEstimatesBP"], expectedPeaks[1:])

    np.testing.assert_allclose(
        [result["CILowerBP"], result["CIUpperBP"]],
        np.quantile(expectedPeaks[1:], [0.025, 0.975]),
    )

    np.testing.assert_allclose(
        result["bootstrapStdBP"], np.std(expectedPeaks[1:], ddof=1)
    )

    if curves.shape == (2, 3) and curves[1, -1] == 1 and smoothingWidthBP == 1:
        assert result["CIUpperBP"] - result["CILowerBP"] == 2 * candidateStepBP


def testEndpointFixtureAndReader():
    data = Path(__file__).parent / "data"
    options = cc.CountOptions(countMode="fiveprime", bamInputMode="reads")
    native = _native.NativeSource(
        data / "endpoints.bam",
        "BAM",
        {"chr1": 325},
        barcodeAllowListFile=str(data / "allow.txt"),
    )

    try:
        for start, forward, reverse in (
            (0, [5, 20, 35], [45, 60, 75, 100]),
            (100, [0, 5, 20, 35], [10, 45, 60, 75]),
        ):
            actualF, actualR, counts = native.blockEndpoints(
                "chr1", start, start + 100, options
            )

            np.testing.assert_array_equal(actualF, forward)
            np.testing.assert_array_equal(actualR, reverse)
            assert all(
                counts[name] == 1
                for name in (
                    "removedForwardCoordinateDuplicates",
                    "removedReverseCoordinateDuplicates",
                    "excludedFlags",
                    "excludedMappingQuality",
                    "excludedSelection",
                )
            )
    finally:
        native.close()

    estimator = cc.FragmentLengthOptions(
        blockWidthBP=100, maxLengthBP=90, bootstrapReplicates=64
    )
    source = cc.SourceSpec(
        data / "endpoints.bam", "BAM", "s", countOptions=options, selection=("A",)
    )
    grid = cc.GridSpec({"chr1": 325}, binSizeBP=10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EstimateWarning)

        with cc.openSources([source], grid, fragmentLengthOptions=estimator) as reader:
            counts = reader.readRegion("chr1", 0, 300).values.copy()

            estimate = reader.estimateFragmentLength("s")
            diagnostics = reader.sourceInfo["s"]["fragmentLengthDiagnostics"]

            assert diagnostics["settings"]["smoothingWidthBP"] == 11

            assert (
                type(estimate) is int and reader.estimateFragmentLength("s") == estimate
            )

            assert (
                diagnostics["readLengthBP"],
                diagnostics["effectiveMinLengthBP"],
                diagnostics["effectiveMaxLengthBP"],
            ) == (10, 11, 90)

            assert (
                diagnostics["sampledBlockCount"]
                == diagnostics["samplingFrameBlockCount"]
                == 5
            )

            assert (
                diagnostics["excludedShortChromosomeBP"]
                == diagnostics["excludedTerminalBP"]
                == 75
            )

            assert (
                diagnostics["chromosomeRepresentation"]["chrExtra"]["sampledBlockCount"]
                == 2
            )
            np.testing.assert_array_equal(diagnostics["selectedBlockIDs"], np.arange(5))
            np.testing.assert_array_equal(
                reader.readRegion("chr1", 0, 300).values, counts
            )

        source = replace(
            source, countOptions=replace(options, inferFragmentLength=True)
        )

        with cc.openSources([source], grid, fragmentLengthOptions=estimator) as reader:
            info = reader.sourceInfo["s"]

            assert (
                info["fragmentLengthBP"]
                == info["countOptions"].extendFrom5pBP
                == estimate
            )
            np.testing.assert_array_equal(
                info["fragmentLengthDiagnostics"]["bootstrapEstimatesBP"],
                diagnostics["bootstrapEstimatesBP"],
            )

            reader.setReadExtension("s", 30)

            assert reader.sourceInfo["s"]["fragmentLengthDiagnostics"] is None

    with pytest.raises(ValueError):
        cc.openSources([source], grid)

    for explicit, step, bounds in (
        (35, 1, (35, 90)),
        (1, 1, (11, 90)),
        (35, 7, (35, 84)),
    ):
        configured = replace(estimator, minLengthBP=explicit, candidateStepBP=step)

        with cc.openSources([source], grid, fragmentLengthOptions=configured) as reader:
            result = reader.sourceInfo["s"]["fragmentLengthDiagnostics"]

            assert (
                result["effectiveMinLengthBP"],
                result["effectiveMaxLengthBP"],
            ) == bounds

    with pytest.raises(ValueError):
        cc.openSources(
            [source], grid, fragmentLengthOptions=replace(estimator, maxLengthBP=10)
        )

    with pytest.warns(EstimateWarning, match=str(EstimateWarning())) as emitted:
        with cc.openSources(
            [source, replace(source, sampleID="other")],
            grid,
            fragmentLengthOptions=estimator,
        ):
            pass
    assert len(emitted) == 1


def testEstimatorSettingsAndFailures():
    assert asdict(cc.FragmentLengthOptions()) == {
        "blockCount": 512,
        "blockWidthBP": 1000000,
        "minLengthBP": None,
        "maxLengthBP": 1000,
        "candidateStepBP": 1,
        "bootstrapReplicates": 1000,
        "randomSeed": 42,
        "smoothingWidthBP": 11,
    }

    for fields in (
        {"blockCount": 0},
        {"blockWidthBP": 1000},
        {"candidateStepBP": 0},
        {"bootstrapReplicates": 1},
        {"minLengthBP": 0},
        {"smoothingWidthBP": 0},
        {"smoothingWidthBP": 2},
        {"smoothingWidthBP": 1.5},
    ):
        with pytest.raises((ValueError, TypeError)):
            cc.FragmentLengthOptions(**fields)

    for curves, counts in (
        (np.full((2, 3), np.nan), [1, 1]),
        (np.ones((2, 3)), [0, 0]),
        ([[0.1, 0.2, 0.1], [np.nan] * 3], [10, 0]),
    ):
        with pytest.raises(ValueError):
            _poolCurves(
                np.asarray(curves),
                np.asarray(counts),
                np.array([100, 200, 300]),
                repetitions=128,
                randomSeed=73,
            )
