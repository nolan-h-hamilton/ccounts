import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pysam
import pytest

import ccounts as cc


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "countMode,expectedCounts,expectedMass",
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
def testEventArithmeticAndBlockOwnership(countSources, dtype, countMode, expectedCounts, expectedMass):
    grid = cc.GridSpec.fromSizesFile(countSources["sizes"], binSizeBP=50)
    sources = [
        cc.SourceSpec(
            countSources["eventsBAM"], "BAM", "bam",
            countOptions=cc.CountOptions(countMode=countMode, bamInputMode="fragments"),
            selection=("A",),
        ),
        cc.SourceSpec(
            countSources["events"], "FRAGMENTS", "fragments",
            countOptions=cc.CountOptions(countMode=countMode, fragmentsUseReadSupport=True),
            selection=("A",),
        ),
    ]
    with cc.openSources(sources, grid, dtype=dtype) as reader:
        whole = reader.readRegion("chr1", 0, 115, returnRawNoiseMass=True)
        np.testing.assert_allclose(whole.values, [expectedCounts] * 2, rtol=2e-7)
        np.testing.assert_allclose(whole.rawNoiseMass, [expectedMass] * 2, rtol=2e-7)
        assert whole.values.dtype == np.dtype(dtype)
        assert whole.rawNoiseMass.dtype == np.dtype(dtype)
        totals = reader.libraryTotals(chromosomes=["chr1"])
        for record in totals.records.values():
            assert record["eventMass"] == (12 if countMode == "fiveprime" else 6)
            assert record["spanBP"] == 220
        for blockSizeBins in (1, 2, 7):
            blocks = list(reader.iterBlocks(["chr1"], blockSizeBins=blockSizeBins, returnRawNoiseMass=True))
            np.testing.assert_array_equal(np.concatenate([block.values for block in blocks], axis=1), whole.values)
            np.testing.assert_array_equal(np.concatenate([block.rawNoiseMass for block in blocks], axis=1), whole.rawNoiseMass)
        for startBP, endBP, column in [(100, 115, 2), (0, 50, 0), (50, 100, 1)]:
            part = reader.readRegion("chr1", startBP, endBP, returnRawNoiseMass=True)
            np.testing.assert_array_equal(part.values[:, 0], whole.values[:, column])
            np.testing.assert_array_equal(part.rawNoiseMass[:, 0], whole.rawNoiseMass[:, column])
        if countMode == "coverage":
            np.testing.assert_allclose(whole.values @ np.array([50, 50, 15]), [220, 220], rtol=2e-7)
        if countMode == "conservedFractionalOverlap":
            np.testing.assert_allclose(whole.values.sum(axis=1), [6, 6], rtol=2e-7)
        ownedValues = whole.values.copy()
        ownedMass = whole.rawNoiseMass.copy()
        reader.readRegion("chr1", 0, 50)
    np.testing.assert_array_equal(whole.values, ownedValues)
    np.testing.assert_array_equal(whole.rawNoiseMass, ownedMass)


@pytest.mark.parametrize("countMode", ["coverage", "binOverlap", "conservedFractionalOverlap", "center", "fiveprime"])
def testNormalizationWidthsAndUnits(countSources, countMode):
    grid = cc.GridSpec({"chr1": 115}, binSizeBP=50)
    source = cc.SourceSpec(
        countSources["events"], "FRAGMENTS", "A",
        countOptions=cc.CountOptions(countMode=countMode, fragmentsUseReadSupport=True),
        selection=("A",),
    )
    with cc.openSources([source], grid, dtype="float64") as reader:
        block = reader.readRegion("chr1", 0, 115, returnRawNoiseMass=True)
        preserved = block.values.copy()
        totals = reader.libraryTotals(chromosomes=["chr1"])
        eventTotal = 12 if countMode == "fiveprime" else 6
        expectedFactors = {
            "CPM": 1e6 / eventTotal,
            "RPKM": None if countMode == "coverage" else 1e9 / eventTotal / 50,
            "RPGC": {"coverage": 115 / 220,
                     "conservedFractionalOverlap": 115 / eventTotal / 50}.get(countMode),
        }
        for method, multiplier in expectedFactors.items():
            options = {"effectiveGenomeSizeBP": 115} if method == "RPGC" else {}
            if multiplier is None:
                with pytest.raises(ValueError):
                    cc.Normalization.fromTotals(totals, method=method, **options)
                continue
            normalization = cc.Normalization.fromTotals(totals, method=method, **options)
            whole = normalization.apply(block)
            np.testing.assert_allclose(whole, preserved * multiplier)
            effectiveFactor = normalization.multipliers(block, sampleID="A")
            assert np.isscalar(effectiveFactor)
            np.testing.assert_allclose(effectiveFactor, multiplier)
            for startBP, endBP, column in [(0, 50, 0), (50, 100, 1), (100, 115, 2)]:
                part = reader.readRegion("chr1", startBP, endBP)
                assert np.isscalar(normalization.multipliers(part, sampleID="A"))
                np.testing.assert_array_equal(normalization.apply(part), whole[:, column:column + 1])
        np.testing.assert_array_equal(block.values, preserved)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(4,), (2, 2)])
def testLogPrecisionAndArrayContracts(dtype, shape):
    with mp.workdps(400):
        reference = np.array([1, 2**20, 2**40, 0], dtype=dtype)
        values = np.nextafter(reference, np.array(np.inf, dtype=dtype)).reshape(shape)
        reference = reference.reshape(shape)
        expected = np.array([
            float(mp.log((mp.mpf(float(x)) + 1) / (mp.mpf(float(r)) + 1), 2))
            for x, r in zip(values.flat, reference.flat)
        ], dtype=dtype).reshape(shape)
    preserved = values.copy()
    result = cc.logTransform(values, reference=reference)
    assert result.shape == shape
    assert result.dtype == dtype
    np.testing.assert_allclose(result, expected, rtol=4 * np.finfo(dtype).eps, atol=0)
    np.testing.assert_array_equal(values, preserved)
    out = np.empty_like(values)
    assert cc.logTransform(values, reference=reference, out=out) is out
    np.testing.assert_array_equal(out, result)
    inPlace = values.copy()
    assert cc.logTransform(inPlace, reference=reference, out=inPlace) is inPlace
    np.testing.assert_array_equal(inPlace, result)
    storage = np.arange(5, dtype=dtype)
    with pytest.raises(ValueError, match="overlap"):
        cc.logTransform(storage[:-1], out=storage[1:])
    with pytest.raises(ValueError, match="overlap"):
        cc.logTransform(np.ones(4, dtype=dtype), reference=storage[:-1], out=storage[1:])
    with pytest.raises(ValueError):
        cc.logTransform(values, out=np.empty(shape, dtype=np.float64 if dtype == np.float32 else np.float32))
    readOnly = np.empty_like(values)
    readOnly.flags.writeable = False
    with pytest.raises(ValueError):
        cc.logTransform(values, out=readOnly)


def testLogExtremeFiniteArguments():
    values = np.array([1e308, np.nextafter(1e308, np.inf), -9e307, 1e-300], dtype=np.float64)
    reference = np.array([np.nextafter(1e308, 0), 1e308, 1e308, 0], dtype=np.float64)
    pseudocount = 1e308
    with mp.workdps(160):
        expectedSingle = np.array([
            float(mp.log(mp.mpf(float(x)) + mp.mpf(pseudocount), 2)) for x in values
        ])
        expectedPair = np.array([
            float(mp.log((mp.mpf(float(x)) + mp.mpf(pseudocount)) / (mp.mpf(float(r)) + mp.mpf(pseudocount)), 2))
            for x, r in zip(values, reference)
        ])
    np.testing.assert_allclose(cc.logTransform(values, pseudocount=pseudocount), expectedSingle, rtol=3e-15)
    np.testing.assert_allclose(cc.logTransform(values, reference=reference, pseudocount=pseudocount), expectedPair, rtol=5e-15, atol=0)
    for pseudocount in (0, -1, math.inf, math.nan, True):
        with pytest.raises(ValueError):
            cc.logTransform(np.ones(2), pseudocount=pseudocount)
    for values in (np.array([-1.0]), np.array([math.nan]), np.array([math.inf])):
        with pytest.raises(ValueError):
            cc.logTransform(values)


@pytest.mark.parametrize("pseudocount", [1e-100, 1e100, 1e308])
def testLogNearNegativeDomainBoundary(pseudocount):
    reference = -pseudocount
    for _ in range(16):
        reference = np.nextafter(reference, 0)
    value = np.nextafter(reference, 0)
    result = cc.logTransform(
        np.array([value]), reference=np.array([reference]), pseudocount=pseudocount,
    )
    np.testing.assert_allclose(result, [math.log2(17 / 16)], rtol=5e-15)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testLogPseudocountPrecisionAtDomainBoundary(dtype):
    pseudocount = 1 + 2**-30
    tolerance = 4 * np.finfo(dtype).eps
    boundary = np.array([-1], dtype=dtype)
    single = cc.logTransform(boundary, pseudocount=pseudocount)
    assert single.dtype == np.dtype(dtype)
    np.testing.assert_allclose(single, [-30], rtol=tolerance, atol=0)
    pairValues = np.array([[0, -1]], dtype=dtype)
    pairReference = np.array([[-1, 0]], dtype=dtype)
    expectedPair = np.array([[30 + math.log2(pseudocount), -30 - math.log2(pseudocount)]], dtype=dtype)
    assert cc.logTransform(pairValues, reference=pairReference, pseudocount=pseudocount, out=pairValues) is pairValues
    np.testing.assert_allclose(pairValues, expectedPair, rtol=tolerance, atol=0)
    for invalidPseudocount in (1.0, 1 - 2**-30):
        for values, reference in ((boundary, None), (np.zeros_like(boundary), boundary)):
            out = np.full_like(values, 7)
            with pytest.raises(ValueError, match="greater than -pseudocount"):
                cc.logTransform(values, reference=reference, pseudocount=invalidPseudocount, out=out)
            np.testing.assert_array_equal(out, np.full_like(values, 7))
    for shape in ((0,), (2, 0)):
        empty = np.empty(shape, dtype=dtype)
        result = cc.logTransform(empty, reference=empty, pseudocount=pseudocount)
        assert result.shape == shape
        assert result.dtype == np.dtype(dtype)
    for pseudocount, entries in ((1e-300, [0]), (1e-100, [0]), (1e300, [-1, 0])):
        values = np.array(entries, dtype=dtype)
        with np.errstate(over="raise", invalid="raise"):
            result = cc.logTransform(values, pseudocount=pseudocount)
        assert result.dtype == np.dtype(dtype)
        expected = np.full(values.shape, math.log2(pseudocount), dtype=dtype)
        np.testing.assert_allclose(result, expected, rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def testLogTinySignedValues(dtype):
    values = np.array([-1e-20, 0, 1e-20], dtype=dtype)
    expected = (np.log1p(values.astype(np.float64)) / math.log(2)).astype(dtype)
    np.testing.assert_allclose(cc.logTransform(values), expected, rtol=3 * np.finfo(dtype).eps, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def testSuppliedFactorsOutputAndOverflow(countSources, dtype):
    grid = cc.GridSpec({"chr1": 115}, binSizeBP=50)
    source = cc.SourceSpec(
        countSources["events"], "FRAGMENTS", "A",
        countOptions=cc.CountOptions(countMode="center", fragmentsUseReadSupport=True),
        selection=("A",),
    )
    with cc.openSources([source], grid, dtype=dtype) as reader:
        block = reader.readRegion("chr1", 0, 115)
        normalization = cc.Normalization.fromFactors(["A"], [2])
        out = np.empty_like(block.values)
        assert normalization.apply(block, out=out) is out
        np.testing.assert_array_equal(out, block.values * 2)
        with pytest.raises(ValueError):
            normalization.forPair("A", "B")
        with pytest.raises((ValueError, OverflowError, FloatingPointError)):
            cc.Normalization.fromFactors(["A"], [np.finfo(dtype).max]).apply(block)
        storage = np.empty((1, 4), dtype=dtype)
        with pytest.raises(ValueError):
            reader.readRegion("chr1", 0, 115, out=storage[:, :2])
        with pytest.raises(ValueError):
            reader.readRegion("chr1", -50, 50)
        with pytest.raises(ValueError):
            reader.readRegion("chr1", 0, 100.0)


def testExactTotalsOwnershipAndReuse(countSources):
    grid = cc.GridSpec({"chrTotals": 115}, binSizeBP=50)
    sources = [
        cc.SourceSpec(
            countSources["eventsBAM"], "BAM", sampleID,
            countOptions=cc.CountOptions(countMode="coverage", bamInputMode="reads"),
        )
        for sampleID in ("A", "B")
    ]
    with cc.openSources(sources, grid, dtype="float64") as reader:
        block = reader.readRegion("chrTotals", 0, 115)
        assert not reader._totals
        totals = reader.libraryTotals(chromosomes=["chrTotals"])
        assert reader.libraryTotals(chromosomes=("chrTotals",)) is totals
        for record in totals.records.values():
            assert record["requestedMethod"] == record["usedMethod"] == "exact"
            assert record["eventMass"] == 7
            assert record["spanBP"] == 26
            assert record["metadata"]["chromosomes"]["chrTotals"] == {
                "scannedRecords": 7, "eventMass": 7, "spanBP": 26,
            }
        np.testing.assert_allclose(block.values @ np.array([50, 50, 15]), [26, 26])
    with cc.openSources(sources[::-1], grid) as reader:
        reordered = reader.libraryTotals(chromosomes=["chrTotals"])
        assert reordered.sampleIDs == ("B", "A")
        assert reordered.records == totals.records


@pytest.mark.parametrize(
    "countMode,counts,mass,eventMass",
    [
        ("coverage", [2, 2, 2], [2, 2, 2], 2),
        ("binOverlap", [2, 2, 2], [2, 2, 2], 2),
        ("conservedFractionalOverlap", [0.8, 0.8, 0.4], [0.32, 0.32, 0.08], 2),
        ("center", [0, 2, 0], [0, 2, 0], 2),
        ("fiveprime", [0, 0, 0], [0, 0, 0], 0),
        ("ffp", [0, 0, 0], [0, 0, 0], 0),
    ],
)
def testShiftedExtensionChromosomeClipping(tmp_path, countMode, counts, mass, eventMass):
    bamPath = tmp_path / "clipped.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 25}]}
    with pysam.AlignmentFile(bamPath, "wb", header=header) as handle:
        for start, flag in [(0, 0), (20, 16)]:
            record = pysam.AlignedSegment()
            record.query_name = f"r{start}"
            record.query_sequence = "AAAAA"
            record.flag = flag
            record.reference_id = 0
            record.reference_start = start
            record.mapping_quality = 60
            record.cigar = [(0, 5)]
            handle.write(record)
    pysam.index(str(bamPath))
    grid = cc.GridSpec({"chr1": 25}, binSizeBP=10)
    source = cc.SourceSpec(
        str(bamPath), "BAM", "A",
        countOptions=cc.CountOptions(
            countMode=countMode, bamInputMode="reads", shiftForward5p=-10,
            shiftReverse5p=-10, extendFrom5pBP=35,
        ),
    )
    with cc.openSources([source], grid, dtype="float64") as reader:
        block = reader.readRegion("chr1", 0, 25, returnRawNoiseMass=True)
        np.testing.assert_allclose(block.values, [counts], rtol=1e-14)
        np.testing.assert_allclose(block.rawNoiseMass, [mass], rtol=1e-14)
        pieces = list(reader.iterBlocks(["chr1"], blockSizeBins=1, returnRawNoiseMass=True))
        np.testing.assert_array_equal(np.concatenate([piece.values for piece in pieces], axis=1), block.values)
        np.testing.assert_array_equal(np.concatenate([piece.rawNoiseMass for piece in pieces], axis=1), block.rawNoiseMass)
        totals = reader.libraryTotals(chromosomes=["chr1"])
        assert totals.records["A"]["eventMass"] == eventMass
        if countMode in ("coverage", "binOverlap"):
            assert totals.records["A"]["spanBP"] == 50


def testEmptyBAMRawPathAndZeroTotals(tmp_path):
    bamPath = tmp_path / "empty.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 25}]}
    with pysam.AlignmentFile(bamPath, "wb", header=header):
        pass
    cc.prepareIndex(bamPath, "BAM")
    assert Path(str(bamPath) + ".bai").exists()
    source = cc.SourceSpec(str(bamPath), "BAM", "empty")
    with cc.openSources([source], cc.GridSpec({"chr1": 25}, binSizeBP=10)) as reader:
        block = reader.readRegion("chr1", 0, 25, returnRawNoiseMass=True)
        np.testing.assert_array_equal(block.values, np.zeros((1, 3), dtype=np.float32))
        np.testing.assert_array_equal(block.rawNoiseMass, np.zeros((1, 3), dtype=np.float32))
        totals = reader.libraryTotals(chromosomes=["chr1"])
        assert totals.records["empty"]["eventMass"] == 0
        with pytest.raises(ValueError):
            cc.Normalization.fromTotals(totals, method="CPM")


@pytest.mark.parametrize("countMode", ["coverage", "binOverlap"])
@pytest.mark.parametrize("sourceName,chromosomeLength,expected", [
    ("cancellation", 200, [2**54] * 10 + [1] * 5 + [0] * 5),
    ("precision", 115, [float(2**54 + 3)] * 12),
])
def testCoverageLargeSupportCancellation(countSources, countMode, sourceName, chromosomeLength, expected):
    source = cc.SourceSpec(
        countSources[sourceName], "FRAGMENTS", "A",
        countOptions=cc.CountOptions(countMode=countMode, fragmentsUseReadSupport=True),
    )
    grid = cc.GridSpec({"chr1": chromosomeLength}, binSizeBP=10)
    expected = np.array(expected, dtype=np.float64)
    with cc.openSources([source], grid, dtype="float64") as reader:
        whole = reader.readRegion("chr1", 0, chromosomeLength, returnRawNoiseMass=True)
        np.testing.assert_array_equal(whole.values[0], expected)
        np.testing.assert_array_equal(whole.rawNoiseMass[0], expected)
        for blockSizeBins in (1, 3, 5, 7):
            blocks = list(reader.iterBlocks(["chr1"], blockSizeBins=blockSizeBins, returnRawNoiseMass=True))
            np.testing.assert_array_equal(np.concatenate([block.values for block in blocks], axis=1)[0], expected)
            np.testing.assert_array_equal(np.concatenate([block.rawNoiseMass for block in blocks], axis=1)[0], expected)
        for startBP in reversed(range(0, chromosomeLength, grid.binSizeBP)):
            part = reader.readRegion(
                "chr1", startBP, min(startBP + grid.binSizeBP, chromosomeLength), returnRawNoiseMass=True,
            )
            column = startBP // grid.binSizeBP
            np.testing.assert_array_equal(part.values[0], expected[column:column + 1])
            np.testing.assert_array_equal(part.rawNoiseMass[0], expected[column:column + 1])



@pytest.mark.parametrize(
    "profile,columns,extraColumns,minRefDist,centerMedian,expected",
    [
        ([[4], [16], [64]], range(0, 100, 10), (), 10, False, [4, 1, 0.25]),
        ([[1e40], [2e40], [4e40]], range(0, 100, 10), (), 10, False, [2, 1, 0.5]),
        ([[1e300], [1e300], [1e300]], range(0, 100, 10), (), 10, False, [1, 1, 1]),
        ([[2**24], [2**24 + 1]], range(0, 100, 10), (), 10, False,
         [math.sqrt(1 + 2**-24), 1 / math.sqrt(1 + 2**-24)]),
        ([[1] * 4 + [4, 16] + [64] * 4, [64] * 4 + [16, 4] + [1] * 4],
         range(10), (), 0, False, [1, 1]),
        ([[1] * 4 + [4, 4, 16] + [64] * 4, [64] * 4 + [16, 16, 4] + [1] * 4],
         range(0, 110, 10), (), 10, True,
         [2 * math.sqrt((0.5 + 1e-8) * (2 + 1e-8)),
          0.5 * math.sqrt((0.5 + 1e-8) * (2 + 1e-8))]),
        ([[1], [1], [1_000_000]], range(0, 100, 10), (), 10, True,
         [1.00000005, 1.00000005, 0.2]),
        ([[4], [16], [64]], [1, 12, 23, 34, 46, 57, 72, 84, 95, 106, 118],
         [3, 19, 39, 50, 65, 75, 87, 99, 109], 10, False, [4, 1, 0.25]),
    ],
)
def testSizeFactorArithmeticAndApplication(profile, columns, extraColumns, minRefDist, centerMedian, expected):
    counts = np.full((len(profile), max(columns) + 1), 0.5, dtype=np.float64)
    if extraColumns:
        counts[:, extraColumns] = np.asarray(profile)[::-1]
    counts[:, columns] = profile
    preserved = counts.copy()
    factors = cc.estimateSizeFactors(counts, centerMedian=centerMedian, minRefDist=minRefDist)
    assert factors.dtype == np.float64
    np.testing.assert_allclose(factors, expected, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(counts, preserved)
    sampleIDs = tuple(f"s{index}" for index in range(len(profile)))
    block = cc.CountBlock(
        np.arange(1, len(profile) * 3 + 1, dtype=np.float64).reshape(-1, 3),
        sampleIDs, cc.GridSpec({"chr1": 115}, 50), "chr1", 0, 115,
        ("eventMass",) * len(profile), ("center",) * len(profile),
    )
    normalization = cc.Normalization.fromFactors(sampleIDs, factors)
    np.testing.assert_allclose(normalization.apply(block), block.values * np.asarray(expected)[:, None], rtol=1e-12)


@pytest.mark.parametrize("shape,badValue,options", [
    ((110,), None, {}), ((0, 110), None, {}), ((3, 0), None, {}),
    ((3, 81), None, {}), ((3, 110), 0.5, {}),
    ((3, 110), -1, {}), ((3, 110), np.nan, {}), ((3, 110), np.inf, {}),
    ((3, 100), np.nextafter(1.0, 0.0), {}),
    ((3, 110), None, {"minRefDist": -1}),
    ((3, 110), None, {"minRefDist": 1.5}),
    ((3, 110), None, {"minRefDist": True}),
    ((3, 110), None, {"centerMedian": 1}),
])
def testSizeFactorInvalidInputs(shape, badValue, options):
    counts = np.ones(shape)
    if badValue is not None and 0 <= badValue < 1:
        counts.fill(badValue)
    elif badValue is not None:
        counts.flat[0] = badValue
    with pytest.raises(ValueError):
        cc.estimateSizeFactors(counts, **options)


def testPrivateLogBindingRejectsInvalidStorage():
    from ccounts import _native

    values = np.asarray([1.0, 2.0], dtype=np.float64)
    readonly = np.empty_like(values)
    readonly.flags.writeable = False
    cases = [
        (values, None, np.empty(2, dtype=np.float32)),
        (values, None, np.empty(1, dtype=np.float64)),
        (np.asarray([1, 2], dtype=np.int64), None, np.empty(2, dtype=np.int64)),
        (np.ones((2, 2), order="F"), None, np.empty((2, 2))),
        (np.ones((2, 2)), None, np.empty((2, 2), order="F")),
        (values, None, readonly),
        (values, np.ones(2, dtype=np.float32), np.empty_like(values)),
        (values, np.ones(1), np.empty_like(values)),
        (np.ones((1, 2)), None, np.empty(2)),
        (np.ones((1, 1, 2)), None, np.empty((1, 1, 2))),
        ([1.0, 2.0], None, np.empty_like(values)),
    ]
    for source, reference, target in cases:
        with pytest.raises(ValueError):
            _native.logValues(source, reference, 1.0, target)
    storage = np.arange(5.0)
    with pytest.raises(ValueError, match="overlap"):
        _native.logValues(storage[:-1], None, 1.0, storage[1:])
    with pytest.raises(ValueError, match="overlap"):
        _native.logValues(np.ones(4), storage[:-1], 1.0, storage[1:])
    for pseudocount in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            _native.logValues(values, None, pseudocount, np.empty_like(values))
    result = values.copy()
    assert _native.logValues(result, None, 1.0, result) is result
    np.testing.assert_allclose(result, np.log2(values + 1.0))
