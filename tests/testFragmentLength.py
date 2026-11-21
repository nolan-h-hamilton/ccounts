from dataclasses import asdict, replace
from pathlib import Path
import os
import warnings
import numpy as np
import pysam
import pytest
import ccounts as cc
from ccounts import _native
from ccounts.exceptions import EstimateError, EstimateWarning


def testChromosomeCrossCorrelation(tmp_path):
    data = Path(__file__).parent / "data"
    path = tmp_path / "crossCorrelation.bam"
    sizes = {"chr1": 325, "chr2": 250, "chr3": 30, "chrExtra": 325}
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in sizes.items()],
    }
    with (
        pysam.AlignmentFile(data / "endpoints.bam") as original,
        pysam.AlignmentFile(path, "wb", header=header) as output,
    ):
        for tid, chromosome in enumerate(("chr1", "chrExtra", "chr1", "chr1")):
            records = list(original.fetch(chromosome))
            if tid == 1:
                for read in records:
                    if read.is_reverse:
                        read.reference_start = (
                            246
                            if read.reference_start == 165
                            else read.reference_start + 10
                        )
                records.sort(key=lambda read: read.reference_start)
            if tid == 2:
                records = records[:2]
            for read in records:
                fields = read.to_string().split("\t")
                fields[2] = output.get_reference_name(tid)
                output.write(
                    pysam.AlignedSegment.fromstring("\t".join(fields), output.header)
                )
    pysam.index(str(path))

    settings = cc.FragmentLengthOptions(minLengthBP=1, maxLengthBP=90)
    options = cc.CountOptions(countMode="fiveprime", bamInputMode="reads")
    source = cc.SourceSpec(path, "BAM", "s", options, selection=("A",))
    grid = cc.GridSpec(sizes, binSizeBP=10)
    forward = [
        [5, 5, 20, 35, 100, 105, 105, 120, 135, 205, 205, 220, 235],
        [5, 5, 20, 35, 105, 105, 120, 135],
    ]
    reverse = [
        [45, 45, 60, 75, 100, 110, 145, 145, 160, 175, 245, 245, 260, 275],
        [55, 55, 70, 85, 155, 155, 170, 250],
    ]
    lags = np.arange(91)
    expected = []
    weights = []
    for length, starts, ends in zip((325, 250), forward, reverse):
        f = np.bincount(starts, minlength=length + 1)
        r = np.bincount(ends, minlength=length + 1)
        expected.append(
            [np.corrcoef(f[: length + 1 - lag], r[lag:])[0, 1] for lag in lags]
        )
        weights.append(len(starts) + len(ends))
    pooled = np.average(expected, axis=0, weights=weights)

    with cc.openSources([source], grid, fragmentLengthOptions=settings) as reader:
        before = reader.readRegion("chr1", 0, 300).values.copy()
        estimate = reader.estimateFragmentLength("s")
        diagnostics = reader.sourceInfo["s"]["fragmentLengthDiagnostics"]
        assert estimate == reader.estimateFragmentLength("s") == 1 + pooled[1:].argmax()
        np.testing.assert_array_equal(reader.readRegion("chr1", 0, 300).values, before)

    assert diagnostics["chromosomeNames"] == ("chr1", "chr2", "chr3")
    assert diagnostics["chromosomeEstimatesBP"] == {
        "chr1": 40,
        "chr2": 50,
        "chr3": None,
    }
    assert diagnostics["chromosomeHeterogeneity"] == {
        "chromosomeCount": 2,
        "minBP": 40,
        "maxBP": 50,
        "medianBP": 45.0,
        "iqrBP": 5.0,
    }
    np.testing.assert_array_equal(diagnostics["candidateLengthsBP"], lags)
    np.testing.assert_array_equal(
        diagnostics["chromosomeReadCounts"], [[13, 14], [8, 8], [2, 0]]
    )
    np.testing.assert_allclose(
        diagnostics["chromosomeCurves"][:2], expected, atol=1e-14
    )
    np.testing.assert_array_equal(diagnostics["chromosomeCurves"][2], np.zeros(91))
    np.testing.assert_allclose(diagnostics["pooledCurve"], pooled, atol=1e-14)

    source = replace(source, countOptions=replace(options, inferFragmentLength=True))
    with cc.openSources([source], grid, fragmentLengthOptions=settings) as reader:
        info = reader.sourceInfo["s"]
        assert (
            info["fragmentLengthBP"] == info["countOptions"].extendFrom5pBP == estimate
        )
        reader.setReadExtension("s", 30)
        assert reader.sourceInfo["s"]["fragmentLengthDiagnostics"] is None

    with pytest.warns(EstimateWarning, match=str(EstimateWarning())) as emitted:
        with cc.openSources(
            [source, replace(source, sampleID="other")],
            grid,
            fragmentLengthOptions=cc.FragmentLengthOptions(
                minLengthBP=estimate, maxLengthBP=estimate
            ),
        ):
            pass
    assert len(emitted) == 1

    with pytest.raises(EstimateError, match="No standard chromosome with both strands"):
        with cc.openSources(
            [replace(source, selection=("B",))], grid, fragmentLengthOptions=settings
        ):
            pass


@pytest.mark.parametrize(
    "budget,allocation,bamCount,threads,expected",
    [
        (16, None, 1, "auto", 4),
        (16, "8", 2, "auto", 2),
        (None, None, 1, "auto", 1),
        (None, "5", 1, "auto", 3),
        (None, None, 1, 5, 5),
    ],
)
def testAutoFragmentLengthThreads(
    monkeypatch, budget, allocation, bamCount, threads, expected
):
    monkeypatch.setattr(os, "process_cpu_count", lambda: budget, raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    if allocation is not None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", allocation)
    data = Path(__file__).parent / "data"
    monkeypatch.chdir(data)
    options = cc.CountOptions(bamInputMode="reads")
    sources = [
        cc.SourceSpec(
            "endpoints.bam",
            "BAM",
            f"sample{i}",
            replace(options, inferFragmentLength=i == 0),
            selection=("A",),
        )
        for i in range(bamCount)
    ]
    settings = cc.FragmentLengthOptions(minLengthBP=1, maxLengthBP=90)
    with cc.openSources(
        sources,
        cc.GridSpec({"chr1": 325}),
        threads=threads,
        fragmentLengthOptions=settings,
    ) as reader:
        monkeypatch.chdir(data.parent)
        for sampleID in reader.sampleIDs:
            assert reader.estimateFragmentLength(sampleID) == 40
            diagnostics = reader.sourceInfo[sampleID]["fragmentLengthDiagnostics"]
            assert diagnostics["settings"]["threads"] == expected


def testEstimatorSettings():
    assert asdict(cc.FragmentLengthOptions()) == {
        "minLengthBP": 100,
        "maxLengthBP": 1000,
    }
    for fields in (
        {"minLengthBP": 0},
        {"maxLengthBP": 99},
        {"maxLengthBP": 2**31},
        {"minLengthBP": 12.5},
    ):
        with pytest.raises((ValueError, TypeError)):
            cc.FragmentLengthOptions(**fields)


@pytest.fixture
def fragmentBAM(tmp_path):
    def write(groups, *, paired=True):
        length = (sum(group[0] for group in groups) + 1) * 4096
        header = {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": length}],
        }
        path = tmp_path / "fragments.bam"
        recordIndex = 0
        with pysam.AlignmentFile(path, "wb", header=header) as output:
            for count, span, readLength, mapQ, extraFlags, barcode in groups:
                for _ in range(count):
                    start = recordIndex * 4096
                    aligned = min(readLength, span) if paired else readLength
                    mates = (
                        (
                            (99, start, start + span - aligned, span),
                            (147, start + span - aligned, start, -span),
                        )
                        if paired
                        else ((0, start, -1, 0),)
                    )
                    for flag, position, mate, templateLength in mates:
                        record = pysam.AlignedSegment()
                        record.query_name = f"read{recordIndex}"
                        record.query_sequence = "A" * readLength
                        record.query_qualities = pysam.qualitystring_to_array(
                            "I" * readLength
                        )
                        record.flag = flag | extraFlags
                        record.reference_id = 0
                        record.reference_start = position
                        record.mapping_quality = mapQ
                        record.cigar = [(0, aligned)]
                        if aligned < readLength:
                            record.cigar += [(4, readLength - aligned)]
                        record.next_reference_id = 0 if paired else -1
                        record.next_reference_start = mate
                        record.template_length = templateLength
                        record.set_tag("CB", barcode)
                        output.write(record)
                    recordIndex += 1
        pysam.index(str(path))
        return path, length

    return write


@pytest.mark.parametrize(
    "groups,optionFields,selection,expected",
    [
        (
            [(600, 3000, 20, 60, 0, "A")],
            {},
            None,
            (3000, 600, 0, None),
        ),
        (
            [(600, 100, 20, 60, 0, "A"), (600, 600, 20, 60, 0, "A")],
            {"minTemplateLengthBP": 500, "maxTemplateLengthBP": 2500},
            None,
            (600, 600, 500, 2500),
        ),
        ([(600, 10, 20, 60, 0, "A")], {}, None, (10, 600, 0, None)),
        (
            [(600, 10, 20, 60, 0, "A"), (600, 60, 20, 60, 0, "A")],
            {"minTemplateLengthBP": -1},
            None,
            (60, 600, 20, None),
        ),
        (
            [
                (600, 100, 20, 60, 0, "A"),
                (1200, 300, 20, 0, 0, "A"),
                (1200, 500, 20, 60, 1024, "A"),
                (1200, 600, 20, 60, 0, "B"),
            ],
            {},
            ("A",),
            (100, 600, 0, None),
        ),
        (
            [(1, 100, 20, 60, 0, "A"), (600, 300, 20, 60, 0, "B")],
            {},
            ("A",),
            (100, 1, 0, None),
        ),
        (
            [(600, 3000, 20, 60, 0, "A")],
            {"maxTemplateLengthBP": 100},
            None,
            (147, 0, 0, 100),
        ),
        (
            [(600, 3000, 20, 60, 0, "A")],
            {"maxTemplateLengthBP": 100},
            ("A",),
            (147, 0, 0, 100),
        ),
        (
            [(600, 100, 20, 0, 0, "A")],
            {"minTemplateLengthBP": -1},
            ("A",),
            (147, 0, None, None),
        ),
        ([(1, 100, 20, 60, 0, "A")], {}, None, (147, 1, 0, None)),
    ],
    ids=[
        "unbounded",
        "minimum",
        "zeroMinimum",
        "readMedianMinimum",
        "filters",
        "selectedMinimum",
        "noTemplates",
        "selectedNoTemplates",
        "excludedMappingQuality",
        "fewTemplates",
    ],
)
def testPairedFragmentFilters(
    fragmentBAM,
    groups,
    optionFields,
    selection,
    expected,
):
    path, length = fragmentBAM(groups)
    source = cc.SourceSpec(
        path,
        "BAM",
        "s",
        cc.CountOptions(bamInputMode="fragments", **optionFields),
        selection=selection,
    )
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", EstimateWarning)
        with cc.openSources([source], cc.GridSpec({"chr1": length})) as reader:
            estimate = reader.estimateFragmentLength("s")
            diagnostics = reader.sourceInfo["s"]["fragmentLengthDiagnostics"]
            assert reader.estimateFragmentLength("s") == estimate

    assert (
        estimate,
        diagnostics["retainedTemplateCount"],
        diagnostics["minimumTemplateLengthBP"],
        diagnostics["maximumTemplateLengthBP"],
    ) == expected
    expectedCount = expected[1]
    insufficient = selection is None and expectedCount < 500 or expectedCount == 0
    expectedWarnings = (
        ["Insufficient paired fragments. Using the 147 bp fallback."]
        if insufficient
        else []
    )
    assert [item.category for item in emitted] == (
        [EstimateWarning] if insufficient else []
    )
    if insufficient:
        assert "147" in str(emitted[0].message)
    assert diagnostics["warnings"] == expectedWarnings


def testReadLengthUsesFilters(fragmentBAM, tmp_path):
    path, length = fragmentBAM(
        [
            (100, 0, 100, 0, 0, "A"),
            (100, 0, 80, 60, 1024, "A"),
            (100, 0, 60, 60, 0, "B"),
            (10, 0, 10, 60, 0, "A"),
            (10, 0, 10, 60, 16, "A"),
        ],
        paired=False,
    )
    allowPath = tmp_path / "allow.txt"
    allowPath.write_text("A\n")
    native = _native.NativeSource(
        path, "BAM", {"chr1": length}, barcodeAllowListFile=str(allowPath)
    )
    options = cc.CountOptions(bamInputMode="reads")
    try:
        assert native.readLength(options) == 10
        assert native.readLength(replace(options, minMappingQuality=0)) == 100
        assert native.readLength(options) == 10
        assert native.readLength(replace(options, flagExclude=0)) == 80
        assert native.readLength(options) == 10
    finally:
        native.close()


def testPairedFallbackOnOpen(fragmentBAM):
    path, length = fragmentBAM([(600, 3000, 20, 60, 0, "A")])
    source = cc.SourceSpec(
        path,
        "BAM",
        "s",
        cc.CountOptions(
            bamInputMode="reads", maxTemplateLengthBP=100, inferFragmentLength=True
        ),
    )
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", EstimateWarning)
        with cc.openSources([source], cc.GridSpec({"chr1": length})) as reader:
            assert reader.estimateFragmentLength("s") == 147
            assert reader.sourceInfo["s"]["countOptions"].extendFrom5pBP == 147
    assert len(emitted) == 1
    assert emitted[0].category is EstimateWarning
    assert "147" in str(emitted[0].message)
