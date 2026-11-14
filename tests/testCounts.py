from pathlib import Path
from shutil import copyfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pysam
import pytest
from anndata.io import write_elem
from scipy import sparse

import ccounts as cc


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def testH5ADSelectionLazyTotalsAndOwnedArrays(tmp_path, storage):
    matrix = np.array([[3, 1, 0, 2, 1], [0, 4, 2, 5, 0], [8, 8, 8, 8, 8]], dtype=np.int64)
    values = matrix if storage == "dense" else getattr(sparse, storage + "_matrix")(matrix)
    obs = pd.DataFrame({"sample": ["s1", "s1", "s2"], "group": ["A", "B", "A"]},
                       index=["cell1", "cell2", "cell3"])
    var = pd.DataFrame({
        "chrom": ["chr1"] * 3 + ["chr2"] * 2,
        "start": [0, 50, 100, 0, 50],
        "end": [50, 100, 115, 50, 60],
    }, index=[f"bin{i}" for i in range(5)])
    path = tmp_path / "counts.h5ad"
    ad.AnnData(values, obs=obs, var=var).write_h5ad(path)
    grid = cc.GridSpec({"chr1": 115, "chr2": 60}, 50)
    source = cc.SourceSpec(
        str(path), "H5AD", "selected",
        selection={"sampleColumn": "sample", "sampleValue": "s1",
                   "groupColumn": "group", "groups": ("A",)},
    )
    with cc.openSources([source], grid, dtype="float64") as reader:
        raw = reader.readRegion("chr1", 0, 115, returnRawNoiseMass=True)
        np.testing.assert_array_equal(raw.values, [[3, 1, 0]])
        assert raw.noiseAvailable == (False,)
        assert np.all(np.isnan(raw.rawNoiseMass))
        assert reader.sourceInfo["selected"]["selectedCellCount"] == 1
        totals = reader.libraryTotals(chromosomes=("chr1", "chr2"))
        assert totals.records["selected"]["eventMass"] == 7
        assert totals.records["selected"]["usedMethod"] == "exact"
        assert totals.records["selected"]["denominatorUnit"] == "storedBinCount"
        assert reader.libraryTotals(chromosomes=("chr2", "chr1")) is totals
        terminalBlock = reader.readRegion("chr2", 0, 60)
        np.testing.assert_array_equal(terminalBlock.values, [[2, 1]])
        for method in ("CPM", "RPKM"):
            norm = cc.Normalization.fromTotals(totals, method=method)
            multiplier = 1e6 / 7 if method == "CPM" else 1e9 / 7 / 50
            expected = raw.values * multiplier
            np.testing.assert_allclose(norm.apply(raw), expected)
            np.testing.assert_allclose(norm.apply(terminalBlock), terminalBlock.values * multiplier)
            assert np.isscalar(norm.multipliers(raw, sampleID="selected"))
            np.testing.assert_allclose(norm.multipliers(raw, sampleID="selected"), multiplier)
        with pytest.raises(ValueError, match="H5AD"):
            cc.Normalization.fromTotals(totals, method="RPGC", effectiveGenomeSizeBP=175)
    np.testing.assert_array_equal(raw.values, [[3, 1, 0]])


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
@pytest.mark.parametrize("barcodes,expectedCount,expectedIDs", [
    (None, 3, ("A",)),
    (("A", "B"), 3, ("A",)),
    (("B",), 0, ()),
])
def testH5ADInternalSelectionIntersections(tmp_path, storage, barcodes, expectedCount, expectedIDs):
    matrix = np.array([[3], [7]], dtype=np.int64)
    values = matrix if storage == "dense" else getattr(sparse, storage + "_matrix")(matrix)
    obs = pd.DataFrame({"sample": ["s", "s"], "group": ["g", "g"], "cellID": ["A", "B"]},
                       index=["first", "second"])
    var = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [50]}, index=["bin"])
    path = tmp_path / "selected.h5ad"
    ad.AnnData(values, obs=obs, var=var).write_h5ad(path)
    allowPath = tmp_path / "allow.txt"
    allowPath.write_text("A\n")
    selection = {"sampleColumn": "sample", "sampleValue": "s", "groupColumn": "group",
                 "groups": ("g",), "barcodeColumn": "cellID", "barcodeAllowListFile": str(allowPath)}
    if barcodes is not None:
        selection["barcodes"] = barcodes
    source = cc.SourceSpec(path, "H5AD", "s", selection=selection)
    with cc.openSources([source], cc.GridSpec({"chr1": 50})) as reader:
        np.testing.assert_array_equal(reader.readRegion("chr1", 0, 50).values, [[expectedCount]])
        info = reader.sourceInfo["s"]
        assert info["selectedBarcodes"] == frozenset(expectedIDs)
        assert info["selectedCellCount"] == len(expectedIDs)
        assert reader.libraryTotals(chromosomes=("chr1",)).records["s"]["eventMass"] == expectedCount
    conflicting = cc.SourceSpec(
        path, "H5AD", "s",
        selection=dict(selection, barcodeGroupMapFile=str(tmp_path / "absent.tsv")),
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        cc.openSources([conflicting], cc.GridSpec({"chr1": 50}))


@pytest.mark.parametrize("identifiers", [(1, 2), ("", "B"), ("A B", "B"), ("A", "A")])
def testH5ADInternalCellIdentifiersFail(tmp_path, identifiers):
    path = tmp_path / "identifiers.h5ad"
    ad.AnnData(
        np.array([[3], [7]], dtype=np.int64),
        obs=pd.DataFrame({"cellID": identifiers}, index=["first", "second"]),
        var=pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [50]}, index=["bin"]),
    ).write_h5ad(path)
    source = cc.SourceSpec(path, "H5AD", "s", selection={"barcodeColumn": "cellID"})
    with pytest.raises(ValueError, match="cell identifier"):
        cc.openSources([source], cc.GridSpec({"chr1": 50}))


def testH5ADValidationIsLimitedToRequestedValues(tmp_path):
    path = tmp_path / "counts.h5ad"
    ad.AnnData(
        np.array([[2, -1]], dtype=np.int64),
        obs=pd.DataFrame(index=["A"]),
        var=pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [0, 0], "end": [10, 10]},
                         index=["a", "b"]),
    ).write_h5ad(path)
    with cc.openSources([cc.SourceSpec(path, "H5AD", "A")],
                        cc.GridSpec({"chr1": 10, "chr2": 10}, 10)) as reader:
        np.testing.assert_array_equal(reader.readRegion("chr1", 0, 10).values, [[2]])
        with pytest.raises(ValueError, match="nonnegative"):
            reader.libraryTotals(chromosomes=("chr1", "chr2"))


def testMetadataSelectionAndPairDepths(tmp_path):
    metadata = tmp_path / "cells.tsv"
    metadata.write_text("cell\tsample\tgroup\nA\ts1\tx\nB\ts1\ty\nA\ts2\ty\n")
    descriptor = {"path": str(metadata), "barcodeColumn": "cell",
                  "sampleColumn": "sample", "groupColumn": "group"}
    assert cc.selectionGroups(descriptor, sampleValue="s1") == {
        "x": frozenset(("A",)), "y": frozenset(("B",))}
    grid = cc.GridSpec({"chr1": 115}, 50)
    records = {
        name: {"eventMass": total, "spanBP": total * 30, "unit": "eventMass",
               "sourceKind": "BAM", "countMode": "conservedFractionalOverlap"}
        for name, total in (("A", 100), ("B", 400), ("C", 200))
    }
    totals = cc.LibraryTotals(records, grid, ("chr1",))
    block = cc.CountBlock(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
        ("A", "B", "C"), grid, "chr1", 0, 115, ("eventMass",) * 3,
        ("conservedFractionalOverlap",) * 3,
    )
    preserved = block.values.copy()
    mismatched = cc.CountBlock(
        block.values, block.sampleIDs, grid, "chr1", 0, 115, block.units,
        ("conservedFractionalOverlap", "center", "conservedFractionalOverlap"),
    )
    reordered = cc.CountBlock(
        block.values[[2, 0, 1]], ("C", "A", "B"), grid, "chr1", 0, 115,
        ("eventMass",) * 3,
        ("center", "conservedFractionalOverlap", "conservedFractionalOverlap"),
    )
    for method in ("CPM", "RPKM", "RPGC"):
        args = {"effectiveGenomeSizeBP": 115} if method == "RPGC" else {}
        normalization = cc.Normalization.fromTotals(totals, method=method, **args)
        pair = normalization.forPair("B", "A")
        assert pair.widthRules == (False, False)
        np.testing.assert_allclose(pair.apply(reordered), [block.values[1] / 4, block.values[0]])
        pair = normalization.forPair("A", "B", fixReference=True)
        np.testing.assert_allclose(pair.apply(block), [block.values[0], block.values[1]])
        pair = normalization.forPair("B", "A", fixReference=True)
        np.testing.assert_allclose(pair.apply(block), [block.values[1] / 4, block.values[0]])
        for checkedNormalization in (normalization, pair):
            out = np.full((len(checkedNormalization.sampleIDs), 3), -7.0)
            with pytest.raises(ValueError):
                checkedNormalization.apply(mismatched, out=out)
            assert np.all(out == -7.0)
            with pytest.raises(ValueError):
                checkedNormalization.multipliers(mismatched, sampleID="A")
        np.testing.assert_array_equal(block.values, preserved)
    factors = cc.Normalization.fromFactors(("C", "A"), (2, 3))
    np.testing.assert_array_equal(factors.apply(reordered), [block.values[2] * 2, block.values[0] * 3])
    with pytest.raises(ValueError):
        factors.apply(block, out=block.values[:2])
    with pytest.raises(ValueError):
        factors.forPair("C", "A")


@pytest.mark.parametrize("change", [
    {"countMode": "midpoint"}, {"countMode": "cutsite"}, {"countMode": "ffp-center"},
    {"flagExclude": -1}, {"minMappingQuality": True}, {"shiftForward5p": 2**31},
    {"extendFrom5pBP": 0}, {"inferFragmentLength": True, "extendFrom5pBP": 100},
])
def testInvalidOptionsFail(change):
    with pytest.raises(ValueError):
        cc.CountOptions(**change)


def testAssemblyConstants():
    expected = {
        "hg19": (93, 23, 3036303846), "hg38": (455, 23, 3031042417),
        "hs1": (25, 23, 3054815472), "mm10": (66, 20, 2633776672),
        "mm39": (61, 20, 2631958877), "dm3": (15, 6, 120381546),
        "dm6": (1870, 6, 133880608), "sacCer2": (18, 16, 12070898),
        "sacCer3": (17, 16, 12071326), "ce10": (7, 6, 100272276),
        "ce11": (7, 6, 100272607), "danRer10": (1061, 25, 1340430591),
        "danRer11": (1923, 25, 1345101833),
    }
    assert set(cc.assemblyChromosomeSizes) == set(cc.standardChromosomes) == set(expected)
    for assembly, dimensions in expected.items():
        sizes = cc.assemblyChromosomeSizes[assembly]
        chromosomes = cc.standardChromosomes[assembly]
        grid = cc.GridSpec({chrom: sizes[chrom] for chrom in chromosomes})
        assert (len(sizes), len(chromosomes), sum(grid.chromosomeSizes.values())) == dimensions, assembly
        assert tuple(grid.chromosomeSizes) == chromosomes
        assert "chrY" not in chromosomes and "chrM" not in chromosomes
        with pytest.raises(TypeError):
            sizes[chromosomes[0]] = sizes[chromosomes[0]]
    for catalog in (cc.assemblyChromosomeSizes, cc.standardChromosomes):
        with pytest.raises(TypeError):
            catalog["hg19"] = catalog["hg19"]
    assert cc.assemblyChromosomeSizes["hg19"]["chrM"] == 16571
    assert cc.assemblyChromosomeSizes["hs1"]["chrY"] == 62460029
    assert cc.assemblyChromosomeSizes["sacCer2"]["2micron"] == 6318


def testRawSourceGridValidation():
    for sizes, width in [({}, 50), ({"chr1": -1}, 50), ({"chr1": 100}, 0)]:
        with pytest.raises(ValueError):
            cc.GridSpec(sizes, width)
    grid = cc.GridSpec({"chr1": 115}, 50)
    for start, end in [(-1, 50), (1, 50), (0, 51), (100, 116), (50, 50)]:
        with pytest.raises(ValueError):
            grid.checkRegion("chr1", start, end)
    with pytest.raises(ValueError):
        cc.SourceSpec("x", "FRAGMENTS", "A", cc.CountOptions(bamInputMode="read1"))
    with pytest.raises(ValueError):
        cc.SourceSpec("x", "BEDGRAPH", "A", cc.CountOptions())


def testH5ADCSCReadsEachColumnBlockOnce(tmp_path):
    rows = 10_240
    path = tmp_path / "columns.h5ad"
    ad.AnnData(
        sparse.csc_matrix(np.ones((rows, 1), dtype=np.int32)),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(rows)]),
        var=pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]}, index=["bin"]),
    ).write_h5ad(path)

    class ReadCounter:
        format = "csc"

        def __init__(self, matrix):
            self.matrix = matrix
            self.readCount = 0

        def __getitem__(self, index):
            self.readCount += 1
            return self.matrix[index]

    with cc.openSources([cc.SourceSpec(path, "H5AD", "A")],
                        cc.GridSpec({"chr1": 10}, 10)) as reader:
        handle = reader._handles["A"]
        counter = ReadCounter(handle.matrix)
        handle.matrix = counter
        np.testing.assert_array_equal(reader.readRegion("chr1", 0, 10).values, [[rows]])
        assert counter.readCount == 1
        counter.readCount = 0
        totals = reader.libraryTotals(chromosomes=("chr1",))
        assert totals.records["A"]["eventMass"] == rows
        assert counter.readCount == 1


def testExplicitIndexUtilitiesPreserveInputs(tmp_path, countSources):
    rawPath = tmp_path / "signal.bedGraph"
    copyfile(countSources["bedGraph"], rawPath)
    rawText = rawPath.read_text()
    compressedPath = tmp_path / "signal.bg.gz"
    result = cc.prepareIndex(rawPath, "BEDGRAPH", targetPath=compressedPath)
    assert result == str(compressedPath)
    assert rawPath.read_text() == rawText
    assert Path(str(compressedPath) + ".tbi").exists()
    grid = cc.GridSpec({"chr1": 105}, 50)
    source = cc.SourceSpec(compressedPath, "BEDGRAPH", "signal")
    with cc.openSources([source], grid) as reader:
        actual = reader.readRegion("chr1", 0, 105).values
    np.testing.assert_array_equal(actual, [[1, 0, 3]])
    preserved = compressedPath.read_bytes()
    with pytest.raises(FileExistsError):
        cc.prepareIndex(rawPath, "BEDGRAPH", targetPath=compressedPath)
    assert compressedPath.read_bytes() == preserved
    with pytest.raises(ValueError):
        cc.prepareIndex(rawPath, "BEDGRAPH", targetPath=rawPath)
    assert rawPath.read_text() == rawText


@pytest.mark.parametrize("text", [
    "chr1\t0\t20\t1\nchr1\t10\t30\t2\n",
    "chr1\t0\t10\tnan\n",
    "chr1\t0\t10\t2\textra\n",
])
def testInvalidIndexedBedGraphFailsAtReaderBoundary(tmp_path, text):
    rawPath = tmp_path / "invalid.bedGraph"
    rawPath.write_text(text)
    target = tmp_path / "invalid.bg.gz"
    cc.prepareIndex(rawPath, "BEDGRAPH", targetPath=target)
    with pytest.raises(ValueError):
        cc.openSources([cc.SourceSpec(target, "BEDGRAPH", "invalid")], cc.GridSpec({"chr1": 100}, 50))


@pytest.mark.parametrize("support", ["0", "-1", "bad", "1.5"])
def testSelectedFragmentSupportMustBePositiveInteger(tmp_path, support):
    plain = tmp_path / "fragments.tsv"
    plain.write_text(f"chr1\t0\t30\tcell\t{support}\n")
    compressed = tmp_path / "fragments.tsv.gz"
    pysam.tabix_compress(str(plain), str(compressed), force=True)
    cc.prepareIndex(compressed, "FRAGMENTS")
    options = cc.CountOptions(fragmentsUseReadSupport=True)
    source = cc.SourceSpec(compressed, "FRAGMENTS", "sample", countOptions=options)
    with cc.openSources([source], cc.GridSpec({"chr1": 100})) as reader:
        with pytest.raises(ValueError, match="column five"):
            reader.readRegion("chr1", 0, 100)
        with pytest.raises(ValueError, match="column five"):
            reader.libraryTotals(chromosomes=("chr1",))


def testMissingIndexesDoNotTriggerImplicitCreation(tmp_path, countSources):
    sourcePath = tmp_path / "unindexed.bam"
    copyfile(countSources["eventsBAM"], sourcePath)
    source = cc.SourceSpec(sourcePath, "BAM", "sample")
    with pytest.raises(ValueError, match="index"):
        cc.openSources([source], cc.GridSpec({"chr1": 115}))
    assert not Path(str(sourcePath) + ".bai").exists()


def testReadLengthUsesEncodedCIGARWithoutSequence(tmp_path):
    bamPath = tmp_path / "cigar.bam"
    with pysam.AlignmentFile(bamPath, "wb", header={"SQ": [{"SN": "chr1", "LN": 100}]}) as handle:
        record = pysam.AlignedSegment()
        record.query_name = "read"
        record.flag = 0
        record.reference_id = 0
        record.reference_start = 10
        record.mapping_quality = 60
        record.cigar = [(0, 20)]
        handle.write(record)
    cc.prepareIndex(bamPath, "BAM")
    source = cc.SourceSpec(bamPath, "BAM", "sample")
    with cc.openSources([source], cc.GridSpec({"chr1": 100})) as reader:
        assert reader.getSourceInfo("sample", inspectLengths=True)["readLength"] == 20
        assert reader.readRegion("chr1", 0, 100).values.sum() == 1


def testBAMSourceRejectsOtherAlignmentFormats(tmp_path):
    path = tmp_path / "alignment.sam"
    path.write_text("@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:100\n")
    with pytest.raises(ValueError, match="BAM file format"):
        cc.openSources([cc.SourceSpec(path, "BAM", "sample")], cc.GridSpec({"chr1": 100}))


@pytest.mark.parametrize("barcodeColumn", [None, "cellID"])
def testExternalH5ADCellMetadataUsesOnlyObs(tmp_path, monkeypatch, barcodeColumn, countSources):
    metadataPath = tmp_path / "cells.h5ad"
    frame = pd.DataFrame(
        {"type": ["typeA", "typeB", "typeB", "typeA"],
         "donor": ["donorA", "donorA", "donorB", "donorB"],
         "cellID": ["shared", "uniqueA", "shared", "uniqueB"]},
        index=["shared", "uniqueA", "shared", "uniqueB"],
    )
    with h5py.File(metadataPath, "w") as handle:
        write_elem(handle, "obs", frame)
        handle.create_dataset("X", data=np.array([[np.nan]], dtype=np.float32))
    originalGetItem = h5py.Group.__getitem__

    def readMetadataOnly(handle, name):
        if name in ("X", "/X"):
            raise AssertionError("Cell metadata selection must not read X")
        return originalGetItem(handle, name)

    monkeypatch.setattr(h5py.Group, "__getitem__", readMetadataOnly)
    descriptor = {"path": str(metadataPath), "barcodeColumn": barcodeColumn,
                  "sampleColumn": "donor", "groupColumn": "type"}
    assert cc.selectionGroups(descriptor, sampleValue="donorA") == {
        "typeA": frozenset({"shared"}), "typeB": frozenset({"uniqueA"})}
    assert cc.selectionGroups(descriptor, sampleValue="donorB") == {
        "typeB": frozenset({"shared"}), "typeA": frozenset({"uniqueB"})}
    selection = dict(descriptor, groups=("typeB",))
    source = cc.SourceSpec(countSources["barcodes"], "FRAGMENTS", "donorA", selection=selection,
                           countOptions=cc.CountOptions(countMode="center"))
    with cc.openSources([source], cc.GridSpec({"chr1": 150})) as reader:
        assert reader.sourceInfo["donorA"]["selectedBarcodes"] == frozenset({"uniqueA"})
        np.testing.assert_array_equal(reader.readRegion("chr1", 0, 150).values, [[0, 1, 0]])


@pytest.mark.parametrize("selection", [
    ("same", "same"),
    {"barcodes": ("same", "same")},
    {"path": "unused.h5ad", "barcodes": ("same", "same")},
])
def testExplicitDuplicateCellIDsFailBeforeSetConversion(selection):
    with pytest.raises(ValueError, match="Duplicate selected cell"):
        cc.SourceSpec("unused.tsv.gz", "FRAGMENTS", "sample", selection=selection)


def testExternalH5ADMetadataRejectsDuplicateSampleCellIdentity(tmp_path):
    path = tmp_path / "duplicated.h5ad"
    frame = pd.DataFrame({"donor": ["sample", "sample"], "type": ["A", "B"]}, index=["same", "same"])
    with h5py.File(path, "w") as handle:
        write_elem(handle, "obs", frame)
    descriptor = {"path": str(path), "sampleColumn": "donor", "groupColumn": "type"}
    with pytest.raises(ValueError, match="Duplicate sample/barcode"):
        cc.selectionGroups(descriptor, sampleValue="sample")


@pytest.mark.parametrize("kind", ["csv", "h5ad"])
def testMetadataWithoutSampleColumnUsesSharedCatalog(tmp_path, kind):
    if kind == "csv":
        path = tmp_path / "cells.csv"
        path.write_text("cellID,type\nA,groupA\nB,groupB\n")
        descriptor = {"path": str(path), "barcodeColumn": "cellID", "groupColumn": "type"}
    else:
        path = tmp_path / "cells.h5ad"
        with h5py.File(path, "w") as handle:
            write_elem(handle, "obs", pd.DataFrame({"type": ["groupA", "groupB"]}, index=["A", "B"]))
        descriptor = {"path": str(path), "groupColumn": "type"}
    assert cc.selectionGroups(descriptor, sampleValue="sourceName") == {
        "groupA": frozenset({"A"}), "groupB": frozenset({"B"})}
