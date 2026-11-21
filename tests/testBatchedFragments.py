from unittest.mock import patch

import numpy as np
import pysam
import pytest

import ccounts as cc
from ccounts import _native


def testBatchedFragmentRowsAndTotals(countSources):
    grid = cc.GridSpec({"chr1": 115, "chrEmpty": 115}, binSizeBP=5)
    sources = [
        cc.SourceSpec(
            countSources["events"],
            "FRAGMENTS",
            name,
            cc.CountOptions(countMode=mode, fragmentsUseReadSupport=support),
            selection=selection,
        )
        for name, mode, support, selection in (
            ("fraction", "conservedFractionalOverlap", True, ("A",)),
            ("overlap", "binOverlap", True, ("A", "B")),
            ("all", "coverage", False, None),
            ("ends", "fiveprime", True, ("B",)),
            ("empty", "center", True, ()),
        )
    ]
    names = ("empty", "overlap", "ends", "all", "fraction")
    chromosomes = ("chr1", "chrEmpty")

    with cc.openSources(sources, grid, dtype="float64") as reader:
        for chromosome, start, end in (
            ("chr1", 0, 115),
            ("chr1", 40, 100),
            ("chrEmpty", 0, 115),
        ):
            query = (chromosome, start, end)
            with patch.object(
                _native, "_fragmentsBatch", wraps=_native._fragmentsBatch
            ) as batch:
                block = reader.readRegion(
                    *query, sampleIDs=names, returnRawNoiseMass=True
                )
                batch.assert_called_once()
            assert block.sampleIDs == names
            assert all(block.noiseAvailable)
            for row, name in enumerate(names):
                single = reader.readRegion(
                    *query, sampleIDs=[name], returnRawNoiseMass=True
                )
                np.testing.assert_array_equal(block.values[row], single.values[0])
                np.testing.assert_array_equal(
                    block.rawNoiseMass[row], single.rawNoiseMass[0]
                )

        with patch.object(
            _native, "_fragmentsBatch", wraps=_native._fragmentsBatch
        ) as batch:
            totals = reader.libraryTotals(chromosomes=chromosomes, sampleIDs=names)
            assert batch.call_count == len(chromosomes)
        for name in names:
            single = reader.libraryTotals(chromosomes=chromosomes, sampleIDs=[name])
            assert totals.records[name] == single.records[name]
        assert (
            totals.records["ends"]["metadata"]["chromosomes"]["chr1"]["scannedRecords"]
            == 4
        )


def testBatchedFragmentRecordErrors(tmp_path):
    text = tmp_path / "fragments.tsv"
    text.write_text("chr1\t5\t15\tA\tbad\nchr1\t20\t30\tB\t4\nchr2\t5\t15\n")
    path = pysam.tabix_index(str(text), preset="bed")
    grid = cc.GridSpec({"chr1": 50, "chr2": 50}, binSizeBP=10)
    options = cc.CountOptions(countMode="center", fragmentsUseReadSupport=True)
    sources = [
        cc.SourceSpec(path, "FRAGMENTS", name, options, selection=selection)
        for name, selection in (("selected", ("B",)), ("bad", ("A",)), ("empty", ()))
    ]

    with cc.openSources(sources, grid, dtype="float64") as reader:
        with pytest.raises(cc.NativeError, match="Fragment support"):
            out = np.full((3, 5), -1.0)
            reader.readRegion("chr1", 0, 50, out=out)
        np.testing.assert_array_equal(out, [[0, 0, 4, 0, 0], [-1] * 5, [-1] * 5])

        retained = ("selected", "empty")
        np.testing.assert_array_equal(
            reader.readRegion("chr1", 0, 50, sampleIDs=retained).values,
            [[0, 0, 4, 0, 0], [0] * 5],
        )
        with pytest.raises(cc.NativeError, match="Invalid input file"):
            reader.readRegion("chr2", 0, 50, sampleIDs=retained)
