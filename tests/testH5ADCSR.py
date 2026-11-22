import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import ccounts as cc


def testSharedCSRSelection(tmp_path, monkeypatch):
    path = tmp_path / "counts.h5ad"
    values = np.array([2**24, 1, np.nan, -2, 1, 3, np.inf, 2], dtype=np.float32)
    indices = np.array([0, 0, 2, 0, 0, 1, 0, 1], dtype=np.int32)
    indptr = np.array([0, 3, 4, 6, 7, 8], dtype=np.int32)
    obs = pd.DataFrame({"group": ["g", "h", "g", "h", "g"]}, index=list("ABCDE"))
    var = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr2"], "start": [0, 50, 0], "end": [50, 100, 50]},
        index=["a", "b", "c"],
    )
    matrix = sparse.csr_matrix((values, indices, indptr), shape=(len(obs), len(var)))
    anndata.AnnData(matrix, obs=obs, var=var).write_h5ad(path, compression="gzip")

    metadata = []
    readElem = anndata.io.read_elem

    def recordRead(node):
        metadata.append(node.name)
        return readElem(node)

    monkeypatch.setattr(anndata.io, "read_elem", recordRead)
    sources = [
        cc.SourceSpec(
            path,
            "H5AD",
            "group",
            selection={"path": path, "groupColumn": "group", "groups": ("g",)},
        ),
        cc.SourceSpec(path, "H5AD", "one", selection=("A",)),
        cc.SourceSpec(path, "H5AD", "empty", selection=()),
        cc.SourceSpec(path, "H5AD", "bad", selection=("B",)),
    ]
    grid = cc.GridSpec({"chr1": 100, "chr2": 50})

    with cc.openSources(sources, grid, dtype="float64") as reader:
        assert metadata == ["/var", "/obs"]
        selected = ("one", "group", "empty")
        expected = [[2**24 + 1, 0], [2**24 + 2, 5], [0, 0]]
        block = reader.readRegion("chr1", 0, 100, sampleIDs=selected)
        np.testing.assert_array_equal(block.values, expected)
        totals = reader.libraryTotals(chromosomes=("chr1",), sampleIDs=selected)
        assert [totals.records[name]["eventMass"] for name in selected] == [
            sum(row) for row in expected
        ]
        for chromosome, end, sample in (("chr1", 100, "bad"), ("chr2", 50, "group")):
            with pytest.raises(cc.CountError):
                reader.readRegion(chromosome, 0, end, sampleIDs=(sample,))


@pytest.mark.parametrize(
    "dtype,first,expected",
    [(np.float64, 2**53, 2**53 + 2), (np.uint64, 2**53 + 1, 2**53 + 4)],
)
def testCSRCountsAcrossSlabs(tmp_path, dtype, first, expected):
    path = tmp_path / "counts.h5ad"
    values = np.ones(8195, dtype=dtype)
    values[0] = first
    indices = np.zeros(values.size, dtype=np.int32)
    indptr = np.array([0, 2, values.size - 1, values.size], dtype=np.int32)
    matrix = sparse.csr_matrix((values, indices, indptr), shape=(3, 1))
    obs = pd.DataFrame(index=["A", "B", "C"])
    var = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [50]}, index=["x"])
    anndata.AnnData(matrix, obs=obs, var=var).write_h5ad(path)
    source = cc.SourceSpec(path, "H5AD", "group", selection=("A", "C"))
    with cc.openSources([source], cc.GridSpec({"chr1": 50}), dtype="float64") as reader:
        assert reader.readRegion("chr1", 0, 50).values.item() == expected
        assert reader.libraryTotals(chromosomes=["chr1"]).records["group"]["eventMass"] == expected
