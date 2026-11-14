from pathlib import Path

import pysam
import pytest


@pytest.fixture(scope="session")
def countSources(tmp_path_factory):
    dataPath = Path(__file__).parent / "data"
    indexedPath = tmp_path_factory.mktemp("countSources")
    sources = {
        "eventsBAM": str(dataPath / "events.bam"),
        "sizes": str(dataPath / "genome.sizes"),
        "bedGraph": str(dataPath / "signal.bedGraph"),
    }
    for name in ("events", "barcodes", "cancellation", "precision"):
        path = indexedPath / f"{name}.fragments.tsv.gz"
        pysam.tabix_compress(str(dataPath / f"{name}.fragments.tsv"), str(path))
        pysam.tabix_index(str(path), preset="bed")
        sources[name] = str(path)
    return sources
