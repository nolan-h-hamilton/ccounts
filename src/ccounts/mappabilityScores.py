from importlib.resources import files

import numpy as np

from ._mappability import _intervalPrefix
from .checks import (
    checkedMappabilityEdges,
    checkedMappabilityIntervals,
    checkedMappabilityReference,
)


class Mappability:
    r"""Quantify unique mapping of 50-base reads for genomic bins.

    Use the scores to identify bins with little uniquely mappable sequence
    or select donor bins with high mappability for count correction.

    Bins can have any positive integer width. The map groups consecutive bases
    with the same mappability value into *runs*.

    - ``multiRead`` divides unique overlapping starts by the bin width plus 49.
    - ``meanMultiRead`` averages the per-base multi-read scores.
    - ``singleRead`` gives the fraction of bases covered by a unique 50-mer.
    - ``uniqueStart`` gives the fraction of positions with a unique 50-mer.

    Use the same reference DNA and complete contig set as the resource.
    Passing ``referenceSizes`` checks the contig names and lengths.
    """

    k = 50

    def __init__(self, assembly, chromosome, *, referenceSizes=None):
        sizes = checkedMappabilityReference(assembly, chromosome, referenceSizes)
        self.assembly = assembly
        self.chromosome = chromosome
        self.length = sizes[chromosome]
        self.referenceSizes = sizes

        resource = files("ccounts") / "mappability" / f"{assembly}.k50.npz"
        with (
            resource.open("rb") as source,
            np.load(source, allow_pickle=False) as archive,
        ):
            runBytes = archive[chromosome]

        edges = checkedMappabilityEdges(runBytes, self.length, self.k)
        starts = edges[1:-1:2]
        ends = edges[2::2]
        widths = ends - starts

        self._starts = np.r_[starts, self.length]
        self._ends = ends.copy()
        self._totals = np.r_[np.int64(0), np.cumsum(widths)]
        self._moments = np.r_[np.int64(0), np.cumsum(widths * (starts + ends - 1) // 2)]

        if starts.size:
            extendedEnds = np.minimum(ends + self.k - 1, self.length)
            groups = np.flatnonzero(np.r_[True, starts[1:] > extendedEnds[:-1]])
            coveredStarts = starts[groups]
            coveredEnds = extendedEnds[np.r_[groups[1:] - 1, ends.size - 1]]
        else:
            coveredStarts = np.empty(0, dtype=np.int64)
            coveredEnds = np.empty(0, dtype=np.int64)

        self._coveredStarts = np.r_[coveredStarts, self.length]
        self._coveredEnds = coveredEnds
        self._coveredTotals = np.r_[np.int64(0), np.cumsum(coveredEnds - coveredStarts)]

    def score(self, starts, ends, *, measure="multiRead"):
        starts, ends = checkedMappabilityIntervals(starts, ends, self.length, measure)
        widths = ends - starts

        if measure == "singleRead":
            positions = np.stack((starts, ends), axis=-1)
            prefix = _intervalPrefix(
                positions, self._coveredStarts, self._coveredEnds, self._coveredTotals
            )
            return (prefix[..., 1] - prefix[..., 0]) / widths

        if measure == "meanMultiRead":
            positions = np.stack(
                (starts, starts - self.k, ends, ends - self.k), axis=-1
            )
            prefix = _intervalPrefix(
                positions, self._starts, self._ends, self._totals, self._moments
            )
            return (
                prefix[..., 2] - prefix[..., 3] - prefix[..., 0] + prefix[..., 1]
            ) / (self.k * widths)

        positions = np.stack(
            (starts - (self.k - 1) if measure == "multiRead" else starts, ends), axis=-1
        )
        prefix = _intervalPrefix(positions, self._starts, self._ends, self._totals)
        denominator = widths + self.k - 1 if measure == "multiRead" else widths
        return (prefix[..., 1] - prefix[..., 0]) / denominator
