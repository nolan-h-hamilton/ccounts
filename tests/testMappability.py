from importlib.resources import files

import numpy as np
import pytest

import ccounts as cc
from ccounts._mappability import _intervalPrefix


def testMappabilityIntervals():
    resourceDir = files("ccounts") / "mappability"
    for assembly, sizes in cc.assemblyChromosomeSizes.items():
        chromosome = next(iter(sizes))
        length = sizes[chromosome]
        scores = cc.Mappability(assembly, chromosome, referenceSizes=sizes)
        with (resourceDir / f"{assembly}.k50.npz").open("rb") as source:
            with np.load(source, allow_pickle=False) as archive:
                runLengths = (
                    np.ascontiguousarray(archive[chromosome].T).view("<u4").reshape(-1)
                )
        runEnds = np.cumsum(runLengths, dtype=np.int64)
        starts = np.array(
            [
                0,
                0,
                1,
                17,
                49,
                50,
                51,
                100,
                length // 2,
                length - 51,
                length - 50,
                length - 1,
            ]
        )
        ends = starts + np.array([1, 50, 1, 97, 1, 50, 101, 300, 250, 50, 50, 1])
        expected = {
            key: []
            for key in ("multiRead", "meanMultiRead", "singleRead", "uniqueStart")
        }
        for start, end in zip(starts, ends):
            left = max(0, start - 49)
            positions = np.arange(left, end)
            flags = (np.searchsorted(runEnds, positions, side="right") & 1).astype(
                np.int64
            )
            coverage = np.convolve(flags, np.ones(50, dtype=np.int64))[
                start - left : end - left
            ]
            width = end - start
            expected["multiRead"].append(int(flags.sum()) / (width + 49))
            expected["meanMultiRead"].append(int(coverage.sum()) / (50 * width))
            expected["singleRead"].append(int((coverage > 0).sum()) / width)
            expected["uniqueStart"].append(int(flags[start - left :].sum()) / width)
        for measure, values in expected.items():
            np.testing.assert_array_equal(
                scores.score(starts, ends, measure=measure), values
            )
            matrixStarts = starts.reshape(2, -1)
            matrixEnds = ends.reshape(2, -1)
            matrixValues = np.asarray(values).reshape(2, -1)
            for queryStarts, queryEnds, queryValues in (
                (matrixStarts, matrixEnds, matrixValues),
                (matrixStarts.T, matrixEnds.T, matrixValues.T),
                (
                    matrixStarts.astype(np.uint64),
                    matrixEnds.astype(np.uint64),
                    matrixValues,
                ),
            ):
                np.testing.assert_array_equal(
                    scores.score(queryStarts, queryEnds, measure=measure), queryValues
                )
            np.testing.assert_array_equal(
                scores.score(
                    np.zeros((2, 1), dtype=np.int64), ends[:2][None, :], measure=measure
                ),
                np.broadcast_to(values[:2], (2, 2)),
            )
            assert scores.score(
                np.empty((2, 0), dtype=np.int64),
                np.empty((1, 0), dtype=np.int64),
                measure=measure,
            ).shape == (2, 0)
            assert (
                scores.score(int(starts[0]), int(ends[0]), measure=measure) == values[0]
            )
        fraction = int(runLengths[1::2].sum()) / length
        assert scores.score(0, length, measure="meanMultiRead") == fraction
        assert scores.score(0, length, measure="uniqueStart") == fraction
        assert scores.referenceSizes == sizes

    for start, end in (
        (0, 0),
        (-1, 1),
        (0, length + 1),
        (1.5, 2),
        (True, 2),
        (
            np.array([0], dtype=np.uint64),
            np.array([2**64 - 1], dtype=np.uint64),
        ),
    ):
        with pytest.raises(cc.InputError):
            scores.score(start, end)
    for measure in ("coverage", None, ["multiRead"]):
        with pytest.raises(cc.InputError):
            scores.score(0, 1, measure=measure)
    for invalidAssembly, invalidChromosome in (
        ("unknown", chromosome),
        (assembly, "unknown"),
        ([], chromosome),
        (assembly, []),
    ):
        with pytest.raises(cc.SourceError):
            cc.Mappability(invalidAssembly, invalidChromosome)
    with pytest.raises(cc.InputError):
        cc.Mappability(assembly, chromosome, referenceSizes={chromosome: length})


def testMappabilityPrefix():
    for starts, ends, chromosomeEnd in (
        ((), (), 12),
        ((0, 7), (4, 10), 14),
        ((2, 9, 15), (6, 12, 20), 24),
    ):
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        totals = np.r_[np.int64(0), np.cumsum(ends - starts)]
        moments = np.r_[
            np.int64(0),
            np.cumsum(
                [np.arange(start, end).sum() for start, end in zip(starts, ends)],
                dtype=np.int64,
            ),
        ]
        covered = np.zeros(chromosomeEnd, dtype=bool)
        for start, end in zip(starts, ends):
            covered[start:end] = True
        baseCoordinates = np.flatnonzero(covered)
        positions = np.arange(-3, chromosomeEnd + 5, dtype=np.int64).reshape(2, -1)
        expectedCounts = (positions[..., None] > baseCoordinates).sum(axis=-1)
        expectedDistances = np.maximum(positions[..., None] - baseCoordinates, 0).sum(
            axis=-1
        )
        starts = np.r_[starts, chromosomeEnd]
        for array in (positions, starts, ends, totals, moments):
            array.setflags(write=False)
        for coordinateSums, expected in (
            (None, expectedCounts),
            (moments, expectedDistances),
        ):
            for queries, expectedValues in (
                (positions, expected),
                (positions.T, expected.T),
            ):
                actual = _intervalPrefix(queries, starts, ends, totals, coordinateSums)
                np.testing.assert_array_equal(actual, expectedValues)
                assert actual.dtype == np.dtype("int64")

    buffers = (positions, starts, ends, totals, moments)
    for bufferIndex, array in enumerate(buffers):
        unaligned = np.ndarray(
            array.shape,
            dtype=np.int64,
            buffer=bytearray(array.nbytes + 1),
            offset=1,
        )
        invalidBuffers = [
            array.astype(np.int32),
            array.astype(array.dtype.newbyteorder("S")),
            unaligned,
        ]
        if bufferIndex:
            invalidBuffers.extend(
                (np.repeat(array, 2)[::2], array[None, :], array[:-1])
            )
        for invalid in invalidBuffers:
            arguments = list(buffers)
            arguments[bufferIndex] = invalid
            with pytest.raises(cc.InputError):
                _intervalPrefix(*arguments)
