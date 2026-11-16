#include "fenwickMedian.h"

#include <stdlib.h>

/*
 * Fenwick update and lower_bound adapted from KACTL's FenwickTree.h.
 * Author: Lukas Polacek. Source license: CC0.
 * https://github.com/kth-competitive-programming/kactl/blob/main/content/data-structures/FenwickTree.h
 */
static inline void ccFenwickUpdate(int64_t *tree, size_t nValues,
                                   size_t position, int difference) {
    // zero-based Fenwick parents, as in KACTL's update
    // (position |= position + 1 moves to next parent)
    for (; position < nValues; position |= position + 1) {
        tree[position] += difference;
    }
}

static inline size_t ccFenwickLowerBound(const int64_t *tree, size_t nValues,
                                         int64_t rank, size_t bit) {
    size_t position = 0;

    // smallest value index whose cumulative frequency reaches rank
    for (; bit; bit >>= 1) {
        size_t step = position + bit;
        if (step <= nValues && tree[step - 1] < rank) {
            rank -= tree[step - 1];
            position = step;
        }
    }
    return position;
}

int ccMedianProfile(const intptr_t *valueIndices, size_t nBins,
                    const double *values, size_t nValues, size_t windowBins,
                    double *result) {
    int64_t *tree = calloc(nValues, sizeof(*tree));
    size_t index, left = 0, right = 0, halfWidth = windowBins / 2;
    size_t targetLeft, targetRight, topBit = 1, medianIndex;
    int64_t rank;

    if (!tree) return -1;

    // largest power of two <= nValues, shared by all rank lookups
    while (topBit <= nValues / 2) topBit <<= 1;

    for (index = 0; index < nBins; ++index) {
        // centered window [targetLeft, targetRight), clipped at both ends
        targetLeft = index > halfWidth ? index - halfWidth : 0;
        targetRight = halfWidth >= nBins - index - 1 ? nBins : index + halfWidth + 1;

        while (right < targetRight) {
            ccFenwickUpdate(tree, nValues, (size_t)valueIndices[right], 1);
            ++right;
        }
        while (left < targetLeft) {
            ccFenwickUpdate(tree, nValues, (size_t)valueIndices[left], -1);
            ++left;
        }

        // 1-based observation rank, taking the upper middle for even windows
        rank = (int64_t)((right - left) / 2 + 1);
        medianIndex = ccFenwickLowerBound(tree, nValues, rank, topBit);
        result[index] = values[medianIndex];
    }

    free(tree);
    return 0;
}
