#ifndef CCOUNTS_FENWICK_MEDIAN_H
#define CCOUNTS_FENWICK_MEDIAN_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Compute upper medians in centered windows, clipped at the ends.
 *
 * valueIndices has nBins entries in [0, nValues), and values is sorted.
 * nBins and nValues must be positive, and windowBins must be positive and odd.
 * The caller checks these inputs and provides nBins result slots.
 *
 * @return 0 on success, -1 if the tree allocation fails
 */
int ccMedianProfile(const intptr_t *valueIndices, size_t nBins,
                    const double *values, size_t nValues, size_t windowBins,
                    double *result);

#endif
