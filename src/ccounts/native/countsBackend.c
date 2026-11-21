#include "countsBackend.h"
#include "messages.h"
#include "barcode_selection.h"

#include <htslib/hts.h>
#include <htslib/sam.h>
#include <htslib/tbx.h>
#include <htslib/bgzf.h>
#include <htslib/kstring.h>

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** @brief Open file, index, and barcode filter for one source. */
struct ccSource {
    int kind;                      ///< 0 = BAM, 1 = FRAGMENTS, 2 = BEDGRAPH

    htsFile *file;                 ///< open input file
    sam_hdr_t *header;             ///< BAM header, NULL for tabix sources
    hts_idx_t *index;              ///< BAM index, NULL for tabix sources
    tbx_t *tabix;                  ///< tabix index for FRAGMENTS/BEDGRAPH, NULL for BAM

    crBarcodeSelection *selection; ///< barcode tag and allow-list filter

    const char **names;            ///< reference names from tabix, NULL for BAM
    int nameCount;                 ///< number of reference sequences

    uint32_t readLengthBP;         ///< cached sampled median read length in bp, 0 if unset
    uint16_t readLengthFlags;
    int readLengthMinMappingQuality;
    int readLengthBAMMode;

    int64_t *maximumTemplateLengths; ///< per-chromosome query padding, -1 until a full scan
};

/** @brief Span, counting points, and weight for one read/fragment. */
typedef struct {
    int64_t start;       ///< 0-based span start, clipped at 0
    int64_t end;         ///< span end (exclusive), clipped at chromosome length

    int64_t point[2];    ///< 0-based positions to count (midpoint or ends)
    int pointCount;      ///< number of usable points, 0..2

    double multiplicity; ///< record weight, 1 unless fragment read support is used
} ccEvent;

/** @brief Settings and buffers for bin counts, totals, and span tracking. */
typedef struct {
    int64_t start;              ///< 0-based start of the output region in bp
    int64_t end;                ///< end of the output region in bp (exclusive)
    int64_t binSize;            ///< bin width in bp
    int64_t chromosomeLength;   ///< chromosome length in bp

    const ccOptions *options;   ///< filtering and counting settings

    double *values;             ///< per-bin output values
    double *mass;               ///< per-bin raw noise mass, NULL if not requested

    double *interior;           ///< difference array for bin contributions, count + 1 slots
    double *interiorCorrection; ///< roundoff corrections for interior, or NULL
    double *interiorTail;
    double *interiorMass;
    double *interiorMassCorrection;
    double *interiorMassTail;
    size_t *interiorEnds;
    size_t firstInterior;
    size_t stopInterior;
    size_t count;               ///< number of output bins

    ccTotals *totals;           ///< totals output, non-NULL selects totals mode

    const int64_t *endpointRegions;
    size_t endpointRegionCount;
    double endpointMass;
    double endpointCorrection;

    int64_t rangeStart;         ///< leftmost retained span start, -1 if none
    int64_t rangeEnd;           ///< rightmost retained span end (exclusive), 0 if none
} ccWork;

typedef struct {
    int64_t left;
    int64_t right;
    const char *barcode;
    const char *readSupport;
} ccFragment;

typedef struct {
    uint32_t position;
    uint32_t count;
} ccEndpointRun;

typedef struct {
    ccEndpointRun *runs;
    size_t size;
    size_t capacity;
    int unordered;
} ccEndpointRuns;

static int ccFail(const char **error, const char *message) {
    *error = message;
    return -1;
}

static int64_t ccMax(int64_t a, int64_t b) { return a > b ? a : b; }

static int64_t ccMin(int64_t a, int64_t b) { return a < b ? a : b; }

static int ccMeta(const char *line) {
    while (*line && isspace((unsigned char)*line)) ++line;
    return !*line || *line == '#' || !strncmp(line, "track ", 6) ||
           !strncmp(line, "browser ", 8);
}

static int ccParseInteger(const char **cursor, int64_t *value) {
    char *stop;
    long long parsed;

    errno = 0;
    if (!**cursor || isspace((unsigned char)**cursor)) return -1;

    parsed = strtoll(*cursor, &stop, 10);
    if (stop == *cursor || errno == ERANGE ||
        (*stop && !isspace((unsigned char)*stop))) return -1;

    *value = (int64_t)parsed;
    *cursor = stop;

    return 0;
}

static int ccColumns(char *line, char **fields, int limit) {
    int count = 0;
    char *cursor = line;

    while (*cursor && count < limit) {
        while (*cursor && isspace((unsigned char)*cursor)) ++cursor;
        if (!*cursor) break;

        fields[count++] = cursor;

        while (*cursor && !isspace((unsigned char)*cursor)) ++cursor;
        if (*cursor) *cursor++ = '\0';
    }

    return count;
}

static int ccParseCoordinate(const char *field, int64_t *value) {
    const char *cursor = field;

    return ccParseInteger(&cursor, value) == 0 && !*cursor && *value >= 0 ? 0 : -1;
}

static int ccBarcodeAllowed(const ccSource *source, const char *barcode) {
    if (!source->selection || !source->selection->active) return 1;
    if (!source->selection->count) return 0;

    return bsearch(&barcode, source->selection->barcodes, source->selection->count,
                   sizeof(char *), crBarcodeCompare) != NULL;
}

static int ccBasicRecord(ccSource *source, bam1_t *record,
                         const ccOptions *options, const char **error) {
    int selected;

    if ((record->core.flag & BAM_FUNMAP) ||
        record->core.tid < 0 ||
        (record->core.flag & options->flagExclude) ||
        record->core.qual < options->minMappingQuality) return 0;

    if (options->bamInputMode == 2 &&
        (!(record->core.flag & BAM_FPAIRED) ||
         !(record->core.flag & BAM_FREAD1) ||
         (record->core.flag & BAM_FMUNMAP))) return 0;

    selected = crBarcodeSelected(source->selection, record);
    if (selected < 0) return ccFail(error, ccInvalidBarcodeType);

    return selected;
}

static int ccResolveBAM(ccSource *source, bam1_t *record, int64_t chromosomeLength,
                         const ccOptions *options, ccEvent *event, const char **error) {
    int accepted = ccBasicRecord(source, record, options, error);
    int reverse;
    int64_t readStart, readEnd, fivePrime, left, right, shift, length;

    if (accepted <= 0) return accepted;

    readStart = record->core.pos;
    readEnd = bam_endpos(record);
    if (readEnd <= readStart) return 0;

    reverse = (record->core.flag & BAM_FREVERSE) != 0;
    shift = reverse ? -options->shiftReverse5p : options->shiftForward5p;
    fivePrime = (reverse ? readEnd - 1 : readStart) + shift;

    if (options->bamInputMode == 1) {
        if (!(record->core.flag & BAM_FPROPER_PAIR) ||
            !(record->core.flag & BAM_FREAD1) ||
            (record->core.flag & BAM_FMUNMAP) ||
            record->core.mtid != record->core.tid) return 0;

        length = record->core.isize;
        if (length == INT64_MIN) return ccFail(error, ccInvalidNumerics);
        if (length < 0) length = -length;

        if (!length || (options->maxTemplateLengthBP > 0 &&
                        length > options->maxTemplateLengthBP)) return 0;

        if (options->minTemplateLengthBP < 0) {
            uint16_t readLengthFlags = options->flagExclude |
                (source->selection->active ? BAM_FREAD2 : 0);

            if (!source->readLengthBP ||
                    source->readLengthFlags != readLengthFlags ||
                    source->readLengthMinMappingQuality != options->minMappingQuality ||
                    source->readLengthBAMMode != options->bamInputMode) {

                ccOptions readOptions = *options;
                readOptions.flagExclude = readLengthFlags;

                if (ccReadLength(source, &readOptions, &source->readLengthBP, error) != 0)
                    return -1;
            }
        }

        if (length < (options->minTemplateLengthBP < 0 ?
                (int64_t)source->readLengthBP : options->minTemplateLengthBP)) return 0;

        if (record->core.isize >= 0) {
            left = readStart + shift;
            right = left + length;
        } else {
            right = readEnd + shift;
            left = right - length;
        }

    } else if (options->extendFrom5pBP > 0) {
        if (reverse) {
            right = fivePrime + 1;
            left = right - options->extendFrom5pBP;
        } else {
            left = fivePrime;
            right = left + options->extendFrom5pBP;
        }
    } else {
        left = readStart + shift;
        right = readEnd + shift;
    }

    event->multiplicity = 1.0;
    event->pointCount = 0;

    if (options->countMode == 3 || options->countMode == 4) {
        if (options->countMode == 4 || options->bamInputMode != 1) {
            event->point[0] = fivePrime;
            event->pointCount = 1;
        } else {
            event->point[0] = left;
            event->point[1] = right - 1;
            event->pointCount = 2;
        }
    }

    event->start = ccMax(0, left);
    event->end = ccMin(chromosomeLength, right);

    if (options->countMode == 2 && event->end > event->start) {
        event->point[0] = event->start + (event->end - event->start) / 2;
        event->pointCount = 1;
    }

    if (options->countMode >= 2 && options->countMode <= 4) {
        int a = 0, b;

        for (b = 0; b < event->pointCount; ++b) {
            if (event->point[b] >= 0 && event->point[b] < chromosomeLength)
                event->point[a++] = event->point[b];
        }

        event->pointCount = a;
        return a > 0;
    }

    return event->end > event->start;
}

static int ccParseFragment(char *line, ccFragment *fragment, const char **error) {
    char *fields[5];
    int n = ccColumns(line, fields, 5);
    if (n < 4 ||
        ccParseCoordinate(fields[1], &fragment->left) ||
        ccParseCoordinate(fields[2], &fragment->right) ||
        fragment->right <= fragment->left || !*fields[3])
        return ccFail(error, ccInvalidFile);

    fragment->barcode = fields[3];
    fragment->readSupport = n == 5 ? fields[4] : NULL;
    return 0;
}

static int ccFragmentSupport(const ccFragment *fragment, int64_t *weight,
                             const char **error) {
    if (!fragment->readSupport ||
        ccParseCoordinate(fragment->readSupport, weight) || *weight <= 0)
        return ccFail(error, ccInvalidCounts);
    return 0;
}

static int ccFragmentEvent(const ccFragment *fragment, int64_t chromosomeLength,
                           int mode, int64_t multiplicity, ccEvent *event) {
    int a, b;

    event->multiplicity = (double)multiplicity;
    event->start = fragment->left;
    event->end = ccMin(chromosomeLength, fragment->right);
    event->pointCount = 0;

    if (mode == 2 && event->end > event->start) {
        event->point[0] = event->start + (event->end - event->start) / 2;
        event->pointCount = 1;
    } else if (mode == 3) {
        event->point[0] = fragment->left;
        event->point[1] = fragment->right - 1;
        event->pointCount = 2;
    }

    if (mode >= 2 && mode <= 4) {
        a = 0;

        for (b = 0; b < event->pointCount; ++b) {
            if (event->point[b] >= 0 && event->point[b] < chromosomeLength)
                event->point[a++] = event->point[b];
        }

        event->pointCount = a;
        return a > 0;
    }

    return event->end > event->start;
}

static int ccResolveFragment(ccSource *source, char *line, int64_t chromosomeLength,
                              const ccOptions *options, ccEvent *event,
                              int64_t *owner, const char **error) {
    ccFragment fragment;
    int64_t multiplicity = 1;
    if (ccParseFragment(line, &fragment, error)) return -1;
    *owner = fragment.left;
    if (!ccBarcodeAllowed(source, fragment.barcode)) return 0;
    if (options->fragmentsUseReadSupport &&
        ccFragmentSupport(&fragment, &multiplicity, error)) return -1;
    return ccFragmentEvent(&fragment, chromosomeLength, options->countMode,
                           multiplicity, event);
}

/**
 * @brief Bin one read/fragment's points in [work->start, work->end),
 * scaled by event->multiplicity.
 */
static void ccPoint(ccWork *work, const ccEvent *event) {
    int k;
    int64_t bins[2] = {-1, -1};

    for (k = 0; k < event->pointCount; ++k) {
        int64_t p = event->point[k];

        if (p >= work->start && p < work->end)
            bins[k] = (p - work->start) / work->binSize;
    }

    if (bins[0] >= 0) {
        int same = event->pointCount == 2 && bins[0] == bins[1];

        // Count both endpoints if they land in this bin, otherwise just one
        work->values[bins[0]] += event->multiplicity * (same ? 2.0 : 1.0);

        // For N fragments with both ends in the bin, left/right endpoint counts are L=R=N.
        // Cov(L,R)=Var(N), so Var(L+R)=Var(L)+Var(R)+2*Cov(L,R)=4*Var(N).
        // Assuming Poisson sampling, estimate Var(N) := N --> add 4 per fragment.
        if (work->mass) work->mass[bins[0]] += event->multiplicity * (same ? 4.0 : 1.0);

        if (same) return;
    }

    if (event->pointCount == 2 && bins[1] >= 0) {
        work->values[bins[1]] += event->multiplicity;
        if (work->mass) work->mass[bins[1]] += event->multiplicity;
    }
}

static inline double ccTwoSum(double *sum, double value) {
    double total = *sum + value;
    double removed = total - *sum;
    double error = (*sum - (total - removed)) + (value - removed);
    *sum = total;
    return error;
}

static inline void ccAccumulateWithTail(double value, double *high, double *low,
                                        double *tail) {
    double error = ccTwoSum(high, value);
    if (tail) *tail += ccTwoSum(low, error);
    else *low += error;
    *low = ccTwoSum(high, *low);
}

/**
 * @brief Keep rounding leftovers when adding to a running total.
 *
 * high holds the rounded total. low keeps what was too small to fit there.
 * Start both at zero and read the total as high + low.
 *
 * @verbatim
 *   add        high        low
 *   +2^54      2^54         0
 *   +1         2^54         1   <- too small to change high
 *   -2^54         1         0   <- the saved 1 survives
 * @endverbatim
 *
 * In contrast, naive double addition would yields 0 for this example.
 *
 * @param[in] value Amount to add
 * @param[in,out] high Rounded part of the total
 * @param[in,out] low Rounding leftovers carried between calls.
 * @see Ogita, Rump and Oishi, Accurate Sum and Dot Product, Algorithm 3.1.
 */
static void ccAccumulate(double value, double *high, double *low) {
    ccAccumulateWithTail(value, high, low, NULL);
}

enum { ccFractionalMinBins = 8 };

static inline void ccAddInterior(double *values, double *corrections, double *tails,
                                 size_t first, size_t stop, double value) {
    if (corrections) {
        ccAccumulateWithTail(value, &values[first], &corrections[first],
                             tails ? &tails[first] : NULL);
        ccAccumulateWithTail(-value, &values[stop], &corrections[stop],
                             tails ? &tails[stop] : NULL);
    } else {
        values[first] += value;
        values[stop] -= value;
    }
}

static void ccSpan(ccWork *work, const ccEvent *event) {
    int64_t left = ccMax(event->start, work->start);
    int64_t right = ccMin(event->end, work->end);
    int64_t first, last, firstFull, stopFull, j, binStart, binEnd;
    double weight;

    if (right <= left) return;

    first = (left - work->start) / work->binSize;
    last = (right - 1 - work->start) / work->binSize;
    firstFull = stopFull = first;

    int mode = work->options->countMode;
    if (mode == 0 || mode == 5) {
        firstFull = first;
        stopFull = last + 1;

        if (mode == 0) {
            binStart = work->start + first * work->binSize;
            binEnd = ccMin(
                work->start + (last + 1) * work->binSize, work->chromosomeLength
            );

            if (left > binStart) ++firstFull;
            if (right < binEnd) --stopFull;
        }

        if (firstFull < stopFull)
            ccAddInterior(work->interior, work->interiorCorrection, NULL,
                          firstFull, stopFull, event->multiplicity);

        if (mode == 5) return;
    } else if (work->interiorEnds && last - first >= ccFractionalMinBins - 1) {
        firstFull = first + (left > work->start + first * work->binSize);
        stopFull = (right - work->start) / work->binSize;

        if (stopFull - firstFull >= ccFractionalMinBins) {
            weight = (double)work->binSize / (double)(event->end - event->start);
            double value = event->multiplicity * weight;
            ccAddInterior(work->interior, work->interiorCorrection, work->interiorTail,
                          firstFull, stopFull, value);
            if (work->interiorMass)
                ccAddInterior(work->interiorMass, work->interiorMassCorrection,
                              work->interiorMassTail, firstFull, stopFull, value * weight);

            size_t stop = (size_t)stopFull;
            size_t start = (size_t)firstFull;
            if (stop > work->interiorEnds[start]) work->interiorEnds[start] = stop;
            if (start < work->firstInterior) work->firstInterior = start;
            if (stop > work->stopInterior) work->stopInterior = stop;
        } else {
            firstFull = stopFull;
        }
    }

    for (j = first; j <= last; ++j) {
        if (j == firstFull && j < stopFull) {
            j = stopFull - 1;
            continue;
        }

        binStart = work->start + j * work->binSize;
        binEnd = ccMin(binStart + work->binSize, work->chromosomeLength);

        weight = (double)(ccMin(right, binEnd) - ccMax(left, binStart));
        weight /= (double)(mode == 0 ?
            binEnd - binStart : event->end - event->start);

        work->values[j] += event->multiplicity * weight;
        if (work->mass) work->mass[j] += event->multiplicity * weight * weight;
    }
}

static int ccEndpointInRegions(const ccWork *work, int64_t point) {
    const int64_t *regions = work->endpointRegions;
    size_t first = 0, stop = work->endpointRegionCount;

    if (point < work->start || point >= work->end) return 0;

    // Find the first interval whose right boundary exceeds the endpoint.
    while (first < stop) {
        size_t middle = first + (stop - first) / 2;

        if (regions[2 * middle + 1] <= point)
            first = middle + 1;
        else
            stop = middle;
    }

    return first < work->endpointRegionCount && point >= regions[2 * first];
}

/**
 * @brief Add one accepted read/fragment to totals or bin counts, using its
 * multiplicity and the selected counting mode.
 * Also, we track the leftmost start and rightmost end of accepted nonempty spans
 * through `work`'s rangeStart and rangeEnd fields.
 */
static void ccConsume(ccWork *work, const ccEvent *event) {
    if (work->endpointRegions) {
        int k, matches = 0;

        for (k = 0; k < event->pointCount; ++k)
            matches += ccEndpointInRegions(work, event->point[k]);

        if (matches)
            ccAccumulate(event->multiplicity * matches, &work->endpointMass,
                         &work->endpointCorrection);

        return;
    }

    if (work->totals) {
        work->totals->eventMass += event->multiplicity *
            (work->options->countMode >= 2 && work->options->countMode <= 4 ?
                event->pointCount : 1);
        work->totals->spanBP += event->multiplicity *
            (double)ccMax(0, event->end - event->start);

    } else if (work->options->countMode >= 2 && work->options->countMode <= 4) {
        ccPoint(work, event);
    } else {
        ccSpan(work, event);
    }

    if (event->end > event->start) {
        if (work->rangeStart < 0 || event->start < work->rangeStart)
            work->rangeStart = event->start;

        if (event->end > work->rangeEnd) work->rangeEnd = event->end;
    }
}

static int64_t ccBAMQueryPadding(const ccSource *source, int tid,
                                 int64_t chromosomeLength, const ccOptions *options) {
    int64_t padding = ccMax(
        llabs(options->shiftForward5p), llabs(options->shiftReverse5p));
    padding = ccMin(padding, chromosomeLength);

    if (options->extendFrom5pBP >= chromosomeLength - padding)
        return chromosomeLength;
    padding += options->extendFrom5pBP;

    if (options->bamInputMode == 1 && options->countMode != 4) {
        int64_t templatePadding = options->maxTemplateLengthBP;
        if (templatePadding <= 0)
            templatePadding = source->maximumTemplateLengths[tid];
        // An unknown template bound requires a full chromosome scan.
        if (templatePadding < 0 || templatePadding >= chromosomeLength - padding)
            return chromosomeLength;
        padding += templatePadding;
    }

    return padding;
}

/**
 * @brief Read BAM/FRAGMENTS records and update counts or totals.
 *
 * owned = 1 assigns each record to [start, end) by its BAM alignment start
 * or FRAGMENTS left coordinate.
 *
 * With owned = 0, BAM queries can extend beyond [start, end), e.g., for shifted
 * reads, extended reads, etc. FRAGMENTS use [start, end).
 *
 * @param source open BAM or FRAGMENTS input
 * @param chromosome reference name to query
 * @param chromosomeLength reference length in bp
 * @param start 0-based region start in bp
 * @param end region end in bp (exclusive)
 * @param options filtering and read/fragment settings
 * @param[in,out] work count buffers or totals, plus retained span bounds
 * @param owned 1 = assign by start, 0 = allow padded BAM queries
 * @param[out] error error message on failure
 * @return 0 on success (including missing references), -1 on failure
 */
static int ccVisit(ccSource *source, const char *chromosome, int64_t chromosomeLength,
                    int64_t start, int64_t end, const ccOptions *options,
                    ccWork *work, int owned, const char **error) {
    int tid, status, accepted, collectTemplateLengths;
    int64_t queryStart = start, queryEnd = end, owner;
    int64_t observedTemplateLength, maximumTemplateLength = 0;
    hts_itr_t *iterator = NULL;
    bam1_t *record = NULL;
    kstring_t line = {0, 0, NULL};
    ccEvent event;

    if (source->kind == 0) {
        tid = sam_hdr_name2tid(source->header, chromosome);
        if (tid < 0) return 0;

        if (!owned) {
            int64_t padding = ccBAMQueryPadding(source, tid, chromosomeLength, options);
            queryStart = start > padding ? start - padding : 0;
            queryEnd = padding >= chromosomeLength - end
                ? chromosomeLength
                : end + padding;
        }

        collectTemplateLengths = queryStart == 0 &&
            queryEnd == chromosomeLength &&
            source->maximumTemplateLengths[tid] < 0;

        iterator = sam_itr_queryi(source->index, tid, queryStart, queryEnd);
        if (!iterator) return ccFail(error, ccInputQueryFailed);

        record = bam_init1();
        if (!record) {
            hts_itr_destroy(iterator);
            return ccFail(error, ccAllocationFailed);
        }

        while ((status = sam_itr_next(source->file, iterator, record)) >= 0) {
            if (collectTemplateLengths) {
                // include rejected reads so changing filters cannot hide longer fragments
                observedTemplateLength = record->core.isize;

                if (observedTemplateLength == INT64_MIN) {
                    observedTemplateLength = chromosomeLength;
                } else {
                    if (observedTemplateLength < 0)
                        observedTemplateLength = -observedTemplateLength;
                    observedTemplateLength = ccMin(
                        observedTemplateLength, chromosomeLength);
                }

                if (observedTemplateLength > maximumTemplateLength)
                    maximumTemplateLength = observedTemplateLength;
            }

            if ((record->core.flag & BAM_FUNMAP) || record->core.tid < 0) continue;
            owner = record->core.pos;

            if (owned && (owner < start || owner >= end)) continue;

            if (work->totals) ++work->totals->scannedRecords;
            accepted = ccResolveBAM(
                source, record, chromosomeLength, options, &event, error);

            if (accepted < 0) {
                status = -2;
                break;
            }

            if (accepted) ccConsume(work, &event);
        }

        bam_destroy1(record);
        hts_itr_destroy(iterator);

        if (collectTemplateLengths && status == -1)
            source->maximumTemplateLengths[tid] = maximumTemplateLength;

        if (status < -1) return *error ? -1 : ccFail(error, ccInputReadFailed);
        return 0;
    }

    tid = tbx_name2id(source->tabix, chromosome);
    if (tid < 0) return 0;

    if (owned) {
        queryStart = start;
        queryEnd = end;
    }

    iterator = tbx_itr_queryi(source->tabix, tid, queryStart, queryEnd);
    if (!iterator) return ccFail(error, ccInputQueryFailed);

    while ((status = tbx_itr_next(
                source->file, source->tabix, iterator, &line)) >= 0) {
        accepted = ccResolveFragment(
            source, line.s, chromosomeLength, options, &event, &owner, error);

        if (accepted < 0) {
            status = -2;
            break;
        }

        if (owned && (owner < start || owner >= end)) continue;

        if (work->totals) ++work->totals->scannedRecords;
        if (accepted) ccConsume(work, &event);
    }

    free(line.s);
    hts_itr_destroy(iterator);

    if (status < -1) return *error ? -1 : ccFail(error, ccInputReadFailed);
    return 0;
}

static int ccBedGraphLine(char *line, char **chromosome, int64_t *start,
                           int64_t *end, double *value, const char **error) {
    char *fields[5], *stop;
    int n;

    if (ccMeta(line)) return 0;

    n = ccColumns(line, fields, 5);
    if (n != 4 ||
            ccParseCoordinate(fields[1], start) ||
            ccParseCoordinate(fields[2], end) ||
            *end <= *start)
        return ccFail(error, ccInvalidBEDGRAPH);

    errno = 0;
    *value = strtod(fields[3], &stop);
    if (stop == fields[3] || *stop || !isfinite(*value))
        return ccFail(error, ccInvalidBEDGRAPH);

    *chromosome = fields[0];
    return 1;
}

static int ccValidateBedGraph(ccSource *source, const char *path,
                              int threads, const char **error) {
    int status, parsed, tid = -1, nextTid;
    int64_t left, right, lastEnd = -1;
    double value;
    char *chromosome;
    unsigned char *seen = calloc((size_t)source->nameCount, 1);
    kstring_t line = {0, 0, NULL};

    if (!seen && source->nameCount) return ccFail(error, ccAllocationFailed);

    while ((status = hts_getline(source->file, '\n', &line)) >= 0) {
        parsed = ccBedGraphLine(line.s, &chromosome, &left, &right, &value, error);
        if (parsed < 0) {
            status = -2;
            break;
        }

        if (!parsed) continue;

        nextTid = tbx_name2id(source->tabix, chromosome);
        if (nextTid < 0) {
            ccFail(error, ccInvalidBEDGRAPH);
            status = -2;
            break;
        }

        if (tid != nextTid) {
            if (seen[nextTid]) {
                ccFail(error, ccInvalidBEDGRAPH);
                status = -2;
                break;
            }

            seen[nextTid] = 1;
            tid = nextTid;
            lastEnd = -1;
        }

        if (left < lastEnd) {
            ccFail(error, ccInvalidBEDGRAPH);
            status = -2;
            break;
        }

        lastEnd = right;
    }

    free(seen);
    free(line.s);

    hts_close(source->file);
    source->file = hts_open(path, "r");
    if (!source->file) return ccFail(error, ccInputOpenFailed);
    if (threads > 1 && hts_set_threads(source->file, threads) != 0)
        return ccFail(error, ccThreadsFailed);

    if (status < -1) return *error ? -1 : ccFail(error, ccInputReadFailed);
    return 0;
}

int ccOpen(const char *path, int kind, const char *allowPath, const char *tag,
           int threads, ccSource **out, const char **error) {
    ccSource *source = calloc(1, sizeof(ccSource));
    int referenceIndex;

    *out = NULL;
    if (!source) return ccFail(error, ccAllocationFailed);

    source->kind = kind;
    source->file = hts_open(path, "r");
    if (!source->file) {
        ccClose(source);
        return ccFail(error, ccInputOpenFailed);
    }

    if (threads > 1 && hts_set_threads(source->file, threads) != 0) {
        ccClose(source);
        return ccFail(error, ccThreadsFailed);
    }

    source->selection = crBarcodeSelectionCreate(tag, allowPath, error);
    if (!source->selection) {
        ccClose(source);
        return -1;
    }

    if (kind == 0) {
        if (hts_get_format(source->file)->format != bam) {
            ccClose(source);
            return ccFail(error, ccInvalidSource);
        }

        source->header = sam_hdr_read(source->file);
        if (!source->header) {
            ccClose(source);
            return ccFail(error, ccHeaderReadFailed);
        }

        source->index = sam_index_load(source->file, path);
        if (!source->index) {
            ccClose(source);
            return ccFail(error, ccMissingIndex);
        }

        source->nameCount = sam_hdr_nref(source->header);
        if (source->nameCount) {
            if ((size_t)source->nameCount >
                    SIZE_MAX / sizeof(*source->maximumTemplateLengths)) {
                ccClose(source);
                return ccFail(error, ccAllocationFailed);
            }

            source->maximumTemplateLengths = malloc(
                (size_t)source->nameCount * sizeof(*source->maximumTemplateLengths));
            if (!source->maximumTemplateLengths) {
                ccClose(source);
                return ccFail(error, ccAllocationFailed);
            }

            for (referenceIndex = 0; referenceIndex < source->nameCount; ++referenceIndex)
                source->maximumTemplateLengths[referenceIndex] = -1;
        }
    } else {
        source->tabix = tbx_index_load(path);
        if (!source->tabix) {
            ccClose(source);
            return ccFail(error, ccMissingIndex);
        }

        source->names = tbx_seqnames(source->tabix, &source->nameCount);

        if (kind == 2 && ccValidateBedGraph(source, path, threads, error) != 0) {
            ccClose(source);
            return -1;
        }
    }

    *out = source;
    return 0;
}

void ccClose(ccSource *source) {
    if (!source) return;

    if (source->index) hts_idx_destroy(source->index);
    if (source->tabix) tbx_destroy(source->tabix);
    if (source->names) free(source->names);
    free(source->maximumTemplateLengths);
    if (source->header) sam_hdr_destroy(source->header);
    if (source->file) hts_close(source->file);
    crBarcodeSelectionDestroy(source->selection);

    free(source);
}

int ccChromosomeCount(ccSource *source) { return source->nameCount; }

const char *ccChromosomeName(ccSource *source, int index) {
    return source->kind == 0
        ? sam_hdr_tid2name(source->header, index)
        : source->names[index];
}

int64_t ccChromosomeLength(ccSource *source, int index) {
    return source->kind == 0 ? sam_hdr_tid2len(source->header, index) : -1;
}

static int ccInitCount(ccWork *work, const char **error) {
    int mode = work->options->countMode;
    int fractional = mode == 1 && work->count >= ccFractionalMinBins;
    if (mode != 0 && mode != 5 && !fractional) return 0;

    int compensate = fractional || work->options->fragmentsUseReadSupport;
    int separateMass = fractional && work->mass;
    // Weighted spans on a fine grid can mix widely separated magnitudes
    int extend = fractional && work->options->fragmentsUseReadSupport &&
        work->chromosomeLength / work->binSize > (1 << 20);
    size_t stride = work->count + 1;
    size_t baseArrays = 1 + compensate + 2 * separateMass;
    size_t arrays = baseArrays + extend * (1 + separateMass);
    work->interior = calloc(stride, arrays * sizeof(double));
    if (!work->interior) return ccFail(error, ccAllocationFailed);

    if (fractional) {
        work->interiorEnds = calloc(work->count, sizeof(*work->interiorEnds));
        if (!work->interiorEnds) {
            free(work->interior);
            work->interior = NULL;
            return ccFail(error, ccAllocationFailed);
        }
    }

    if (compensate) work->interiorCorrection = work->interior + stride;
    if (separateMass) {
        work->interiorMass = work->interior + 2 * stride;
        work->interiorMassCorrection = work->interior + 3 * stride;
    }
    if (extend) {
        work->interiorTail = work->interior + baseArrays * stride;
        if (separateMass) work->interiorMassTail = work->interiorTail + stride;
    }
    work->firstInterior = fractional ? work->count : 0;
    work->stopInterior = fractional ? 0 : work->count;
    return 0;
}

/**
 * @brief Add span contributions to the output bins and free the scratch arrays.
 *
 * In a finite-difference array add the span's weight at its first bin and subtract
 * it at its exclusive end. Use prefix sums of these changes to update counts.
 * Recording N spans, then scanning B bins like this is O(N + B) time,
 * whereas counting B bins for each span may be O(NB)
 *
 * @verbatim
 *   bin                0           1           2           3           4           5
 *   A (weight 2)                   [===================================)
 *   B (weight 1)                                           [=======================)
 *   A's changes        0          +2           0           0          -2           0
 *   B's changes        0           0           0          +1           0          -1
 *   difference A+B     0          +2           0          +1          -2          -1
 *
 *   prefix sum     0 + 0   0   0 + 2 = 2   2 + 0 = 2   2 + 1 = 3   3 - 2 = 1   1 - 1 = 0
 *   count in bin       1           0           1           0           2           1
 *   count += sum   1 + 0 = 1   0 + 2 = 2   1 + 2 = 3   0 + 3 = 3   2 + 1 = 3   1 + 0 = 1
 * @endverbatim
 *
 * @param[in,out] work Output bins and scratch arrays for one region.
 */
static void ccFinishCount(ccWork *work) {
    if (!work->interior) return;

    double running = 0, correction = 0, tail = 0;
    double mass = 0, massCorrection = 0, massTail = 0;
    double *tailPointer = work->interiorTail ? &tail : NULL;
    double *massTailPointer = work->interiorMassTail ? &massTail : NULL;
    size_t coveredEnd = 0;

    for (size_t bin = work->firstInterior; bin < work->stopInterior; ++bin) {
        if (work->interiorEnds) {
            size_t end = work->interiorEnds[bin];
            if (end > coveredEnd) coveredEnd = end;
            // no interior spans cover the bin
            if (bin >= coveredEnd) {
                running = correction = tail = 0;
                mass = massCorrection = massTail = 0;
                continue;
            }
        }

        if (work->interiorCorrection) {
            ccAccumulateWithTail(work->interior[bin], &running, &correction, tailPointer);
            ccAccumulateWithTail(work->interiorCorrection[bin], &running, &correction,
                                 tailPointer);
            if (tailPointer) tail += work->interiorTail[bin];
        } else {
            running += work->interior[bin];
        }


        double contribution = running + (correction + tail);
        work->values[bin] += contribution;

        // keep small noise contributions from rounding away when large span weights cancel
        if (work->interiorMass) {
            ccAccumulateWithTail(work->interiorMass[bin], &mass, &massCorrection,
                                 massTailPointer);
            ccAccumulateWithTail(work->interiorMassCorrection[bin],
                                 &mass, &massCorrection, massTailPointer);

            if (massTailPointer) {
                massTail += work->interiorMassTail[bin];
            }

            work->mass[bin] += mass + (massCorrection + massTail);

        }
        // FFR: do we ever actually use these?
        else if (work->mass) {
            work->mass[bin] += contribution;
        }
    }

    free(work->interiorEnds);
    free(work->interior);
}

int ccCount(ccSource *source, const char *chromosome, int64_t chromosomeLength,
            int64_t start, int64_t end, int64_t binSize, const ccOptions *options,
            double *values, double *mass, size_t count, const char **error) {
    ccWork work = {0};
    int status, tid, parsed;
    int64_t left, right, j, first, last, binStart, binEnd;
    double value;
    char *recordChromosome;
    kstring_t line = {0, 0, NULL};
    hts_itr_t *iterator;

    work.start = start;
    work.end = end;
    work.binSize = binSize;
    work.chromosomeLength = chromosomeLength;
    work.options = options;
    work.values = values;
    work.mass = mass;
    work.count = count;
    work.rangeStart = -1;

    if (source->kind == 2) {
        tid = tbx_name2id(source->tabix, chromosome);
        if (tid < 0) return 0;

        iterator = tbx_itr_queryi(source->tabix, tid, start, end);
        if (!iterator) return ccFail(error, ccInputQueryFailed);

        while ((status = tbx_itr_next(
                    source->file, source->tabix, iterator, &line)) >= 0) {
            parsed = ccBedGraphLine(
                line.s, &recordChromosome, &left, &right, &value, error);

            if (parsed < 0) {
                status = -2;
                break;
            }
            if (!parsed) continue;

            if (right > chromosomeLength) {
                ccFail(error, ccInvalidRegion);
                status = -2;
                break;
            }

            left = ccMax(start, left);
            right = ccMin(end, right);
            if (right <= left) continue;

            first = (left - start) / binSize;
            last = (right - 1 - start) / binSize;

            for (j = first; j <= last; ++j) {
                binStart = start + j * binSize;
                binEnd = ccMin(binStart + binSize, chromosomeLength);
                //
                values[j] += value *
                    ((double)(ccMin(right, binEnd) - ccMax(left, binStart)) /
                     (double)(binEnd - binStart));
            }
        }

        free(line.s);
        hts_itr_destroy(iterator);

        if (status < -1) return *error ? -1 : ccFail(error, ccInputReadFailed);
        return 0;
    }

    if (ccInitCount(&work, error)) return -1;
    status = ccVisit(
        source, chromosome, chromosomeLength, start, end, options, &work, 0, error);
    ccFinishCount(&work);
    return status;
}

int ccRegionTotals(ccSource *source, const char *chromosome, int64_t chromosomeLength,
                   int64_t start, int64_t end, const ccOptions *options,
                   ccTotals *totals, const char **error) {
    ccWork work = {0};

    if (source->kind == 2) return ccFail(error, ccInvalidUnits);

    memset(totals, 0, sizeof(*totals));
    work.options = options;
    work.totals = totals;
    work.rangeStart = -1;

    return ccVisit(
        source, chromosome, chromosomeLength, start, end, options, &work, 1, error);
}

/**
 * @brief Scan a region ONCE for targets that share a 10x frag file
 *
 * Parses each row once and apply each target's barcode filter and count settings
 */
static void ccVisitFragmentsBatch(ccFragmentTarget *targets, ccWork *work,
                                  size_t targetCount, const char *chromosome,
                                  int owned) {
    ccSource *source = targets[0].source;
    int64_t start = work[0].start, end = work[0].end;
    int tid = tbx_name2id(source->tabix, chromosome), status;
    uint64_t scannedRecords = 0;
    const char *scanError = NULL;
    kstring_t line = {0, 0, NULL};
    ccFragment fragment;
    ccEvent event;

    if (tid < 0) return;
    hts_itr_t *iterator = tbx_itr_queryi(source->tabix, tid, start, end);

    if (!iterator) {
        scanError = ccInputQueryFailed;
    } else {
        while ((status = tbx_itr_next(source->file, source->tabix, iterator, &line)) >= 0) {
            if (ccParseFragment(line.s, &fragment, &scanError)) break;

            int inRegion = !owned || (fragment.left >= start && fragment.left < end);
            int64_t weight = 0;
            const char *supportError = NULL;
            if (inRegion) ++scannedRecords;

            for (size_t i = 0; i < targetCount; ++i) {
                const ccOptions *options = work[i].options;
                if (targets[i].error || !ccBarcodeAllowed(targets[i].source, fragment.barcode))
                    continue;

                int64_t multiplicity = 1;
                if (options->fragmentsUseReadSupport) {
                    if (!weight && !supportError)
                        ccFragmentSupport(&fragment, &weight, &supportError);
                    if (supportError) {
                        targets[i].error = supportError;
                        continue;
                    }
                    multiplicity = weight;
                }

                if (inRegion && ccFragmentEvent(&fragment, work[i].chromosomeLength,
                                                 options->countMode, multiplicity, &event))
                    ccConsume(&work[i], &event);
            }
        }

        if (status < -1 && !scanError) scanError = ccInputReadFailed;
        free(line.s);
        hts_itr_destroy(iterator);
    }

    for (size_t i = 0; i < targetCount; ++i) {
        if (!targets[i].error) targets[i].error = scanError;
        if (owned) work[i].totals->scannedRecords = scannedRecords;
    }
}

int ccFragmentsBatch(ccFragmentTarget *targets, size_t targetCount,
                     const char *chromosome, int64_t chromosomeLength,
                     int64_t start, int64_t end, int64_t binSize, size_t count,
                     double *values, double *mass, ccTotals *totals, const char **error) {
    if (!targetCount || targetCount > SIZE_MAX / sizeof(ccWork))
        return ccFail(error, ccAllocationFailed);

    ccWork *work = malloc(targetCount * sizeof(*work));
    if (!work) return ccFail(error, ccAllocationFailed);

    int status = 0;
    size_t i;
    for (i = 0; i < targetCount; ++i) {
        work[i] = (ccWork){
            .start = start, .end = end, .binSize = binSize,
            .chromosomeLength = chromosomeLength, .options = &targets[i].options,
            .values = values ? values + i * count : NULL,
            .mass = mass ? mass + i * count : NULL,
            .count = count, .totals = totals ? &totals[i] : NULL, .rangeStart = -1,
        };
        targets[i].error = NULL;
        if (totals) {
            totals[i] = (ccTotals){0};
        } else {
            status = ccInitCount(&work[i], error);
            if (status) break;
        }
    }

    if (!status) ccVisitFragmentsBatch(targets, work, targetCount, chromosome, totals != NULL);
    if (!totals) while (i > 0) ccFinishCount(&work[--i]);
    free(work);
    return status;
}

int ccRegionEndpointTotal(ccSource *source, const char *chromosome,
                          int64_t chromosomeLength, const int64_t *regions,
                          size_t regionCount, const ccOptions *options,
                          double *total, const char **error) {
    ccWork work = {0};
    int64_t start, end = 0;
    int64_t maxGap = ccMin(chromosomeLength, 64 * 1024);
    size_t i, first, stop;

    if (source->kind == 2 || options->countMode != 3)
        return ccFail(error, ccInvalidUnits);
    if (!regionCount) return ccFail(error, ccInvalidRegion);

    for (i = 0; i < regionCount; ++i) {
        start = regions[2 * i];
        if (start < end) return ccFail(error, ccInvalidRegion);
        end = regions[2 * i + 1];
        if (end <= start || end > chromosomeLength)
            return ccFail(error, ccInvalidRegion);
    }

    if (source->kind == 0 && regionCount > 1) {
        int tid = sam_hdr_name2tid(source->header, chromosome);
        if (tid < 0) {
            *total = 0.0;
            return 0;
        }
        int64_t padding = ccBAMQueryPadding(source, tid, chromosomeLength, options);
        // group nearby intervals and any whose padded queries overlap
        maxGap = padding >= (chromosomeLength - maxGap) / 2
            ? chromosomeLength : maxGap + 2 * padding;
    }

    work.options = options;
    for (first = 0; first < regionCount; first = stop) {
        // we scan regions together to save calls, splitting at gaps over maxGap
        for (stop = first + 1; stop < regionCount; ++stop) {
            if (regions[2 * stop] - regions[2 * stop - 1] > maxGap) break;
        }

        work.start = regions[2 * first];
        work.end = regions[2 * stop - 1];

        // count endpoints in these regions, skipping the gaps
        work.endpointRegions = regions + 2 * first;
        work.endpointRegionCount = stop - first;

        int status = ccVisit(source, chromosome, chromosomeLength, work.start, work.end,
                             options, &work, 0, error);
        if (status) return status;
    }

    *total = work.endpointMass + work.endpointCorrection;
    return 0;
}

static int ccCompareReadLengths(const void *first, const void *second) {
    uint32_t left = *(const uint32_t *)first;
    uint32_t right = *(const uint32_t *)second;

    return (left > right) - (left < right);
}

int ccReadLength(ccSource *source, const ccOptions *options, uint32_t *readLength,
                 const char **error) {
    samFile *fileHandle = NULL;
    sam_hdr_t *header = NULL;
    bam1_t *record = NULL;
    uint32_t lengths[100];
    int readCount = 0, iterCount = 0, accepted, midIndex, result = 0;
    hts_pos_t queryLength;

    if (source->kind != 0) {
        *readLength = 0;
        return ccFail(error, ccInvalidSource);
    }

    if (source->readLengthBP &&
        source->readLengthFlags == options->flagExclude &&
        source->readLengthMinMappingQuality == options->minMappingQuality &&
        source->readLengthBAMMode == options->bamInputMode) {
        *readLength = source->readLengthBP;
        return 0;
    }

    *readLength = 0;

    fileHandle = sam_open(source->file->fn, "r");
    if (!fileHandle) return ccFail(error, ccInputOpenFailed);

    header = sam_hdr_read(fileHandle);
    if (!header) {
        result = ccFail(error, ccHeaderReadFailed);
        goto finish;
    }

    record = bam_init1();
    if (!record) {
        result = ccFail(error, ccAllocationFailed);
        goto finish;
    }

    while ((source->selection->active ? readCount : iterCount) < 1000 &&
           sam_read1(fileHandle, header, record) >= 0) {
        ++iterCount;
        accepted = ccBasicRecord(source, record, options, error);
        if (accepted < 0) {
            result = -1;
            goto finish;
        }

        if (!accepted) continue;

        queryLength = record->core.l_qseq;
        if (queryLength <= 0 && record->core.n_cigar > 0)
            queryLength = bam_cigar2qlen(
                record->core.n_cigar, bam_get_cigar(record));
        if (queryLength <= 0) continue;

        lengths[readCount++] = (uint32_t)queryLength;
        if (readCount >= 100) break;
    }

    if (!readCount) {
        result = ccFail(error, ccNoEstimate);
        goto finish;
    }

    qsort(lengths, (size_t)readCount, sizeof(uint32_t), ccCompareReadLengths);
    midIndex = readCount / 2;
    *readLength = (readCount & 1) ? lengths[midIndex] :
        (uint32_t)((lengths[midIndex - 1] + lengths[midIndex]) / 2U);

    source->readLengthBP = *readLength;
    source->readLengthFlags = options->flagExclude;
    source->readLengthMinMappingQuality = options->minMappingQuality;
    source->readLengthBAMMode = options->bamInputMode;

finish:
    if (record) bam_destroy1(record);
    if (header) sam_hdr_destroy(header);
    sam_close(fileHandle);

    return result;
}

int ccInspect(ccSource *source, const ccOptions *options, int maxRecords,
              int *paired, const char **error) {
    int t, status, accepted, sampled = 0;
    bam1_t *record;
    hts_itr_t *iterator;

    *paired = 0;
    if (source->kind != 0) return ccFail(error, ccInvalidSource);
    if (source->selection->active && source->selection->count == 0) return 0;

    record = bam_init1();
    if (!record) return ccFail(error, ccAllocationFailed);

    for (t = 0; t < source->nameCount && sampled < maxRecords; ++t) {
        iterator = sam_itr_queryi(
            source->index, t, 0, sam_hdr_tid2len(source->header, t));
        if (!iterator) {
            bam_destroy1(record);
            return ccFail(error, ccInputQueryFailed);
        }

        while (sampled < maxRecords &&
               (status = sam_itr_next(source->file, iterator, record)) >= 0) {
            accepted = ccBasicRecord(source, record, options, error);
            if (accepted < 0) {
                hts_itr_destroy(iterator);

                bam_destroy1(record);
                return -1;
            }

            if (!accepted) continue;

            ++sampled;
            if (record->core.flag & BAM_FPAIRED) *paired = 1;
        }

        hts_itr_destroy(iterator);
        if (sampled < maxRecords && status < -1) {
            bam_destroy1(record);
            return ccFail(error, ccInputReadFailed);
        }
    }

    bam_destroy1(record);

    return 0;
}

int ccSourceRange(ccSource *source, const char *chromosome, uint64_t chromLength,
                  uint16_t flagExclude, uint64_t *startOut, uint64_t *endOut,
                  const char **error)
{
    bam1_t *record = NULL;
    hts_itr_t *iteratorHandle = NULL;
    int32_t tid = -1;
    uint64_t tailStart = 0;
    uint64_t alignmentEnd;
    int seenAlignment = 0;
    int seenTailAlignment = 0;

    if (startOut != NULL)
    {
        *startOut = 0;
    }

    if (endOut != NULL)
    {
        *endOut = 0;
    }

    if (source->kind == 2)
    {
        kstring_t lineBuffer = {0, 0, NULL};
        hts_itr_t *bedGraphIterator = NULL;
        char regionBuffer[1024];
        char *fields[3];
        int seenRecord = 0;
        int64_t bgStart = 0;
        int64_t bgEnd = 0;
        uint64_t minStart = 0;
        uint64_t maxEnd = 0;

        if (source->file == NULL)
        {
            return ccFail(error, ccClosedSource);
        }

        snprintf(regionBuffer, sizeof(regionBuffer), "%s:%d-%llu",
                 chromosome, 1, (unsigned long long)chromLength);
        bedGraphIterator = tbx_itr_querys(source->tabix, regionBuffer);
        if (bedGraphIterator != NULL)
        {
            while (tbx_itr_next(source->file, source->tabix,
                               bedGraphIterator, &lineBuffer) >= 0)
            {
                if (ccMeta(lineBuffer.s) ||
                    ccColumns(lineBuffer.s, fields, 3) < 3 ||
                    ccParseCoordinate(fields[1], &bgStart) ||
                    ccParseCoordinate(fields[2], &bgEnd) ||
                    bgEnd <= bgStart ||
                    strcmp(fields[0], chromosome))
                {
                    continue;
                }

                if (!seenRecord)
                {
                    minStart = (uint64_t)bgStart;
                    maxEnd = (uint64_t)bgEnd;
                    seenRecord = 1;
                }
                else
                {
                    if ((uint64_t)bgStart < minStart)
                    {
                        minStart = (uint64_t)bgStart;
                    }
                    if ((uint64_t)bgEnd > maxEnd)
                    {
                        maxEnd = (uint64_t)bgEnd;
                    }
                }
            }

            hts_itr_destroy(bedGraphIterator);
        }

        free(lineBuffer.s);

        if (startOut != NULL && seenRecord)
        {
            *startOut = minStart;
        }
        if (endOut != NULL && seenRecord)
        {
            *endOut = maxEnd;
        }

        return 0;
    }

    if (source->kind == 1)
    {
        kstring_t lineBuffer = {0, 0, NULL};
        char regionBuffer[1024];
        char *fields[3];
        hts_itr_t *iteratorHandle = NULL;
        int64_t fragStart = 0;
        int64_t fragEnd = 0;
        int seenFragment = 0;

        if (source->tabix == NULL || source->file == NULL)
        {
            return ccFail(error, ccMissingIndex);
        }

        snprintf(regionBuffer, sizeof(regionBuffer), "%s:%d-%llu",
                 chromosome, 1, (unsigned long long)chromLength);
        iteratorHandle = tbx_itr_querys(source->tabix, regionBuffer);
        if (iteratorHandle != NULL)
        {
            while (tbx_itr_next(source->file, source->tabix,
                               iteratorHandle, &lineBuffer) >= 0)
            {
                if (ccColumns(lineBuffer.s, fields, 3) < 3 ||
                    ccParseCoordinate(fields[1], &fragStart) ||
                    ccParseCoordinate(fields[2], &fragEnd) ||
                    fragEnd <= fragStart)
                {
                    continue;
                }

                if (!seenFragment)
                {
                    if (startOut != NULL)
                    {
                        *startOut = (uint64_t)fragStart;
                    }
                    seenFragment = 1;
                }

                if (endOut != NULL && (uint64_t)fragEnd > *endOut)
                {
                    *endOut = (uint64_t)fragEnd;
                }
            }

            hts_itr_destroy(iteratorHandle);
        }

        free(lineBuffer.s);

        return 0;
    }

    if (source->index == NULL)
    {
        return ccFail(error, ccMissingIndex);
    }

    tid = sam_hdr_name2tid(source->header, chromosome);
    if (tid < 0)
    {
        return ccFail(error, ccMissingSource);
    }

    record = bam_init1();
    if (record == NULL)
    {
        return ccFail(error, ccAllocationFailed);
    }

    iteratorHandle = sam_itr_queryi(source->index, tid, 0, (hts_pos_t)chromLength);
    if (iteratorHandle != NULL)
    {
        while (sam_itr_next((htsFile *)source->file, iteratorHandle, record) >= 0)
        {
            if ((record->core.flag & flagExclude) != 0)
            {
                continue;
            }

            if (startOut != NULL)
            {
                *startOut = (uint64_t)record->core.pos;
            }
            if (endOut != NULL)
            {
                *endOut = (uint64_t)bam_endpos(record);
            }

            seenAlignment = 1;
            break;
        }

        hts_itr_destroy(iteratorHandle);
    }

    tailStart = chromLength > 1000000ULL ? chromLength - 1000000ULL : 0;
    iteratorHandle = sam_itr_queryi(
        source->index,
        tid,
        (hts_pos_t)tailStart,
        (hts_pos_t)chromLength);
    if (iteratorHandle != NULL)
    {
        while (sam_itr_next((htsFile *)source->file, iteratorHandle, record) >= 0)
        {
            if ((record->core.flag & flagExclude) != 0)
            {
                continue;
            }

            if (endOut != NULL)
            {
                alignmentEnd = (uint64_t)bam_endpos(record);
                if (alignmentEnd > *endOut)
                {
                    *endOut = alignmentEnd;
                }
            }
            seenTailAlignment = 1;
        }

        hts_itr_destroy(iteratorHandle);
    }

    if (seenAlignment && !seenTailAlignment)
    {
        hts_pos_t scanStart = startOut != NULL ? (hts_pos_t)(*startOut) : 0;

        iteratorHandle = sam_itr_queryi(
            source->index,
            tid,
            scanStart,
            (hts_pos_t)chromLength);
        if (iteratorHandle != NULL)
        {
            while (sam_itr_next((htsFile *)source->file, iteratorHandle, record) >= 0)
            {
                if ((record->core.flag & flagExclude) != 0)
                {
                    continue;
                }

                if (endOut != NULL)
                {
                    alignmentEnd = (uint64_t)bam_endpos(record);
                    if (alignmentEnd > *endOut)
                    {
                        *endOut = alignmentEnd;
                    }
                }
            }

            hts_itr_destroy(iteratorHandle);
        }
    }

    bam_destroy1(record);

    return 0;
}

int ccChromRange(ccSource *source, const char *chromosome, int64_t chromosomeLength,
                 const ccOptions *options, int64_t *start, int64_t *end,
                 const char **error) {
    ccWork work = {0};
    ccTotals totals = {0};
    int status;

    if (source->kind == 2) {
        int tid = tbx_name2id(source->tabix, chromosome), parsed;
        int64_t left, right;
        double value;
        char *recordChromosome;
        kstring_t line = {0, 0, NULL};
        hts_itr_t *iterator;

        *start = 0;
        *end = 0;
        if (tid < 0) return 0;

        iterator = tbx_itr_queryi(source->tabix, tid, 0, chromosomeLength);
        if (!iterator) return ccFail(error, ccInputQueryFailed);

        while ((status = tbx_itr_next(
                    source->file, source->tabix, iterator, &line)) >= 0) {
            parsed = ccBedGraphLine(
                line.s, &recordChromosome, &left, &right, &value, error);
            if (parsed < 0) {
                status = -2;
                break;
            }
            if (!parsed) continue;

            if (right > chromosomeLength) {
                ccFail(error, ccInvalidRegion);
                status = -2;
                break;
            }

            if (!*end || left < *start) *start = left;
            if (right > *end) *end = right;
        }

        free(line.s);
        hts_itr_destroy(iterator);

        if (status < -1) return *error ? -1 : ccFail(error, ccInputReadFailed);
        return 0;
    }

    work.options = options;
    work.totals = &totals;
    work.rangeStart = -1;

    status = ccVisit(source, chromosome, chromosomeLength, 0, chromosomeLength,
                     options, &work, 1, error);

    *start = work.rangeStart < 0 ? 0 : work.rangeStart;
    *end = work.rangeStart < 0 ? 0 : work.rangeEnd;

    return status;
}

static int ccEndpointCompare(const void *left, const void *right) {
    uint32_t a = ((const ccEndpointRun *)left)->position;
    uint32_t b = ((const ccEndpointRun *)right)->position;

    return (a > b) - (a < b);
}

static int ccAppendEndpoint(ccEndpointRuns *strand, uint32_t position, const char **error) {
    size_t size = strand->size;
    ccEndpointRun *runs = strand->runs;
    if (size && position == runs[size - 1].position) {
        if (runs[size - 1].count == UINT32_MAX) return ccFail(error, ccInvalidNumerics);
        ++runs[size - 1].count;
        return 0;
    }
    if (size && position < runs[size - 1].position) strand->unordered = 1;

    if (size == strand->capacity) {
        if (size > SIZE_MAX / sizeof(*runs) / 2) return ccFail(error, ccAllocationFailed);
        size_t capacity = size ? size * 2 : 1024;
        runs = realloc(runs, capacity * sizeof(*runs));
        if (!runs) return ccFail(error, ccAllocationFailed);
        strand->runs = runs;
        strand->capacity = capacity;
    }
    runs[size].position = position;
    runs[size].count = 1;
    strand->size = size + 1;
    return 0;
}

static int ccSortEndpoints(ccEndpointRuns *strand) {
    if (!strand->unordered) return 0;

    ccEndpointRun *runs = strand->runs;
    qsort(runs, strand->size, sizeof(*runs), ccEndpointCompare);
    size_t unique = 1;
    for (size_t i = 1; i < strand->size; ++i) {
        if (runs[i].position == runs[unique - 1].position) {
            if (UINT32_MAX - runs[unique - 1].count < runs[i].count) return -1;
            runs[unique - 1].count += runs[i].count;
        } else {
            runs[unique++] = runs[i];
        }
    }
    strand->size = unique;
    return 0;
}

static void ccCorrelateEndpoints(const ccEndpointRuns *forward, const ccEndpointRuns *reverse,
                                 uint64_t dimension, size_t maxLag, double *curve) {
    const ccEndpointRun *f = forward->runs, *r = reverse->runs;
    size_t reverseStart = 0, forwardEnd = forward->size;
    double sumF = 0.0, sumR = 0.0, squareF = 0.0, squareR = 0.0;

    for (size_t i = 0; i < forward->size; ++i) {
        uint32_t position = f[i].position;
        double count = f[i].count;
        sumF += count;
        squareF += count * count;
        while (reverseStart < reverse->size && r[reverseStart].position < position)
            ++reverseStart;

        for (size_t j = reverseStart; j < reverse->size; ++j) {
            uint32_t lag = r[j].position - position;
            if (lag > maxLag) break;
            curve[lag] += count * r[j].count;
        }
    }

    for (size_t j = 0; j < reverse->size; ++j) {
        double count = r[j].count;
        sumR += count;
        squareR += count * count;
    }

    // Trim the margins at each lag to compute Pearson correlation on the overlap.
    reverseStart = 0;
    for (size_t lag = 0; lag <= maxLag; ++lag) {
        uint64_t overlap = dimension - lag;
        while (forwardEnd && f[forwardEnd - 1].position >= overlap) {
            double count = f[--forwardEnd].count;
            sumF -= count;
            squareF -= count * count;
        }
        while (reverseStart < reverse->size && r[reverseStart].position < lag) {
            double count = r[reverseStart++].count;
            sumR -= count;
            squareR -= count * count;
        }

        double varianceF = squareF - sumF * sumF / overlap;
        double varianceR = squareR - sumR * sumR / overlap;
        curve[lag] = varianceF > 0.0 && varianceR > 0.0
            ? (curve[lag] - sumF * sumR / overlap) / sqrt(varianceF * varianceR)
            : 0.0;
    }
}

int ccChromosomeCorrelations(ccSource *source, const char *chromosome,
                             const ccOptions *options, size_t maxLag,
                             double *curve, uint64_t *readCounts, const char **error) {
    if (source->kind != 0) return ccFail(error, ccInvalidSource);

    int tid = sam_hdr_name2tid(source->header, chromosome);
    if (tid < 0) return ccFail(error, ccMissingSource);
    hts_pos_t length = sam_hdr_tid2len(source->header, tid);
    if (length <= 0 || length > UINT32_MAX) return ccFail(error, ccInvalidRegion);

    memset(curve, 0, (maxLag + 1) * sizeof(*curve));
    readCounts[0] = 0;
    readCounts[1] = 0;
    if (source->selection->active && !source->selection->count) return 0;

    hts_itr_t *iterator = sam_itr_queryi(source->index, tid, 0, length);
    if (!iterator) return ccFail(error, ccInputQueryFailed);
    bam1_t *record = bam_init1();
    if (!record) {
        hts_itr_destroy(iterator);
        return ccFail(error, ccAllocationFailed);
    }

    ccEndpointRuns strands[2] = {0};
    int status;
    while ((status = sam_itr_next(source->file, iterator, record)) >= 0) {
        int accepted = ccBasicRecord(source, record, options, error);
        if (accepted == 0) continue;
        if (accepted < 0) {
            status = -2;
            break;
        }

        int reverse = (record->core.flag & BAM_FREVERSE) != 0;
        hts_pos_t position = record->core.pos;
        // The conceptual chromosome vector includes the right endpoint at length.
        if (reverse) {
            position += bam_cigar2rlen(record->core.n_cigar, bam_get_cigar(record));
            position = ccMin(position, length);
        } else {
            position = ccMax(position, 0);
        }
        if (ccAppendEndpoint(&strands[reverse], (uint32_t)position, error) != 0) {
            status = -2;
            break;
        }
        ++readCounts[reverse];
    }

    bam_destroy1(record);
    hts_itr_destroy(iterator);
    if (status == -1 && readCounts[0] && readCounts[1]) {
        if (ccSortEndpoints(&strands[0]) != 0 || ccSortEndpoints(&strands[1]) != 0) {
            ccFail(error, ccInvalidNumerics);
            status = -2;
        } else {
            size_t usableLag = maxLag < (uint64_t)length ? maxLag : (size_t)length - 1;
            ccCorrelateEndpoints(&strands[0], &strands[1], (uint64_t)length + 1,
                                 usableLag, curve);
        }
    }
    free(strands[0].runs);
    free(strands[1].runs);

    if (status < -1) return *error ? -1 : ccFail(error, ccInputReadFailed);
    return 0;
}

int ccBuildIndex(const char *path, int kind, const char *targetPath,
                 int threads, const char **error) {
    htsFile *input;
    BGZF *output;
    kstring_t line = {0, 0, NULL};
    int status, closed;

    if (kind == 0) {
        if (sam_index_build3(path, NULL, 0, threads) != 0)
            return ccFail(error, ccIndexBuildFailed);

        return 0;
    }

    if (kind == 1) {
        if (tbx_index_build3(path, NULL, 0, threads, &tbx_conf_bed) != 0)
            return ccFail(error, ccIndexBuildFailed);

        return 0;
    }

    input = hts_open(path, "r");
    if (!input) return ccFail(error, ccInputOpenFailed);

    output = bgzf_open(targetPath, "w");
    if (!output) {
        hts_close(input);
        return ccFail(error, ccOutputOpenFailed);
    }

    if (threads > 1 && bgzf_mt(output, threads, 256) != 0) {
        bgzf_close(output);
        hts_close(input);
        return ccFail(error, ccThreadsFailed);
    }

    while ((status = hts_getline(input, '\n', &line)) >= 0) {
        if (ccMeta(line.s)) continue;

        if (bgzf_write(output, line.s, line.l) != (ssize_t)line.l ||
            bgzf_write(output, "\n", 1) != 1) {
            free(line.s);
            bgzf_close(output);
            hts_close(input);
            return ccFail(error, ccOutputWriteFailed);
        }
    }

    free(line.s);
    closed = bgzf_close(output);
    hts_close(input);

    if (status < -1 || closed != 0) return ccFail(error, ccCompressionFailed);

    if (tbx_index_build3(targetPath, NULL, 0, threads, &tbx_conf_bed) != 0)
        return ccFail(error, ccIndexBuildFailed);

    return 0;
}
