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
    size_t count;               ///< number of output bins

    ccTotals *totals;           ///< totals output, non-NULL selects totals mode

    int64_t rangeStart;         ///< leftmost retained span start, -1 if none
    int64_t rangeEnd;           ///< rightmost retained span end (exclusive), 0 if none
} ccWork;

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

static int ccResolveFragment(ccSource *source, char *line, int64_t chromosomeLength,
                              const ccOptions *options, ccEvent *event,
                              int64_t *owner, const char **error) {
    char *fields[6];
    int n = ccColumns(line, fields, 6), a, b;
    int64_t left, right, multiplicity = 1;

    if (n < 4 ||
        ccParseCoordinate(fields[1], &left) ||
        ccParseCoordinate(fields[2], &right) ||
        right <= left || !*fields[3])
        return ccFail(error, ccInvalidFile);

    *owner = left;
    if (!ccBarcodeAllowed(source, fields[3])) return 0;

    if (options->fragmentsUseReadSupport &&
        (n < 5 ||
         ccParseCoordinate(fields[4], &multiplicity) ||
         multiplicity <= 0))
        return ccFail(error, ccInvalidCounts);

    event->multiplicity = (double)multiplicity;
    event->start = ccMax(0, left);
    event->end = ccMin(chromosomeLength, right);
    event->pointCount = 0;

    if (options->countMode == 2 && event->end > event->start) {
        event->point[0] = event->start + (event->end - event->start) / 2;
        event->pointCount = 1;
    } else if (options->countMode == 3) {
        event->point[0] = left;
        event->point[1] = right - 1;
        event->pointCount = 2;
    }

    if (options->countMode >= 2 && options->countMode <= 4) {
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
    double total = *high + value;
    double removed = total - *high;
    double error = (*high - (total - removed)) + (value - removed);

    double correction = *low + error;
    double combined = total + correction;

    removed = combined - total;
    *low = (total - (combined - removed)) + (correction - removed);
    *high = combined;
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

    if (work->options->countMode == 0 || work->options->countMode == 5) {
        firstFull = first;
        stopFull = last + 1;

        if (work->options->countMode == 0) {
            binStart = work->start + first * work->binSize;
            binEnd = ccMin(
                work->start + (last + 1) * work->binSize, work->chromosomeLength
            );

            if (left > binStart) ++firstFull;
            if (right < binEnd) --stopFull;
        }

        if (firstFull < stopFull) {
            if (work->interiorCorrection) {
                ccAccumulate(
                    event->multiplicity, &work->interior[firstFull],
                    &work->interiorCorrection[firstFull]
                );
                ccAccumulate(
                    -event->multiplicity, &work->interior[stopFull],
                    &work->interiorCorrection[stopFull]
                );
            } else {
                work->interior[firstFull] += event->multiplicity;
                work->interior[stopFull] -= event->multiplicity;
            }
        }

        if (work->options->countMode == 5) return;
    }

    for (j = first; j <= last; ++j) {
        if (j == firstFull && j < stopFull) {
            j = stopFull - 1;
            continue;
        }

        binStart = work->start + j * work->binSize;
        binEnd = ccMin(binStart + work->binSize, work->chromosomeLength);

        weight = (double)(ccMin(right, binEnd) - ccMax(left, binStart));
        weight /= (double)(work->options->countMode == 0 ?
            binEnd - binStart : event->end - event->start);

        work->values[j] += event->multiplicity * weight;
        if (work->mass) work->mass[j] += event->multiplicity * weight * weight;
    }
}

/**
 * @brief Add one accepted read/fragment to totals or bin counts, using its
 * multiplicity and the selected counting mode.
 * Also, we track the leftmost start and rightmost end of accepted nonempty spans
 * through `work`'s rangeStart and rangeEnd fields.
 */
static void ccConsume(ccWork *work, const ccEvent *event) {
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
    int64_t queryStart = start, queryEnd = end, padding = 0, owner;
    int64_t templatePadding, observedTemplateLength, maximumTemplateLength = 0;
    hts_itr_t *iterator = NULL;
    bam1_t *record = NULL;
    kstring_t line = {0, 0, NULL};
    ccEvent event;

    if (source->kind == 0) {
        tid = sam_hdr_name2tid(source->header, chromosome);
        if (tid < 0) return 0;

        if (!owned) {
            padding = ccMax(
                llabs(options->shiftForward5p), llabs(options->shiftReverse5p));
            padding = ccMin(padding, chromosomeLength);

            if (options->extendFrom5pBP >= chromosomeLength - padding)
                padding = chromosomeLength;
            else
                padding += options->extendFrom5pBP;

            if (options->bamInputMode == 1 && options->countMode != 4) {
                templatePadding = options->maxTemplateLengthBP;

                if (templatePadding <= 0) {
                    templatePadding = source->maximumTemplateLengths[tid];
                    // The first unbounded query scans the chromosome to learn its bound
                    if (templatePadding < 0) templatePadding = chromosomeLength;
                }

                if (templatePadding >= chromosomeLength - padding)
                    padding = chromosomeLength;
                else
                    padding += templatePadding;
            }

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

int ccCount(ccSource *source, const char *chromosome, int64_t chromosomeLength,
            int64_t start, int64_t end, int64_t binSize, const ccOptions *options,
            double *values, double *mass, size_t count, const char **error) {
    ccWork work = {0};
    int status, tid, parsed;
    int64_t left, right, j, first, last, binStart, binEnd;
    size_t k;
    double value, running = 0, correction = 0, contribution;
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

    if (options->countMode == 0 || options->countMode == 5) {
        work.interior = calloc(count + 1, sizeof(double));
        if (!work.interior) return ccFail(error, ccAllocationFailed);

        if (options->fragmentsUseReadSupport) {
            work.interiorCorrection = calloc(count + 1, sizeof(double));
            if (!work.interiorCorrection) {
                free(work.interior);
                return ccFail(error, ccAllocationFailed);
            }
        }
    }

    status = ccVisit(
        source, chromosome, chromosomeLength, start, end, options, &work, 0, error);

    if (work.interior) {
        for (k = 0; k < count; ++k) {
            if (work.interiorCorrection) {
                ccAccumulate(work.interior[k], &running, &correction);
                ccAccumulate(
                    work.interiorCorrection[k], &running, &correction);
                contribution = running + correction;
            } else {
                running += work.interior[k];
                contribution = running;
            }

            values[k] += contribution;
            if (mass) mass[k] += contribution;
        }

        free(work.interiorCorrection);
        free(work.interior);
    }

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
    int64_t a = *(const int64_t *)left;
    int64_t b = *(const int64_t *)right;

    return (a > b) - (a < b);
}

int ccBlockEndpoints(ccSource *source, const char *chromosome, int64_t start,
                     int64_t end, const ccOptions *options, int64_t **forward,
                     int64_t **reverse, size_t *nF, size_t *nR,
                     ccEndpointCounts *counts, const char **error) {
    int tid, status, accepted, strand;
    int64_t position;
    int64_t *endpoints[2] = {NULL, NULL};
    int64_t *grown;
    size_t capacity[2] = {0, 0}, count[2] = {0, 0};
    size_t nextCapacity, i, uniqueCount;
    bam1_t *record;
    hts_itr_t *iterator;

    *forward = NULL;
    *reverse = NULL;
    *nF = 0;
    *nR = 0;
    memset(counts, 0, sizeof(*counts));

    if (source->kind != 0) return ccFail(error, ccInvalidSource);

    tid = sam_hdr_name2tid(source->header, chromosome);
    if (tid < 0) return ccFail(error, ccMissingSource);
    if (start < 0 || end <= start || end > sam_hdr_tid2len(source->header, tid))
        return ccFail(error, ccInvalidRegion);

    iterator = sam_itr_queryi(source->index, tid, start, end);
    if (!iterator) return ccFail(error, ccInputQueryFailed);

    record = bam_init1();
    if (!record) {
        hts_itr_destroy(iterator);
        return ccFail(error, ccAllocationFailed);
    }

    while ((status = sam_itr_next(source->file, iterator, record)) >= 0) {
        ++counts->queriedRecords;
        if ((record->core.flag & (options->flagExclude | BAM_FUNMAP)) ||
            record->core.tid < 0) {
            ++counts->excludedFlags;
            continue;
        }

        if (record->core.qual < options->minMappingQuality) {
            ++counts->excludedMappingQuality;
            continue;
        }

        accepted = ccBasicRecord(source, record, options, error);
        if (accepted < 0) {
            status = -2;
            break;
        }

        if (!accepted) {
            ++counts->excludedSelection;
            continue;
        }

        strand = (record->core.flag & BAM_FREVERSE) != 0;
        if (strand) {
            if (bam_cigar2rlen(
                    record->core.n_cigar, bam_get_cigar(record)) <= 0) {
                ccFail(error, ccInvalidFile);
                status = -2;
                break;
            }

            position = bam_endpos(record);
            if (position <= start || position > end) {
                ++counts->outsideEndpointOwnership;
                continue;
            }
        } else {
            position = record->core.pos;
            if (position < start || position >= end) {
                ++counts->outsideEndpointOwnership;
                continue;
            }
        }

        if (count[strand] == capacity[strand]) {
            if (capacity[strand] > SIZE_MAX / sizeof(int64_t) / 2) {
                ccFail(error, ccInvalidNumerics);
                status = -2;
                break;
            }

            nextCapacity = capacity[strand] ? capacity[strand] * 2 : 1024;
            grown = realloc(endpoints[strand], nextCapacity * sizeof(int64_t));
            if (!grown) {
                ccFail(error, ccAllocationFailed);
                status = -2;
                break;
            }

            endpoints[strand] = grown;
            capacity[strand] = nextCapacity;
        }

        endpoints[strand][count[strand]++] = position - start;
    }

    bam_destroy1(record);
    hts_itr_destroy(iterator);

    if (status < -1) {
        free(endpoints[0]);
        free(endpoints[1]);
        return *error ? -1 : ccFail(error, ccInputReadFailed);
    }

    counts->preDedupForwardRecords = count[0];
    counts->preDedupReverseRecords = count[1];

    for (strand = 0; strand < 2; ++strand) {
        if (count[strand] > 1)
            qsort(endpoints[strand], count[strand], sizeof(int64_t), ccEndpointCompare);

        uniqueCount = 0;
        for (i = 0; i < count[strand]; ++i) {
            if (i == 0 || endpoints[strand][i] != endpoints[strand][i - 1])
                endpoints[strand][uniqueCount++] = endpoints[strand][i];
        }

        count[strand] = uniqueCount;
    }

    counts->retainedForwardEndpoints = count[0];
    counts->retainedReverseEndpoints = count[1];
    counts->removedForwardCoordinateDuplicates =
        counts->preDedupForwardRecords - count[0];
    counts->removedReverseCoordinateDuplicates =
        counts->preDedupReverseRecords - count[1];

    *forward = endpoints[0];
    *reverse = endpoints[1];
    *nF = count[0];
    *nR = count[1];

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
