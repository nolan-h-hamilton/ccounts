#include "countsBackend.h"
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

struct ccSource {
    int kind;
    htsFile *file;
    sam_hdr_t *header;
    hts_idx_t *index;
    tbx_t *tabix;
    crBarcodeSelection *selection;
    const char **names;
    int nameCount;
    uint32_t readLengthBP;
};

typedef struct {
    int64_t start;
    int64_t end;
    int64_t point[2];
    int pointCount;
    double multiplicity;
} ccEvent;

typedef struct {
    int64_t start;
    int64_t end;
    int64_t binSize;
    int64_t chromosomeLength;
    const ccOptions *options;
    double *values;
    double *mass;
    double *interior;
    double *interiorCorrection;
    size_t count;
    ccTotals *totals;
    int64_t rangeStart;
    int64_t rangeEnd;
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
    if (stop == *cursor || errno == ERANGE || (*stop && !isspace((unsigned char)*stop))) return -1;
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
    if ((record->core.flag & BAM_FUNMAP) || record->core.tid < 0 ||
        (record->core.flag & options->flagExclude) ||
        record->core.qual < options->minMappingQuality) return 0;
    if (options->bamInputMode == 2 &&
        (!(record->core.flag & BAM_FPAIRED) || !(record->core.flag & BAM_FREAD1) ||
         (record->core.flag & BAM_FMUNMAP))) return 0;
    selected = crBarcodeSelected(source->selection, record);
    if (selected < 0) return ccFail(error, "barcodeTag requires SAM type Z");
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
            (record->core.flag & BAM_FMUNMAP) || record->core.mtid != record->core.tid) return 0;
        length = record->core.isize;
        if (length == INT64_MIN) return ccFail(error, "template length exceeds signed native range");
        if (length < 0) length = -length;
        if (!length || (options->maxTemplateLengthBP > 0 && length > options->maxTemplateLengthBP)) return 0;
        if (options->minTemplateLengthBP < 0 && !source->readLengthBP) {
            if (ccReadLength(source, options->flagExclude |
                    (source->selection->active ? BAM_FREAD2 : 0),
                    &source->readLengthBP, error) != 0) return -1;
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
    if (n < 4 || ccParseCoordinate(fields[1], &left) ||
        ccParseCoordinate(fields[2], &right) || right <= left || !*fields[3])
        return ccFail(error, "invalid fragments record");
    *owner = left;
    if (!ccBarcodeAllowed(source, fields[3])) return 0;
    if (options->fragmentsUseReadSupport &&
        (n < 5 || ccParseCoordinate(fields[4], &multiplicity) || multiplicity <= 0))
        return ccFail(error, "fragmentsUseReadSupport requires positive integer column five");
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

static void ccPoint(ccWork *work, const ccEvent *event) {
    int k;
    int64_t bins[2] = {-1, -1};
    for (k = 0; k < event->pointCount; ++k) {
        int64_t p = event->point[k];
        if (p >= work->start && p < work->end) bins[k] = (p - work->start) / work->binSize;
    }
    if (bins[0] >= 0) {
        int same = event->pointCount == 2 && bins[0] == bins[1];
        work->values[bins[0]] += event->multiplicity * (same ? 2.0 : 1.0);
        if (work->mass) work->mass[bins[0]] += event->multiplicity * (same ? 4.0 : 1.0);
        if (same) return;
    }
    if (event->pointCount == 2 && bins[1] >= 0) {
        work->values[bins[1]] += event->multiplicity;
        if (work->mass) work->mass[bins[1]] += event->multiplicity;
    }
}

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
            binEnd = ccMin(work->start + (last + 1) * work->binSize, work->chromosomeLength);
            if (left > binStart) ++firstFull;
            if (right < binEnd) --stopFull;
        }
        if (firstFull < stopFull) {
            if (work->interiorCorrection) {
                ccAccumulate(event->multiplicity, &work->interior[firstFull], &work->interiorCorrection[firstFull]);
                ccAccumulate(-event->multiplicity, &work->interior[stopFull], &work->interiorCorrection[stopFull]);
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
        weight /= (double)(work->options->countMode == 0 ? binEnd - binStart : event->end - event->start);
        work->values[j] += event->multiplicity * weight;
        if (work->mass) work->mass[j] += event->multiplicity * weight * weight;
    }
}

static void ccConsume(ccWork *work, const ccEvent *event) {
    if (work->totals) {
        work->totals->eventMass += event->multiplicity *
            (work->options->countMode >= 2 && work->options->countMode <= 4 ? event->pointCount : 1);
        work->totals->spanBP += event->multiplicity * (double)ccMax(0, event->end - event->start);
    } else if (work->options->countMode >= 2 && work->options->countMode <= 4) {
        ccPoint(work, event);
    } else {
        ccSpan(work, event);
    }
    if (event->end > event->start) {
        if (work->rangeStart < 0 || event->start < work->rangeStart) work->rangeStart = event->start;
        if (event->end > work->rangeEnd) work->rangeEnd = event->end;
    }
}

static int ccVisit(ccSource *source, const char *chromosome, int64_t chromosomeLength,
                    int64_t start, int64_t end, const ccOptions *options,
                    ccWork *work, int owned, const char **error) {
    int tid, status, accepted;
    int64_t queryStart = start, queryEnd = end, padding = 0, owner;
    hts_itr_t *iterator = NULL;
    bam1_t *record = NULL;
    kstring_t line = {0, 0, NULL};
    ccEvent event;
    if (source->kind == 0) {
        tid = sam_hdr_name2tid(source->header, chromosome);
        if (tid < 0) return 0;
        if (!owned) {
            padding = ccMax(llabs(options->shiftForward5p), llabs(options->shiftReverse5p));
            padding += options->extendFrom5pBP;
            if (options->bamInputMode == 1 && options->countMode != 4) {
                if (options->maxTemplateLengthBP <= 0) padding = chromosomeLength;
                else padding += options->maxTemplateLengthBP;
            }
            queryStart = ccMax(0, start - padding);
            queryEnd = ccMin(chromosomeLength, end + padding);
        }
        iterator = sam_itr_queryi(source->index, tid, queryStart, queryEnd);
        if (!iterator) return ccFail(error, "failed BAM region query");
        record = bam_init1();
        if (!record) { hts_itr_destroy(iterator); return ccFail(error, "failed BAM record allocation"); }
        while ((status = sam_itr_next(source->file, iterator, record)) >= 0) {
            if ((record->core.flag & BAM_FUNMAP) || record->core.tid < 0) continue;
            owner = record->core.pos;
            if (owned && (owner < start || owner >= end)) continue;
            if (work->totals) ++work->totals->scannedRecords;
            accepted = ccResolveBAM(source, record, chromosomeLength, options, &event, error);
            if (accepted < 0) { status = -2; break; }
            if (accepted) ccConsume(work, &event);
        }
        bam_destroy1(record);
        hts_itr_destroy(iterator);
        if (status < -1) return *error ? -1 : ccFail(error, "failed to read BAM region");
        return 0;
    }
    tid = tbx_name2id(source->tabix, chromosome);
    if (tid < 0) return 0;
    if (owned) {
        queryStart = start;
        queryEnd = end;
    }
    iterator = tbx_itr_queryi(source->tabix, tid, queryStart, queryEnd);
    if (!iterator) return ccFail(error, "failed fragments region query");
    while ((status = tbx_itr_next(source->file, source->tabix, iterator, &line)) >= 0) {
        accepted = ccResolveFragment(source, line.s, chromosomeLength, options, &event, &owner, error);
        if (accepted < 0) { status = -2; break; }
        if (owned && (owner < start || owner >= end)) continue;
        if (work->totals) ++work->totals->scannedRecords;
        if (accepted) ccConsume(work, &event);
    }
    free(line.s);
    hts_itr_destroy(iterator);
    if (status < -1) return *error ? -1 : ccFail(error, "failed to read fragments region");
    return 0;
}

static int ccBedGraphLine(char *line, char **chromosome, int64_t *start,
                           int64_t *end, double *value, const char **error) {
    char *fields[5], *stop;
    int n;
    if (ccMeta(line)) return 0;
    n = ccColumns(line, fields, 5);
    if (n != 4 || ccParseCoordinate(fields[1], start) || ccParseCoordinate(fields[2], end) || *end <= *start)
        return ccFail(error, "invalid bedGraph interval");
    errno = 0;
    *value = strtod(fields[3], &stop);
    if (stop == fields[3] || *stop || !isfinite(*value)) return ccFail(error, "invalid bedGraph value");
    *chromosome = fields[0];
    return 1;
}

static int ccValidateBedGraph(ccSource *source, const char *path, int threads, const char **error) {
    int status, parsed, tid = -1, nextTid;
    int64_t left, right, lastEnd = -1;
    double value;
    char *chromosome;
    unsigned char *seen = calloc((size_t)source->nameCount, 1);
    kstring_t line = {0, 0, NULL};
    if (!seen && source->nameCount) return ccFail(error, "failed bedGraph validation allocation");
    while ((status = hts_getline(source->file, '\n', &line)) >= 0) {
        parsed = ccBedGraphLine(line.s, &chromosome, &left, &right, &value, error);
        if (parsed < 0) { status = -2; break; }
        if (!parsed) continue;
        nextTid = tbx_name2id(source->tabix, chromosome);
        if (nextTid < 0) { ccFail(error, "bedGraph index differs from source"); status = -2; break; }
        if (tid != nextTid) {
            if (seen[nextTid]) { ccFail(error, "bedGraph chromosomes must form contiguous groups"); status = -2; break; }
            seen[nextTid] = 1;
            tid = nextTid;
            lastEnd = -1;
        }
        if (left < lastEnd) { ccFail(error, "bedGraph intervals overlap or are unsorted"); status = -2; break; }
        lastEnd = right;
    }
    free(seen);
    free(line.s);
    hts_close(source->file);
    source->file = hts_open(path, "r");
    if (!source->file) return ccFail(error, "failed to reopen validated bedGraph");
    if (threads > 1 && hts_set_threads(source->file, threads) != 0)
        return ccFail(error, "failed to configure input threads");
    if (status < -1) return *error ? -1 : ccFail(error, "failed to validate bedGraph input");
    return 0;
}

int ccOpen(const char *path, int kind, const char *allowPath, const char *tag,
           int threads, ccSource **out, const char **error) {
    ccSource *source = calloc(1, sizeof(ccSource));
    *out = NULL;
    if (!source) return ccFail(error, "failed native source allocation");
    source->kind = kind;
    source->file = hts_open(path, "r");
    if (!source->file) { ccClose(source); return ccFail(error, "failed to open source"); }
    if (threads > 1 && hts_set_threads(source->file, threads) != 0) {
        ccClose(source); return ccFail(error, "failed to configure input threads");
    }
    source->selection = crBarcodeSelectionCreate(tag, allowPath, error);
    if (!source->selection) { ccClose(source); return -1; }
    if (kind == 0) {
        if (hts_get_format(source->file)->format != bam) {
            ccClose(source); return ccFail(error, "BAM source requires BAM file format");
        }
        source->header = sam_hdr_read(source->file);
        if (!source->header) { ccClose(source); return ccFail(error, "failed BAM header read"); }
        source->index = sam_index_load(source->file, path);
        if (!source->index) { ccClose(source); return ccFail(error, "BAM arbitrary access requires an index"); }
        source->nameCount = sam_hdr_nref(source->header);
    } else {
        source->tabix = tbx_index_load(path);
        if (!source->tabix) { ccClose(source); return ccFail(error, "tabix source requires an index"); }
        source->names = tbx_seqnames(source->tabix, &source->nameCount);
        if (kind == 2 && ccValidateBedGraph(source, path, threads, error) != 0) { ccClose(source); return -1; }
    }
    *out = source;
    return 0;
}

void ccClose(ccSource *source) {
    if (!source) return;
    if (source->index) hts_idx_destroy(source->index);
    if (source->tabix) tbx_destroy(source->tabix);
    if (source->names) free(source->names);
    if (source->header) sam_hdr_destroy(source->header);
    if (source->file) hts_close(source->file);
    crBarcodeSelectionDestroy(source->selection);
    free(source);
}

int ccChromosomeCount(ccSource *source) { return source->nameCount; }
const char *ccChromosomeName(ccSource *source, int index) {
    return source->kind == 0 ? sam_hdr_tid2name(source->header, index) : source->names[index];
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
    work.start = start; work.end = end; work.binSize = binSize;
    work.chromosomeLength = chromosomeLength; work.options = options;
    work.values = values; work.mass = mass; work.count = count;
    work.rangeStart = -1;
    if (source->kind == 2) {
        tid = tbx_name2id(source->tabix, chromosome);
        if (tid < 0) return 0;
        iterator = tbx_itr_queryi(source->tabix, tid, start, end);
        if (!iterator) return ccFail(error, "failed bedGraph query");
        while ((status = tbx_itr_next(source->file, source->tabix, iterator, &line)) >= 0) {
            parsed = ccBedGraphLine(line.s, &recordChromosome, &left, &right, &value, error);
            if (parsed < 0) { status = -2; break; }
            if (!parsed) continue;
            if (right > chromosomeLength) { ccFail(error, "bedGraph interval exceeds chromosome size"); status = -2; break; }
            left = ccMax(start, left); right = ccMin(end, right);
            if (right <= left) continue;
            first = (left - start) / binSize; last = (right - 1 - start) / binSize;
            for (j = first; j <= last; ++j) {
                binStart = start + j * binSize;
                binEnd = ccMin(binStart + binSize, chromosomeLength);
                values[j] += value * ((double)(ccMin(right, binEnd) - ccMax(left, binStart)) / (double)(binEnd - binStart));
            }
        }
        free(line.s); hts_itr_destroy(iterator);
        if (status < -1) return *error ? -1 : ccFail(error, "failed bedGraph read");
        return 0;
    }
    if (options->countMode == 0 || options->countMode == 5) {
        work.interior = calloc(count + 1, sizeof(double));
        if (!work.interior) return ccFail(error, "failed interval accumulation allocation");
        if (options->fragmentsUseReadSupport) {
            work.interiorCorrection = calloc(count + 1, sizeof(double));
            if (!work.interiorCorrection) {
                free(work.interior);
                return ccFail(error, "failed compensated interval allocation");
            }
        }
    }
    status = ccVisit(source, chromosome, chromosomeLength, start, end, options, &work, 0, error);
    if (work.interior) {
        for (k = 0; k < count; ++k) {
            if (work.interiorCorrection) {
                ccAccumulate(work.interior[k], &running, &correction);
                ccAccumulate(work.interiorCorrection[k], &running, &correction);
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
    if (source->kind == 2) return ccFail(error, "bedGraph has no event denominators");
    memset(totals, 0, sizeof(*totals));
    work.options = options; work.totals = totals; work.rangeStart = -1;
    return ccVisit(source, chromosome, chromosomeLength, start, end, options, &work, 1, error);
}

static int ccCompareReadLengths(const void *first, const void *second) {
    uint32_t left = *(const uint32_t *)first;
    uint32_t right = *(const uint32_t *)second;
    return (left > right) - (left < right);
}

int ccReadLength(ccSource *source, uint16_t flagExclude, uint32_t *readLength,
                 const char **error) {
    samFile *fileHandle = NULL;
    sam_hdr_t *header = NULL;
    bam1_t *record = NULL;
    uint32_t lengths[100];
    int readCount = 0, iterCount = 0, selected, midIndex, result = 0;
    hts_pos_t queryLength;
    *readLength = 0;
    if (source->kind != 0) return ccFail(error, "read-length inspection requires BAM");
    fileHandle = sam_open(source->file->fn, "r");
    if (!fileHandle) return ccFail(error, "failed to open BAM for read-length inspection");
    header = sam_hdr_read(fileHandle);
    if (!header) {
        result = ccFail(error, "failed BAM header read");
        goto finish;
    }
    record = bam_init1();
    if (!record) {
        result = ccFail(error, "failed BAM record allocation");
        goto finish;
    }
    while ((source->selection->active ? readCount : iterCount) < 1000 &&
           sam_read1(fileHandle, header, record) >= 0) {
        ++iterCount;
        if (record->core.flag & flagExclude) continue;
        selected = crBarcodeSelected(source->selection, record);
        if (selected < 0) {
            result = ccFail(error, "barcodeTag requires SAM type Z");
            goto finish;
        }
        if (!selected) continue;
        queryLength = record->core.l_qseq;
        if (queryLength <= 0 && record->core.n_cigar > 0)
            queryLength = bam_cigar2qlen(record->core.n_cigar, bam_get_cigar(record));
        if (queryLength <= 0) continue;
        lengths[readCount++] = (uint32_t)queryLength;
        if (readCount >= 100) break;
    }
    if (!readCount) {
        result = ccFail(error, "failed to estimate read length");
        goto finish;
    }
    qsort(lengths, (size_t)readCount, sizeof(uint32_t), ccCompareReadLengths);
    midIndex = readCount / 2;
    *readLength = (readCount & 1) ? lengths[midIndex] :
        (uint32_t)((lengths[midIndex - 1] + lengths[midIndex]) / 2U);
    source->readLengthBP = *readLength;
finish:
    if (record) bam_destroy1(record);
    if (header) sam_hdr_destroy(header);
    sam_close(fileHandle);
    return result;
}

int ccInspect(ccSource *source, const ccOptions *options, int maxRecords,
              int *paired, double *readLength, const char **error) {
    int t, status, accepted, sampled = 0, lengths = 0;
    int64_t queryLength;
    double length = 0;
    bam1_t *record;
    hts_itr_t *iterator;
    *paired = 0; *readLength = 0;
    if (source->kind != 0) return ccFail(error, "BAM inspection requires BAM");
    if (source->selection->active && source->selection->count == 0) return 0;
    record = bam_init1();
    if (!record) return ccFail(error, "failed BAM record allocation");
    for (t = 0; t < source->nameCount && sampled < maxRecords; ++t) {
        iterator = sam_itr_queryi(source->index, t, 0, sam_hdr_tid2len(source->header, t));
        if (!iterator) { bam_destroy1(record); return ccFail(error, "failed BAM inspection query"); }
        while (sampled < maxRecords && (status = sam_itr_next(source->file, iterator, record)) >= 0) {
            accepted = ccBasicRecord(source, record, options, error);
            if (accepted < 0) { hts_itr_destroy(iterator); bam_destroy1(record); return -1; }
            if (!accepted) continue;
            ++sampled;
            if (record->core.flag & BAM_FPAIRED) *paired = 1;
            queryLength = record->core.l_qseq > 0 ? record->core.l_qseq :
                bam_cigar2qlen(record->core.n_cigar, bam_get_cigar(record));
            if (queryLength > 0) { length += queryLength; ++lengths; }
        }
        hts_itr_destroy(iterator);
        if (sampled < maxRecords && status < -1) { bam_destroy1(record); return ccFail(error, "failed BAM inspection read"); }
    }
    bam_destroy1(record);
    if (lengths) *readLength = length / lengths;
    return 0;
}

int ccFragmentLengths(ccSource *source, const ccOptions *options, int maxRecords,
                      int64_t *lengths, int *count, const char **error) {
    int t, status, accepted;
    int64_t length;
    bam1_t *record;
    hts_itr_t *iterator;
    *count = 0;
    if (source->kind != 0) return ccFail(error, "fragment-length inference requires BAM");
    record = bam_init1();
    if (!record) return ccFail(error, "failed BAM record allocation");
    for (t = 0; t < source->nameCount && *count < maxRecords; ++t) {
        iterator = sam_itr_queryi(source->index, t, 0, sam_hdr_tid2len(source->header, t));
        if (!iterator) { bam_destroy1(record); return ccFail(error, "failed fragment-length query"); }
        while (*count < maxRecords && (status = sam_itr_next(source->file, iterator, record)) >= 0) {
            accepted = ccBasicRecord(source, record, options, error);
            if (accepted < 0) { hts_itr_destroy(iterator); bam_destroy1(record); return -1; }
            if (!accepted || !(record->core.flag & BAM_FPROPER_PAIR) ||
                !(record->core.flag & BAM_FREAD1) || (record->core.flag & BAM_FMUNMAP) ||
                record->core.mtid != record->core.tid) continue;
            length = record->core.isize;
            if (length == INT64_MIN) { hts_itr_destroy(iterator); bam_destroy1(record); return ccFail(error, "template length exceeds signed native range"); }
            if (length < 0) length = -length;
            if (length <= 0 || length < options->minTemplateLengthBP ||
                (options->maxTemplateLengthBP > 0 && length > options->maxTemplateLengthBP)) continue;
            lengths[(*count)++] = length;
        }
        hts_itr_destroy(iterator);
        if (*count < maxRecords && status < -1) { bam_destroy1(record); return ccFail(error, "failed fragment-length read"); }
    }
    bam_destroy1(record);
    return 0;
}

static int ccSourceRangeInt64Field(
    const char *fieldStart,
    size_t fieldLength,
    int64_t *valueOut)
{
    size_t fieldIndex = 0U;
    int sign = 1;
    int64_t value = 0;

    if (valueOut != NULL)
    {
        *valueOut = 0;
    }
    if (fieldStart == NULL || fieldLength == 0U)
    {
        return 0;
    }
    if (fieldStart[0] == '-')
    {
        sign = -1;
        fieldIndex = 1U;
    }
    for (; fieldIndex < fieldLength; ++fieldIndex)
    {
        if (fieldStart[fieldIndex] < '0' || fieldStart[fieldIndex] > '9')
        {
            return 0;
        }
        value = (10 * value) + (int64_t)(fieldStart[fieldIndex] - '0');
    }
    if (valueOut != NULL)
    {
        *valueOut = sign > 0 ? value : -value;
    }
    return 1;
}

static int ccSourceRangeFieldEquals(
    const char *fieldStart,
    size_t fieldLength,
    const char *target)
{
    size_t targetLength = 0U;
    if (fieldStart == NULL || target == NULL)
    {
        return 0;
    }
    targetLength = strlen(target);
    return targetLength == fieldLength && strncmp(fieldStart, target, fieldLength) == 0;
}

int ccSourceRange(ccSource *source, const char *chromosome, uint64_t chromLength,
                  uint16_t flagExclude, uint64_t *startOut, uint64_t *endOut,
                  const char **error)
{
    bam1_t *record = NULL;
    hts_itr_t *iteratorHandle = NULL;
    int32_t tid = -1;
    uint64_t tailStart = 0;
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
        int useBedGraphIterator = 0;
        int iteratorCode = 0;
        int seenRecord = 0;
        int64_t bgStart = 0;
        int64_t bgEnd = 0;
        uint64_t minStart = 0;
        uint64_t maxEnd = 0;

        if (source->file == NULL)
        {
            return ccFail(error, "bedGraph source handle is not initialized");
        }

        if (source->tabix != NULL)
        {
            useBedGraphIterator = 1;
            snprintf(
                regionBuffer,
                sizeof(regionBuffer),
                "%s:%d-%llu",
                chromosome,
                1,
                (unsigned long long)chromLength);
            bedGraphIterator = tbx_itr_querys(source->tabix, regionBuffer);
        }

        while ((iteratorCode = (useBedGraphIterator
                                    ? (bedGraphIterator != NULL
                                           ? tbx_itr_next(
                                                 source->file,
                                                 source->tabix,
                                                 bedGraphIterator,
                                                 &lineBuffer)
                                           : -1)
                                    : hts_getline(source->file, '\n', &lineBuffer))) >= 0)
        {
            const char *cursor = lineBuffer.s;
            const char *lineStart = NULL;
            const char *fieldStart = NULL;
            size_t fieldLength = 0U;
            const char *chromStart = NULL;
            size_t chromLength = 0U;
            int fieldIndex = 0;
            int validRecord = 1;

            while (*cursor != '\0' && isspace((unsigned char)(*cursor)))
            {
                ++cursor;
            }
            lineStart = cursor;
            if (*lineStart == '\0' || *lineStart == '#' ||
                strncmp(lineStart, "track", 5U) == 0 ||
                strncmp(lineStart, "browser", 7U) == 0)
            {
                continue;
            }

            while (fieldIndex < 3 && *cursor != '\0' && *cursor != '\n' && *cursor != '\r')
            {
                fieldStart = cursor;
                while (*cursor != '\0' && *cursor != '\n' && *cursor != '\r' &&
                       !isspace((unsigned char)(*cursor)))
                {
                    ++cursor;
                }
                fieldLength = (size_t)(cursor - fieldStart);
                if (fieldIndex == 0)
                {
                    chromStart = fieldStart;
                    chromLength = fieldLength;
                }
                else if (fieldIndex == 1 && !ccSourceRangeInt64Field(fieldStart, fieldLength, &bgStart))
                {
                    validRecord = 0;
                    break;
                }
                else if (fieldIndex == 2 && !ccSourceRangeInt64Field(fieldStart, fieldLength, &bgEnd))
                {
                    validRecord = 0;
                    break;
                }
                while (*cursor != '\0' && isspace((unsigned char)(*cursor)))
                {
                    ++cursor;
                }
                ++fieldIndex;
            }

            if (!validRecord || fieldIndex < 3 || bgStart < 0 || bgEnd <= bgStart ||
                !ccSourceRangeFieldEquals(chromStart, chromLength, chromosome))
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
        if (lineBuffer.s != NULL)
        {
            free(lineBuffer.s);
        }
        if (bedGraphIterator != NULL)
        {
            hts_itr_destroy(bedGraphIterator);
        }
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
        hts_itr_t *iteratorHandle = NULL;
        int64_t fragStart = 0;
        int64_t fragEnd = 0;
        int iteratorCode = 0;
        int seenFragment = 0;

        if (source->tabix == NULL || source->file == NULL)
        {
            return ccFail(error, "fragments index is required for chromosome range queries");
        }

        snprintf(regionBuffer, sizeof(regionBuffer), "%s:%d-%llu", chromosome, 1, (unsigned long long)chromLength);
        iteratorHandle = tbx_itr_querys(source->tabix, regionBuffer);
        if (iteratorHandle != NULL)
        {
            while ((iteratorCode = tbx_itr_next(source->file, source->tabix, iteratorHandle, &lineBuffer)) >= 0)
            {
                const char *cursor = lineBuffer.s;
                const char *fieldStart = NULL;
                const char *fieldEnd = NULL;
                size_t fieldLength = 0U;
                int fieldIndex = 0;
                while (fieldIndex < 3)
                {
                    fieldStart = cursor;
                    while (*cursor != '\0' && *cursor != '\t' && *cursor != '\n' && *cursor != '\r')
                    {
                        ++cursor;
                    }
                    fieldEnd = cursor;
                    fieldLength = (size_t)(fieldEnd - fieldStart);
                    if (fieldIndex == 1 && !ccSourceRangeInt64Field(fieldStart, fieldLength, &fragStart))
                    {
                        fieldLength = 0U;
                        break;
                    }
                    if (fieldIndex == 2 && !ccSourceRangeInt64Field(fieldStart, fieldLength, &fragEnd))
                    {
                        fieldLength = 0U;
                        break;
                    }
                    if (*cursor == '\t')
                    {
                        ++cursor;
                    }
                    ++fieldIndex;
                }
                if (fieldLength == 0U || fragEnd <= fragStart)
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
                if (endOut != NULL)
                {
                    *endOut = (uint64_t)fragEnd;
                }
            }
            hts_itr_destroy(iteratorHandle);
        }
        if (lineBuffer.s != NULL)
        {
            free(lineBuffer.s);
        }
        return 0;
    }
    if (source->index == NULL)
    {
        return ccFail(error, "alignment index is required for chromosome range queries");
    }

    tid = sam_hdr_name2tid(source->header, chromosome);
    if (tid < 0)
    {
        return ccFail(error, "chromosome not found in alignment header");
    }

    record = bam_init1();
    if (record == NULL)
    {
        return ccFail(error, "failed to allocate bam record");
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
                *endOut = (uint64_t)bam_endpos(record);
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
                    *endOut = (uint64_t)bam_endpos(record);
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
        *start = 0; *end = 0;
        if (tid < 0) return 0;
        iterator = tbx_itr_queryi(source->tabix, tid, 0, chromosomeLength);
        if (!iterator) return ccFail(error, "failed bedGraph range query");
        while ((status = tbx_itr_next(source->file, source->tabix, iterator, &line)) >= 0) {
            parsed = ccBedGraphLine(line.s, &recordChromosome, &left, &right, &value, error);
            if (parsed < 0) { status = -2; break; }
            if (!parsed) continue;
            if (right > chromosomeLength) { ccFail(error, "bedGraph interval exceeds chromosome size"); status = -2; break; }
            if (!*end || left < *start) *start = left;
            if (right > *end) *end = right;
        }
        free(line.s); hts_itr_destroy(iterator);
        if (status < -1) return *error ? -1 : ccFail(error, "failed bedGraph range read");
        return 0;
    }
    work.options = options; work.totals = &totals; work.rangeStart = -1;
    status = ccVisit(source, chromosome, chromosomeLength, 0, chromosomeLength, options, &work, 1, error);
    *start = work.rangeStart < 0 ? 0 : work.rangeStart;
    *end = work.rangeStart < 0 ? 0 : work.rangeEnd;
    return status;
}

int ccStrandPositions(ccSource *source, const char *chromosome, int64_t start,
                      int64_t end, const ccOptions *options, double *forward,
                      double *reverse, const char **error) {
    int tid, status, accepted;
    int64_t p;
    bam1_t *record;
    hts_itr_t *iterator;
    if (source->kind != 0) return ccFail(error, "strand tracks require BAM");
    tid = sam_hdr_name2tid(source->header, chromosome);
    if (tid < 0) return 0;
    iterator = sam_itr_queryi(source->index, tid, start, end);
    if (!iterator) return ccFail(error, "failed strand-track query");
    record = bam_init1();
    if (!record) { hts_itr_destroy(iterator); return ccFail(error, "failed BAM record allocation"); }
    while ((status = sam_itr_next(source->file, iterator, record)) >= 0) {
        accepted = ccBasicRecord(source, record, options, error);
        if (accepted < 0) { status = -2; break; }
        if (!accepted) continue;
        p = (record->core.flag & BAM_FREVERSE) ? bam_endpos(record) - 1 : record->core.pos;
        if (p < start || p >= end) continue;
        if (record->core.flag & BAM_FREVERSE) reverse[p - start] += 1;
        else forward[p - start] += 1;
    }
    bam_destroy1(record); hts_itr_destroy(iterator);
    if (status < -1) return *error ? -1 : ccFail(error, "failed strand-track read");
    return 0;
}

int ccBuildIndex(const char *path, int kind, const char *targetPath,
                 int threads, const char **error) {
    htsFile *input;
    BGZF *output;
    kstring_t line = {0, 0, NULL};
    int status, closed;
    if (kind == 0) {
        if (sam_index_build3(path, NULL, 0, threads) != 0) return ccFail(error, "failed BAM index build");
        return 0;
    }
    if (kind == 1) {
        if (tbx_index_build3(path, NULL, 0, threads, &tbx_conf_bed) != 0)
            return ccFail(error, "failed fragments index build");
        return 0;
    }
    input = hts_open(path, "r");
    if (!input) return ccFail(error, "failed bedGraph indexing input");
    output = bgzf_open(targetPath, "w");
    if (!output) { hts_close(input); return ccFail(error, "failed bedGraph indexing output"); }
    if (threads > 1 && bgzf_mt(output, threads, 256) != 0) {
        bgzf_close(output); hts_close(input); return ccFail(error, "failed compression thread setup");
    }
    while ((status = hts_getline(input, '\n', &line)) >= 0) {
        if (ccMeta(line.s)) continue;
        if (bgzf_write(output, line.s, line.l) != (ssize_t)line.l || bgzf_write(output, "\n", 1) != 1) {
            free(line.s); bgzf_close(output); hts_close(input); return ccFail(error, "failed compressed bedGraph write");
        }
    }
    free(line.s); closed = bgzf_close(output); hts_close(input);
    if (status < -1 || closed != 0) return ccFail(error, "failed bedGraph compression");
    if (tbx_index_build3(targetPath, NULL, 0, threads, &tbx_conf_bed) != 0)
        return ccFail(error, "failed bedGraph index build");
    return 0;
}

int ccDensity(ccSource *source, const char *chromosome, int64_t binSize,
              const ccOptions *options, double *density, size_t count,
              const char **error) {
    int tid, status, accepted;
    int64_t bin;
    bam1_t *record;
    hts_itr_t *iterator;
    if (source->kind != 0) return ccFail(error, "strand density requires BAM");
    tid = sam_hdr_name2tid(source->header, chromosome);
    if (tid < 0) return 0;
    iterator = sam_itr_queryi(source->index, tid, 0, sam_hdr_tid2len(source->header, tid));
    if (!iterator) return ccFail(error, "failed density query");
    record = bam_init1();
    if (!record) { hts_itr_destroy(iterator); return ccFail(error, "failed BAM record allocation"); }
    while ((status = sam_itr_next(source->file, iterator, record)) >= 0) {
        accepted = ccBasicRecord(source, record, options, error);
        if (accepted < 0) { status = -2; break; }
        if (!accepted) continue;
        bin = record->core.pos / binSize;
        if (bin >= 0 && (size_t)bin < count) density[bin] += 1;
    }
    bam_destroy1(record); hts_itr_destroy(iterator);
    if (status < -1) return *error ? -1 : ccFail(error, "failed density read");
    return 0;
}
