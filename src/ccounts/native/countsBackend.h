#ifndef CONSENRICH_COUNTS_BACKEND_H
#define CONSENRICH_COUNTS_BACKEND_H
#include <stdint.h>
#include <stddef.h>

typedef struct ccSource ccSource;
typedef struct {
    int countMode;
    int bamInputMode;
    uint16_t flagExclude;
    int minMappingQuality;
    int64_t minTemplateLengthBP;
    int64_t maxTemplateLengthBP;
    int64_t shiftForward5p;
    int64_t shiftReverse5p;
    int64_t extendFrom5pBP;
    int fragmentsUseReadSupport;
} ccOptions;
typedef struct {
    uint64_t scannedRecords;
    double eventMass;
    double spanBP;
} ccTotals;

typedef struct {
    uint64_t queriedRecords;
    uint64_t excludedFlags;
    uint64_t excludedMappingQuality;
    uint64_t excludedSelection;
    uint64_t outsideEndpointOwnership;
    uint64_t preDedupForwardRecords;
    uint64_t preDedupReverseRecords;
    uint64_t removedForwardCoordinateDuplicates;
    uint64_t removedReverseCoordinateDuplicates;
    uint64_t retainedForwardEndpoints;
    uint64_t retainedReverseEndpoints;
} ccEndpointCounts;

int ccOpen(const char *path, int kind, const char *allowPath, const char *tag,
           int threads, ccSource **out, const char **error);
void ccClose(ccSource *source);
int ccChromosomeCount(ccSource *source);
const char *ccChromosomeName(ccSource *source, int index);
int64_t ccChromosomeLength(ccSource *source, int index);
int ccCount(ccSource *source, const char *chromosome, int64_t chromosomeLength,
            int64_t start, int64_t end, int64_t binSize, const ccOptions *options,
            double *values, double *mass, size_t count, const char **error);
int ccRegionTotals(ccSource *source, const char *chromosome, int64_t chromosomeLength,
                   int64_t start, int64_t end, const ccOptions *options,
                   ccTotals *totals, const char **error);
int ccInspect(ccSource *source, const ccOptions *options, int maxRecords,
              int *paired, const char **error);
int ccReadLength(ccSource *source, uint16_t flagExclude, uint32_t *readLength,
                 const char **error);
int ccSourceRange(ccSource *source, const char *chromosome, uint64_t chromLength,
                  uint16_t flagExclude, uint64_t *start, uint64_t *end,
                  const char **error);
int ccChromRange(ccSource *source, const char *chromosome, int64_t chromosomeLength,
                 const ccOptions *options, int64_t *start, int64_t *end,
                 const char **error);
int ccBlockEndpoints(ccSource *source, const char *chromosome, int64_t start,
                     int64_t end, const ccOptions *options, int64_t **forward,
                     int64_t **reverse, size_t *nF, size_t *nR,
                     ccEndpointCounts *counts, const char **error);
int ccBuildIndex(const char *path, int kind, const char *targetPath,
                 int threads, const char **error);
#endif
