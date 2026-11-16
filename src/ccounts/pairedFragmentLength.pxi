from libc.stdint cimport int32_t, uint8_t

cdef extern from "htslib/hts.h":
    ctypedef struct htsFile
    ctypedef struct hts_idx_t
    ctypedef struct hts_itr_t
    ctypedef long long hts_pos_t
    int hts_set_threads(htsFile *fp, int n)
    void hts_idx_destroy(hts_idx_t *idx)
    void hts_itr_destroy(hts_itr_t *itr)

cdef extern from "htslib/sam.h":
    ctypedef struct samFile
    ctypedef struct bam1_core_t:
        uint16_t flag
        int32_t l_qseq
        uint32_t n_cigar
        int64_t isize
    ctypedef struct bam1_t:
        bam1_core_t core
    ctypedef struct sam_hdr_t:
        int32_t n_targets
        uint32_t *target_len
    samFile *sam_open(const char *fn, const char *mode)
    int sam_close(samFile *fp)
    sam_hdr_t *sam_hdr_read(samFile *fp)
    void sam_hdr_destroy(sam_hdr_t *h)
    bam1_t *bam_init1()
    void bam_destroy1(bam1_t *b)
    hts_idx_t *sam_index_load(htsFile *fp, const char *fn)
    hts_itr_t *sam_itr_queryi(hts_idx_t *idx, int tid, hts_pos_t beg, hts_pos_t end)
    int sam_itr_next(htsFile *fp, hts_itr_t *itr, bam1_t *record)
    uint32_t *bam_get_cigar(bam1_t *b)
    hts_pos_t bam_cigar2qlen(int n_cigar, uint32_t *cigar)

cdef extern from "native/barcode_selection.h":
    ctypedef struct crBarcodeSelection:
        int active
    crBarcodeSelection *crBarcodeSelectionCreate(const char *tag, const char *path, const char **error)
    void crBarcodeSelectionDestroy(crBarcodeSelection *selection)
    int crBarcodeSelected(const crBarcodeSelection *selection, const bam1_t *record)

cdef dict _pairedFragmentLength(
    str bamFile,
    uint16_t samThreads,
    uint16_t samFlagExclude,
    int64_t maxInsertSize,
    str barcodeAllowListFile,
    str barcodeTag,
):
    cdef int readFlag, selected, status, kTop, contigTid, tlenN = 0
    cdef int64_t queryLength, contigLen, tlen, minimum, estimate
    cdef int readLengthCount = 0
    cdef double readLengthSum = 0.0
    cdef Py_ssize_t contigIndex
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lengths, templateLengths
    cdef object orderedContigs
    cdef bytes bamBytes = bamFile.encode("utf-8")
    cdef bytes allowBytes = barcodeAllowListFile.encode("utf-8")
    cdef bytes tagBytes = barcodeTag.encode("utf-8")
    cdef samFile *fileHandle = NULL
    cdef sam_hdr_t *header = NULL
    cdef hts_idx_t *indexHandle = NULL
    cdef hts_itr_t *iteratorHandle = NULL
    cdef bam1_t *record = NULL
    cdef crBarcodeSelection *selection = NULL
    cdef const char *selectionError = NULL
    cdef bint hasSelection = bool(barcodeAllowListFile)
    if maxInsertSize < 1:
        maxInsertSize = 1
    fileHandle = sam_open(bamBytes, "r")
    if fileHandle == NULL:
        raise SourceError(kind="io")
    try:
        selection = crBarcodeSelectionCreate(tagBytes, allowBytes, &selectionError)
        if selection == NULL:
            raise NativeError((<bytes>selectionError).decode("utf-8"))
        if samThreads > 1 and hts_set_threads(<htsFile *>fileHandle, samThreads) < 0:
            raise SourceError(kind="io")
        header = sam_hdr_read(fileHandle)
        if header == NULL or header.n_targets <= 0:
            raise SourceError()
        indexHandle = sam_index_load(<htsFile *>fileHandle, bamBytes)
        if indexHandle == NULL:
            raise SourceError(kind="index")
        record = bam_init1()
        if record == NULL:
            raise AllocationError()
        lengths = np.empty(header.n_targets, dtype=np.int64)
        for contigIndex in range(header.n_targets):
            lengths[contigIndex] = header.target_len[contigIndex]
        kTop = header.n_targets if hasSelection else min(3, header.n_targets)
        orderedContigs = np.argpartition(lengths, -kTop)[-kTop:]
        orderedContigs = orderedContigs[np.argsort(lengths[orderedContigs])[::-1]]
        for contigIndex in orderedContigs:
            contigTid = <int>contigIndex
            contigLen = lengths[contigTid]
            if contigLen <= 0:
                continue
            iteratorHandle = sam_itr_queryi(indexHandle, contigTid, 0, contigLen)
            if iteratorHandle == NULL:
                raise SourceError(kind="io")
            while True:
                status = sam_itr_next(<htsFile *>fileHandle, iteratorHandle, record)
                if status < 0:
                    if status < -1:
                        raise SourceError(kind="io")
                    break
                readFlag = record.core.flag
                if readFlag & samFlagExclude:
                    continue
                if hasSelection and readFlag & 1 and not readFlag & 64:
                    continue
                selected = crBarcodeSelected(selection, record)
                if selected < 0:
                    raise InputError(kind="selection")
                if not selected:
                    continue
                if readLengthCount >= 1000:
                    break
                queryLength = record.core.l_qseq
                if queryLength <= 0 and record.core.n_cigar > 0:
                    queryLength = bam_cigar2qlen(record.core.n_cigar, bam_get_cigar(record))
                if queryLength <= 0:
                    continue
                readLengthSum += queryLength
                readLengthCount += 1
            hts_itr_destroy(iteratorHandle)
            iteratorHandle = NULL
            if readLengthCount >= 1000:
                break
        if readLengthCount == 0:
            if hasSelection:
                raise EstimateError()
            return {"estimateBP": 147, "status": "insufficientData", "retainedTemplateCount": 0,
                    "minimumTemplateLengthBP": None, "maximumTemplateLengthBP": int(maxInsertSize)}
        minimum = <int64_t>(readLengthSum / readLengthCount)
        minimum = max(1, min(minimum, maxInsertSize))
        templateLengths = np.empty(2000, dtype=np.int64)
        for contigIndex in orderedContigs:
            if tlenN >= 2000:
                break
            contigTid = <int>contigIndex
            contigLen = lengths[contigTid]
            if contigLen <= 0:
                continue
            iteratorHandle = sam_itr_queryi(indexHandle, contigTid, 0, contigLen)
            if iteratorHandle == NULL:
                raise SourceError(kind="io")
            while True:
                status = sam_itr_next(<htsFile *>fileHandle, iteratorHandle, record)
                if status < 0:
                    if status < -1:
                        raise SourceError(kind="io")
                    break
                if tlenN >= 2000:
                    break
                readFlag = record.core.flag
                if readFlag & samFlagExclude or not readFlag & 64:
                    continue
                selected = crBarcodeSelected(selection, record)
                if selected < 0:
                    raise InputError(kind="selection")
                if not selected or not readFlag & 2:
                    continue
                tlen = record.core.isize
                if tlen == -9223372036854775807 - 1:
                    raise NumericalError()
                if tlen < 0:
                    tlen = -tlen
                if tlen == 0 or tlen < minimum or tlen > maxInsertSize:
                    continue
                templateLengths[tlenN] = tlen
                tlenN += 1
            hts_itr_destroy(iteratorHandle)
            iteratorHandle = NULL
        if tlenN < (1 if hasSelection else 500):
            if hasSelection:
                raise EstimateError()
            return {"estimateBP": 147, "status": "insufficientData", "retainedTemplateCount": tlenN,
                    "minimumTemplateLengthBP": int(minimum), "maximumTemplateLengthBP": int(maxInsertSize)}
        estimate = np.partition(templateLengths[:tlenN], tlenN // 2)[tlenN // 2]
        return {"estimateBP": int(estimate), "status": "observedTemplate", "retainedTemplateCount": tlenN,
                "minimumTemplateLengthBP": int(minimum), "maximumTemplateLengthBP": int(maxInsertSize)}
    finally:
        crBarcodeSelectionDestroy(selection)
        if iteratorHandle != NULL:
            hts_itr_destroy(iteratorHandle)
        if record != NULL:
            bam_destroy1(record)
        if indexHandle != NULL:
            hts_idx_destroy(indexHandle)
        if header != NULL:
            sam_hdr_destroy(header)
        if fileHandle != NULL:
            sam_close(fileHandle)
