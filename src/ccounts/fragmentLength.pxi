import os
from numpy.random import default_rng
from libc.stdint cimport int32_t, uint8_t, uint32_t

cdef extern from "htslib/hts.h":
    ctypedef struct htsFile
    ctypedef struct hts_idx_t
    ctypedef struct hts_itr_t
    ctypedef long long hts_pos_t

    int hts_set_threads(htsFile* fp, int n)
    void hts_idx_destroy(hts_idx_t* idx)
    void hts_itr_destroy(hts_itr_t* itr)


cdef extern from "htslib/sam.h":
    ctypedef struct samFile

    ctypedef struct bam1_core_t:
        int32_t tid
        int32_t pos
        uint16_t bin
        uint8_t qual
        uint16_t l_qname
        uint16_t n_cigar
        uint16_t flag
        int32_t l_qseq
        int32_t mtid
        int32_t mpos
        int64_t isize

    ctypedef struct bam1_t:
        bam1_core_t core

    ctypedef struct sam_hdr_t:
        int32_t n_targets
        uint32_t* target_len
        char** target_name

    samFile* sam_open(const char* fn, const char* mode)
    int sam_close(samFile* fp)
    sam_hdr_t* sam_hdr_read(samFile* fp)
    void sam_hdr_destroy(sam_hdr_t* h)
    bam1_t* bam_init1()
    void bam_destroy1(bam1_t* b)
    int sam_read1(samFile* fp, sam_hdr_t* h, bam1_t* b)
    hts_idx_t* sam_index_load(htsFile* fp, const char* fn)
    int sam_hdr_name2tid(sam_hdr_t* h, const char* ref)
    hts_itr_t* sam_itr_queryi(hts_idx_t* idx, int tid, hts_pos_t beg, hts_pos_t end)
    int sam_itr_next(htsFile* htsfp, hts_itr_t* itr, bam1_t* r)
    hts_pos_t bam_endpos(bam1_t* b)
    uint32_t* bam_get_cigar(bam1_t* b)
    hts_pos_t bam_cigar2qlen(int n_cigar, uint32_t* cigar)


cdef extern from "native/barcode_selection.h":
    ctypedef struct crBarcodeSelection:
        int active
    crBarcodeSelection* crBarcodeSelectionCreate(const char* tag, const char* path, const char** error)
    void crBarcodeSelectionDestroy(crBarcodeSelection* selection)
    int crBarcodeSelected(const crBarcodeSelection* selection, const bam1_t* record)


cdef object _medianFilterNearest(object values, int width):
    cdef Py_ssize_t start, stop
    cdef int radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    result = np.empty_like(values)
    for start in range(0, len(values), 8192):
        stop = min(start + 8192, len(values))
        result[start:stop] = np.partition(windows[start:stop], radius, axis=1)[:, radius]
    return result


cdef int64_t _fragmentLength(
    str bamFile,
    uint16_t samThreads=0,
    uint16_t samFlagExclude=3844,
    int64_t maxInsertSize=1000,
    int64_t iters=1000,
    int64_t blockSize=5000,
    int64_t fallBack=147,
    int64_t rollingChunkSize=250,
    int64_t lagStep=5,
    int64_t earlyExit=250,
    int64_t randSeed=42,
    str barcodeAllowListFile="",
    str barcodeTag="CB",
):
    cdef object rng = default_rng(randSeed)
    cdef int64_t regionLen, numRollSteps
    cdef int numChunks
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rawArr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] medArr
    cdef list blockCenters
    cdef list bestLags
    cdef int i, j, idxVal
    cdef int startIdx, endIdx
    cdef int winSize, takeK
    cdef int blockHalf, readFlag
    cdef int maxValidLag
    cdef int strand
    cdef int samThreadsInternal
    cdef object cpuCountObj
    cdef int cpuCount
    cdef int64_t blockStartBP, blockEndBP, readStart, readEnd
    cdef int64_t med
    cdef double score
    cdef cnp.ndarray[cnp.intp_t, ndim=1] topContigsIdx
    cdef cnp.ndarray[cnp.intp_t, ndim=1] unsortedIdx, sortedIdx
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unsortedVals
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] seen
    cdef bint isPairedEnd = <bint>0
    cdef double avgReadLength = <double>0.0
    cdef int64_t numReadLengthSamples = <int64_t>0
    cdef int64_t minInsertSize
    cdef int64_t requiredSamplesPE
    cdef int64_t tlen
    cdef cnp.ndarray[cnp.int64_t, ndim=1] lengthsArr
    cdef Py_ssize_t contigIdx
    cdef int contigTid
    cdef int64_t contigLen
    cdef int kTop
    cdef cnp.ndarray[cnp.float64_t, ndim=1] fwd
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rev
    cdef double[::1] fwdView
    cdef double[::1] revView
    cdef double fwdSum
    cdef double revSum
    cdef double fwdMean
    cdef double revMean
    cdef double bestScore
    cdef int bestLag
    cdef int blockLen
    cdef int localMinLag
    cdef int localMaxLag
    cdef int localLagStep
    cdef int nFFT
    cdef object Ff
    cdef object Fr
    cdef object corr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tlenArr
    cdef cnp.ndarray[cnp.int32_t, ndim=1] tlenWork
    cdef cnp.int32_t[::1] tlenWorkView
    cdef int tlenN
    cdef int midIdx
    cdef cnp.int32_t medPE
    cdef cnp.ndarray[cnp.uint32_t, ndim=1] bestLagsArr
    cdef bytes bamFileBytes = bamFile.encode("utf-8")
    cdef samFile* fileHandle = NULL
    cdef sam_hdr_t* header = NULL
    cdef hts_idx_t* indexHandle = NULL
    cdef hts_itr_t* iteratorHandle = NULL
    cdef bam1_t* record = NULL
    cdef hts_pos_t queryLength = 0
    cdef bytes allowListBytes = barcodeAllowListFile.encode("utf-8")
    cdef bytes tagBytes = barcodeTag.encode("utf-8")
    cdef const char* selectionError = NULL
    cdef crBarcodeSelection* selection = NULL
    cdef int selected
    cdef bint hasSelection = bool(barcodeAllowListFile)

    earlyExit = min(earlyExit, iters)
    samThreadsInternal = <int>samThreads
    cpuCountObj = os.cpu_count()
    if cpuCountObj is None:
        cpuCount = 1
    else:
        cpuCount = <int>cpuCountObj
        if cpuCount < 1:
            cpuCount = 1

    if samThreads < 1:
        samThreadsInternal = <int>min(max(1, cpuCount // 2), 4)

    if maxInsertSize < 1:
        maxInsertSize = 1
    if iters < 1:
        if hasSelection:
            raise ValueError("could not estimate fragment length from selected BAM records")
        return <int64_t>fallBack
    if blockSize < 64:
        blockSize = 64
    if rollingChunkSize < 1:
        rollingChunkSize = 1
    if lagStep < 1:
        lagStep = 1

    fileHandle = sam_open(bamFileBytes, "r")
    if fileHandle == NULL:
        if hasSelection:
            raise ValueError("could not estimate fragment length from selected BAM records")
        return <int64_t>fallBack

    try:
        selection = crBarcodeSelectionCreate(tagBytes, allowListBytes, &selectionError)
        if selection == NULL:
            raise ValueError((<bytes>selectionError).decode("utf-8"))
        if samThreadsInternal > 1:
            hts_set_threads(<htsFile*>fileHandle, samThreadsInternal)

        header = sam_hdr_read(fileHandle)
        if header == NULL or header.n_targets <= 0:
            if hasSelection:
                raise ValueError("could not estimate fragment length from selected BAM records")
            return <int64_t>fallBack

        indexHandle = sam_index_load(<htsFile*>fileHandle, bamFileBytes)
        if indexHandle == NULL:
            if hasSelection:
                raise ValueError("could not estimate fragment length from selected BAM records")
            return <int64_t>fallBack

        record = bam_init1()
        if record == NULL:
            if hasSelection:
                raise ValueError("could not estimate fragment length from selected BAM records")
            return <int64_t>fallBack

        lengthsArr = np.empty(header.n_targets, dtype=np.int64)
        for contigIdx in range(header.n_targets):
            lengthsArr[contigIdx] = <int64_t>header.target_len[contigIdx]

        kTop = header.n_targets if hasSelection else (3 if header.n_targets >= 3 else (2 if header.n_targets >= 2 else 1))
        topContigsIdx = np.argpartition(lengthsArr, -kTop)[-kTop:]
        topContigsIdx = topContigsIdx[np.argsort(lengthsArr[topContigsIdx])[::-1]]

        for contigIdx in topContigsIdx:
            contigTid = <int>contigIdx
            contigLen = <int64_t>lengthsArr[contigTid]
            if contigLen <= 0:
                continue

            iteratorHandle = sam_itr_queryi(indexHandle, contigTid, 0, <hts_pos_t>contigLen)
            if iteratorHandle == NULL:
                continue

            while sam_itr_next(<htsFile*>fileHandle, iteratorHandle, record) >= 0:
                readFlag = <int>record.core.flag
                if (readFlag & samFlagExclude) != 0:
                    continue
                if hasSelection and (readFlag & 1) != 0 and (readFlag & 64) == 0:
                    continue
                selected = crBarcodeSelected(selection, record)
                if selected < 0:
                    raise ValueError("barcodeTag requires SAM type Z")
                if not selected:
                    continue

                if not isPairedEnd and (readFlag & 1) != 0:
                    isPairedEnd = <bint>1

                if numReadLengthSamples >= iters:
                    break

                queryLength = record.core.l_qseq
                if queryLength <= 0 and record.core.n_cigar > 0:
                    queryLength = bam_cigar2qlen(record.core.n_cigar, bam_get_cigar(record))
                if queryLength <= 0:
                    continue

                avgReadLength += <double>queryLength
                numReadLengthSamples += 1

            hts_itr_destroy(iteratorHandle)
            iteratorHandle = NULL

            if numReadLengthSamples >= iters:
                break

        if numReadLengthSamples <= 0:
            if hasSelection:
                raise ValueError("could not estimate fragment length from selected BAM records")
            return <int64_t>fallBack

        avgReadLength /= <double>numReadLengthSamples
        minInsertSize = <int64_t>(avgReadLength)
        if minInsertSize < 1:
            minInsertSize = 1
        if minInsertSize > maxInsertSize:
            minInsertSize = maxInsertSize

        if isPairedEnd:
            requiredSamplesPE = max(iters, 2000)
            tlenArr = np.empty(requiredSamplesPE, dtype=np.int32)
            tlenN = 0

            for contigIdx in topContigsIdx:
                if tlenN >= requiredSamplesPE:
                    break

                contigTid = <int>contigIdx
                contigLen = <int64_t>lengthsArr[contigTid]
                if contigLen <= 0:
                    continue

                iteratorHandle = sam_itr_queryi(indexHandle, contigTid, 0, <hts_pos_t>contigLen)
                if iteratorHandle == NULL:
                    continue

                while sam_itr_next(<htsFile*>fileHandle, iteratorHandle, record) >= 0:
                    if tlenN >= requiredSamplesPE:
                        break

                    readFlag = <int>record.core.flag
                    if (readFlag & samFlagExclude) != 0:
                        continue
                    if (readFlag & 64) == 0:
                        continue
                    selected = crBarcodeSelected(selection, record)
                    if selected < 0:
                        raise ValueError("barcodeTag requires SAM type Z")
                    if not selected:
                        continue
                    if (readFlag & 2) == 0:
                        continue

                    tlen = <int64_t>record.core.isize
                    if tlen == 0:
                        continue
                    if tlen < 0:
                        tlen = -tlen

                    if tlen < minInsertSize or tlen > maxInsertSize:
                        continue

                    tlenArr[tlenN] = <cnp.int32_t>tlen
                    tlenN += 1

                hts_itr_destroy(iteratorHandle)
                iteratorHandle = NULL

            if tlenN < (1 if hasSelection else max(500, requiredSamplesPE // 5)):
                if hasSelection:
                    raise ValueError("could not estimate fragment length from selected BAM records")
                return <int64_t>fallBack

            midIdx = tlenN // 2
            tlenWork = tlenArr[:tlenN].copy()
            tlenWork = np.partition(tlenWork, midIdx)
            tlenWorkView = tlenWork
            medPE = <cnp.int32_t>tlenWorkView[midIdx]

            if medPE < <cnp.int32_t>minInsertSize:
                return <int64_t>minInsertSize
            if medPE > <cnp.int32_t>maxInsertSize:
                if hasSelection:
                    raise ValueError("could not estimate fragment length from selected BAM records")
                return <int64_t>fallBack
            return <int64_t>medPE

        bestLags = []
        blockHalf = blockSize // 2

        fwd = np.zeros(blockSize, dtype=np.float64, order="C")
        rev = np.zeros(blockSize, dtype=np.float64, order="C")
        fwdView = fwd
        revView = rev

        nFFT = 1
        while nFFT < (2 * blockSize):
            nFFT <<= 1

        for contigIdx in topContigsIdx:
            contigTid = <int>contigIdx
            contigLen = <int64_t>lengthsArr[contigTid]
            regionLen = contigLen

            if regionLen < blockSize or regionLen <= 0:
                continue

            numRollSteps = regionLen // rollingChunkSize
            if numRollSteps <= 0:
                numRollSteps = 1
            numChunks = <int>numRollSteps

            rawArr = np.zeros(numChunks, dtype=np.float64)
            medArr = np.zeros(numChunks, dtype=np.float64)

            iteratorHandle = sam_itr_queryi(indexHandle, contigTid, 0, <hts_pos_t>contigLen)
            if iteratorHandle == NULL:
                continue

            while sam_itr_next(<htsFile*>fileHandle, iteratorHandle, record) >= 0:
                readFlag = <int>record.core.flag
                if (readFlag & samFlagExclude) != 0:
                    continue
                selected = crBarcodeSelected(selection, record)
                if selected < 0:
                    raise ValueError("barcodeTag requires SAM type Z")
                if not selected:
                    continue
                j = <int>(record.core.pos // rollingChunkSize)
                if 0 <= j < numChunks:
                    rawArr[j] += 1.0

            hts_itr_destroy(iteratorHandle)
            iteratorHandle = NULL

            winSize = <int>(blockSize // rollingChunkSize)
            if winSize < 1:
                winSize = 1
            if (winSize & 1) == 0:
                winSize += 1
            medArr[:] = _medianFilterNearest(rawArr, winSize)

            takeK = iters if iters < numChunks else numChunks
            unsortedIdx = np.argpartition(medArr, -takeK)[-takeK:]
            unsortedVals = medArr[unsortedIdx]
            sortedIdx = unsortedIdx[np.argsort(unsortedVals)[::-1]]

            seen = np.zeros(numChunks, dtype=np.uint8)
            blockCenters = []
            for i in range(takeK):
                idxVal = <int>sortedIdx[i]
                startIdx = idxVal - (winSize // 2)
                endIdx = startIdx + winSize
                if startIdx < 0:
                    startIdx = 0
                    endIdx = winSize if winSize < numChunks else numChunks
                if endIdx > numChunks:
                    endIdx = numChunks
                    startIdx = endIdx - winSize if winSize <= numChunks else 0
                for j in range(startIdx, endIdx):
                    if seen[j] == 0:
                        seen[j] = 1
                        blockCenters.append(j)

            if len(blockCenters) > 1:
                rng.shuffle(blockCenters)

            for idxVal in blockCenters:
                blockStartBP = idxVal*rollingChunkSize + (rollingChunkSize // 2) - blockHalf
                if blockStartBP < 0:
                    blockStartBP = 0
                blockEndBP = blockStartBP + blockSize
                if blockEndBP > contigLen:
                    blockEndBP = contigLen
                    blockStartBP = blockEndBP - blockSize
                    if blockStartBP < 0:
                        continue

                fwd.fill(0.0)
                rev.fill(0.0)

                iteratorHandle = sam_itr_queryi(
                    indexHandle,
                    contigTid,
                    <hts_pos_t>blockStartBP,
                    <hts_pos_t>blockEndBP,
                )
                if iteratorHandle == NULL:
                    continue

                while sam_itr_next(<htsFile*>fileHandle, iteratorHandle, record) >= 0:
                    readFlag = <int>record.core.flag
                    if (readFlag & samFlagExclude) != 0:
                        continue
                    selected = crBarcodeSelected(selection, record)
                    if selected < 0:
                        raise ValueError("barcodeTag requires SAM type Z")
                    if not selected:
                        continue

                    readStart = <int64_t>record.core.pos
                    readEnd = <int64_t>bam_endpos(record)
                    if readStart < blockStartBP or readEnd > blockEndBP:
                        continue
                    if readEnd <= readStart:
                        continue

                    strand = readFlag & 16
                    if strand == 0:
                        i = <int>(readStart - blockStartBP)
                        if 0 <= i < blockSize:
                            fwdView[i] += 1.0
                    else:
                        i = <int>((readEnd - 1) - blockStartBP)
                        if 0 <= i < blockSize:
                            revView[i] += 1.0

                hts_itr_destroy(iteratorHandle)
                iteratorHandle = NULL

                maxValidLag = maxInsertSize if (maxInsertSize < blockSize) else (blockSize - 1)
                localMinLag = <int>minInsertSize
                localMaxLag = <int>maxValidLag
                if localMaxLag < localMinLag:
                    continue
                localLagStep = <int>lagStep

                fwdSum = 0.0
                revSum = 0.0
                for i in range(blockSize):
                    fwdSum += fwdView[i]
                    revSum += revView[i]

                if fwdSum < 10.0 or revSum < 10.0:
                    continue

                fwdMean = fwdSum / (<double>blockSize)
                revMean = revSum / (<double>blockSize)

                for i in range(blockSize):
                    fwdView[i] = fwdView[i] - fwdMean
                    revView[i] = revView[i] - revMean

                Ff = np.fft.rfft(fwd, nFFT)
                Fr = np.fft.rfft(rev, nFFT)
                corr = np.fft.irfft(np.conj(Ff) * Fr, nFFT)

                bestScore = -1e308
                bestLag = -1

                for lag in range(localMinLag, localMaxLag + 1, localLagStep):
                    blockLen = blockSize - lag
                    if blockLen <= 0:
                        continue

                    score = <double>corr[lag]
                    if score > bestScore:
                        bestScore = score
                        bestLag = lag

                if bestLag > 0 and bestScore != 0.0:
                    bestLags.append(bestLag)
                if len(bestLags) >= earlyExit:
                    break

            if len(bestLags) >= earlyExit:
                break

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

    if len(bestLags) < 3:
        if hasSelection:
            raise ValueError("could not estimate fragment length from selected BAM records")
        return fallBack

    bestLagsArr = np.asarray(bestLags, dtype=np.uint32)
    med = <int64_t>(np.median(bestLagsArr) + 1.0 + 0.5)

    if med < minInsertSize:
        med = <int>minInsertSize
    elif med > maxInsertSize:
        med = <int>maxInsertSize

    return <int64_t>med
