#ifndef CCOUNTS_MESSAGES_H
#define CCOUNTS_MESSAGES_H

static const char ccAllocationFailed[] = "Allocation failed";
static const char ccInputOpenFailed[] = "Could not open input";
static const char ccInputReadFailed[] = "Could not read input";
static const char ccInputQueryFailed[] = "Could not query indexed input";
static const char ccHeaderReadFailed[] = "Could not read BAM header";
static const char ccThreadsFailed[] = "Could not configure IO threads";
static const char ccMissingIndex[] = "Source index required";
static const char ccIndexBuildFailed[] = "Could not build source index";
static const char ccOutputOpenFailed[] = "Could not open output";
static const char ccOutputWriteFailed[] = "Could not write output";
static const char ccCompressionFailed[] = "Could not compress bedGraph";
static const char ccInvalidSource[] = "Invalid source";
static const char ccInvalidBarcodeTag[] = "Invalid barcodeTag";
static const char ccInvalidBarcodeType[] = "barcodeTag requires SAM type Z";
static const char ccInvalidCounts[] = "Fragment support requires positive integer column five";
static const char ccInvalidRegion[] = "Invalid genomic region";
static const char ccInvalidFile[] = "Invalid input file";
static const char ccInvalidBEDGRAPH[] = "Invalid bedGraph input";
static const char ccInvalidUnits[] = "Event units unavailable";
static const char ccNoEstimate[] = "Read length estimate unavailable";
static const char ccClosedSource[] = "Source handle is not initialized";
static const char ccMissingSource[] = "Unknown source or chromosome";
static const char ccInvalidNumerics[] = "Invalid numerical result";

#endif
