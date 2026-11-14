# Sequencing fixtures

Coordinates are zero-based and half-open.

| File | Purpose |
| --- | --- |
| `genome.sizes` | BAM chromosomes chr1 and chrTotals, each 115 bp long. |
| `events.fragments.tsv` | Four intervals on chr1, length 115 bp. Barcode A has supports 3, 2, and 1. Barcode B has support 7. |
| `events.bam`, `events.bam.bai` | Paired alignments for the same intervals, with one template per support unit and matching CB tags. |
| `barcodes.fragments.tsv` | The identifiers shared, uniqueA, and uniqueB occupy three bins. |
| `cancellation.fragments.tsv` | Overlapping intervals with supports `2**54` and `1`. |
| `precision.fragments.tsv` | Four equal-span intervals with supports `2**54`, `1`, `1`, and `1`. |

BAM chromosome chr1 has 26 retained alignments with flags 99 or 147 and MAPQ 60. A MAPQ-zero read1 and a duplicate-flagged read1 supply two excluded records. Chromosome chrTotals has length 115 bp and seven single-end alignments, with six 1 bp reads and one 20 bp read. Its header has no program-command metadata.

The `countSources` session fixture indexes TSV copies in pytest's temporary directory. The BAM and index are read directly from the fixture directory.

`signal.bedGraph` contains two intervals separated by gaps. Its values test zero-filled bin means and a short terminal bin.
