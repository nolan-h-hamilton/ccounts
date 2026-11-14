# ccounts

Genomic bin counting with explicit depth normalization and log2 transformation.
The package has no dependency on Consenrich.

## Raw counting

~~~python
import ccounts as cc

grid = cc.GridSpec.fromSizesFile("genome.sizes", binSizeBP=50)
source = cc.SourceSpec(
    "sample.bam", "BAM", "sample",
    countOptions=cc.CountOptions(countMode="center"),
)
with cc.openSources([source], grid) as reader:
    raw = reader.readRegion("chr1", 0, 100_000)
    values = raw.values
~~~

Reading returns raw source-by-bin arrays. Opening a reader does not calculate
library totals, normalize values, or evaluate a transform. Use indexed BAM,
fragments, or bedGraph files. H5AD input requires the optional h5ad dependencies.

Canonical counting modes are conservedFractionalOverlap, coverage, binOverlap,
center, fiveprime, and ffp. Fractional mass and mean interval coverage can be
noninteger. Bins use a chromosome-origin grid and actual chromosome-end widths.

| Mode | Contribution per retained interval |
| --- | --- |
| `coverage` | Overlapping bases divided by actual bin width |
| `binOverlap` | One per bin with positive overlap |

Input multiplicity scales each contribution. For `binOverlap`, optional squared
weight sums equal raw counts. Library totals count each retained interval once,
independently of the number of bins it touches. CPM and RPKM support `binOverlap`.
Its raw bin sum can exceed the retained-event total, and its CPM bin sum can
exceed one million. Automatic RPGC rejects `binOverlap`.

Extension inference is explicit. With bamInputMode="auto" and
inferFragmentLength=True, single-end reads use inferred extension and paired
templates use the recorded template span. sourceInfo records requested and
resolved inference settings. Explicit fragment-span mode rejects read-extension
inference. Coordinates and chromosome lengths use the supported signed 32-bit
genomic range.

External cell metadata selects exact cell identifiers. An H5AD source matches
external identifiers to its observation index. Repeated source identifiers
require the explicit sample column for disambiguation. Metadata stored in the
source H5AD can use its explicit observation identifier column.
Internal H5AD sample and group filters intersect explicit cell IDs and allow-list
files. An empty intersection selects zero cells. Selected identifiers must be
nonempty strings without whitespace. Numeric identifiers are rejected.

## Assembly constants

`assemblyChromosomeSizes[assembly]` supplies complete physical sequence lengths.
`standardChromosomes[assembly]` supplies an explicit ordered nuclear chromosome
preset. Each mapping is read-only, and chromosome-size mappings are read-only.

~~~python
assembly = "hg38"
sizes = cc.assemblyChromosomeSizes[assembly]
chromosomes = cc.standardChromosomes[assembly]
grid = cc.GridSpec({chrom: sizes[chrom] for chrom in chromosomes}, binSizeBP=50)
~~~

| Assemblies | Standard chromosomes |
| --- | --- |
| `hg19`, `hg38`, `hs1` | chr1–chr22, chrX |
| `mm10`, `mm39` | chr1–chr19, chrX |
| `dm3`, `dm6` | chr2L, chr2R, chr3L, chr3R, chr4, chrX |
| `sacCer2`, `sacCer3` | chrI–chrXVI |
| `ce10`, `ce11` | chrI, chrII, chrIII, chrIV, chrV, chrX |
| `danRer10`, `danRer11` | chr1–chr25 |

The size tables retain every UCSC record, with Y, mitochondrial, alternate, and
unplaced sequences where supplied. Pass the chosen mapping to `GridSpec`.
Assembly keys and chromosome names are exact. T2T-CHM13v2.0 uses the key `hs1`.
The tables are bundled and need no network access. Physical lengths do not
specify effective genome sizes. [Resource sources](src/ccounts/assemblies/README.md).

## Size-factor estimation

~~~python
factors = cc.estimateSizeFactors(raw.values, centerMedian=True, minRefDist=10)
normalization = cc.Normalization.fromFactors(raw.sampleIDs, factors)
signal = normalization.apply(raw)
~~~

The supplied matrix defines the reference territory. Reference bins require a
count of at least one in every source, with greedy minimum spacing in bin units.
The estimator uses median log ratios to geometric-mean references, size-factor
bounds of 0.2 and 5, and optional median-log centering. Returned factors are
float64 row multipliers. Reference selection preserves supplied count values,
and calculation uses float64. Insufficient reference support raises an error.
The input matrix is preserved, and estimation does not calculate library totals.

## Explicit preprocessing

~~~python
with cc.openSources([source], grid) as reader:
    totals = reader.libraryTotals(chromosomes=("chr1",))
    normalization = cc.Normalization.fromTotals(totals, method="CPM")
    raw = reader.readRegion("chr1", 0, 100_000)
    signal = normalization.apply(raw)
    cc.logTransform(signal, out=signal)
~~~

CPM, RPKM, RPGC, and supplied positive factors are supported. Library totals
use full scans over the explicitly requested chromosomes. Sampling is not part
of the library-total interface. Log2 is the sole transform. Its configurable
pseudocount defaults to one. The pseudocount retains float64 precision for
validation and calculation. Array output preserves the input dtype.

RPKM and RPGC for conservedFractionalOverlap counts divide by GridSpec.binSizeBP,
even for a short chromosome-end bin. Mean base coverage and bedGraph bin means
use actual bin width for raw counting.

Normalization derived from library totals requires matching source IDs, grids,
measurement units, and counting modes. The same per-source mode binding applies
to paired normalization. Supplied factors support any counting mode.

Raw reads can optionally return exact squared event-weight sums using
returnRawNoiseMass=True. That optional counting statistic does not select
priors, approximate variance, or calculate uncertainty floors.

Returned arrays own storage and survive reader closure. Each reader serializes
access to its native handles. Parallel workers use separate readers.

## Continuous integration

`tests.yml` runs the package tests for Python 3.10–3.14 on Linux x86_64 and
arm64, and macOS Intel and Apple Silicon. Pushes to `main`, pull requests, and
manual dispatch run the test workflow.

`wheels.yml` runs by manual dispatch and uploads tested wheels and one source
archive as workflow artifacts. It builds manylinux and musllinux wheels for
Linux x86_64 and aarch64, plus macOS x86_64 and arm64 wheels. It has no publishing
job.

Small sequencing inputs for the tests live in [tests/data](tests/data/README.md)
and ship in the source archive.

## Building distributions

Build portable wheels with `python -m cibuildwheel --output-dir wheelhouse`.
The macOS repair step bundles required external libraries and removes absolute
library search paths. The source archive excludes generated HTSlib configuration
and Finder files. HTSlib compiler and linker reporting strings are empty.

A direct macOS wheel build requires repair before distribution. For arm64:

~~~sh
python -m build --sdist --wheel
delocate-wheel --sanitize-rpaths --require-archs arm64 -w dist/repaired \
  dist/ccounts-0.1.0a0-cp312-cp312-macosx_11_0_arm64.whl
~~~

Distribute the repaired wheel from `dist/repaired`.
