from pathlib import Path
from types import MappingProxyType

from .sources import readChromSizes


standardChromosomes = MappingProxyType({
    **dict.fromkeys(("hg19", "hg38", "hs1"), tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)),
    **dict.fromkeys(("mm10", "mm39"), tuple(f"chr{i}" for i in range(1, 20)) + ("chrX",)),
    **dict.fromkeys(("dm3", "dm6"), ("chr2L", "chr2R", "chr3L", "chr3R", "chr4", "chrX")),
    **dict.fromkeys(("sacCer2", "sacCer3"), (
        "chrI", "chrII", "chrIII", "chrIV", "chrV", "chrVI", "chrVII", "chrVIII",
        "chrIX", "chrX", "chrXI", "chrXII", "chrXIII", "chrXIV", "chrXV", "chrXVI",
    )),
    **dict.fromkeys(("ce10", "ce11"), ("chrI", "chrII", "chrIII", "chrIV", "chrV", "chrX")),
    **dict.fromkeys(("danRer10", "danRer11"), tuple(f"chr{i}" for i in range(1, 26))),
})

assemblyChromosomeSizes = MappingProxyType({
    assembly: MappingProxyType(readChromSizes(
        Path(__file__).with_name("assemblies") / f"{assembly}.chrom.sizes"
    ))
    for assembly in standardChromosomes
})
