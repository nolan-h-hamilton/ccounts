from pathlib import Path
import os
import shutil
import subprocess
import sys
import sysconfig

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

rootDir = Path(__file__).resolve().parent
vendorDir = rootDir / "vendor" / "htslib"


class BuildNative(build_ext):
    def build_extensions(self):
        buildDir = Path(self.build_temp).resolve() / "vendor" / "htslib"
        if buildDir.exists():
            shutil.rmtree(buildDir)
        buildDir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vendorDir, buildDir, ignore=shutil.ignore_patterns("*.o", "*.a", ".git", ".DS_Store", "config.h", "config.mk", "config_vars.h", "htscodecs.mk"))
        cflags = "-O3 -fvisibility=hidden -fPIC"
        if sys.platform == "darwin":
            target = os.environ.get("MACOSX_DEPLOYMENT_TARGET") or sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
            if target:
                cflags += f" -mmacosx-version-min={target}"
        (buildDir / "config.mk").write_text(
            f"CC = {os.environ.get('CC', 'cc')}\n"
            f"RANLIB = {os.environ.get('RANLIB', 'ranlib')}\n"
            f"CPPFLAGS = -I{sys.prefix}/include\n"
            f"CFLAGS = {cflags}\n"
            "LDFLAGS =\nVERSION_SCRIPT_LDFLAGS =\nLIBS = -lz -lm\n"
            "NONCONFIGURE_OBJS =\nplugin_OBJS =\nnoplugin_LDFLAGS =\nnoplugin_LIBS =\n"
            "REF_CACHE_PROGRAMS =\nHTS_CFLAGS_AVX2 =\nHTS_CFLAGS_AVX512 =\nHTS_CFLAGS_SSE4 =\n"
        )
        (buildDir / "config.h").write_text(
            "#ifndef _XOPEN_SOURCE\n#define _XOPEN_SOURCE 600\n#endif\n"
            "#define HAVE_DRAND48 1\n"
            "#if defined __x86_64__\n#define HAVE_X86INTRIN_H 1\n#endif\n"
            "#if defined __x86_64__ || defined __arm__ || defined __aarch64__\n#define HAVE_ATTRIBUTE_CONSTRUCTOR 1\n#endif\n"
            "#if defined __linux__\n#define HAVE_GETAUXVAL 1\n#elif defined __FreeBSD__\n#define HAVE_ELF_AUX_INFO 1\n#elif defined __OpenBSD__\n#define HAVE_OPENBSD 1\n#endif\n"
        )
        (buildDir / "config_vars.h").write_text(
            '#define HTS_CC ""\n'
            '#define HTS_CPPFLAGS ""\n'
            '#define HTS_CFLAGS ""\n'
            '#define HTS_LDFLAGS ""\n'
            '#define HTS_LIBS ""\n'
        )
        subprocess.run(["make", "-C", str(buildDir), "-j2", "lib-static"], check=True)
        for extension in self.extensions:
            extension.include_dirs.extend([str(buildDir), str(buildDir / "htslib")])
            extension.extra_objects.append(str(buildDir / "libhts.a"))
        super().build_extensions()


extension = Extension(
    "ccounts._native",
    sources=["src/ccounts/_native.pyx", "src/ccounts/native/countsBackend.c"],
    include_dirs=[numpy.get_include(), "src/ccounts"],
    libraries=["z"] + (["m", "pthread"] if sys.platform.startswith("linux") else []),
    extra_objects=[],
    extra_compile_args=["-O3", "-fno-trapping-math", "-fno-math-errno"],
)
setup(ext_modules=cythonize([extension], build_dir="build/cython", language_level="3"), cmdclass={"build_ext": BuildNative})
