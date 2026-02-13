from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import sys, os, os.path
import numpy as np
import glob
import subprocess
import pybind11

if sys.platform in ("win32", "win64"):
    print("Windows is not a supported platform.")
    sys.exit(1)

# Check for CUDA support
def find_cuda():
    """Find CUDA installation and return (cuda_home, cuda_available)"""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    # Try common locations
    if not cuda_home:
        for path in ["/usr/local/cuda", "/opt/cuda",
                     os.path.expandvars("$SROOT/cuda"),
                     "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/pweigel/MEOWS_ML/cuda_12.2"]:
            if os.path.exists(os.path.join(path, "include", "cuda_runtime.h")):
                cuda_home = path
                break

    if cuda_home and os.path.exists(os.path.join(cuda_home, "include", "cuda_runtime.h")):
        return cuda_home, True
    return None, False

cuda_home, use_cuda = find_cuda()
if use_cuda:
    print(f"CUDA found at: {cuda_home}")
else:
    print("CUDA not found - building without GPU acceleration")

gollum_build_path = os.environ.get("GOLLUMBUILDPATH", "/usr/local/")
cvmfs_root = os.environ.get("SROOT", "/usr/local/")
prefix = os.environ.get("PREFIX", "/usr/local/")

include_dirs = [
    gollum_build_path + "/include",
    cvmfs_root + "/include",
    "/usr/local/include",
    prefix + "/include",
    np.get_include(),
    "../include/",
    "../src/",
    pybind11.get_include(),
]

libraries = [
    "boost_filesystem", "boost_iostreams", "boost_system", "boost_regex",
    "LeptonWeighter", "photospline",
    "SQuIDS", "nuSQuIDS",
    "gsl", "gslcblas", "m", "z",
    "hdf5", "hdf5_hl", "PhysTools", "cfitsio", "GollumFit",
]

architecture = os.uname().machine
library_dirs = [
    gollum_build_path,  # Library may be directly in build directory
    gollum_build_path + f"/lib/python{sys.version_info[0]}.{sys.version_info[1]}/site-packages",
    gollum_build_path + "/lib",
    gollum_build_path + "/lib64",
    cvmfs_root + "/lib",
    cvmfs_root + "/lib64",
    "/usr/local/lib",
    "/usr/local/lib64",
    prefix + "/lib",
    prefix + "/lib64",
    f"/usr/lib/{architecture}-linux-gnu/hdf5/serial",
]

def pkgconfig_flags(package, flag):
    try:
        out = subprocess.check_output(["pkg-config", flag, package], encoding="utf-8")
        return out.strip().split()
    except Exception:
        return []

for pkg in ("gsl", "hdf5", "cfitsio"):
    include_dirs += [f[2:] for f in pkgconfig_flags(pkg, "--cflags") if f.startswith("-I")]
    library_dirs += [f[2:] for f in pkgconfig_flags(pkg, "--libs-only-L") if f.startswith("-L")]
    libraries += [f[2:] for f in pkgconfig_flags(pkg, "--libs-only-l") if f.startswith("-l")]

gollum_space_path = os.environ.get("GOLLUMSPACE", "..")
extra_objs = glob.glob(gollum_space_path + "/lib/*.o")

extra_link_args = []
if sys.platform == "darwin":
    extra_link_args += ["-Wl,-rpath,@loader_path"]
    for d in library_dirs:
        extra_link_args += [f"-Wl,-rpath,{d}"]
else:
    # On Linux, set rpath to find the correct library - prioritize build directory
    extra_link_args += [f"-Wl,-rpath,{gollum_build_path}"]
    for d in library_dirs[:5]:  # Add first few important paths
        extra_link_args += [f"-Wl,-rpath,{d}"]

# Base compile arguments
extra_compile_args = ["-v", "-O3", "-fPIC", "-std=c++17", "-fpermissive"]

# Add CUDA support if available
if use_cuda:
    print("Enabling CUDA support in Python bindings")
    extra_compile_args.append("-DGOLLUMFIT_USE_CUDA")
    include_dirs.append(os.path.join(cuda_home, "include"))
    library_dirs.append(os.path.join(cuda_home, "lib64"))
    library_dirs.append(os.path.join(cuda_home, "lib"))
    libraries.append("cudart")

ext = Pybind11Extension(
    "GollumFitPy",
    ["GollumFitPy.cpp"],
    library_dirs=library_dirs,
    libraries=libraries,
    include_dirs=include_dirs,
    extra_objects=extra_objs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(ext_modules=[ext])