# Easy Installation

This guide helps you install PGT with basic features.  We recommend building PGT with [Docker](#container-deployment) to avoid dependency issues. We recommend compiling PGT(and possibly its requirements) from the source code using the latest compiler for the best performace. You can also deploy PGT **without building** by [Docker](#container-deployment) . Please note that PGT only supports Linux; for Windows users, please consider using [WSL](https://learn.microsoft.com/en-us/windows/wsl/) or docker.

## Prerequisites

To build PGT, please make sure that the following dependencies are present:

biopython==1.81
certifi==2023.7.22
cffi 
charset-normalizer==3.3.1
dgl==0.6.1
docopt==0.6.2
future 
h5py==3.8.0
idna==3.4
networkx==2.6.3
numpy 
psutil==5.9.6
pycparser 
requests==2.31.0
scipy==1.7.3
torch==1.10
tqdm==4.66.1
typing_extensions 
urllib3==2.0.7
yarg==0.1.9

Some of these packages can be installed with popular package management system, such as pip:

```
pip install -r requirements.txt
```

If you can install the above dependencies, you can go on . If not, you need to check if the versions of the relevant libraries are correct.
## Install requirements

Some of these packages can be installed with popular package management system, such as pip:

```
pip install -r requirements.txt
```

## Get PSAIA source code

Of course a copy of PGT source code is required, which can be obtained via one of the following choices:

- Clone the whole repo with git: `git clone https://github.com/JiaXingou/MIALAB.git
- Clone the minimum required part of repo: `git clone https://github.com/JiaXingou/MIALAB.git `
- Get the source code of a stable version from [here](https://github.com/JiaXingou/MIALAB)

### Update to latest release

Please check the [release page](https://github.com/JiaXingou/MIALAB) for the release note of a new version.

It is OK to download the new source code from beginning following the previous step.

To update your cloned git repo in-place:

```bash
git remote -v
# Check if the output contains the line below
# origin https://github.com/deepmodeling/abacus-develop.git (fetch)
# The remote name is marked as "upstream" if you clone the repo from your own fork.

# Replace "origin" with "upstream" or the remote name corresponding to deepmodeling/abacus-develop if necessary
git fetch origin
git checkout v3.2.0 # Replace the tag with the latest version
git describe --tags # Verify if the tag has been successfully checked out
```

Then proceed to the [Build and Install](#build-and-install) part. If you encountered errors, try remove the `build` directory first and reconfigure.

To use the codes under active development:

```bash
git checkout develop
git pull
```

## Configure

The basic command synopsis is:

```bash
cd abacus-develop
cmake -B build [-D <var>=<value>] ...
```

Here, 'build' is the path for building ABACUS; and '-D' is used for setting up some variables for CMake indicating optional components or requirement positions.

- `CMAKE_INSTALL_PREFIX`: the path of ABACUS binary to install; `/usr/local/bin/abacus` by default
- Compilers
  - `CMAKE_CXX_COMPILER`: C++ compiler; usually `g++`(GNU C++ compiler) or `icpx`(Intel C++ compiler). Can also set from environment variable `CXX`. It is OK to use MPI compiler here.
  - `MPI_CXX_COMPILER`: MPI wrapper for C++ compiler; usually `mpicxx` or `mpiicpc`(for Intel MPI).
- Requirements: Unless indicated, CMake will try to find under default paths.
  - `MKLROOT`: If environment variable `MKLROOT` exists, `cmake` will take MKL as a preference, i.e. not using `LAPACK`, `ScaLAPACK` and `FFTW`. To disable MKL, unset environment variable `MKLROOT`, or pass `-DMKLROOT=OFF` to `cmake`.
  - `LAPACK_DIR`: Path to OpenBLAS library `libopenblas.so`(including BLAS and LAPACK)
  - `SCALAPACK_DIR`: Path to ScaLAPACK library `libscalapack.so`
  - `ELPA_DIR`: Path to ELPA install directory; should be the folder containing 'include' and 'lib'.
  > Note: If you install ELPA from source, please add a symlink to avoid the additional include file folder with version name: `ln -s elpa/include/elpa-2021.05.002/elpa elpa/include/elpa`. This is a known behavior of ELPA.

  - `FFTW3_DIR`: Path to FFTW3.
  - `CEREAL_INCLUDE_DIR`: Path to the parent folder of `cereal/cereal.hpp`. Will download from GitHub if absent.
  - `Libxc_DIR`: (Optional) Path to Libxc.
  > Note: Building Libxc from source with Makefile does NOT support using it in CMake here. Please compile Libxc with CMake instead.
  - `LIBRI_DIR`: (Optional) Path to LibRI.
  - `LIBCOMM_DIR`: (Optional) Path to LibComm.

- Components: The values of these variables should be 'ON', '1' or 'OFF', '0'. The default values are given below.
  - `ENABLE_LCAO=ON`: Enable LCAO calculation. If SCALAPACK, ELPA or CEREAL is absent and only require plane-wave calculations, the feature of calculating LCAO basis can be turned off.
  - `ENABLE_LIBXC=OFF`: [Enable Libxc](../advanced/install.md#add-libxc-support) to suppport variety of functionals. If `Libxc_DIR` is defined, `ENABLE_LIBXC` will set to 'ON'.
  - `ENABLE_LIBRI=OFF`: [Enable LibRI](../advanced/install.md#add-libri-support) to suppport variety of functionals. If `LIBRI_DIR` and `LIBCOMM_DIR` is defined, `ENABLE_LIBRI` will set to 'ON'.
  - `USE_OPENMP=ON`: Enable OpenMP support. Building ABACUS without OpenMP is not fully tested yet.
  - `BUILD_TESTING=OFF`: [Build unit tests](../advanced/install.md#build-unit-tests).
  - `ENABLE_MPI=ON`: Enable MPI parallel compilation. If set to `OFF`, a serial version of ABACUS with PW basis only will be compiled. Currently serial version of ABACUS with LCAO basis is not supported yet, so `ENABLE_LCAO` will be automatically set to `OFF`.
  - `ENABLE_COVERAGE=OFF`: Build ABACUS executable supporting [coverage analysis](../CONTRIBUTING.md#generating-code-coverage-report). This feature has a drastic impact on performance.
  - `ENABLE_ASAN=OFF`: Build with Address Sanitizer. This feature would help detecting memory problems. Only supports GCC.
  - `USE_ELPA=ON`: Use ELPA library in LCAO calculations. If this value is set to OFF, ABACUS can be compiled without ELPA library.

Here is an example:

```bash
CXX=mpiicpc cmake -B build -DCMAKE_INSTALL_PREFIX=~/abacus -DELPA_DIR=~/elpa-2016.05.004/build -DCEREAL_INCLUDE_DIR=~/cereal/include
```

## Build and Install

After configuring, build and install by:

```bash
cmake --build build -j`nproc`
cmake --install build
```

You can change the number after `-j` on your need: set to the number of CPU cores(`nproc`) to reduce compilation time.

## Run

Put pdb files of proteins in the folder named /example/.

Then run the following order to run:

```
cd ..
python predict.py T0805 ./example/.
```

You can change the number of top patches in  sort_patch.sh

Then, you will get the inter-chain contact map and the top 20 residue contact pair prediction in the folder: /example/

> Note: Some Intel CPU has a feature named Hyper-Threading(HT). This feature enables one physical core switch fastly between two logical threads. It would benefits from I/O bound tasks: when a thread is blocked by I/O, the CPU core can work on another thread. However, it helps little on CPU bound tasks, like PGT and many other scientific computing softwares. We recommend using the physical CPU core number.
> To determine if HT is turned on, execute `lscpu | grep 'per core'` and see if 'Thread(s) per core' is 2.

## Container Deployment

> Please note that containers target at developing and testing, but not massively parallel computing for production. Docker has a bad support to MPI, which may cause performance degradation.

We've built a ready-for-use version of ABACUS with docker [here](https://github.com/deepmodeling/abacus-develop/pkgs/container/abacus). For a quick start: pull the image, prepare the data, run container. Instructions on using the image can be accessed in [Dockerfile](../../Dockerfile). A mirror is available by `docker pull registry.dp.tech/deepmodeling/abacus`.

We also offer a pre-built docker image containing all the requirements for development. Please refer to our [Package Page](https://github.com/orgs/deepmodeling/packages?repo_name=abacus-develop).

The project is ready for VS Code development container. Please refer to [Developing inside a Container](https://code.visualstudio.com/docs/remote/containers#_quick-start-try-a-development-container). Choose `Open a Remote Window -> Clone a Repository in Container Volume` in VS Code command palette, and put the [git address](https://github.com/deepmodeling/abacus-develop.git) of `ABACUS` when prompted.

For online development environment, we support [GitHub Codespaces](https://github.com/codespaces): [Create a new Codespace](https://github.com/codespaces/new?machine=basicLinux32gb&repo=334825694&ref=develop&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=SouthEastAsia)

We also support [Gitpod](https://www.gitpod.io/): [Open in Gitpod](https://gitpod.io/#https://github.com/deepmodeling/abacus-develop)

## Install by conda

Conda is a package management system with a separated environment, not requiring system privileges. A pre-built ABACUS binary with all requirements is available at [conda-forge](https://anaconda.org/conda-forge/abacus). Conda will install the GPU-accelerated version of ABACUS if a valid GPU driver is present.

```bash
# Install
# We recommend installing ABACUS in a new environment to avoid potential conflicts:
conda create -n abacus_env abacus -c conda-forge

# Run
conda activate abacus_env
OMP_NUM_THREADS=1 mpirun -n 4 abacus

# Update
conda update -n abacus_env abacus -c conda-forge
```

For more details on building a conda package of ABACUS, please refer to the [conda recipe file](https://github.com/deepmodeling/abacus-develop/blob/develop/conda/meta.yaml).

> Note: The [deepmodeling conda channel](https://anaconda.org/deepmodeling/abacus) offers historical versions of ABACUS.
