# Easy Installation

This guide helps you install PGT with basic features.  We recommend building PGT with [Docker](#container-deployment) to avoid dependency issues. We recommend compiling PGT(and possibly its requirements) from the source code using the latest compiler for the best performace. You can also deploy PGT **without building** by [Docker](#container-deployment) . Please note that PGT only supports Linux; for Windows users, please consider using [WSL](https://learn.microsoft.com/en-us/windows/wsl/) or docker.

## Prerequisites

To build PGT, please make sure that the following dependencies are present:

```
biopython==1.81
certifi==2023.7.22
cffi 
charset-normalizer==3.3.1
dgl
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
torch
tqdm==4.66.1
typing_extensions 
urllib3==2.0.7
yarg==0.1.9

```
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

## Get PGT source code

Of course a copy of PGT source code is required, which can be obtained via one of the following choices:

- Clone the whole repo with git: `git clone https://github.com/RUC-MIALAB/PGT.git
- Clone the minimum required part of repo: `git clone https://github.com/RUC-MIALAB/PGT.git --depth=1 `
- Get the source code of a stable version from [here](https://github.com/RUC-MIALAB/PGT)

### Update to latest release

Please check the [release page](https://github.com/RUC-MIALAB/PGT/releases) for the release note of a new version.

It is OK to download the new source code from beginning following the previous step.

To update your cloned git repo in-place:

```bash
git remote -v
# Check if the output contains the line below
# origin https://github.com/RUC-MIALAB/PGT.git (fetch)
# The remote name is marked as "main" if you clone the repo from your own fork.

git fetch origin
git checkout v0.0.1 # Replace the tag with the latest version
git describe --tags # Verify if the tag has been successfully checked out
```



## Container Deployment

We've built a ready-for-use version of PGT with docker. For a quick start: pull the image, prepare the data, run container. Instructions on using the image can be accessed in [Dockerfile](../../../Dockerfile). 
