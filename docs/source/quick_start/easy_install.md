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

- Clone the whole repo with git: `git clone https://github.com/JiaXingou/mialab.git
- Clone the minimum required part of repo: `git clone https://github.com/JiaXingou/mialab.git `
- Get the source code of a stable version from [here](https://github.com/JiaXingou/mialab)

### Update to latest release

Please check the [release page](https://github.com/JiaXingou/mialab) for the release note of a new version.

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

## Build and Install

## Run

Put pdb files of proteins in the folder named /example/.

Then run the following order to run:

```
cd contact
python predict.py . ./example/.
```

Then, you will get the inter-chain contact map and the top 20 residue contact pair prediction in the folder: ./example/


## Container Deployment

> Please note that containers target at developing and testing, but not massively parallel computing for production. Docker has a bad support to MPI, which may cause performance degradation.

We've built a ready-for-use version of ABACUS with docker [here](https://github.com/deepmodeling/abacus-develop/pkgs/container/abacus). For a quick start: pull the image, prepare the data, run container. Instructions on using the image can be accessed in [Dockerfile](../../Dockerfile). A mirror is available by `docker pull registry.dp.tech/deepmodeling/abacus`.

We also offer a pre-built docker image containing all the requirements for development. Please refer to our [Package Page](https://github.com/orgs/deepmodeling/packages?repo_name=abacus-develop).

The project is ready for VS Code development container. Please refer to [Developing inside a Container](https://code.visualstudio.com/docs/remote/containers#_quick-start-try-a-development-container). Choose `Open a Remote Window -> Clone a Repository in Container Volume` in VS Code command palette, and put the [git address](https://github.com/deepmodeling/abacus-develop.git) of `ABACUS` when prompted.

For online development environment, we support [GitHub Codespaces](https://github.com/codespaces): [Create a new Codespace](https://github.com/codespaces/new?machine=basicLinux32gb&repo=334825694&ref=develop&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=SouthEastAsia)

We also support [Gitpod](https://www.gitpod.io/): [Open in Gitpod](https://gitpod.io/#https://github.com/deepmodeling/abacus-develop)
