# Introduction

This is a program to predict inter-chain contact map for homologous protein complex , named PGT (P is Protein, G is Graph attention network and T is Triangular multiplication update).

# Data

This program needs pdb-format files of homologous protein complex proteins, you can download them from [RCSB PDB: Homepage](https://www.rcsb.org/).

# Prerequisites

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

# Run

Put pdb files of proteins in the folder named /example/.

Then run the following order to run:

```
cd ..
python predict.py T0805 ./example/.
```

You can change the number of top patches in  sort_patch.sh

Then, you will get the inter-chain contact map and the top 20 residue contact pair prediction in the folder: /example/

# Contact

If you have any questions, please contact with: 13168@ruc.edu.cn