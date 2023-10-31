from Bio.PDB import *
from Bio.Cluster import distancematrix
import esm
import os
import numpy as np
import h5py
import torch
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
# 
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# cuda=torch.cuda.is_available()


def feature(pdb_path,pdb_name):


    parser = PDBParser()
    structure = parser.get_structure(pdb_name,pdb_path)

    # seq
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        seq = pp.get_sequence()

    #seq_feature
    repre,attention=seq_feature(seq, pdb_name)
    one_hot_2d= get_fasta_2d(seq)

    # index，记录每个残基的原子个数
    residues = structure.get_residues()
    residues_length = len(seq)
    index = [0]
    for i, residue_i in enumerate(residues):
        index.append(index[i] + len(residue_i))

    #计算距离矩阵，原子距离最小值
    atoms = structure.get_atoms()
    coord = [atom.get_coord() for atom in atoms]
    dist_map = distancematrix(coord)
    dist_map = [np.sqrt(i * 3) for i in dist_map]

    dismap = np.zeros(shape=(residues_length, residues_length))
    contact = np.zeros((residues_length, residues_length))
    for i in range(residues_length):
        for j in range(residues_length):
            if i > j:
                dis_matrix = dist_map[index[i]:index[i + 1]]
                dis = np.array([d[index[j]:index[j + 1]] for d in dis_matrix])
                dismap[i][j] = round(dis.min(), 3)
                dismap[j][i] = dismap[i][j]

                if dismap[i][j] < 8:
                    contact[i][j] = 1
                    contact[j][i] = 1

    #save contact_map
    f_contact = h5py.File('./example/' + pdb_name + '_contact.h5', 'w')  # 写入文件
    f_contact[pdb_name] = contact # 名称



    return repre,attention,torch.tensor(one_hot_2d),torch.tensor(contact)


def seq_feature(seq,pdb_name):

    data=[(pdb_name,seq)]

    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)


        attention=results['apc_sym_attentions']         #这里是经过apc和对称处理后的attention
        repre=results['representations'][33]             #1,301,1280

        #save
        f_repre = h5py.File('./example/' + pdb_name + '_repre.h5', 'w')  # 写入文件
        repre = repre.squeeze()
        repre = repre[1:-1, :]
        f_repre[pdb_name] = repre

        #save
        f_att = './example/'+pdb_name+'_attention.pkl'
        attention = attention.squeeze()
        torch.save(attention, f_att)

    return repre,attention

def get_fasta_2d(seq):
    one_hot_feat = one_hot(seq)
    temp = one_hot_feat[None, :, :]
    temp = np.tile(temp, (temp.shape[1], 1, 1))
    feature = np.concatenate([temp, np.transpose(temp, [1, 0, 2])], 2)
    return feature

def one_hot(seq):
    RNN_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])
    return feat

#测试
pdb_name = 'T0805'
pdb_path = './example/T0805_A.pdb'
# pdb_path = './example/8e4r_A.pdb'
# pdb_name='8e4r'
repre,attention,one_hot_2d,contact= feature(pdb_path,pdb_name)     #299,1280;  1,299,299,660; 299,299,40; 299,299
# 1