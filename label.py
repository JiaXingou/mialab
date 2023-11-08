from Bio.PDB import *
from Bio.Cluster import distancematrix
import numpy as np
import h5py
import os
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
# 
warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def label(pdb_path, pdb_path_monomer, pdb_name):
    parser = PDBParser()

    structure_monomer = parser.get_structure(pdb_name, pdb_path_monomer)

    structure = parser.get_structure(pdb_name, pdb_path)

    monomer_chain_list = Selection.unfold_entities(structure_monomer, 'R')
    monomer_resseq_res_list = [residue.get_id()[1] for residue in monomer_chain_list]
    seq_len=len(monomer_resseq_res_list)
    mask = np.ones((seq_len, seq_len))

    # 记录残基号
    chain_list = Selection.unfold_entities(structure, 'C')
    res_list_A = Selection.unfold_entities(chain_list[0], 'R')
    res_list_B = Selection.unfold_entities(chain_list[1], 'R')
    resseq_res_list_A = [residue.get_id()[1] for residue in res_list_A]
    resseq_res_list_B = [residue.get_id()[1] for residue in res_list_B]
    l_A = len(resseq_res_list_A)
    l_B = len(resseq_res_list_B)
    residues_length = l_A + l_B


    # B相对于monomer多出的残基的index,删掉距离矩阵中相应的
    index_del_B = [resseq_res_list_B.index(i) for i in resseq_res_list_B if i not in monomer_resseq_res_list]
    # monomer相对于B多出的残基,补0到距离矩阵中相应的
    index_add_B = [monomer_resseq_res_list.index(i) for i in monomer_resseq_res_list if i not in resseq_res_list_B]

    # B相对于monomer多出的残基的index,删掉距离矩阵中相应的
    index_del_A = [resseq_res_list_A.index(i) for i in resseq_res_list_A if i not in monomer_resseq_res_list]
    # monomer相对于B多出的残基,补0到距离矩阵中相应的
    index_add_A = [monomer_resseq_res_list.index(i) for i in monomer_resseq_res_list if i not in resseq_res_list_B]


    # index，记录每个残基的原子个数
    residues = structure.get_residues()
    index = [0]
    for i, residue_i in enumerate(residues):
        index.append(index[i] + len(residue_i))

    # 计算距离矩阵，原子距离最小值
    atoms = structure.get_atoms()
    coord = [atom.get_coord() for atom in atoms]
    dist_map = distancematrix(coord)
    dist_map = [np.sqrt(i * 3) for i in dist_map]

    dismap = np.zeros(shape=(residues_length, residues_length))
    for i in range(residues_length):
        for j in range(residues_length):
            if i > j:
                dis_matrix = dist_map[index[i]:index[i + 1]]
                dis = np.array([d[index[j]:index[j + 1]] for d in dis_matrix])
                dismap[i][j] = round(dis.min(), 3)
                dismap[j][i] = dismap[i][j]

                # if dismap[i][j] < 8:
                #     inter_contact[i][j] = 1
                #     inter_contact[j][i] = 1
    # 取链间
    inter_dismap = dismap[:l_A, l_A:]

    # 由于真实结构中A,B链的残基有时候会有缺失，假如我们预测时以A链序列作为输入，为了对比groundtruth和pred，我们调整groundtruth使得与pred的各个残基对应
    flag = 0

    if len(index_del_A) > 0:
        inter_dismap = np.delete(inter_dismap, index_del_A, 1)
    if len(index_del_B) > 0:
        inter_dismap = np.delete(inter_dismap, index_del_B, 1)


    if len(index_add_A) > 0 :
        for i in index_add_A:
            x, y = inter_dismap.shape
            inter_dismap = np.insert(inter_dismap, i, -1 * np.ones(y), 0)
            mask[:, i] = np.zeros(seq_len)  # mask用来记住这部分的值都是-1，不用做评估的
            flag = 1
    if len(index_add_B) > 0 :
        for i in index_add_B:
            x, y = inter_dismap.shape
            inter_dismap = np.insert(inter_dismap, i, -1 * np.ones(x), 1)
            mask[:, i] = np.zeros(seq_len)  # mask用来记住这部分的值都是-1，不用做评估的
            flag = 1

    # save groundtruth
    np.savetxt(os.path.join('./example', pdb_name + '_label.cmap'), inter_dismap, fmt='%d')

    if flag == 1:
        np.savetxt(os.path.join('./example', pdb_name + '_mask.cmap'), mask, fmt='%d')

    inter_contact = np.zeros((seq_len, seq_len))
    for i in range(0,seq_len):
        for j in range(0, seq_len):

            if inter_dismap[i][j] <= 8 and inter_dismap[i][j]>-1:
                # and inter_dismap[i][j] > -1:
                inter_contact[i][j] = 1
            else:
                inter_contact[i][j] = 0
    np.savetxt(os.path.join('./example', pdb_name + '_inter_contact.cmap'), inter_dismap, fmt='%d')
    return inter_dismap, inter_contact


# 测试
# pdb_path = './example/T0805.pdb'
# pdb_path_A = './example/T0805_A.pdb'
# pdb_name = 'T0805'
# # pdb_path = './example/8e4r.pdb'
# # pdb_name='8e4r'
# gt = label(pdb_path, pdb_path_A, pdb_name)  # label
# 1



