import torch
import scipy.sparse as spp
import torch.nn as nn
import dgl
from dgl.nn import GATConv
from dgl.nn.pytorch.glob import MaxPooling,AvgPooling

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.gcn1 = GATConv(1280,128 ,4)
        self.gcn2 = GATConv(128*4, 128, 4)
        self.fc_g1 = torch.nn.Linear(128 * 4, 64)
        # self.maxpooling = MaxPooling()
        self.elu = nn.ELU()
    def forward(self,contact,feature):
        G=loader(contact)
        out=self.gcn1(G,feature)        #G是图，feature是1280维
        out=self.elu(out)
        out = out.reshape(-1, 128*4)
        out=self.gcn2(G,out)
        out = self.elu(out)
        out = out.reshape(-1, 128 * 4)
        G.ndata['feat'] = out
        # out = self.maxpooling(G, out)
        out=self.fc_g1(out)
        out=out.unsqueeze(0)
        # out=self.ins(out)
        # out = out.squeeze(0)
        out= self.elu(out)
        return out

def loader(contact):
    #根据邻接矩阵建图
    seq_len=len(contact)
    for i in range(0, seq_len):  # 自环
        if contact[i][i] != 1:
            contact[i][i] = 1
    adj = spp.coo_matrix(contact)
    u=adj.row
    v=adj.col
    G = dgl.graph((u, v))

    return G
