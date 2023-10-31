import argparse
import torch
import torch.nn as nn
import h5py
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import pickle
import numpy as np
import os
from resnet import ResNet,block
import GAT_module as GAT
from tri_model import tri_model
from feature import feature


cuda=torch.cuda.is_available()
# name_date='8_14'
# device = torch.device('cpu')
# 超参
length=400
# epoch=20
batch_size=1
num_workers=0


##########################TopPrediction##########################
def TopPrediction(pred=None, gt=None,top=10, diag=False, outfile=None):
    if pred is None:
        print('please provide a predicted contact matrix')
        exit(1)
    if outfile is None:
        print('please provide the output file name')
        exit(1)

    # avg_pred = (pred + pred.transpose((1, 0))) / 2.0

    seqLen = pred.shape[0]

    index = np.zeros((seqLen, seqLen, 2))
    for i in range(seqLen):
        for j in range(seqLen):
            index[i, j, 0] = i
            index[i, j, 1] = j

    pred_index = np.dstack((pred, index))
    out_matrix = np.zeros((seqLen, seqLen))

    M1s = np.ones_like(pred, dtype=np.int16)
    if diag:
        mask = np.triu(M1s, 0)
    else:
        mask = np.triu(M1s, 1)

    accs = []
    res = pred_index[(mask > 0)]
    if res.size == 0:
        print("ERROR: No prediction")
        exit()

    res_sorted = res[(-res[:, 0]).argsort()]
    if top == 'all':
        top = res_sorted.shape[0]

    with open(outfile, 'w') as f:
        f.write('#The top'+str(top)+'predictions:')
        f.write('\n')
        # print(f, "#The top", top, " predictions:")
        # print(f, "Number  Residue1  Residue2  Predicted_Score")
        f.write("Number  Residue1  Residue2  Score")
        # f.write("Number  Residue1  Residue2  Score  gt")
        f.write('\n')
        for i in range(top):
            # try:
            # a = int(res_sorted[i, 1])
            # b = int(res_sorted[i, 2])
            #
            # if  gt[a][b]==1:
            #     flag = 1
            # else:
            #     flag = 0
            # f.write("%-8d%-10d%-10d%-10.4f%-10d%-10.3f" % (
            #     i + 1, int(res_sorted[i, 1]) + 1, int(res_sorted[i, 2]) + 1, res_sorted[i, 0], flag, gt[a][b]))
            # except :
            f.write("%-8d%-10d%-10d%-10.4f" % (
            i + 1, int(res_sorted[i, 1]) + 1, int(res_sorted[i, 2]) + 1, res_sorted[i, 0]))
            f.write('\n')

    return None

#####################################################data

def fix_pred_map(_input):
    len = _input.shape[0]
    out = np.zeros((len, len))
    for i in range(len):
        for j in range(len):
            temp = float((_input[i][j] + _input[j][i])) / 2
            out[i][j] = temp
            out[j][i] = temp
    return out



class Dataset(Dataset.Dataset):
#####################序列和representation
    def __init__(self,pdb_name,pdb_path):

#路径下所有文件
        self.pdb_path=pdb_path
        self.pdb_name = pdb_name

    def __getitem__(self):
        feature_dict={}
        repre, attention, one_hot_2d, monomer_contact = feature(self.pdb_path, self.pdb_name)
        seq_len = len(monomer_contact)

        #padding
        if seq_len < length:
            padding = length-seq_len
            pad_1 = nn.ZeroPad2d(padding=(0, 0, 0, padding, 0, padding))    #两种padding
            # pad_2 = nn.ZeroPad2d(padding=(0, padding, 0, padding))
            attention = pad_1(attention)
            one_hot_2d = pad_1(one_hot_2d)
        else:
            attention = attention[:length, :length, :]
            one_hot_2d = one_hot_2d[:length, :length, :]


        # G=GAT.default_loader(monomer_contact,seq_len)     #根据contact建图

        #feature_dict
        feature_dict['repre'] = repre.unsqueeze(0)
        feature_dict['attention'] = attention.unsqueeze(0)
        feature_dict['one_hot_2d'] = one_hot_2d.unsqueeze(0)
        feature_dict['monomer_contact'] = monomer_contact
        feature_dict['seq_len'] = seq_len
        # feature_dict['G'] = G
        # print(G)

        return feature_dict


def _collate_fn(samples):

    repre,name,seq,label,attention,hot,G,mask= map(list, zip(*samples))
    return repre,name,seq,label,attention,hot,G,mask

#####################################################model
# ###################处理repre,把m*64变成m*m*128,两个蛋白各个残基的差异和乘积和conv，处理完了repre
class FullyConnected_1(nn.Module):


    def __init__(self, embed_dim=64, hidden_dim=64, activation=nn.ReLU()):
        super(FullyConnected_1, self).__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv1 = nn.Conv2d(2 * self.D, self.H, 1)
        # self.batchnorm1 = nn.BatchNorm2d(self.H)
        self.batchnorm1= nn.InstanceNorm2d(self.H)
        self.activation1= activation

    def forward(self, z0):


        # z0 is (b,N,d), z1 is (b,M,d)
        z0 = z0.transpose(1, 2)
        # z1 = z1.transpose(1, 2)
        # z0 is (b,d,N), z1 is (b,d,M)

        z_dif = torch.abs(z0.unsqueeze(3) - z0.unsqueeze(2))
        z_mul = z0.unsqueeze(3) * z0.unsqueeze(2)
        z_cat = torch.cat([z_dif, z_mul], 1)

        b = self.conv1(z_cat)
        b = self.batchnorm1(b)
        b = self.activation1(b)


        return b

######################得到attention map
class model_model(nn.Module):
    def __init__(self,in_features=764,bias=True):
        super(model_model, self).__init__()
        self.GGAT = GAT.Net()\
            # .to(device)

        self.F1 = FullyConnected_1()

        self.conv1 = nn.Conv2d(in_features,64, kernel_size=1, stride=1, padding=0, bias=False)
        self.ins = nn.InstanceNorm2d(64)
        self.elu = nn.ELU()

        self.tri_model1=tri_model(64,32)         #输入是128，隐藏层是64
        self.tri_model2 = tri_model(64, 32)
        self.tri_model3 = tri_model(64, 32)  # 输入是128，隐藏层是64
        self.tri_model4 = tri_model(64, 32)
        self.tri_model5 = tri_model(64, 32)  # 输入是128，隐藏层是64
        self.tri_model6 = tri_model(64, 32)
        self.tri_model7 = tri_model(64, 32)  # 输入是128，隐藏层是64
        self.tri_model8 = tri_model(64, 32)

        self.M=ResNet(block, [8], in_features,length * length)\
            # .to(device)

        self.last_layer = nn.Conv2d(64,1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_dict):
        out_repre= self.GGAT(feature_dict['monomer_contact'], feature_dict['repre'])\
            # .to(device)
        out_repre = self.F1(out_repre)
        out_repre= out_repre.permute(0, 2, 3, 1)
        seq_len = feature_dict['seq_len']

        #padding
        if seq_len< length:
            padding = length-seq_len
            pad_1 = nn.ZeroPad2d(padding=(0, 0, 0, padding, 0, padding))
            out_repre = pad_1(out_repre)
        else:
            out_repre= out_repre[:, :length, :length, :]

        feature=torch.cat((out_repre,feature_dict['attention'],feature_dict['one_hot_2d']),axis=3)  #attention_map是1，208，208，660
        feature = feature.permute(0, 3, 1, 2)
        feature = self.conv1(feature)
        feature = self.ins(feature)
        feature = self.elu(feature)
        feature = feature.permute(0, 2, 3, 1)
        feature = self.tri_model1(feature)
        feature = self.tri_model2(feature)
        feature = self.tri_model3(feature)
        feature = self.tri_model4(feature)
        feature = self.tri_model5(feature)
        feature = self.tri_model5(feature)
        feature = self.tri_model7(feature)
        feature = self.tri_model8(feature)
        feature = feature.permute(0, 3, 1, 2)
        out=self.M(feature)
        out = self.last_layer(out)
        out = self.sigmoid(out)
        return out

#训练
def main(pdb_name,pdb_path):


    feature = Dataset(pdb_name,pdb_path)
    feature_dict=feature.__getitem__()


    #model
    model = model_model()
    # .\
        # to(device)
    model_path='./weight/model.pt'
    model.load_state_dict((torch.load(model_path, map_location='cpu')))



    #测试
    model.eval()
    with torch.no_grad():

        # feature_dict=feature_dict.to(device)
        outputs = model(feature_dict)  # repre是1,208,1280，attention是1，208，208，660
        outputs = outputs.squeeze(0)

        seq_len=feature_dict['seq_len']
        if  seq_len < length:
            outputs = outputs[:, 0:seq_len, 0:seq_len]
        else:
            outputs = outputs[:, 0:length, 0:length]

        del feature_dict
        torch.cuda.empty_cache()

        outputs = outputs.squeeze()

        outputs = outputs.cpu().detach().numpy()

        outputs=fix_pred_map(outputs)

        np.savetxt(os.path.join('./example',pdb_name + '_predict.cmap'), outputs)


        TopPrediction(pred=outputs, top=20, outfile='./example/'+ pdb_name + '_TopPrediction.txt')      
        os.remove('./example/'+ pdb_name +'_attention.pkl')
        os.remove('./example/'+ pdb_name +'contact.h5')
        os.remove('./example/'+ pdb_name +'_repre.h5')
        print("output TopPrediction")
       
import sys
if __name__ == "__main__":

    pdb_name=sys.argv[1]
    pdb_path = sys.argv[2]
    # pdb_name = 'T0805'
    # pdb_path = './example/T0805_A.pdb'

    main(pdb_name,pdb_path)

























