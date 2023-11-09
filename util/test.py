import numpy as np
def TopPrediction(pred=None, gt=None,top=10, diag=True, outfile=None):
    if pred is None:
        print('please provide a predicted contact matrix')
        exit(1)
    if outfile is None:
        print('please provide the output file name')
        exit(1)

    # avg_pred = (pred + pred.transpose((1, 0))) / 2.0

    seqLen = pred.shape[0]

    index = np.zeros((seqLen, seqLen, 2))    #214,214,2
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
        # f.write("Number  Residue1  Residue2  Score")
        f.write("Number  Residue1  Residue2  Score   gt   distance")
        f.write('\n')
        for i in range(top):
            a = int(res_sorted[i, 1])
            b = int(res_sorted[i, 2])
            if  gt[a][b]<=8:
                flag = 1
            else:
                flag = 0
            f.write("%-8d%-10d%-10d%-10.4f%-10d%-10.3f" % (
                i + 1, int(res_sorted[i, 1]) + 1, int(res_sorted[i, 2]) + 1, res_sorted[i, 0], flag, gt[a][b]))
            f.write('\n')

    return None

import os
import matplotlib.pyplot as plt
pdb_name='T0805'
pred=np.loadtxt(os.path.join('./example', pdb_name + '_predict.cmap'))

label =np.loadtxt(os.path.join('./example', pdb_name + '_label.cmap'))

mask = np.loadtxt(os.path.join('./example', pdb_name + '_mask.cmap'))

pred=pred*mask


TopPrediction(pred=pred, gt=label, top=20, outfile='./example/' + pdb_name + '_TopPrediction.txt')


