import torch
import math
import time
import argparse
import glob
import os
import random

import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from random import sample
from utils import encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor

nnodes = 200  # 42
batch_size = 40  # 64


def load_edges(path='../Dataset/train/edges/0/', adjes=[], tag='train'):  # .tag='train',dataset = '/edges/1/dos'
    tag = path[11:-9]
    val_count = [99, 100, 118, 171, 187]
    test_count = [0, 127, 150, 219, 239, 0, 0, 0, 0]
    nnodes = 200  # 35 + 6 + 1
    files = os.listdir(path)  # 读入文件夹
    while ('.ipynb_checkpoints' in files):
        files.remove('.ipynb_checkpoints')
    num_png = len(files)  # 统计文件夹中的文件个数

    for i in range(num_png):  # 从边的.csv文件读取边数据
        if tag == 'val':
            edges = np.genfromtxt("{}{}.csv".format(path, str(val_count[int(path[-2])] + i)), delimiter=',',
                                  dtype=np.int32)
        elif tag == 'test':
            edges = np.genfromtxt("{}{}.csv".format(path, str(test_count[int(path[-2])] + i)), delimiter=',',
                                  dtype=np.int32)
        else:
            edges = np.genfromtxt("{}{}.csv".format(path, str(i)), delimiter=',', dtype=np.int32)
        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                            shape=(8000, 8000), dtype=np.float32)  # 8000 = batch_size * nnodes
        #         adj = sp.coo_matrix((edges[:, 2], (edges[:, 0] - 8000 * i, edges[:, 1] - 8000 * i)),
        #                             shape=(8000, 8000), dtype=np.float32)  #

        # 邻接矩阵对称化和归一化
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize (features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        # 分配训练、验证和测试集，总共64个时刻的数据，以4个时刻为一个mini-batch，总共16个batches，前10个batches为训练集，第11至第13个为验证集，第14至第16个为测试集
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adjes.append(adj)

    print('already load {} edges:'.format(tag), i)
    return adjes


def load_nodes(path='../Dataset/train_nodes.csv'):
    #     path = "../../0class/train_nodes.csv"
    print('Loading {} dataset of Novel nodes...'.format(path[13:-4]))

    idx_features_labels = np.genfromtxt(path, delimiter=',',
                                        dtype=np.dtype(np.float32))  # "{}{}.csv".format(path,dataset)

    # 从节点的.csv文件读取特征和标签数据
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    nsamples = len(labels)
    print('nsamples : ', len(labels))  # 3665600
    nnodes = 200;
    batch_size = 40  # 35 + 6 + 1  # =42
    nbatches = nsamples / (batch_size * nnodes)  # 2688 = 64 * 42

    print("Number of samples: %d" % nsamples)  # 1870848
    print("Number of batches: %d" % nbatches)  # 696

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  ##取每一列的第一个（第0个元素）
    idx_map = {j: i for i, j in enumerate(idx)}

    # 构建mini-batches列表
    batches = []
    for i in range(int(nbatches)):
        batches.append([features[i * batch_size * nnodes: (i + 1) * batch_size * nnodes, :],
                        labels[i * batch_size * nnodes: (i + 1) * batch_size * nnodes]])

    # 分配训练、验证和测试集，总共64个时刻的数据，以4个时刻为一个mini-batch，总共16个batches，前10个batches为训练集，第11至第13个为验证集，第14至第16个为测试集

    # 将特征、标签、邻接矩阵、训练集、验证集和测试集处理成tensor，作为函数返回值
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    for batch in batches:
        batch[0] = torch.FloatTensor(np.array(batch[0].todense()))
        batch[1] = torch.LongTensor(np.where(batch[1])[1])
    batches = batches[1:]
    return batches


print('Start loading')

# ''' Load edges data of train'''
edge_path = '../Dataset/train/edges/'
train_adjes = []
for j in range(5):
    train_adjes = load_edges(path=edge_path + str(j) + '/', adjes=train_adjes)
print('len(train_adjes)', len(train_adjes))

# ''' Load edges data of val'''
edge_path = '../Dataset/val/edges/'
val_adjes = []
for j in range(5):
    val_adjes = load_edges(path=edge_path + str(j) + '/', adjes=val_adjes)
print('len(val_adjes)', len(val_adjes))

####################################################################

# ''' Load Novel nodes data of train'''
train_batches = load_nodes(path="../Dataset/train_nodes1.csv")
print('len(train_batches)', len(train_batches))

# ''' Load Novel nodes data of val'''
val_batches = load_nodes(path="../Dataset/val_nodes1.csv")
print('len(val_batches)', len(val_batches))

''' 模型设置 '''


def accuracy(output, labels):
    # preds = output.max (1)[1].type_as (labels)
    preds = output
    # print ("Preds: ", preds)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  ##矩阵相乘  XW
        output = torch.spmm(adj, support)  # 稀疏矩阵相乘 AXW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class NN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(NN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        #         self.gc2 = GraphConvolution(nhid, int(nhid/2))
        #         self.fc = nn.Linear(int(nhid/2), nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = self.fc(x)
        # print ("output shape: \n", x.shape)
        return F.log_softmax(x, dim=1)


'''
NN(
  (gc1): GraphConvolution (5 -> 128)
  (gc2): GraphConvolution (128 -> 128)
  (fc): Linear(in_features=256, out_features=2, bias=True)
)'''

''' Load data '''

nnodes = 200  # 42
batch_size = 40  # 64

#############################################################################模型改这里！#############################################################
model = NN(nfeat=12, nhid=32, nclass=5, dropout=0.5)
# nfeat=8   nclass=train_batches[0][1].max().item() + 1 (8000条数据里的不同类)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
# model.load_state_dict(torch.load('gcn5_1.pkl'))

print(model)


def train(epoch):
    model.train()
    i = 0
    t = time.time()

    x = list(range(len(train_batches)))
    random.shuffle(x)
    print(len(train_batches[0]))  #2
    print(train_batches[0][0].shape)  #(8000,11)
    for j in x:
        batch = train_batches[j]
        adj = train_adjes[j]
        i += 1
        optimizer.zero_grad()
        output = model.forward(batch[0], adj)

        # print('output',output)
        # print((batch[1]))
        loss_train = F.nll_loss(output, batch[1])
        acc_train = accuracy(output.max(1)[1], batch[1])
        loss_train.backward()
        optimizer.step()

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time_train:{}'.format(time.time() - t)
          )
    #
    t = time.time()

    labels, preds = [], []

    for j in range(len(val_batches)):
        batch = val_batches[j]
        adj = val_adjes[j]
        output = model.forward(batch[0], adj)

        loss_val = F.nll_loss(output, batch[1])

        labels.append(batch[1])
        preds.append(output.max(1)[1])

    labels = torch.stack(labels, dim=0)
    preds = torch.stack(preds, dim=0)

    labels = torch.reshape(labels, (-1,))
    preds = torch.reshape(preds, (-1,))

    # print ("labels_val: \n", labels.shape)
    # print ("preds_val: \n", preds.shape)

    acc_val = accuracy(preds, labels)
    recall_val = recall_score(labels, preds, average='macro')
    precision_val = precision_score(labels, preds, average='macro')
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
    #     auc_val = roc_auc_score(labels, preds)

    print('         loss_val:{:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'recall_val: {:.4f}'.format(recall_val.item()),
          "precision = {:.4f}".format(precision_val.item()),
          "f1 = {:.4f}".format(f1_val.item()),  # ,"roc_auc = {:.4f}".format(auc_val.item())
          "val_time = {}".format(time.time() - t)  # ,"roc_auc = {:.4f}".format(auc_val.item())
          )

    row = str(epoch + 1) + "," + str(acc_val.item()) + "," + str(recall_val) + "," + str(precision_val) + "," + str(
        f1_val) + "\n"
    # f.write (row)
    return acc_val.item()  # loss_val.data.item()


acc_values = []
save_values = []  # 用来记录最好的模型的过程
bad_counter = 0
best = 0  # 10 + 1
# best_epoch = 0


print('Start Training!')
for i in range(10):

    print("i: ", i)
    t_total = time.time()
    for epoch in range(20):  ##原来是20

        acc_values.append(train(epoch))
        #         print('i = {} of epoch = {}, with time :{}'.format(i,epoch,time.time()-t_total))
        t_total = time.time()
        if best == 0:
            best = acc_values[-1]
            save_values.append(epoch + i * 100)
            torch.save(model.state_dict(), 'gcn0_32_first.pkl')
        elif acc_values[-1] > best:
            if os.path.exists('gcn0_32_first.pkl'):
                os.remove('gcn0_32_first.pkl')
            print('once over best of epcoh = ', epoch + 1)
            best = acc_values[-1]
            #             best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), 'gcn0_32_12.pkl')
            save_values.append(epoch + i * 100)

        else:
            bad_counter += 1

        if bad_counter == 100:
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('Loading {}th epoch\n -----------------------------'.format(save_values[-1]))
    model.load_state_dict(torch.load('gcn0_32_12.pkl'))

print(save_values)



