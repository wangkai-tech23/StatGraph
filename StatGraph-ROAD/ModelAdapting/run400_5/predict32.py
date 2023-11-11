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
from torchmetrics.classification import MulticlassConfusionMatrix


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from random import sample
from utils import encode_onehot, normalize, sparse_mx_to_torch_sparse_tensor



def load_nodes(path='../../Dataset400_5/train_nodes.csv'):
    print('Loading {} dataset of nodes...'.format(path))

    idx_features_labels = np.genfromtxt(path, delimiter=',',dtype=np.dtype(np.float32)) 

    # Read feature and label data from csv file
    features = sp.csr_matrix(idx_features_labels[1:, :-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[1:, -1])
    nsamples = len(labels)
    print('nsamples : ', len(labels))  
    nnodes = 400;
    batch_size = 5
    nbatches = nsamples / (batch_size * nnodes)  

    print("Number of samples: %d" % nsamples)  
    print("Number of batches: %d" % nbatches)  

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  
    idx_map = {j: i for i, j in enumerate(idx)}

    batches = []
    for i in range(int(nbatches)):
        batches.append([features[i * batch_size * nnodes: (i + 1) * batch_size * nnodes, :],
                        labels[i * batch_size * nnodes: (i + 1) * batch_size * nnodes]])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    for batch in batches:
        batch[0] = torch.FloatTensor(np.array(batch[0].todense()))
        batch[1] = torch.LongTensor(np.where(batch[1])[1])
    return batches


def load_edges(path='../../Dataset400_5/train/edges/0/', adjes=[], tag='train'):  # .tag='train',dataset = '/edges/1/dos'
    tag = path[19:-9]
    ''' train/val/test_count: the number of files in each folder '''
    train_count = [0, 13, 70, 23, 38, 16]
    val_count = [0, 7, 31, 18, 38, 4]  
    test_count = [0, 4, 28, 36, 44, 3]
    files = os.listdir(path)  # Read the folder
    num_png = len(files)  # Count the number of files in a folder

    if path[-2] == '0':
        start = 0
        for j in range(1,6):  # j：Normal data generated from 1-5
            if tag == 'train':
                num = train_count[j]
            elif tag == 'val':
                num = val_count[j]
                if j == 5:
                    start = 16
            elif tag == 'test':
                num = test_count[j]
                if j == 5:
                    start = 20

            for i in range(num):   
                edges = np.genfromtxt("{}{}_{}.csv".format(path,str(j),str(start + i)), delimiter=',',dtype=np.int32)
                adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),shape=(batch_size * nnodes, batch_size * nnodes), dtype=np.float32)

                # build symmetric adjacency matrix
                adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

                # features = normalize (features)
                adj = normalize(adj + sp.eye(adj.shape[0]))

                adj = sparse_mx_to_torch_sparse_tensor(adj)
                adjes.append(adj)
    else:
        files = os.listdir(path)  # Read the folder
        num = len(files)
        if tag == 'test' and path[-2] == '5':
            start = 3
        elif tag =='val' and path[-2] == '5':
            start = 2
        else:
            start = 0
        for i in range(num):  
            edges = np.genfromtxt("{}{}.csv".format(path, str(start +i)), delimiter=',',
                                      dtype=np.int32)
            adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                                shape=(batch_size * nnodes, batch_size * nnodes), dtype=np.float32)
            # build symmetric adjacency matrix
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

            # features = normalize (features)
            adj = normalize(adj + sp.eye(adj.shape[0]))

            adj = sparse_mx_to_torch_sparse_tensor(adj)
            adjes.append(adj)

    print('already load {} edges:'.format(tag), i)
    return adjes

def test():
    model.eval()
    labels, preds = [], []
    print('Start Testing!')
    t = time.time()
    test_conf = MulticlassConfusionMatrix(num_classes=6)

    for j in range(len(test_batches)):
        batch = test_batches[j]
        adj = test_adjes[j]
        output = model.forward(batch[0], adj)
        labels.append(batch[1])
        preds.append(output.max(1)[1])
        test_conf.update(output.max(1)[1], batch[1])

    labels = torch.stack(labels, dim=0)
    preds = torch.stack(preds, dim=0)

    labels = torch.reshape(labels, (-1,))
    preds = torch.reshape(preds, (-1,))


    acc_test = accuracy(preds, labels)
    recall_test = recall_score(labels, preds, average='macro')
    precision_test = precision_score(labels, preds, average='macro')
    f1_test = 2 * precision_test * recall_test / (precision_test + recall_test)
    total_conf = test_conf.compute()
    print("Test set results:\n",
          "accuracy = {:.4f}".format(acc_test.item()),
          "recall = {:.4f}".format(recall_test.item()),
          "precision = {:.4f}".format(precision_test.item()),
          "f1 = {:.4f}".format(f1_test.item()),
          "Test time = {} \n ".format(time.time() - t),
          'confusion matrix:\n {}'.format(total_conf)
          )
    row = "Test," + str(acc_test.item()) + "," + str(recall_test) + "," + str(precision_test) + "," + str(
        f1_test) + "\n"


''' test '''
def accuracy(output, labels):
    preds = output
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
        support = torch.mm(input, self.weight)  # Matrix multiplication  XW
        output = torch.spmm(adj, support)  # Sparse matrix multiplication AXW
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
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

nnodes = 400
batch_size = 5


''' Load edges data of test'''
edge_path = '../../Dataset400_5/test/edges/'
test_adjes = []
for j in range(6):
    test_adjes = load_edges(path=edge_path + str(j) + '/', adjes=test_adjes)
print('len(adjes)', len(test_adjes))

''' Load nodes data of test'''
test_batches = load_nodes(path="../../Dataset400_5/test_nodes.csv")
print('len(test_batches)', len(test_batches))

nnodes = 400
batch_size = 5
model = NN(nfeat=12, nhid=32, nclass=6, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
print(model)
model.load_state_dict(torch.load('gcn32.pkl'))

t = time.time()
test()

