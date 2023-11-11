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

nnodes = 50
batch_size = 40

def load_edges2(path='../../Dataset50_40/train/edges/0/', adjes=[], tag='train'):  # .tag='train',dataset = '/edges/1/dos'
    tag = path[19:-9]
    ''' val/test_count: the number of files in each folder '''
    val_count = [395, 381, 453, 664, 726]
    test_count = [0, 490, 583, 854, 934, 0, 0, 0, 0]
    files = os.listdir(path)  # Read the folder
    num_png = len(files)  # Count the number of files in a folder

    for i in range(num_png):
        if tag == 'val':
            edges = np.genfromtxt("{}{}.csv".format(path, str(val_count[int(path[-2])] + i)), delimiter=',',
                                  dtype=np.int32)
        elif tag == 'test':
            edges = np.genfromtxt("{}{}.csv".format(path, str(test_count[int(path[-2])] + i)), delimiter=',',
                                  dtype=np.int32)
        else:
            edges = np.genfromtxt("{}{}.csv".format(path, str(i)), delimiter=',', dtype=np.int32)
        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                            shape=(batch_size * nnodes, batch_size * nnodes), dtype=np.float32)  # 8000 = batch_size * nnodes

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize (features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adjes.append(adj)

    print('already load {} edges:'.format(tag), i)
    return adjes


def load_nodes2(path='../../Dataset50_40/train_nodes.csv'):
    print('Loading {} dataset of Novel nodes...'.format(path[11:-4]))

    idx_features_labels = np.genfromtxt(path, delimiter=',',
                                        dtype=np.dtype(np.float32))

    features = sp.csr_matrix(idx_features_labels[1:, :-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[1:, -1])
    nsamples = len(labels)
    print('nsamples : ', len(labels))

    nbatches = nsamples / (batch_size * nnodes)

    print("Number of samples: %d" % nsamples)
    print("Number of batches: %d" % nbatches)

    # build graph
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


def test():
    model.eval()
    labels, preds = [], []
    print('Start Testing!')
    t = time.time()
    test_conf = MulticlassConfusionMatrix(num_classes=5)

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
          'confusion matrix:{}'.format(total_conf)
          )

    row = "Test," + str(acc_test.item()) + "," + str(recall_test) + "," + str(precision_test) + "," + str(
        f1_test) + "\n"

edge_path = '../Dataset/test/edges/' #
# ''' Load Novel nodes data of test'''
test_batches = load_nodes2(path="../../Dataset50_40/test_nodes.csv")
print('len(test_batches)', len(test_batches))

# ''' Load edges data of test'''
edge_path =  '../../Dataset50_40/test/edges/'
test_adjes = []
for j in range(1, 9):
    test_adjes = load_edges2(path=edge_path + str(j) + '/', adjes=test_adjes)
print('len(adjes)', len(test_adjes))

''' 进行test '''


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
        output = torch.spmm(adj, support)   # Sparse matrix multiplication AXW
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
        return F.log_softmax(x, dim=1)

nnodes = 50
batch_size = 40   
model = NN(nfeat=12, nhid=32, nclass=5, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
print(model)
model.load_state_dict(torch.load('gcn32new.pkl'))


t = time.time()
test()


# Start Testing!
# Test set results:
#  accuracy = 0.9970 recall = 0.9533 precision = 0.9275 f1 = 0.9403 Test time = 19.084044456481934
#   confusion matrix:tensor([[10584788,     1679,     7499,     7210,     4505],
#         [     274,    58553,        0,        0,        0],
#         [   11012,        0,    39141,        0,        0],
#         [      25,        0,        0,    59565,        0],
#         [     448,        0,        0,        0,    65301]])
