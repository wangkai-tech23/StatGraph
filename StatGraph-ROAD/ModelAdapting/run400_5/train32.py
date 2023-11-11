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

nnodes = 400
batch_size = 5


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


def load_nodes(path='../../Dataset400_5/train_nodes.csv'):
    print('Loading {} dataset of nodes...'.format(path[19:-4]))

    idx_features_labels = np.genfromtxt(path, delimiter=',',dtype=np.dtype(np.float32))

    features = sp.csr_matrix(idx_features_labels[1:, :-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[1:, -1])
    nsamples = len(labels)
    print('nsamples : ', len(labels))
    nnodes = 400
    batch_size = 5
    nbatches = nsamples / (batch_size * nnodes)

    print("Number of samples: %d" % nsamples)
    print("Number of batches: %d" % nbatches)

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    # Build a mini-batches list
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


print('Start loading')

# ''' Load edges data of train'''
edge_path = '../../Dataset400_5/train/edges/'
train_adjes = []
for j in range(6):
    train_adjes = load_edges(path=edge_path + str(j) + '/', adjes=train_adjes)
print('len(train_adjes)', len(train_adjes))

# ''' Load edges data of val'''
edge_path = '../../Dataset400_5/val/edges/'
val_adjes = []
for j in range(6):
    val_adjes = load_edges(path=edge_path + str(j) + '/', adjes=val_adjes)
print('len(val_adjes)', len(val_adjes))



# ''' Load nodes data of train'''
train_batches = load_nodes(path="../../Dataset400_5/train_nodes.csv")
print('len(train_batches)', len(train_batches))

# ''' Load nodes data of val'''
val_batches = load_nodes(path="../../Dataset400_5/val_nodes.csv")
print('len(val_batches)', len(val_batches))


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

''' Load data '''

nnodes = 400
batch_size = 5

model = NN(nfeat=12, nhid=32, nclass=6, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

def train(epoch):
    model.train()
    i = 0
    t = time.time()

    x = list(range(len(train_batches)))
    random.shuffle(x)
    for j in x:
        batch = train_batches[j]
        adj = train_adjes[j]
        i += 1
        optimizer.zero_grad()
        output = model.forward(batch[0], adj)

        loss_train = F.nll_loss(output, batch[1])
        acc_train = accuracy(output.max(1)[1], batch[1])
        loss_train.backward()
        optimizer.step()

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time_train:{}'.format(time.time() - t)
          )

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

    acc_val = accuracy(preds, labels)
    recall_val = recall_score(labels, preds, average='macro')
    precision_val = precision_score(labels, preds, average='macro')
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)

    print('         loss_val:{:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'recall_val: {:.4f}'.format(recall_val.item()),
          "precision = {:.4f}".format(precision_val.item()),
          "f1 = {:.4f}".format(f1_val.item()),
          "val_time = {}".format(time.time() - t)
          )

    row = str(epoch + 1) + "," + str(acc_val.item()) + "," + str(recall_val) + "," + str(precision_val) + "," + str(
        f1_val) + "\n"

    return acc_val.item()


acc_values = []
save_values = []  # save_values: document the best models
bad_counter = 0
best = 0

print('Start Training!')
for i in range(10):

    print("i: ", i)
    t_total = time.time()
    for epoch in range(20):
        acc_values.append(train(epoch))
        t_total = time.time()
        if best == 0:
            best = acc_values[-1]
            save_values.append(epoch + i * 100)
            torch.save(model.state_dict(), 'gcn32_first.pkl')
        elif acc_values[-1] > best:
            if os.path.exists('gcn32_first.pkl'):
                os.remove('gcn32_first.pkl')
            print('once over best of epcoh = ', epoch + 1)
            best = acc_values[-1]
            bad_counter = 0
            torch.save(model.state_dict(), 'gcn32.pkl')
            save_values.append(epoch+1 + i * 100)

        else:
            bad_counter += 1

        if bad_counter == 50:
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('Loading {}th epoch\n -----------------------------'.format(save_values[-1]))
    model.load_state_dict(torch.load('gcn32.pkl'))

print(save_values)




