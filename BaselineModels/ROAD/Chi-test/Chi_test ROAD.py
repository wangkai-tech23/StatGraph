import numpy as np
import csv
import os
import random
from scipy.stats import chisquare
from scipy.stats import chi2

def chi2_fitting(data, alpha, sp=None):
    chis, p_value = chisquare(data, axis=sp)
    i, j = data.shape  # j is DOF
    if j == 0:
        print('DOF should >= 1')
    elif j == 1:
        cv = chi2.isf(alpha, j)
    else:
        cv = chi2.isf(alpha, j - 1)
    if chis > cv:
        re = 1  #  hypothesis is rejected
    else:
        re = 0  #  hypothesis is accepted
    return chis, p_value, cv, j-1, re

 
def find_medium(data):  # Find the median of the array
    data.sort()
    medium  = data.shape[1]/2 
    medium = 0.5*(data[0][int(data.shape[1]/2)]+data[0][int(data.shape[1]/2)-1]) if data.shape[1]%2 == 0 else data[0][int(data.shape[1]/2)]
    return medium

def fine_H1(base,test):   #Find the chi-square test values of test and base
    matrix = np.concatenate((base,test),axis=0)
    k = (matrix.shape[0]-1)*(matrix.shape[1]-1) # DOF = （row-1）*（column-1）
    tobase =  base.sum()/matrix.sum()  #
    totest = test.sum()/matrix.sum()  #
    value = matrix.sum(axis=0)
    base = np.square(base-value*tobase) / (value*tobase)
    test = np.square(test-value*totest) / (value*totest)
    H1 = base.sum() +test.sum()
    return H1

def kafang_test(base,test,alpha):  # Perform chi-square and median tests and output test results
    data = np.concatenate((base,test),axis=0)
    chis, p_value, cv, dof, re = chi2_fitting(data, alpha)
    if re:
        return re
    medium_base = find_medium(base)
    sigma_base = np.var(base, axis=1)[0]  #
    medium_test = find_medium(test)
    if medium_test > medium_base + 3 * sigma_base:
        return 1
    return 0


dic_search = {'':0}
dic_search.clear()
P_kafang = [0.995,0.99,0.975,0.95,0.9,0.75,0.5,0.25,0.1,0.05,0.025,0.01,0.005] #13个
Threshold_kafang = [0.41,0.55,0.83,1.15,1.61,2.67,4.35,6.63,9.24,11.07,12.83,15.09,16.75]#13个
dic_search = {P_kafang[i]: Threshold_kafang[i] for i in range(13)}

alpha = 0.1 #confidence coefficient
alpha_degree = alpha
alpha_weight = 0.01
print('confidence coefficient (alpha):{},alpha_degree:{},alpha_weight:{}\n'.format(alpha,alpha_degree,alpha_weight))
predict = [] # predict
label = [] # real


for tag in range(1,6):
    filepath = './ROAD dealed/graph_list0.csv'
    filepath2 =  './ROAD dealed/graph_list'+str(tag)+'.csv'
    print('tag', tag)

    normal_csvreader = csv.reader(open(filepath,encoding = 'utf-8'))
    attack_csvreader = csv.reader(open(filepath2,encoding = 'utf-8'))
    maxedge1 = []
    maxdegree1 = []
    maxweight1 = []
    line = []
    lin = []
    li = []
    i = -1
    for row in normal_csvreader:
    #     line.append(row)
        if i == -1:
            i += 1
            continue
        line.append(row[1])
        lin.append(row[2])
        li.append(row[3])
        if (i+1) % 6 == 0:
            maxedge1.append(line)
            maxdegree1.append(lin)
            maxweight1.append(li)
            line = [];lin = [];li = []
        i += 1
    maxedge2 = []
    maxdegree2 = []
    maxweight2 = []
    line = []
    lin = []
    li = []
    i = -1
    flag = False
    j = 0;
    rv= 0

    for row in attack_csvreader:
        if i == -1:
            i += 1
            continue
        line.append(row[1])
        lin.append(row[2])
        li.append(row[3])
        if row[-1] == 'T':
            flag = True
        if (i+1) % 6 == 0:
            maxedge2.append(line)
            maxdegree2.append(lin)
            maxweight2.append(li)
            line = [];  lin = [];li = []
            if flag:
                label.append(1)
                flag = False
            else:
                label.append(0)
        i += 1

    print('length1',len(maxedge1))
    print('length2',len(maxedge2))

    if len(maxedge1)<len(maxedge2):
        M = len(maxedge1)
        for k in range(M,len(maxedge2)):
            maxedge1.append(maxedge1[k-M])
            maxdegree1.append(maxdegree1[k-M])
    print('The supplementary length1:',len(maxedge1))


    for i,row in enumerate(maxedge2):
        ba = np.reshape(np.array(maxedge1[i]),(1,-1)).astype(float)
        te = np.reshape(np.array(maxedge2[i]),(1,-1)).astype(float)

        ba2 = np.reshape(np.array(maxdegree1[i]),(1,-1)).astype(float)
        te2 = np.reshape(np.array(maxdegree2[i]),(1,-1)).astype(float)

        result = kafang_test(ba,te,alpha) | kafang_test(ba2,te2,alpha_degree)
        predict.append(result)

    import torch
    def count_pre(label, outputs):
        label = torch.tensor(label)
        outputs = torch.tensor(outputs)
        all_P = torch.eq(outputs, 1).sum()  # TP+FP  =1+2
        correct = (torch.eq(outputs, label)).sum()  ##TP+TN  =1+4
        wenhao = torch.eq(label, 0).sum()  # FP+TN = 2+4
        all_N = torch.eq(outputs, 0).sum()  # TN+FN  =3+4
        TP = (all_P + correct - wenhao) / 2
        FP = (all_P - correct + wenhao) / 2
        TN = correct - TP
        FN = all_N - TN
        return TP, TN, FN, FP

    if True:
        TP = torch.zeros(1)
        TN = torch.zeros(1)
        FN = torch.zeros(1)
        FP = torch.zeros(1)
        tp, tn, fn, fp = count_pre(predict, label)
        TP += tp
        TN += tn
        FN += fn
        FP += fp
        acc_test = (TP + TN) / (TP + FP + TN + FN)
        if TP == 0:
            precision_test = torch.zeros(1)
            recall_test = torch.zeros(1)
            f1_test = torch.zeros(1)
        else:
            precision_test = TP / (TP + FP)
            recall_test = TP / (TP + FN)
            f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

    print('TP:{}, TN:{}, FN:{}, FP:{}'.format(TP.item(), TN.item(), FN.item(), FP.item()))
    print('acc: {:.3f}, pre: {:.3f}, recall: {:.3f}, f1: {:.3f},'.format(acc_test.item(),precision_test.item(), recall_test.item(), f1_test.item()))
    print('----------------------')


