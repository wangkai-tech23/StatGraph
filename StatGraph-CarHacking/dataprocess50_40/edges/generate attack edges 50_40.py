''' edge(adjacent matrix) generation'''

import numpy as np
import csv
import os
import shutil

def write_csv(filepath, way, row):
    if filepath is None:
        filepath = "preprocess_well_origin.csv"
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

'''loading data'''

''' TAG: the number of abnormal data
 DoS_Attack_dataset                    TAG = 1
 Fuzzy_Attack_dataset                  TAG = 2
 Spoofing_the_drive_gear_dataset       TAG = 3
 Spoofing_the_RPM_gauge_dataset        TAG = 4
 '''
TAG = 1
file_path_list = ['','DoS_Attack_dataset','Fuzzy_Attack_dataset','Spoofing_the_drive_gear_dataset','Spoofing_the_RPM_gauge_dataset']
Slice = ['','train','val','test']
div = 1 # div \in [1,2,3]
'''Slice:  divide data into the training set, the validation set and the test set
 train      div = 1
 val        div = 2
 test       div = 3
'''

for TAG in range(1,5):
    filepath = '../../../Dataset/Car Hacking Dataset/'+ file_path_list[TAG]  +'.csv'
    path = '../../Dataset50_40/train/edges/'
    batchsize = 40
    nodes = 50

    csvreader = csv.reader(open(filepath, encoding='utf-8'))
    dataset = [];
    labelset = []
    line = [];
    labeline = []
    i = 0

    for i, row in enumerate(csvreader):
        if i % nodes == 0 and i != 0:
            dataset.append(line)
            labelset.append(labeline)
            line = [];
            labeline = []
        line.append(row[1]) # Keep only the IDs to create the graph
        labeline.append(row[-1])

    label = []
    for i in range(len(labelset)):
        if 'T' in labelset[i]:
            label.append(2)   # Indicates that this graph contains abnormal data
        else:
            label.append(0)

    attack_path = path + str(TAG) + '/'
    normal_path = path +'0_'+ str(TAG) + '/'
    if not os.path.exists(attack_path):
        os.makedirs(attack_path)
    if not os.path.exists(normal_path):
        os.makedirs(normal_path)

    dic_search = {'': 0}  # Create a dictionary
    normalset = [];
    attackset = []
    step = 0
    j = 0
    attack_num = 0;
    normal_num = 0
    count_attack = 0;
    count_normal = 0

    for row in dataset:  # ID sequence only
        i = 0
        dic_search.clear()
        if label[step] != 0:
            for j in range(len(row)):
                if i == 0:
                    i = 1
                    last = row[j]
                    dic_search[last] = [j]
                    continue
                now = row[j]
                yz = (j + count_attack * nodes, j - 1 + count_attack * nodes, 1)
                attackset.append(list(yz))
                if not (now in dic_search.keys()):
                    dic_search[now] = [j]
                else:
                    for k, sam in enumerate(dic_search[now]):
                        if k > 4:  # If the dictionary is too long, only take the last five to reduce the amount of computation
                            break
                        yz = (j + count_attack * nodes, sam + count_attack * nodes, 1)
                        attackset.append(list(yz))
                    dic_search[now].insert(0, j)
                last = now
            count_attack += 1
        else:
            for j in range(len(row)):
                if i == 0:
                    i = 1
                    last = row[j]
                    dic_search[last] = [j]
                    continue
                now = row[j]
                yz = (j + count_normal * nodes, j - 1 + count_normal * nodes, 1)
                normalset.append(list(yz))
                if not (now in dic_search.keys()):
                    dic_search[now] = [j]
                else:
                    for k, sam in enumerate(dic_search[now]):
                        if k > 4:  # If the dictionary is too long, only take the last five to reduce the amount of computation
                            break
                        yz = (j + count_normal * nodes, sam + count_normal * nodes, 1)
                        normalset.append(list(yz))
                    dic_search[now].insert(0, j)
                last = now
            count_normal += 1

        if count_attack == batchsize:
            write_path = attack_path + str(attack_num) + '.csv'
            for rr in attackset:
                write_csv(write_path, 'at', rr)
            count_attack = 0
            attackset = []
            attack_num += 1
        if count_normal == batchsize:
            write_path = normal_path + str(normal_num) + '.csv'
            for rr in normalset:
                write_csv(write_path, 'at', rr)
            count_normal = 0
            normalset = []
            normal_num += 1

        step += 1


    print('-------load over {} ,num_attack= {},num_normal={},total num = {}'.format(filepath, attack_num, normal_num,
                                                                             len(label) / batchsize))
    print('       TAG= {}, path for data generation:{}'.format(TAG,path))


    '''move part of train/TAG/ files into val/TAG/ and test/TAG/'''
    path = '../../Dataset50_40/train/edges/'  # Enter the storage folder address
    orignpath = path + str(TAG) + '/'
    files = os.listdir(orignpath)  # Read the folder
    num_png = len(files)  # Count the number of files in a folder
    lenval = int(num_png * 0.7)
    lentest = int(num_png * 0.9)
    print('       location of the split data: {},{},{}'.format(lenval, lentest,num_png))   # Print the location of the split data

    v_goal = path[:-12] + "val/edges/" + str(TAG) + "/"
    t_goal = path[:-12] + "test/edges/" + str(TAG) + "/"

    if not os.path.exists(v_goal):
        os.makedirs(v_goal)
    if not os.path.exists(t_goal):
        os.makedirs(t_goal)

    for i in range(lenval, lentest):
        shutil.move(path + str(TAG) + "/" + str(i) + ".csv", v_goal)

    for i in range(lentest, num_png):
        shutil.move(path + str(TAG) + "/" + str(i) + ".csv", t_goal)

    vfiles = os.listdir(v_goal)  # Read the folder
    num_pngv = len(vfiles)  # Print the number of files
    tfiles = os.listdir(t_goal)  # Read the folder
    num_pngt = len(tfiles)  # Print the number of files

    print('       |val set|: {}, |test set|: {}'.format(num_pngv,num_pngt))


    '''remove files from train/0_TAG/ to test/(TAG+4)/'''
    path2 = '../../Dataset50_40/train/edges/0_'  # Enter the storage folder address
    orignpath2 = path2 + str(TAG) + '/'
    files2 = os.listdir(orignpath2)  # Read the folder
    t_goal = path2[:-14] + "test/edges/" + str(TAG+4) + '/'

    if os.path.exists(t_goal):
        for file in files2:
            shutil.move(orignpath2+file, t_goal+file)
    else:
        os.rename(orignpath2, t_goal)
    print('       movement of 0_TAG: {}'.format(t_goal))
