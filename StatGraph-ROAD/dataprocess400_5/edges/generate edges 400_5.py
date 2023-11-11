''' edge(adjacent matrix) generation'''

import numpy as np
import csv
import os


def write_csv(filepath, way, row):   # writing fuction
    if filepath is None:
        filepath = "preprocess_well_origin.csv"
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

'''loading data'''

''' TAG: the number of abnormal data
 Deal_correlated_signal_attack         TAG = 1
 Deal_max_speedometer_attack           TAG = 2
 Deal_reverse_light_oFF_attack         TAG = 3
 Deal_reverse_light_on_attack          TAG = 4
 Deal_max_engine_coolant_temp_attack   TAG = 5
 '''
TAG = 1
file_path_list = ['','Deal_correlated_signal_attack_','Deal_max_speedometer_attack_','Deal_reverse_light_off_attack_','Deal_reverse_light_on_attack_','Deal_max_engine_coolant_temp_attack']
Slice = ['','train','val','test']
div = 1 # div \in [1,2,3]
'''Slice:  divide data into the training set, the validation set and the test set
 train      div = 1
 val        div = 2
 test       div = 3
'''
for TAG in range(1,6):
    for div in range(1,4):
        filepath = '../../../Dataset/ROAD/'+ file_path_list[TAG] + str(div) +'_masquerade.csv'
        path = '../../Dataset400_5/'+ Slice[div] +'/edges/'
        if TAG == 5:
            filepath = '../../../Dataset/ROAD/' + file_path_list[TAG] + '_masquerade.csv'
            if div == 2:
                break
        nodes = 400
        batchsize = 5

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
            line.append(row[1])  # Keep only the IDs to create the graph
            labeline.append(int(float(row[-1])))

        label = []
        for i in range(len(labelset)):
            if 1 in labelset[i]:
                label.append(1) # Indicates that this graph contains abnormal data
            else:
                label.append(0)


        ''' label: distinguish between normal data and abnormal data'''

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
                            if k > 4:    # If the dictionary is too long, only take the last five to reduce the amount of computation
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
                            if k > 4:    # If the dictionary is too long, only take the last five to reduce the amount of computation
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
                write_path = normal_path + str(TAG) + '_' + str(normal_num) + '.csv'
                for rr in normalset:
                    write_csv(write_path, 'at', rr)
                count_normal = 0
                normalset = []
                normal_num += 1
            step += 1

        print('------load over {} ,num_normal={},num_attack= {}'.format(filepath, normal_num, attack_num))
        print('TAG=', TAG, 'path=',path)

print('------load over {} ,num_normal={},num_attack= {}'.format(filepath, normal_num, attack_num))
print('TAG=', TAG, 'path=',path)



''' splice data with tag = 5 into train, val and test'''
import os
import shutil
tag = 5
'''deal data in train/0_5/'''
path = '../../Dataset400_5/train/edges/0_' # Enter the storage folder address
orignpath = path + str(tag) +'/'
files = os.listdir(orignpath)  # Read the folder
num_png = len(files)       # Count the number of files in a folder
lenval = int(num_png *0.7)
lentest = int(num_png *0.9)
print(lenval,lentest,num_png)   # Print the location of the split data

v_goal = path[:-14] + "val/edges/0_" + str(tag) + "/"
t_goal = path[:-14] + "test/edges/0_" + str(tag) + "/"

if not os.path.exists(v_goal):
    os.makedirs(v_goal)
if not os.path.exists(t_goal):
    os.makedirs(t_goal)

for i in range(lenval,lentest):
    shutil.move(orignpath + str(tag) + "_" +str(i)+".csv", v_goal)

for i in range(lentest,num_png):
    shutil.move(orignpath + str(tag) + "_" +str(i)+".csv", t_goal)

# # Check whether the data remove successfully
# vfiles = os.listdir(v_goal)   # Read the folder
# num_pngv = len(vfiles)        # Print the number of files
# print('v',num_pngv)
#
# tfiles = os.listdir(t_goal)   # Read the folder
# num_pngt = len(tfiles)        # Print the number of files
# print('t',num_pngt)

'''deal data in train/5/'''
path2 = '../../Dataset400_5/train/edges/' # Enter the storage folder address
orignpath2 = path2 + str(tag) +'/'
files = os.listdir(orignpath2)  # Read the folder
num_png = len(files)       # Count the number of files in a folder
lenval = int(num_png *0.7)
lentest = int(num_png *0.9)
print(lenval,lentest,num_png)   # Print the location of the split data

v_goal = path2[:-13] + "/val/edges/" + str(tag) + "/"
t_goal = path2[:-13] + "/test/edges/" + str(tag) + "/"
# print('v_goal',v_goal,'t_goal',t_goal)

if not os.path.exists(v_goal):
    os.makedirs(v_goal)
if not os.path.exists(t_goal):
    os.makedirs(t_goal)

for i in range(lenval,lentest):
    shutil.move(orignpath2 + str(i)+".csv", v_goal)

for i in range(lentest,num_png):
    shutil.move(orignpath2 + str(i)+".csv", t_goal)





''' next, remove document from test/0_TAG to test/0'''
import os
import shutil

Slice = ['','train','val','test']
print('----------------')
for div in range(1,4):
    movepath = '../../Dataset400_5/'+Slice[div]+'/edges/0_' # Enter the storage folder address
    goal = '../../Dataset400_5/'+Slice[div]+'/edges/0'
    print(' destination folder:',goal)

    if not os.path.exists(goal):
        os.makedirs(goal)

    for tag in range(1,6):
        orignpath = movepath + str(tag) +'/'
        files = os.listdir(orignpath)   # Read the folder
        num_png = len(files)            # Count the number of files in a folder
        print('TAG = ',tag,'number of files',num_png)                  # Print the number of files

        for file in files:
            shutil.move(orignpath +file, goal)

    tfiles = os.listdir(goal)   # Read the folder
    num_pngt = len(tfiles)      # Print the number of files
    print('total number in ',Slice[div],'/0/ :',num_pngt)