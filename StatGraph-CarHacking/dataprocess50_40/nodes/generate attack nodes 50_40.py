''' node vectors generation'''
'''one node vector = 1（ID）+ 8(data field)+  3（Number of edges, maximum degree, number of nodes）'''
import csv
import numpy as np
import time
import os
import shutil


def hex_to_int(lis):  # data cleaning function
    data = []
    # lis1 = lis
    # print('lis1',lis1)
    # while('' in lis1):
    #     lis = lis1.remove('')

    if len(lis) == 8:
        for row in lis:
            row = int(row, 16)
            data.append(row)
        return data
    elif len(lis) == 5:
        for j in range(3):
            lis.append('-1')
    elif len(lis) == 2:
        for j in range(6):
            lis.append('-1')
    else:
        for j in range(8 - len(lis)):
            lis.append('255')
    for row in lis:
        row = int(row, 16)
        data.append(row)
    return data


def write_csv(filepath, way, row): # writing fuction
    if filepath is None:
        filepath = "preprocess_well_origin.csv"
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


class Graph():
    def __init__(self, num_of_nodes, N= 50, directed=True):
        self.num_of_nodes = num_of_nodes  # Number of nodes
        self.directed = directed
        self.list_of_edges = []  # List of edges

        self.edge_matrix = np.zeros((N, N))  # Adjacent matrix
        self.weight_matrix = np.zeros((N, N))  # Weight matrix

        self.adjacency_list = {node: set() for node in range(num_of_nodes)}

    def add_node(self):

        self.num_of_nodes += 1

    def add_edge(self, node1, node2, weight):
        if self.edge_matrix[node1][node2]:  # If node1 and node2 are connected
            self.weight_matrix[node1][node2] += weight
            self.adjacency_list[node1] = [node1, node2, self.adjacency_list[node1][2] + weight]
        else:  # If node1 and node2 are not connected
            self.edge_matrix[node1][node2] = 1
            self.weight_matrix[node1][node2] = weight
            self.adjacency_list[node1] = [node1, node2, weight]

    def record(self):  # Number of edges, maximum degree, number of nodes
        rec = []
        rec.append(np.sum(self.edge_matrix))
        rec.append(np.max((self.weight_matrix)))
        rec.append(self.num_of_nodes)
        return rec


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
div = 1
'''Slice:  divide data into the training set, the validation set and the test set
 train      div = 1
 val        div = 2
 test       div = 3
'''
batch_size = 40
nodes = 50

for TAG in range(1,5):
    filepath = '../../../Dataset/Car Hacking Dataset/'+ file_path_list[TAG]  +'.csv'
    path = '../../Dataset50_40/train/nodes/'

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
        line.append(row[1])  # 根据ID构建图
        labeline.append(row[-1])

    label = []
    for i in range(len(labelset)):
        if 'T' in labelset[i]:
            label.append(1)  # Indicates that this graph contains abnormal data
        else:
            label.append(0)

    actual_label = np.array(labelset)
    actual_label = actual_label.flatten()

    chakan = []
    dic_search = {'': 0}  # Create a dictionary
    node_dataset = []
    att_dataset = []
    buchong = []
    step = 0
    j = 0
    for row in dataset:  # Generate timing correlation graphs and extract the graph attributes
        i = 0
        graph = Graph(0, nodes)
        dic_search.clear()
        for now in row:
            if i == 0:
                i = 1
                last = now
                continue
            if not (last in dic_search.keys()):
                dic_search[last] = len(dic_search)
                graph.add_node()
            if not (now in dic_search.keys()):
                dic_search[now] = len(dic_search)
                graph.add_node()

            graph.add_edge(dic_search[now], dic_search[last], 1)
            last = now

        buchong.append(graph.record())  # Record graph attribute
        step += 1

    attack_path = path + str(TAG) + '/'
    normal_path = path +'0_'+ str(TAG) + '/'
    if not os.path.exists(attack_path):
        os.makedirs(attack_path)
    if not os.path.exists(normal_path):
        os.makedirs(normal_path)

    normalset = [];
    attackset = []
    attack_num = 0;
    normal_num = 0
    count_attack = 0;
    count_normal = 0
    dic_label = {'T': TAG,'R':0}

    tt = []
    csvreader = csv.reader(open(filepath, encoding='utf-8')) # Iterate again, recording the ID and payload value of the point
    for step, row in enumerate(csvreader):
        while ('' in row):
            row.remove('')
        if (step + 1) > (len(buchong) * nodes):
            # print('step', step)
            break
        if label[int(step / nodes)] != 0:
            tt =  hex_to_int(row[3:-1])  # Add payload
            tt.insert(0, int(row[1], 16))  # Add ID
            tt.extend(buchong[int(step  /nodes )])  # Add graph attributes
            tt.append(dic_label[actual_label[step]])
            attackset.append(tt)

        else:
            tt = hex_to_int(row[3:-1])
            tt.insert(0, int(row[1], 16))
            tt.extend(buchong[int(step  / nodes )])
            tt.append(0)
            normalset.append(tt)

        if (step + 1) % nodes == 0:
            if label[int(step / nodes)] != 0:
                count_attack += 1
            else:
                count_normal += 1

        if count_attack == batch_size:
            write_path = attack_path + str(attack_num) + '.csv'
            for rr in attackset:
                write_csv(write_path, 'at', rr)
            count_attack = 0
            attackset = []
            attack_num += 1
        if count_normal == batch_size:
            write_path = normal_path + str(normal_num) + '.csv'
            for rr in normalset:
                write_csv(write_path, 'at', rr)
            count_normal = 0
            normalset = []
            normal_num += 1

    print('-------load over {} ,num_attack= {},num_normal={},total num = {}'.format(filepath, attack_num, normal_num,
                                                                             len(label) / batch_size))
    print('       TAG= {}, path for data generation:{}'.format(TAG,path))


    '''move part of train/TAG/ files into val/TAG/ and test/TAG/'''
    path = '../../Dataset50_40/train/nodes/'  # Enter the storage folder address
    orignpath = path + str(TAG) + '/'
    files = os.listdir(orignpath)  # Read the folder
    num_png = len(files)  # Count the number of files in a folder
    lenval = int(num_png * 0.7)
    lentest = int(num_png * 0.9)
    print('       location of the split data: {},{},{}'.format(lenval, lentest,num_png))   # Print the location of the split data

    v_goal = path[:-12] + "val/nodes/" + str(TAG) + "/"
    t_goal = path[:-12] + "test/nodes/" + str(TAG) + "/"

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
    path2 = '../../Dataset50_40/train/nodes/0_'  # Enter the storage folder address
    orignpath2 = path2 + str(TAG) + '/'
    files2 = os.listdir(orignpath2)  # Read the folder
    t_goal = path2[:-14] + "test/nodes/" + str(TAG+4) + '/'

    if os.path.exists(t_goal):
        for file in files2:
            shutil.move(orignpath2+file, t_goal+file)
    else:
        os.rename(orignpath2, t_goal)
    print('       movement of 0_TAG: {}'.format(t_goal))
