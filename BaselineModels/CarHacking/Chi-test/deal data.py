import csv
import os
import numpy as np
import time

class Graph():
    def __init__(self, num_of_nodes, N=200, directed=True):
        self.num_of_nodes = num_of_nodes
        self.directed = directed
        self.list_of_edges = []
        self.out_degree = []
        self.in_degree = []
        self.all_degree = []
        self.edge_matrix = np.zeros((N, N))
        self.weight_matrix = np.zeros((N, N))
        self.record = []  # 点数，边数，最大度，最大入度，最大出度

    def add_node(self):
        self.num_of_nodes += 1

    def add_edge(self, node1, node2, weight):
        if self.edge_matrix[node1][node2]:
            self.weight_matrix[node1][node2] += weight
        else:
            self.edge_matrix[node1][node2] = 1
            self.weight_matrix[node1][node2] = weight

    def record_degree(self):
        Dmatrix = np.array(self.edge_matrix)
        self.in_degree = np.sum(Dmatrix, axis=0)
        self.out_degree = np.sum(Dmatrix, axis=1)
        self.all_degree = self.in_degree + self.out_degree
        num_of_edges = np.sum(Dmatrix)
        self.record = [self.num_of_nodes, num_of_edges, np.max(self.all_degree), np.max(self.in_degree),
                       np.max(self.out_degree), np.max(self.weight_matrix)]

def write_csv(filepath, way, row):
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def generatedGraph(tag):
    filepath = ['normal_16_id.csv','DoS_Attack_dataset.csv','Fuzzy_Attack_dataset.csv','Spoofing_the_drive_gear_dataset.csv', 'Spoofing_the_RPM_gauge_dataset.csv' ]
    dataset = []
    labelset = []

    csvreader = csv.reader(open('../../../Dataset/Car Hacking Dataset/'+filepath[tag], encoding='utf-8'))
    line = []
    i = 0
    label = 0
    dic_search = {'': 0}
    graph_list = []
    if tag == 0:
        for i, row in enumerate(csvreader):
            if i == 0:
                i = 1
                continue
            if i % 200 == 0:
                dataset.append(line)
                line = []
            line.append(row[1])  # CAN: row[1]  ROAD: row[0]
        # print('len(dataset)', len(dataset))
        for row in dataset:  # Only ID sequence, every 200 IDs is a row
            i = 0
            graph = Graph(0, 200)
            dic_search.clear()
            for now in row:  # Build a graph for every 200 IDs
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

            graph.record_degree()
            graph_list.append(graph.record)


    else:
        csvreader = csv.reader(open('../../../Dataset/Car Hacking Dataset/'+filepath[tag], encoding='utf-8'))
        for i, row in enumerate(csvreader):
            if i == 0:
                i = 1
                continue
            if i % 200 == 0:
                if label == 1:
                    labelset.append('T')
                else:
                    labelset.append('R')
                dataset.append(line)
                line = []
                label = 0
            while( row[-1]== ''):
                row = row[:-1]
            line.append(row[1])  # CAN是row[1]  ROAD是row[0]
            if row[-1] == "T":
                label = 1
        for row in dataset:  # Only ID sequence, every 200 IDs is a row
            i = 0
            graph = Graph(0, 200)
            dic_search.clear()
            for now in row:  # Build a graph for every 200 IDs
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
            graph.record_degree()
            graph_list.append(graph.record)

    return graph_list, labelset




path = './CAN dealed' # Path for generate data
if not os.path.exists(path):
        os.makedirs(path)

for tag in range(5):
    graph_list, labelset = generatedGraph(tag)
    if os.path.exists(path + '/graph_list' + str(tag) + '.csv'):
        os.remove(path + '/graph_list' + str(tag) + '.csv')

    time_start = time.perf_counter()
    k = 0
    for i, row in enumerate(graph_list):
        if labelset != []:
            row.append(labelset[i])
        write_csv(path + '/graph_list' + str(tag) + '.csv', 'at', row)
        k = k + 1
    print('tag:',tag,'writing time %f s' % (time.perf_counter() - time_start),'len',k)

