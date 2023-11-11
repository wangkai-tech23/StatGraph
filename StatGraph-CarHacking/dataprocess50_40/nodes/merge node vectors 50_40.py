import glob
import time
import csv
import os
import pandas as pd
from tqdm import tqdm
def mul_to_whole(path = "../../Dataset50_40/train/nodes/0/", df_list = [],tag = 'train'):
    tag = path[19:-9]
    ''' val/test_count: the number of files in each folder '''
    val_count = [395, 381, 453, 664, 726]
    test_count = [0, 490, 583, 854, 934, 0, 0, 0, 0]
    files = os.listdir(path)
    print('tag',tag)
    num_png = len(files)
    for i in range(num_png):
        if tag == 'val':
            df = pd.read_csv("{}{}.csv".format(path,str(val_count[int(path[-2])]+i)),encoding='utf-8',header=None)
            df_list.append(df)
        elif tag == 'test':
            df = pd.read_csv("{}{}.csv".format(path,str(test_count[int(path[-2])]+i)),encoding='utf-8',header=None)
            df_list.append(df)
        else:
            df = pd.read_csv("{}{}.csv".format(path,str(i)),encoding='utf-8',header=None)
            df_list.append(df)
    df = pd.concat(df_list)
    print("Loading Over",path)
    return df

''' train dealing'''
data = []
data = mul_to_whole(path = "../../Dataset50_40/train/nodes/0/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/train/nodes/train_0.csv",index=False)
print('0len(data)',len(data))


data = []
data = mul_to_whole(path = "../../Dataset50_40/train/nodes/1/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/train/nodes/train_1.csv",index=False)
print('1len(data)',len(data))

data = []
data = mul_to_whole(path = "../../Dataset50_40/train/nodes/2/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/train/nodes/train_2.csv",index=False)
print('2len(data)',len(data))

data = []
data = mul_to_whole(path = "../../Dataset50_40/train/nodes/3/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/train/nodes/train_3.csv",index=False)
print('3len(data)',len(data))

data = []
data = mul_to_whole(path = "../../Dataset50_40/train/nodes/4/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/train/nodes/train_4.csv",index=False)
print('4len(data)',len(data))



''' val dealing'''
data = []
data = mul_to_whole(path = "../../Dataset50_40/val/nodes/0/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/val/nodes/val_0.csv",index=False)
print('0len(data)',len(data))


data = []
data = mul_to_whole(path = "../../Dataset50_40/val/nodes/1/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/val/nodes/val_1.csv",index=False)
print('1len(data)',len(data))

data = []
data = mul_to_whole(path = "../../Dataset50_40/val/nodes/2/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/val/nodes/val_2.csv",index=False)
print('2len(data)',len(data))

data = []
data = mul_to_whole(path = "../../Dataset50_40/val/nodes/3/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/val/nodes/val_3.csv",index=False)
print('3len(data)',len(data))

data = []
data = mul_to_whole(path = "../../Dataset50_40/val/nodes/4/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset50_40/val/nodes/val_4.csv",index=False)
print('4len(data)',len(data))



''' test dealing'''
data = []
data = mul_to_whole(path="../../Dataset50_40/test/nodes/1/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test/nodes/test_1.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset50_40/test/nodes/2/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test/nodes/test_2.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')
data = []
data = mul_to_whole(path="../../Dataset50_40/test/nodes/3/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test/nodes/test_3.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset50_40/test/nodes/4/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test/nodes/test_4.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset50_40/test/nodes/5/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test/nodes/test_5.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset50_40/test/nodes/6/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test/nodes/test_6.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset50_40/test/nodes/7/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test/nodes/test_7.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset50_40/test/nodes/8/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test/nodes/test_8.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')



import pandas as pd
from glob import glob

def hebing(path="../../Dataset50_40/train/nodes/", df_list=[], tag='test'):
    tag = path[19:-7]
    files = glob(r'{}*.csv'.format(path))  # Read the folder
    num_png = len(files)  # Count the number of files in a folder
    print('文件个数：', num_png)
    for i in range(num_png):
        df = pd.read_csv("{}{}_{}.csv".format(path, tag, str(i)), encoding='utf-8', header=None)  # load in order
        df = df[1:]
        df_list.append(df)
    df = pd.concat(df_list)
    print("Loading Over", path)
    return df

def hebing_test(path="../../Dataset50_40/train/nodes/", df_list=[], tag='test'):
    tag = path[19:-7]
    files = glob(r'{}*.csv'.format(path))  # Read the folder
    num_png = len(files)  # Count the number of files in a folder
    print('文件个数：', num_png)
    for i in range(1, num_png + 1):
        df = pd.read_csv("{}{}_{}.csv".format(path, tag, str(i)), encoding='utf-8', header=None)   # load in order
        df = df[1:]
        df_list.append(df)
    df = pd.concat(df_list)
    print("Loading Over", path)
    return df


data = []
data = hebing(path="../../Dataset50_40/train/nodes/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/train_nodes.csv", index=False)
print('train:len(data)', len(data))
print('---------------------')

data = []
data = hebing(path="../../Dataset50_40/val/nodes/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/val_nodes.csv", index=False)
print('val:len(data)', len(data))
print('---------------------')

data = []
data = hebing_test(path="../../Dataset50_40/test/nodes/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset50_40/test_nodes.csv", index=False)
print('test:len(data)', len(data))
print('---------------------')
