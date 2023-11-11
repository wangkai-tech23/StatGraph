import glob
import time
import csv
import os
import pandas as pd
from tqdm import tqdm
def mul_to_whole(path = '../../Dataset400_5/train/nodes/0/', df_list = [],tag = 'train'):
    tag = path[19:-9]
    ''' train/val/test_count: the number of files in each folder '''
    train_count = [0, 13, 70, 23, 38, 16]
    val_count = [0, 7, 31, 18, 38, 4]  
    test_count = [0, 4, 28, 36, 44, 3]
    print('tag',tag)
    if path[-2] == '0':
        for j in range(1,6):
            start = 0
            if tag == 'train':
                num_png = train_count[j]
            elif tag == 'val':
                if j == 5:
                    start = 16
                num_png = val_count[j]
            elif tag == 'test':
                if j == 5:
                    start = 20
                num_png = test_count[j]

            for i in range(num_png):
                df = pd.read_csv("{}{}_{}.csv".format(path, str(j), str(start + i)), encoding='utf-8', header=None)
                df_list.append(df)
    else:
        files = os.listdir(path)  # Read the folder
        num_png = len(files)
        if tag == 'test' and path[-2] == '5':
            start = 3
        elif tag =='val' and path[-2] == '5':
            start = 2
        else:
            start = 0

        for i in range(num_png):
            df = pd.read_csv("{}{}.csv".format(path, str(start +i)),encoding='utf-8', header=None)
            df_list.append(df)

    df2 = pd.concat(df_list)
    print("Loading Over",path)
    return df2

'''train merge'''
data = []
data = mul_to_whole(path = "../../Dataset400_5/train/nodes/0/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/train/nodes/train_0.csv",index=False)
print('0len(data)',len(data))
print('--------------------')

data = []
data = mul_to_whole(path = "../../Dataset400_5/train/nodes/1/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/train/nodes/train_1.csv",index=False)
print('1len(data)',len(data))
print('--------------------')
#
data = []
data = mul_to_whole(path = "../../Dataset400_5/train/nodes/2/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/train/nodes/train_2.csv",index=False)
print('2len(data)',len(data))
print('--------------------')

data = []
data = mul_to_whole(path = "../../Dataset400_5/train/nodes/3/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/train/nodes/train_3.csv",index=False)
print('3len(data)',len(data))
print('--------------------')

data = []
data = mul_to_whole(path = "../../Dataset400_5/train/nodes/4/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/train/nodes/train_4.csv",index=False)
print('4len(data)',len(data))

data = []
data = mul_to_whole(path = "../../Dataset400_5/train/nodes/5/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/train/nodes/train_5.csv",index=False)
print('4len(data)',len(data))


'''val merge'''
data = []
data = mul_to_whole(path = "../../Dataset400_5/val/nodes/0/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/val/nodes/val_0.csv",index=False)
print('0len(data)',len(data))
print('--------------------')


data = []
data = mul_to_whole(path = "../../Dataset400_5/val/nodes/1/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/val/nodes/val_1.csv",index=False)
print('1len(data)',len(data))
print('--------------------')

data = []
data = mul_to_whole(path = "../../Dataset400_5/val/nodes/2/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/val/nodes/val_2.csv",index=False)
print('2len(data)',len(data))
print('--------------------')

data = []
data = mul_to_whole(path = "../../Dataset400_5/val/nodes/3/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/val/nodes/val_3.csv",index=False)
print('3len(data)',len(data))
print('--------------------')

data = []
data = mul_to_whole(path = "../../Dataset400_5/val/nodes/4/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/val/nodes/val_4.csv",index=False)
print('4len(data)',len(data))
print('--------------------')

data = []
data = mul_to_whole(path = "../../Dataset400_5/val/nodes/5/",df_list = data) #
data.to_csv(path_or_buf="../../Dataset400_5/val/nodes/val_5.csv",index=False)
print('4len(data)',len(data))
print('--------------------')


''' test merge '''
data = []
data = mul_to_whole(path="../../Dataset400_5/test/nodes/0/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/test/nodes/test_0.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset400_5/test/nodes/1/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/test/nodes/test_1.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset400_5/test/nodes/2/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/test/nodes/test_2.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')
data = []
data = mul_to_whole(path="../../Dataset400_5/test/nodes/3/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/test/nodes/test_3.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset400_5/test/nodes/4/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/test/nodes/test_4.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

data = []
data = mul_to_whole(path="../../Dataset400_5/test/nodes/5/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/test/nodes/test_5.csv", index=False)
print('len(data)', len(data))
print('-----------------------------')

import pandas as pd
from glob import glob

def hebing(path="../../Dataset400_5/train/nodes/", df_list=[], tag='test'):
    tag = path[19:-7]
    files = glob(r'{}*.csv'.format(path))   # Read the folder
    num_png = len(files)  # Count the number of files in a folder
    print('the number of files：', num_png)
    for i in range(num_png):
        df = pd.read_csv("{}{}_{}.csv".format(path, tag, str(i)), encoding='utf-8', header=None)  # load in order
        df = df[1:]
        df_list.append(df)
    df = pd.concat(df_list)
    print("Loading Over", path)
    return df

def hebing_singletest(path="../../Dataset400_5/train/nodes/", df_list=[], tag='test'):
    tag = path[19:-7]
    files = glob(r'{}*.csv'.format(path))    # Read the folder
    num_png = len(files)  # Count the number of files in a folder
    numList = [0,5]
    for i in numList:
        df = pd.read_csv("{}{}_{}.csv".format(path, tag, str(i)), encoding='utf-8', header=None)  # load in order
        df = df[1:]
        df_list.append(df)
    df = pd.concat(df_list)
    print("Loading Over", path)
    return df


data = []
data = hebing(path="../../Dataset400_5/train/nodes/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/train_nodes.csv", index=False)
print('train:len(data)', len(data))
print('---------------------')

data = []
data = hebing(path="../../Dataset400_5/val/nodes/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/val_nodes.csv", index=False)
print('val:len(data)', len(data))
print('---------------------')

data = []
data = hebing(path="../../Dataset400_5/test/nodes/", df_list=data)  #
data.to_csv(path_or_buf="../../Dataset400_5/test_nodes.csv", index=False)
print('train:len(data)', len(data))
print('---------------------')

