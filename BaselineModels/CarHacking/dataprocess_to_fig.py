''' preprocess normal Car Hacking Dataset '''
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

file_path = '../../Dataset/Car Hacking Dataset/PreprocessedData/normal_dataset.csv'
df = pd.read_csv(file_path)
# Transform all features into the scale of [0,1]
numeric_features = df.dtypes[df.dtypes != 'object'].index
scaler = QuantileTransformer()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Multiply the feature values by 255 to transform them into the scale of [0,255]
df[numeric_features] = df[numeric_features].apply(lambda x: (x * 255))
# # Generate 9*9 color images for class 0 (Normal)
# # Change the numbers 9 to the number of features n in your dataset if you use a different dataset, reshape(n,n,3)

count = 0
ims = []
label = 0
count_attack = 0
count_normal = 0

image_path_normal = "../data9_9_3-each27/train/0/"

if not os.path.exists(image_path_normal):
    os.makedirs(image_path_normal)

for i in range(0, len(df)):
    count = count + 1
    if count < 27:

        if df.loc[i, 'Label'] == 'T':
            label = 1
        im = df.iloc[i].drop(['Label']).values
        ims = np.append(ims, im)
    else:
        if df.loc[i, 'Label'] == 'T':
            label = 1
        im = df.iloc[i].drop(['Label']).values
        ims = np.append(ims, im)
        ims = np.array(ims).reshape(9, 9, 3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        if label == 1:
            count_attack += 1
            # new_image.save(image_path_attack+str(count_attack)+'.png')
        else:
            count_normal += 1
            new_image.save(image_path_normal + str(count_normal) + '.png')
        count = 0
        ims = []
        label = 0

print('count normal:', count_normal)

tag = 0
vpath = '../data9_9_3-each27/val/' + str(tag) + '/'
if not os.path.exists(vpath):
    os.makedirs(vpath)

path = '../data9_9_3-each27/train/' + str(tag) + '/'
files = os.listdir(path)  # Read the folder
num_png = len(files)  # Count the number of files in a folder
print('normal raw train num', num_png)  # print the number of files

lenval = int(num_png * 0.8)

print(lenval)
for i in range(lenval, num_png):
    shutil.move(path + str(i) + ".png", vpath)

vfiles = os.listdir(vpath)  # Read the folder
num_pngv = len(vfiles)  # Count the number of files in a folder
print('normal v num:', num_pngv)  #0-29298, 29299-36624(7325)

print('-----------------------')

# import numpy as np
# import pandas as pd
# import os
# import shutil
# from sklearn.preprocessing import QuantileTransformer
# from PIL import Image
# import warnings
# warnings.filterwarnings("ignore")

''' preprocess Car Hacking Datasets with attacks'''

file_path = ['','DoS_dataset.csv','Fuzzy_dataset.csv','Gear_dataset.csv', 'RPM_dataset.csv']
dic = {1:36186,2:42992,3:63081,4:68973}

for i in range(1,5):
    print('start deal:', file_path[i])

    # Read dataset
    df = pd.read_csv('../../Dataset/Car Hacking Dataset/PreprocessedData/'+file_path[i])

    # Transform all features into the scale of [0,1]
    numeric_features = df.dtypes[df.dtypes != 'object'].index
    scaler = QuantileTransformer()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Multiply the feature values by 255 to transform them into the scale of [0,255]
    df[numeric_features] = df[numeric_features].apply(lambda x: (x * 255))

    # # Generate 9*9 color images for class 0 (Normal)
    # # Change the numbers 9 to the number of features n in your dataset if you use a different dataset, reshape(n,n,3)

    count = 0
    ims = []
    label = 0
    count_attack = 0
    count_normal = 0


    tag = i
    image_path_attack = "../data9_9_3-each27/train/" + str(tag) + "/"
    image_path_normal = "../data9_9_3-each27/test/0/"

    label_list = []  ##############
    test_attack_record = []  ##############
    labeldic = {'T': tag, 'R': 0}

    if not os.path.exists(image_path_attack):
        os.makedirs(image_path_attack)
    if not os.path.exists(image_path_normal):
        os.makedirs(image_path_normal)



    for j in range(0, len(df)):
        count = count + 1
        if count < 27:
            label_list.append(labeldic[df.loc[j, 'Label']]) ##############
            if df.loc[j, 'Label'] == 'T':
                label = 1
            im = df.iloc[j].drop(['Label']).values
            ims = np.append(ims, im)

        else:
            label_list.append(labeldic[df.loc[j, 'Label']])   ##########
            if df.loc[j, 'Label'] == 'T':
                label = 1
            im = df.iloc[j].drop(['Label']).values
            ims = np.append(ims, im)
            ims = np.array(ims).reshape(9, 9, 3)
            array = np.array(ims, dtype=np.uint8)
            new_image = Image.fromarray(array)
            if label == 1:
                count_attack += 1
                if count_attack > dic[tag] - 1:   ##########   只要测试部分的
                    test_attack_record.append(label_list)   ##########
                new_image.save(image_path_attack + str(count_attack) + '.png')
            else:
                count_normal += 1
                new_image.save(image_path_normal + str(tag) + '_' + str(count_normal) + '.png')
            count = 0
            ims = []
            label = 0
            label_list = []  ##########

    print('i:',i,'already deal:', file_path[i])
    print('i:',i,'count_attack', count_attack)
    print('i:',i,'count_normal', count_normal)

    test_attack_record = np.array(test_attack_record)
    DFattack_record = pd.DataFrame(test_attack_record)
    DFattack_record.to_csv("../data9_9_3-each27/IG" + str(tag) + ".csv", index=False)


    vpath = '../data9_9_3-each27/val/' + str(tag) + '/'
    tpath = '../data9_9_3-each27/test/' + str(tag) + '/'
    if not os.path.exists(vpath):
        os.makedirs(vpath)
    if not os.path.exists(tpath):
        os.makedirs(tpath)

    path = image_path_attack  # = "../data9_9_3-each27/train/" +str(tag) + "/"
    files = os.listdir(path)  # read the folder
    num_png = len(files)  # Count the number of files in a folder
    print('i:',i,'raw train num', num_png)  # print the number of files

    lenval = int(num_png * 0.7)
    lentest = int(num_png * 0.9)
    print(lenval, lentest)
    for j in range(lenval, lentest):
        shutil.move(path + str(j) + ".png", vpath)

    for j in range(lentest, num_png):
        shutil.move(path + str(j) + ".png", tpath)

    vfiles = os.listdir(vpath)  # read the folder
    num_pngv = len(vfiles)  #  Count the number of files in a folder
    print('i:',i,'v num:', num_pngv)

    tfiles = os.listdir(tpath)  # read the folder
    num_pngt = len(tfiles)  #  Count the number of files in a folder
    print('i:',i,'t num:', num_pngt) # print the number of files
    print('-----------------------')





