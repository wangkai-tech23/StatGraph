'''' deal processed ROAD data to fig'''
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

file_path = '../../Dataset/ROAD/Deal_correlated_signal_attack_3_masquerade.csv'  # 3drive gear  4 RPM

ChooseFile =  ['','correlated_signal_attack_1_masquerade.csv', 'max_speedometer_attack_1_masquerade.csv','reverse_light_off_attack_1_masquerade.csv','reverse_light_on_attack_1_masquerade.csv','max_engine_coolant_temp_attack_masquerade.csv']
banben = {1: "train/", 2: "val/", 3: "test/"}
t = 1
tag = 5
for tag in range(1,6):
    for t in range(1,4):
        if tag == 5:
            choose_path = ChooseFile[tag]
        elif t == 1:
            choose_path = ChooseFile[tag]
        else:
            choose_path = ChooseFile[tag]
            index = choose_path.index('1')
            choose_path = choose_path[:index] + str(t) + choose_path[index + 1:]



        file_path = '../../Dataset/ROAD/Deal_' + choose_path
        print('------start deal:', file_path)
        df = pd.read_csv(file_path)
        numeric_features = df.dtypes[df.dtypes != 'object'].index
        scaler = QuantileTransformer()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        df[numeric_features] = df[numeric_features].apply(lambda x: (x * 255))

        count = 0
        ims = []
        label = 0
        count_attack = 0
        count_normal = 0
        label_list = []
        test_attack_record = []
        labeldic = {255: tag, 0.0: 0}

        image_path_attack = '../ROAD9_9_3-each27/' + banben[t] + str(tag) + "/"
        image_path_normal = '../ROAD9_9_3-each27/' + banben[t] + "0/"

        if not os.path.exists(image_path_attack):
            os.makedirs(image_path_attack)
        if not os.path.exists(image_path_normal):
            os.makedirs(image_path_normal)

        for i in range(0, len(df)):

            count = count + 1
            if count < 27:
                label_list.append(df.loc[i, 'Label'])
                if df.loc[i, 'Label'] == 255:
                    label = 1
                im = df.iloc[i].drop(['Label']).values
                ims = np.append(ims, im)
            else:
                label_list.append(df.loc[i, 'Label'])
                if df.loc[i, 'Label'] == 255:
                    label = 1
                im = df.iloc[i].drop(['Label']).values
                ims = np.append(ims, im)
                ims = np.array(ims).reshape(9, 9, 3)
                array = np.array(ims, dtype=np.uint8)
                new_image = Image.fromarray(array)
                if label == 1:
                    count_attack += 1
                    test_attack_record.append(label_list)
                    new_image.save(image_path_attack + str(count_attack) + '.png')
                else:
                    count_normal += 1
                    new_image.save(image_path_normal + str(tag) + '_' + str(count_normal) + '.png')

                ims = []
                count = 0
                label = 0
                label_list = []

        print('tag',tag, 'type',banben[t],'already deal:', file_path)
        print('tag',tag, 'type',banben[t],'count_attack', count_attack)
        print('tag',tag, 'type',banben[t],'count_normal', count_normal)


        test_attack_record = np.array(test_attack_record)
        DFattack_record = pd.DataFrame(test_attack_record)
        if t == 3:
            if os.path.exists("../ROAD9_9_3-each27/IG" + str(tag) + ".csv"):
                os.remove("../ROAD9_9_3-each27/IG" + str(tag) + ".csv")
            DFattack_record.to_csv("../ROAD9_9_3-each27/IG" + str(tag) + ".csv", index=False)
        if tag == 5:
            break

    print('------deal over------')

if tag == 5:
    t = 1
    image_path_attack = '../ROAD9_9_3-each27/' + banben[t] + str(tag) + "/"
    image_path_normal = '../ROAD9_9_3-each27/' + banben[t] + "0/"
    path = [ image_path_attack  ,image_path_normal]
    for j in range(2):
        vpath = ['../ROAD9_9_3-each27/val/' + str(tag) + '/','../ROAD9_9_3-each27/val/0/']
        tpath = ['../ROAD9_9_3-each27/test/' + str(tag) + '/','../ROAD9_9_3-each27/test/0/']
        if not os.path.exists(vpath[j]):
            os.makedirs(vpath[j])
        if not os.path.exists(tpath[j]):
            os.makedirs(tpath[j])

        files = os.listdir(path[j])  # read the folder
        num_png = len(files)  # Count the number of files in a folder
        if j == 1:
            num_png = 2073
        print(j, 'raw train num', num_png)  # print the number of files

        lenval = int(num_png * 0.7)
        lentest = int(num_png * 0.9)
        print(j,vpath[j],tpath[j],lenval, lentest)

        for k in range(lenval, lentest+1):
            if j == 0:
                shutil.move(path[j] + str(k) + ".png", vpath[j])
            else:
                shutil.move(path[j] + str(tag)+ '_' +str(k) + ".png", vpath[j])

        for k in range(lentest+1, num_png+1):
            # shutil.move(path[j] + str(k) + ".png", tpath[j])
            if j == 0:
                shutil.move(path[j] + str(k) + ".png", tpath[j])
            else:
                shutil.move(path[j] + str(tag)+ '_' +str(k) + ".png", tpath[j])


        vfiles = os.listdir(vpath[j])  # read the folder
        num_pngv = len(vfiles)  # Count the number of files in a folder
        print('j:', j, 'v num:', num_pngv)

        tfiles = os.listdir(tpath[j])  # read the folder
        num_pngt = len(tfiles)  # Count the number of files in a folder
        print('j:', j, 't num:', num_pngt)  # print the number of files
        print('-----------------------')

        DFattack_record = pd.DataFrame(test_attack_record[lentest:])
        if os.path.exists("../ROAD9_9_3-each27/IG" + str(tag) + ".csv"):
            os.remove("../ROAD9_9_3-each27/IG" + str(tag) + ".csv")
        DFattack_record.to_csv("../ROAD9_9_3-each27/IG" + str(tag) + ".csv", index=False)


#read IG*.csv code:
    # Alabel = pd.read_csv("../ROAD9_9_3-each27/IG" + str(tag) + ".csv", header=None)
    # Alabel = Alabel[1:]
    # A = Alabel.to_numpy()
    # print('A.shape', A.shape)


# ------start deal: ../../Dataset/ROAD/Deal_correlated_signal_attack_1_masquerade.csv
# tag 1 type train/ already deal: ../../Dataset/ROAD/Deal_correlated_signal_attack_1_masquerade.csv
# tag 1 type train/ count_attack 1511
# tag 1 type train/ count_normal 1200
# ------start deal: ../../Dataset/ROAD/Deal_correlated_signal_attack_2_masquerade.csv
# tag 1 type val/ already deal: ../../Dataset/ROAD/Deal_correlated_signal_attack_2_masquerade.csv
# tag 1 type val/ count_attack 1550
# tag 1 type val/ count_normal 770
# ------start deal: ../../Dataset/ROAD/Deal_correlated_signal_attack_3_masquerade.csv
# tag 1 type test/ already deal: ../../Dataset/ROAD/Deal_correlated_signal_attack_3_masquerade.csv
# tag 1 type test/ count_attack 986
# tag 1 type test/ count_normal 405
# ------deal over------
# ------start deal: ../../Dataset/ROAD/Deal_max_speedometer_attack_1_masquerade.csv
# tag 2 type train/ already deal: ../../Dataset/ROAD/Deal_max_speedometer_attack_1_masquerade.csv
# tag 2 type train/ count_attack 1829
# tag 2 type train/ count_normal 5403
# ------start deal: ../../Dataset/ROAD/Deal_max_speedometer_attack_2_masquerade.csv
# tag 2 type val/ already deal: ../../Dataset/ROAD/Deal_max_speedometer_attack_2_masquerade.csv
# tag 2 type val/ count_attack 2331
# tag 2 type val/ count_normal 2571
# ------start deal: ../../Dataset/ROAD/Deal_max_speedometer_attack_3_masquerade.csv
# tag 2 type test/ already deal: ../../Dataset/ROAD/Deal_max_speedometer_attack_3_masquerade.csv
# tag 2 type test/ count_attack 4809
# tag 2 type test/ count_normal 2308
# ------deal over------
# ------start deal: ../../Dataset/ROAD/Deal_reverse_light_off_attack_1_masquerade.csv
# tag 3 type train/ already deal: ../../Dataset/ROAD/Deal_reverse_light_off_attack_1_masquerade.csv
# tag 3 type train/ count_attack 525
# tag 3 type train/ count_normal 1781
# ------start deal: ../../Dataset/ROAD/Deal_reverse_light_off_attack_2_masquerade.csv
# tag 3 type val/ already deal: ../../Dataset/ROAD/Deal_reverse_light_off_attack_2_masquerade.csv
# tag 3 type val/ count_attack 1786
# tag 3 type val/ count_normal 1548
# ------start deal: ../../Dataset/ROAD/Deal_reverse_light_off_attack_3_masquerade.csv
# tag 3 type test/ already deal: ../../Dataset/ROAD/Deal_reverse_light_off_attack_3_masquerade.csv
# tag 3 type test/ count_attack 1803
# tag 3 type test/ count_normal 2936
# ------deal over------
# ------start deal: ../../Dataset/ROAD/Deal_reverse_light_on_attack_1_masquerade.csv
# tag 4 type train/ already deal: ../../Dataset/ROAD/Deal_reverse_light_on_attack_1_masquerade.csv
# tag 4 type train/ count_attack 1539
# tag 4 type train/ count_normal 2956
# ------start deal: ../../Dataset/ROAD/Deal_reverse_light_on_attack_2_masquerade.csv
# tag 4 type val/ already deal: ../../Dataset/ROAD/Deal_reverse_light_on_attack_2_masquerade.csv
# tag 4 type val/ count_attack 2907
# tag 4 type val/ count_normal 3002
# ------start deal: ../../Dataset/ROAD/Deal_reverse_light_on_attack_3_masquerade.csv
# tag 4 type test/ already deal: ../../Dataset/ROAD/Deal_reverse_light_on_attack_3_masquerade.csv
# tag 4 type test/ count_attack 1872
# tag 4 type test/ count_normal 3396
# ------deal over------
# ------start deal: ../../Dataset/ROAD/Deal_max_engine_coolant_temp_attack_masquerade.csv
# tag 5 type train/ already deal: ../../Dataset/ROAD/Deal_max_engine_coolant_temp_attack_masquerade.csv
# tag 5 type train/ count_attack 43
# tag 5 type train/ count_normal 2073
# ------deal over------
# 0 raw train num 43
# 0 ../ROAD9_9_3-each27/val/5/ ../ROAD9_9_3-each27/test/5/ 30 38
# j: 0 v num: 9
# j: 0 t num: 5
# -----------------------
# 1 raw train num 13413
# 1 ../ROAD9_9_3-each27/val/0/ ../ROAD9_9_3-each27/test/0/ 1451 1865
# j: 1 v num: 8306
# j: 1 t num: 9253
# -----------------------