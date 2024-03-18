'''' about ROAD from 0-1 to .csv'''
import numpy as np
import pandas as pd
import os
import csv

def write_csv(filepath, way, row):
    with open(filepath, way, encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def dealDatafield(line):  #line = lines[2][-21:-1].split('#',1)
    line = line.split('#',1)
    ID = int(line[0],16)
    data = []
    data.append(int(line[1][0:2],16))
    data.append(int(line[1][2:4], 16))
    data.append(int(line[1][4:6], 16))
    data.append(int(line[1][6:8], 16))
    data.append(int(line[1][8:10], 16))
    data.append(int(line[1][10:12], 16))
    data.append(int(line[1][12:14], 16))
    data.append(int(line[1][14:], 16))
    # print(data)
    data.insert(0,ID)
    return data

txt_path = '../../Dataset/ROAD/raw ROAD/txt/attacks/'
csv_path = '../../Dataset/ROAD/raw ROAD/csv/attacks/'

ChooseFile =  ['','correlated_signal_attack_1_masquerade', 'max_speedometer_attack_1_masquerade','reverse_light_off_attack_1_masquerade','reverse_light_on_attack_1_masquerade','max_engine_coolant_temp_attack_masquerade']
banben = {1: "train/", 2: "val/", 3: "test/"}
t = 2
tag = 1
for tag in range(1,6):
    for t in range(1,4):
        if tag == 5:
            choosedata = ChooseFile[tag]
        elif t == 1:
            choosedata = ChooseFile[tag]
        else:
            choosedata = ChooseFile[tag]
            index = choosedata.index('1')
            choosedata = choosedata[:index] + str(t) + choosedata[index+1:]

        # print('choosedata',choosedata)
        '''load .txt file'''
        f = open(txt_path  + choosedata + ".log",'r')
        lines = f.readlines()
        f.close()

        finaldata = []
        pd.set_option('display.precision', 5)
        Datacsv = pd.read_csv(csv_path  + choosedata + ".csv")

        print('------Start load------\n',choosedata)

        j = 0
        for index, row in Datacsv.iterrows():
            txtdata = dealDatafield(lines[j][-21:])

            standard = j-index
            cord = 0
            while(float(format(row["Time"], '.6f'))  != float(lines[j][4:18]) or txtdata[0] != int(row["ID"])):
                j += 1
                txtdata = dealDatafield(lines[j][-21:])
                cord += 1
                if cord > 10:
                    j = index + standard
                    break

            if float(format(row["Time"], '.6f'))  == float(lines[j][4:18]) and txtdata[0] == int(row["ID"]):
                txtdata.append(row['Label'])
                finaldata.append(txtdata)
            j += 1

        print('------load over-----')

        finaldata.insert(0,['ID','Data[0]','Data[1]','Data[2]','Data[3]','Data[4]','Data[5]','Data[6]','Data[7]','Label'])
        write_path = '../../Dataset/ROAD/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        for rr in finaldata:
            write_csv(write_path+'Deal_' + choosedata + ".csv", 'at', rr)
        print('------write over-----')
        if tag == 5:
            break
