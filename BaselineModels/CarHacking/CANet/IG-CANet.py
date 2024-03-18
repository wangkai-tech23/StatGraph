import os
import sys
import json
import pandas as pd

import torch
import torchvision
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from CANet import CANet
from torchsummary import summary
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix

def main():

    batch_size = 32
    num_classes = 5
    input_size = 9
    hidden_size = 32
    num_layers = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device.")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../data9_9_3-each27"))  # get data root path
    image_path = os.path.join(data_root, "test")  # flower data set path

    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    net = CANet(input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, num_classes=num_classes, init_weights=False)
    model_weight_path = "./CANet.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))  #
    net.eval()
    startdic = {1: 36186, 2: 42992, 3: 63081, 4: 68973}
    rangedic = {1: 4021, 2: 4777, 3: 7009, 4: 7664}
    range0dic = {1: 95562, 2: 94411, 3: 94470, 4: 94537}

    trydic = {0: torch.full((27,), 0), 1: torch.full((27,), 1), 2: torch.full((27,), 2), 3: torch.full((27,), 3),
              4: torch.full((27,), 4)}
    test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
    b = torch.full((27,), 0)

    for tag in range(1, 5):
        # print(tag)
        path = '../../data9_9_3-each27/test/0/' + str(tag) + '_'

        for i in range(1, range0dic[tag] + 1):
            tpath = os.path.join(r'../../data9_9_3-each27/test/0/' + str(tag) + '_' + str(i) + '.png')
            fopen = Image.open(tpath)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            data = transform(fopen)  # Pre-processed 'data' it can be fed into the model for training
            output = net(data)
            predict_y = torch.max(output, dim=1)[1]
            if predict_y.item() == 0:
                test_acc.update(trydic[predict_y.item()], b)
            else:
                test_acc.update(trydic[1], b)


    for tag in range(1, 5):
        # print(tag+4)
        Alabel = pd.read_csv("../../data9_9_3-each27/IG" + str(tag) + ".csv", header=None)
        Alabel = Alabel[2:]
        A = Alabel.to_numpy()
        A = torch.tensor(A)

        # path = '../data9_9_3-each27/test/' + str(tag) + '/'

        for i in range(rangedic[tag]):  # rangedic[tag] is the number of pictures
            m = i + startdic[tag]
            tpath = os.path.join(
                r'../../data9_9_3-each27/test/' + str(tag) + '/' + str(m) + '.png')
            fopen = Image.open(tpath)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            data = transform(fopen)
            output = net(data)
            predict_y = torch.max(output, dim=1)[1]
            if predict_y.item() == tag:
                test_acc.update(trydic[predict_y.item()], A[i])
            else:
                test_acc.update(trydic[1], b)

    total_acc = test_acc.compute()
    print("torch IR:", total_acc)
    test_acc.reset()

if __name__ == '__main__':
    main()
