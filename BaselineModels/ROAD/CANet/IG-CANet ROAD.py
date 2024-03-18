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
from CANet_ROAD import CANet

from torchsummary import summary
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix

def main():
    batch_size = 32
    num_classes = 6
    input_size = 9
    hidden_size = 32
    num_layers = 1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device.")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../ROAD9_9_3-each27"))  # get data root path
    image_path = os.path.join(data_root, "test")

    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    nw = 0#min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    net = CANet(input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, num_classes=num_classes, init_weights=False)
    model_weight_path = "./CANet_ROAD.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))  #
    net.eval()

    rangedic = {1: 986, 2: 4809, 3: 1803, 4: 1872, 5: 5}
    range0dic = {1: 405, 2: 2308, 3: 2936, 4: 3396, 5: 208}

    trydic = {0: torch.full((27,), 0), 1: torch.full((27,), 1), 2: torch.full((27,), 2), 3: torch.full((27,), 3),
              4: torch.full((27,), 4), 5: torch.full((27,), 5)}
    test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    b = torch.full((27,), 0)

    for tag in range(1, 6):
        # print(tag)
        if tag == 5:
            mm = 1865
        else:
            mm = 0

        for i in range(1, range0dic[tag] + 1):
            tpath = os.path.join(r'../../ROAD9_9_3-each27/test/0/' + str(tag) + '_' + str(i + mm) + '.png')
            fopen = Image.open(tpath)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            data = transform(fopen)
            output = net(data)
            predict_y = torch.max(output, dim=1)[1]
            if predict_y.item() == 0:
                test_acc.update(trydic[predict_y.item()], b)
            else:
                test_acc.update(trydic[1], trydic[3])

    # print('--------------')
    for tag in range(1, 6):
        Alabel = pd.read_csv("../../ROAD9_9_3-each27/IG" + str(tag) + ".csv", header=None)
        Alabel = Alabel[1:]
        A = Alabel.to_numpy()
        A = torch.tensor(A)/255 * tag
        # print(tag, 'A.shape', A.shape)

        for i in range(1, rangedic[tag]):
            if tag == 5:
                m = i + 38
            else:
                m = i
            tpath = os.path.join(r'../../ROAD9_9_3-each27/test/' + str(tag) + '/' + str(m) + '.png')
            fopen = Image.open(tpath)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            data = transform(fopen)
            output = net(data)
            predict_y = torch.max(output, dim=1)[1]

            if predict_y.item() == tag:
                test_acc.update(trydic[predict_y.item()], A[i])
            else:
                test_acc.update(trydic[2], trydic[3])

    total_acc = test_acc.compute()
    print("torch metrics acc:", total_acc)
    test_acc.reset()

if __name__ == '__main__':
    main()
