import os
import sys
import json
import torch
import pandas as pd
import torchvision
import torchmetrics
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
from torchsummary import summary
from EfficientNet import efficientnet_b0

def main():
    batch_size = 32
    num_classes = 6

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device.")

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))  # get data root path
    image_path = os.path.join(data_root, "ROAD9_9_3-each27")  # flower data set path
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    predict_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                           transform=transform)
    predict_num = len(predict_dataset)
    predict_loader = torch.utils.data.DataLoader(predict_dataset,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=nw)

    print(predict_loader)

    # create model
    net = efficientnet_b0(num_classes=num_classes)

    model_weight_path = "./EfficientNet_ROAD.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.eval()

    test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    test_conf = MulticlassConfusionMatrix(num_classes=num_classes)

    rangedic = {1: 986, 2: 4809, 3: 1803, 4: 1872, 5: 5}
    # range0dic = {1: 405, 2: 2308, 3: 2936, 4: 3396, 5: 208}

    trydic = {0: torch.full((27,), 0), 1: torch.full((27,), 1), 2: torch.full((27,), 2), 3: torch.full((27,), 3),
              4: torch.full((27,), 4), 5: torch.full((27,), 5)}
    test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
    b = torch.full((27,), 0)
    if True:
        data0 = torch.full((1, 3, 9, 9), 0)
        count = 0
        path = '../../ROAD9_9_3-each27/test/0/'
        file_names = os.listdir(path)
        file_list = []
        for name in file_names:
            file_list.append(name)
        for name in file_list:
            count += 1
            if count < batch_size:
                tpath = os.path.join(path, name)
                fopen = Image.open(tpath)
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                data = transform(fopen)
                data = data.unsqueeze(0)
                #                 print('data',data.shape)
                data0 = torch.cat([data0, data], 0)
            else:
                count = 0
                tpath = os.path.join(path,name)
                fopen = Image.open(tpath)
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                data = transform(fopen)
                data = data.unsqueeze(0)
                data0 = torch.cat([data0, data], 0)

                output = net(data0[1:])
                predict_y = torch.max(output, dim=1)[1]
                for step, gr in enumerate(predict_y):
                    if gr.item() == 0:
                        test_acc.update(trydic[gr.item()], b)
                    else:
                        test_acc.update(trydic[1], b)
                data0 = torch.full((1, 3, 9, 9), 0)

    # print('---------------------')
    for tag in range(1, 6):
        Alabel = pd.read_csv("../../ROAD9_9_3-each27/IG" + str(tag) + ".csv", header=None)
        Alabel = Alabel[1:]
        A = Alabel.to_numpy()
        A = torch.tensor(A)
        # print(tag)

        count = 0
        data0 = torch.full((1, 3, 9, 9), 0)

        for i in range(1, rangedic[tag]):
            count += 1
            if tag == 5:
                m = 38
            else:
                m = 0
            if count < batch_size:
                tpath = os.path.join(r'../../ROAD9_9_3-each27/test/' + str(tag) + '/' + str(i + m) + '.png')
                fopen = Image.open(tpath)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                data = transform(fopen)
                data = data.unsqueeze(0)
                data0 = torch.cat([data0, data], 0)
            else:
                tpath = os.path.join(r'../../ROAD9_9_3-each27/test/' + str(tag) + '/' + str( i + m) + '.png')
                fopen = Image.open(tpath)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                data = transform(fopen)
                data = data.unsqueeze(0)
                data0 = torch.cat([data0, data], 0)
                count = 0
                mydata = data0[1:]
                data0 = torch.full((1, 3, 9, 9), 0)
                output = net(mydata)
                predict_y = torch.max(output, dim=1)[1]
                for step, gr in enumerate(predict_y):
                    if gr.item() == tag:
                        test_acc.update(trydic[gr.item()], A[i - 31 + step])
                    else:
                        test_acc.update(trydic[1], b)

    total_acc = test_acc.compute()

    print("torch metrics acc:", total_acc)

    test_acc.reset()

if __name__ == '__main__':
    main()
