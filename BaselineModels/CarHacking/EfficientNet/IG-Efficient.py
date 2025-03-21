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
    num_classes = 5

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using {device} device.")

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))  # get data root path
    image_path = os.path.join(data_root, "data9_9_3-each27")  # flower data set path
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
    model_weight_path = "./EfficientNet.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.eval()

    startdic = {1:36186,2:42992,3:63081,4:68973}
    rangedic = {1:4021,2:4777,3:7009,4:7664}
    range0dic = {1:95562,2:94411,3:94470,4:94537}

    trydic = {0:torch.full((27,), 0),1:torch.full((27,), 1),2:torch.full((27,), 2),3:torch.full((27,), 3),4:torch.full((27,), 4),5:torch.full((27,), 5)}
    test_acc = torchmetrics.Accuracy(task = 'multiclass',num_classes = num_classes)
    b = torch.full((27,), 0)
    for tag in range(1,5):
        # print(tag)
        path = '../../data9_9_3-each27/test/0/' +str(tag) + '_'
        count = 0
        data0 = torch.full((1,3,9,9), 0)
        for i in range(1,range0dic[tag]+1):
            count += 1
            if count < batch_size:
                m = i + startdic[tag]
                tpath=os.path.join(r'../../data9_9_3-each27/test/0/' +str(tag) + '_' + str(i) +'.png')
                fopen = Image.open(tpath)
                transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                data=transform(fopen)
                data =data.unsqueeze(0)
                data0 = torch.cat([data0, data], 0)
            else:
                count = 0
                output = net(data0[1:]) #放入模型进行测试
                predict_y = torch.max(output, dim=1)[1]
                for step,gr in enumerate(predict_y):
                    test_acc.update(trydic[gr.item()], b)
                data0 = torch.full((1,3,9,9), 0)


    for tag in range(1,5):
        Alabel = pd.read_csv("../../data9_9_3-each27/IG" + str(tag) + ".csv",header = None)
        Alabel = Alabel[2:]
        A = Alabel.to_numpy()
        A = torch.tensor(A)
        # print(tag+4)

        path = '../data9_9_3-each27/test/' +str(tag) + '/'
        count = 0
        data0 = torch.full((1,3,9,9), 0)

        for i in range(rangedic[tag]):
            count += 1

            if count < batch_size:
                m = i + startdic[tag]
                tpath=os.path.join(r'../../data9_9_3-each27/test/' +str(tag) + '/'+ str(m) +'.png')
                fopen = Image.open(tpath)
                transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                data=transform(fopen)
                data =data.unsqueeze(0)
                data0 = torch.cat([data0, data], 0)
            else:
                count = 0
                mydata = data0[1:]
                data0 = torch.full((1,3,9,9), 0)
                output = net(mydata)
                predict_y = torch.max(output, dim=1)[1]
                for step,gr in enumerate(predict_y):

                    test_acc.update(trydic[gr.item()], A[i-31+step])


    total_acc = test_acc.compute()

    print("torch RG:", total_acc)

    test_acc.reset()


if __name__ == '__main__':
    main()