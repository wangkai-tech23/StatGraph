import os
import sys
import json

import torch
import torchvision
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
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

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    predict_dataset = datasets.ImageFolder(root=image_path,
                                           transform=transform)
    predict_num = len(predict_dataset)
    predict_loader = torch.utils.data.DataLoader(predict_dataset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=nw)
    print(predict_loader)

    net = CANet(input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, num_classes=num_classes, init_weights=False)
    model_weight_path = "./CANet.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))  #
    net.eval()

    test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5)
    test_recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=num_classes)
    test_precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=num_classes)
    test_f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=num_classes)
    test_auc = torchmetrics.AUROC(task='multiclass', average="macro", num_classes=num_classes)
    test_conf = MulticlassConfusionMatrix(num_classes=5)

    with torch.no_grad():
        predict_bar = tqdm(predict_loader, file=sys.stdout)
        for predict_data in predict_bar:
            predict_images, predict_labels = predict_data
            outputs = net(predict_images)
            predict_y = torch.max(outputs, dim=1)[1]
            test_acc.update(predict_y, predict_labels)
            test_auc.update(outputs, predict_labels)
            test_recall(predict_y, predict_labels)
            test_precision(predict_y, predict_labels)
            test_f1(predict_y, predict_labels)
            test_conf.update(predict_y, predict_labels)

    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auc = test_auc.compute()
    total_f1 = test_f1.compute()
    total_conf = test_conf.compute()

    print("torch metrics acc:", total_acc)
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1-score of every test dataset class: ", total_f1)
    print("auc:", total_auc.item())
    print('confusion matrix:', total_conf)

    # 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    test_auc.reset()


main()