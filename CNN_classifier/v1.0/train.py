import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataloader import *


class MNISTNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        # 28 x 28
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 3, stride= 1, padding= 1)
        self.batchnorm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True)
        self.relu1 = nn.ELU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride= 1, padding= 1)
        self.batchnorm2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True)
        self.relu2 = nn.ELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.dropout = nn.Dropout2d(0.25)


        # 14 x 14
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride= 1, padding= 1)
        self.batchnorm3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
        self.relu3 = nn.ELU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride= 1, padding= 1)
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True)
        self.relu4 = nn.ELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        self.dropout = nn.Dropout2d(0.25)


        # 7 x 7
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 1, stride= 1)
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
        self.relu5 = nn.ELU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size = 1, stride= 1)
        self.batchnorm6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True)
        self.relu6 = nn.ELU()
        self.dropout = nn.Dropout2d(0.25)


        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        res = self.conv1(x)
        res = self.batchnorm1(res)
        res = self.relu1(res)
        res = self.conv2(res)
        res = self.batchnorm2(res)
        res = self.relu2(res)
        res = self.maxpool1(res)

        res = self.conv3(res)
        res = self.batchnorm3(res)
        res = self.relu3(res)
        res = self.conv4(res)
        res = self.batchnorm4(res)
        res = self.relu4(res)
        res = self.maxpool2(res)

        res = self.conv5(res)
        res = self.batchnorm5(res)
        res = self.relu5(res)
        res = self.conv6(res)
        res = self.dropout(res)
        res = self.batchnorm6(res)
        res = self.relu6(res)

        res = res.view(res.size(0), -1)
        res = self.fc1(res)
        res = self.fc2(res)
        # print res.size(), res.type()
        # res = F.softmax(res)
        return res

def init_weights(m):
    if type(m) == nn.Conv2d :
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))

def train(resume,train):
    learning_rate = 1e-3
    epoches = 20
    batch_size = 2000

    train_data = dataset(test=False)

    test_data = dataset(test=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    if resume:
        print "resume"
        model = torch.load("MNISTNet_iter120.pt")
        # model = torch.load("MNISTNet_iter60.pt")

    else:
        model = MNISTNet(num_classes=10)
        model.apply(init_weights)

    model.cuda()

    criterian = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    running_loss_list = []

    res_list = []

    if train:
        print("start training")

        for i in range(epoches):

            running_loss = 0.
            running_acc = 0.
            exp_lr_scheduler.step()

            start_time = time.time()
            count = 0
            for data in train_loader:
                count +=1

                img , label = data
                img = Variable(img).view(batch_size,1,img.shape[1],img.shape[2]).float().cuda()
                label = Variable(label).cuda()
                
                optimizer.zero_grad()
                output = model(img)

                # print label.type(), output.type()

                loss = criterian(output,label)

                loss.backward()
                optimizer.step()

                running_loss += loss.to(torch.device("cpu")).item()

                if count%100 == 0 :
                    print time.strftime("%H:%M:%S"),'loss', loss.to(torch.device("cpu")).item() , 'count' , count*batch_size
            running_loss /= len(train_data)
            print np.floor(time.time() - start_time)
            print("[%d/%d] Loss: %.5f" % (i + 1, epoches, running_loss))

    #     running_loss_list.append(running_loss)
        torch.save(model, 'MNISTNet_iter' + str(i+1) + '.pt')

    # file = open("loss_record.txt","w")
    # for l in running_loss_list:
    #     file.write("%s\n" % l)
    
    # file.close()

        

    # else:
        sub = pd.read_csv("sample_submission.csv")
        print("testing")
        model.eval()
        print "result",sub.shape
        batch_size = test_loader.batch_size

        for idx, data in enumerate(test_loader):
            img = data

            img = Variable(img).view(batch_size,1,img.shape[1],img.shape[2]).float().cuda()

            output = model(img)

            res = output.to(torch.device("cpu")).argmax()

            res = res.numpy()
            # res_list.append(res)
            sub.iloc[idx, 1] = res
        sub.to_csv("sample_submission.csv",index=False)


if __name__ == '__main__':
    print(torch.__version__)
    train(resume = False, train = True)
