import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils import data


class dataset(data.Dataset):
    def __init__(self,test = False):
        self.test = test
        if not self.test:
            data = pd.read_csv("train.csv")
            self.data, self.label, self.length = convert_data(data, self.test)
        else:
            data = pd.read_csv("test.csv")
            self.data, self.length = convert_data(data, self.test)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not self.test:
            image = self.data[index].reshape((28,28))
            label = self.label[index]

            # plt.imshow(image)
            # plt.show()
            # print label

            return image,label

        else:
            image = self.data[index].reshape((28,28))
            return image

def convert_data(data, test):
    mnist = data.as_matrix()
    # if not test:
    #     length = 40000
    #     label = mnist[:length,0]
    #     mnist_data = mnist[:length,1:]

    #     return mnist_data, label, length
    
    # else:
    #     length = 2000
    #     label = mnist[40000:,0]
    #     mnist_data = mnist[40000:,1:]

    #     return mnist_data, label, length
    if not test:
        length = data.shape[0]
        label = mnist[:,0]
        mnist_data = mnist[:,1:]
        return mnist_data, label, length

    else:
        length = data.shape[0]
        mnist_data = mnist[:,:]

        return mnist_data, length


if __name__ == '__main__':
    train_data = dataset(test=True)
