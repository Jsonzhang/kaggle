# coding=utf-8
import numpy as np
import pandas as pd


# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

# Parameters
LEARNING_RATE = 1e-4
TRAINNING_ITERATIONS = 2500
DROPOUT = 0.5
BATCH_SIZE = 50
VALIDATION_SIZE = 2000
IMAGE_TO_DISPLAY = 10

data = pd.read_csv('./data/train.csv')

## 将数据存储为 0-1 区间的浮点格式
images = data.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)

## 计算总共有几幅图,每幅图横高值,有利于卷积矩阵运算
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)


## 总共有几种输出
labels_flat = data['label'].values
labels_count = np.unique(labels_flat).shape[0]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot