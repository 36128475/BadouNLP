"""
改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""

import torch
import torch.nn as nn
import numpy as np

class MyMoudle(nn.Moudle):
  def __init__(self, inputs):
    super(self, MyMoudle).__init__()
    
    

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return nn.Softmax(x)

def build_sample():
  x = randam

def train():
  #超参数
  epoch_num = 100
  batch_size = 24
  train_sample = 5000
  inputs_size = 5
  learning_rate = 0.01

  model = MyMoudle(inputs_size)
  
