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



def train():
  #超参数
  epoch_num = 100
  batch_size = 24
  train_sample = 5000
  inputs_size = 5
  learning_rate = 0.01

  model = MyMoudle(inputs_size)
  
