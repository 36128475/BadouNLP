import torch
import torch.nn as nn
import numpy as np

class MyMoudle(nn.Moudle):
  def __init__(self, inputs, hidden1, hidden2, outputs):
    super(self, MyMoudle).__init__()
    self.layer1 = nn.Linear(inputs, hidden1)
    self.layer2 = nn.Linear(hidden1,hidden2)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return nn.Softmax(x)

