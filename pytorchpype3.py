# steps in trining pipeline in pytorch
# 1. design Model :input size, output size, forward pass(layerss)
# 2.loss and optimization
# 3.training loop:
#   forward pass
#   prediction
#   gradient
#   backward pass

import torch
from torch import nn
import torch.nn as nn
x=torch.tensor([1,2,3,4,5], dtype=torch.float32)
y=torch.tensor([2,4,6,8,10], dtype=torch.float32)

#WEIGHT needs to be a tesnot too. to keep track in order to ind gradient, make true the para
w=torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
learning_rate=0.01
epochs=10


def forward(self, input):
    return w*input
#using toch module, a callable function
loss=nn.MSELoss()
optimizer= torch.optim.SGD([w],lr=learning_rate)

#traiinging
for epochs in range(epochs):
    y_prediction=forward(x)
    l=loss(y_prediction)
    #gradient = backward padd
    l.backward() #this will calculate gradient of loss wrt. weight dl/dw
    #update weights
    # with torch.no_grad(): #to make not part of tracking. computational graph
    #   w = -(learning_rate* w.grad)
    optimizer.step()
    optimizer.zero_grad()
    print(f"epochs is {epochs}\n loss is {l}")

#model
#prediction
#loss
#gradient

