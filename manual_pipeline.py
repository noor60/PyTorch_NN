# steps in trining pipeline in pytorch
# 1. design Model :input size, output size, forward pass(layerss)
# 2.loss and optimization
# 3.training loop:
#   forward pass
#   prediction
#   gradient
#   backward pass

import torch
import numpy as np
x=np.array([1,2,3,4,5], dtype=np.float32)
y=np.array([2,4,6,8,10], dtype=np.float32)

w=0.0
learning_rate=0.1
epochs=10

def forward(x):
    return w*x

def loss(y_prediction):
    return ((y_prediction-y)**2).mean()

def gradient(x,y,y_prediction):
    return np.dot(2*w,y_prediction-y).mean()

#traiinging
for epochs in range(epochs):
    y_prediction= forward(x)

    l=loss(y_prediction)
    grad= gradient(x,y,y_prediction)
    #update weights
    w = -(learning_rate*grad)
    print(f'epochs is {epochs}\n loss is {l} weight is {w}')

#model
#prediction
#loss
#gradient

