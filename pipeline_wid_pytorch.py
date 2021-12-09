# steps in trining pipeline in pytorch
# 1. design Model :input size, output size, forward pass(layerss)
# 2.loss and optimization
# 3.training loop:
#   forward pass
#   prediction
#   gradient
#   backward pass

import torch
x=torch.tensor([1,2,3,4,5], dtype=torch.float32)
y=torch.tensor([2,4,6,8,10], dtype=torch.float32)

#WEIGHT needs to be a tesnot too. to keep track in order to ind gradient, make true the para
w=torch.tensor([0.0],dtype=torch.float32,requires_grad=True)
learning_rate=0.01
epochs=10

def forward(x):
    return w*x

def loss(y_prediction):
    return ((y_prediction-y)**2).mean()


#traiinging
for epochs in range(epochs):
    y_prediction= forward(x)

    l=loss(y_prediction)
    #gradient = backward padd
    l.backward() #this will calculate gradient of loss wrt. weight dl/dw
    #update weights
    with torch.no_grad(): #to make not part of tracking. computational graph
      w = -(learning_rate* w.grad)
    w.grad.zero_()
    print(f"epochs is {epochs}\n loss is {l}")

#model
#prediction
#loss
#gradient

