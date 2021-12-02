import torch

x= torch.randn(4, requires_grad=True)
print(x)
y=x+2
z=y+y
v=z+2
z.backward(v)

print(x.grad)