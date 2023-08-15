import torch
from torch import nn


def main():
    print("Hello, nn!")
    fnc=nn.Linear(in_features=4,out_features=3)
    print(" linear weigth size :\t ",fnc.weight.size())
    
    print(" weight :\t ",fnc.weight)
    print(" linear weigth  :\t ",fnc.weight.shape)
    x1=torch.tensor([1,2,3,4],dtype=torch.float32)
    print("first input ",x1)
    print(fnc(x1))
    x2=torch.tensor([[1,2,3,4],[5,6,7,8]],dtype=torch.float32)
    print("second input",x2)
    print(fnc(x2))

    x3=torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]],dtype=torch.float32)
    print("third input",x3)
    print(fnc(x3))


if __name__ == "__main__":
    main()

