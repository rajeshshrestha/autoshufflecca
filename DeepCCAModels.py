from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from objectives import cca_loss


class CNet(nn.Module):
    def __init__(self, input_size):
        super(CNet, self).__init__()
        self.input_size = input_size
        layers = []

        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3),stride=1, bias=True),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),stride=1, bias=True),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),stride=1, bias=True),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),stride=1, bias=True),
            nn.MaxPool2d(3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=1, stride=1, kernel_size=(3,3),bias=True),
            nn.MaxPool2d(3, stride=1),
            nn.Flatten()
            # nn.Linear(in_features= 2048, out_features=10)
        ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PermNet(nn.Module):
    def __init__(self):
        super(PermNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
         )
        self.final_layer = nn.Sequential(
            nn.Linear(in_features= 384, out_features=64*64),
            nn.ReLU(),
            # nn.Linear(in_features= 128, out_features=64),
            # nn.ReLU()
        )
        self.layers = nn.ModuleList([self.conv_layer, self.final_layer])
    
    def forward(self, x1, x2):
        o1 = self.layers[0](x1)
        o2 = self.layers[0](x2)
        M = self.layers[1](torch.cat((o1,o2),-1))
        M = M.view(-1,64,64)
        for k in range(1):
            M = F.normalize(M+1e-12*torch.rand_like(M),p=1, dim=1)
            M = F.normalize(M+1e-12*torch.rand_like(M), p=1, dim=2)

        return M

class DeepCCA(nn.Module):
    def __init__(self,
     input_size1,
     input_size2,
     k_eigen_check_num,
     use_all_singular_values,
       device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = CNet(input_size1).double()
        self.model2 = CNet(input_size2).double()
        self.permutationmodel = PermNet().double()

        self.loss = cca_loss(k_eigen_check_num, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        permutation_out = self.permutationmodel(x1,x2)
        output1 = torch.matmul(permutation_out, output1.view(-1,64,1)).view(-1,64)

        # print(permutation_out)

        # print(f"M: {permutation_out}")
        # for param in self.permutationmodel.parameters():
        #     print(param)
        # print("Loop completed")
        return output1, output2, permutation_out
