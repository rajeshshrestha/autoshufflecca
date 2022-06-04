from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from objectives import cca_loss
from torchvision import models
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_resnet_build_comp(model_type='resnet18', channel_num=1):
    if model_type == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=False)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=False)
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=False)
    else:
        raise Exception(f"Unknown model passed: {model_type}")
    model.conv1 = nn.Conv2d(channel_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = Identity()
    return model


class CNet(nn.Module):
    def __init__(self, input_size):
        super(CNet, self).__init__()
        self.input_size = input_size
        layers = []

        layers.append(nn.Sequential(
            get_resnet_build_comp('resnet18')
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
            get_resnet_build_comp('resnet18'),
            nn.Flatten(),
         )
        self.final_layer = nn.Sequential(
            nn.Linear(in_features= 1024, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features= 2048, out_features=512*512),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([self.conv_layer, self.final_layer])
    
    def forward(self, x1, x2):
        o1 = self.layers[0](x1)
        o2 = self.layers[0](x2)
        M = self.layers[1](torch.cat((o1,o2),-1))
        M = M.view(-1,512,512)
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

        print(x1.shape, output1.shape)
        print(x2.shape, output2.shape)

        permutation_out = self.permutationmodel(x1,x2)

        print(permutation_out.shape)
        
        output1 = torch.matmul(permutation_out, output1.view(-1,512,1)).view(-1,512)

        # print(permutation_out)

        # print(f"M: {permutation_out}")
        # for param in self.permutationmodel.parameters():
        #     print(param)
        # print("Loop completed")
        return output1, output2, permutation_out
