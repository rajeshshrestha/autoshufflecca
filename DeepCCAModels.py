import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss
import torch.nn.functional as F


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PermNet(nn.Module):
    def __init__(self, layer_sizes, input_size1, input_size2, output_size):
        super(PermNet, self).__init__()
        self.layer_sizes = layer_sizes
        self.input_size = input_size1+input_size2
        self.ouput_size = output_size
        self.model = MlpNet(layer_sizes, self.input_size)
    
    def forward(self, x1, x2):
        M = self.model(torch.cat((x1,x2),-1))
        M = M.view(-1, self.ouput_size, self.ouput_size)
        for k in range(1):
            M = F.normalize(M+1e-12*torch.rand_like(M),p=1, dim=1)
            M = F.normalize(M+1e-12*torch.rand_like(M), p=1, dim=2)

        return M


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, layer_sizes3, input_size1, input_size2, outdim_size, use_all_singular_values, lambda_M, device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.outdim_size = outdim_size
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()
        self.permModel = PermNet(layer_sizes3, input_size1, input_size2, outdim_size)
        self.loss = cca_loss(outdim_size, use_all_singular_values, device, lambda_M=lambda_M).loss

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        M = self.permModel(x1, x2)
        output2 = torch.matmul(M, output2.view(-1, self.outdim_size, 1)).view(-1, self.outdim_size)

        return output1, output2,M
