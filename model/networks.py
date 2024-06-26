import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

from torch.autograd import Variable
from model import utils 



class HyperGCN(nn.Module):
    def __init__(self, V, E, X, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = args.d, args.depth, args.c
        cuda = args.cuda and torch.cuda.is_available()

        h = [d]
        for i in range(l-1):
            power = l - i + 2
            if args.dataset == 'citeseer': power = l - i + 4
            h.append(2**power)
        h.append(c)

        if args.fast:
            reapproximate = False
            structure = utils.Laplacian(V, E, X, args.mediators)        
        else:
            reapproximate = True
            structure = E
            
        self.layers = nn.ModuleList([utils.HyperGraphConvolution(h[i], h[i+1], reapproximate, cuda) for i in range(l)])
        self.do, self.l = args.dropout, args.depth
        self.structure, self.m = structure, args.mediators

        # self.linear = nn.Sequential(*[nn.Linear(d, 64),
        #                               nn.ReLU(),
        #                               nn.Linear(64, 64),
        #                               nn.ReLU(),
        #                               nn.Linear(64, d)])

    def forward(self, H):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        # H = self.linear(H)  
    
        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                V = H
                H = F.dropout(H, do, training=self.training)

        # 각 행의 평균, 표준편차 계산
        epsilon = 1e-10
        mean_vals = H.mean(dim=1, keepdim=True)
        std_vals = H.std(dim=1, keepdim=True) + epsilon

        # Z-정규화
        H_normalized = (H - mean_vals) / std_vals

        return F.log_softmax(H_normalized, dim=1)
    
        # return F.log_softmax(H, dim=1)


        return F.softmax(H, dim=1)
   