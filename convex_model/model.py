import torch
import torch.nn as nn

import numpy as np

class GradDescent(torch.nn.Module):
    # GradDescent (weights for learning c1A1 + c2A2 + ... + cnAn = M, to minimize ||Mx - b||
    # constrained by c1 = 1 - c2 - c3 ... - cn

    def __init__(self, matrices, device):
        super(GradDescent, self).__init__()

        # Store matrices
        self.matrices = torch.tensor(np.array(matrices).T, device=device, requires_grad=False)
        print('Matrix shape:', self.matrices.shape)
        #print(len(self.matrices), self.matrices[0].shape)

        # Create weights
        self.weights = nn.Parameter(torch.ones(self.matrices.shape[1]) * (1/self.matrices.shape[1]))
        #print('sum:',self.weights.sum(),self.weights.shape)

    def forward(self, x):

        #print('Grad stuff')
        #print(self.matrices.requires_grad)
        #print(self.weights.requires_grad)

        #print('Forward shapes')
        #print('Input:',x.shape)
        #print('Weights:',self.weights.shape)
        #print('Matrices:',self.matrices.shape)

        #print(self.matrices[:5, :5])
        #print(self.weights[:5])

        # Forward pass
        weighted_matrices = self.weights.softmax(0) * self.matrices
        summed_matrix = torch.sum(weighted_matrices, dim=1)
        #summed_matrix = torch.sum(self.matrices, dim=1)
        transformed_input = summed_matrix * x

        #print('Weighted matrices:', weighted_matrices.shape)
        #print('Summed matrix:', summed_matrix.shape)



     
        return transformed_input