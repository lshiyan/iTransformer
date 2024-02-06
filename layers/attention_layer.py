import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    
    #Input is N length vector.
    
    def __init__(self, input_dimension, output_dimension):
        super(SelfAttention, self).__init__()
        
        self.N=len(input)
        self.D=len(input[0])
        self.input=input
        self.key_weights=nn.Parameter(torch.rand(self.D, self.N), requires_grad=True)
        self.value_weights=nn.Parameter(torch.rand(self.D, self.N), requires_grad=True)
        self.query_weights=nn.Parameter(torch.rand(self.D, self.N), requires_grad=True)
        
        self.keys=torch.matmul(input, self.key_weights)
        self.values=torch.matmul(input, self.value_weights)
        self.queries=torch.matmul(input, self.query_weights)
        
    def forward(x)