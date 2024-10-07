# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:36:32 2024

@author: Santosh
"""
import torch
import torch.nn as nn
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self, no_inputs, hidden_layers, output_layers):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(no_inputs, hidden_layers)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers, output_layers)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x) 
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
    
NO_INPUTS = 2
HIDDEN_LAYERS = 4
OUTPUT_LAYERS = 2

model = SimpleModel(NO_INPUTS, HIDDEN_LAYERS, OUTPUT_LAYERS)

criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01)


input_value = np.array([[6,7]])
input_tensor = torch.from_numpy(input_value).float()

outputs = model(input_tensor)
print(outputs)


torch.save(model, "C:\\Tutorials\\NeuralNetworks\\torch_model.pth")
        





































