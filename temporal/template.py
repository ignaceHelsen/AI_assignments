# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:57:11 2022

@author: Simon
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:05:21 2022

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

from torch import nn, optim
from torch import nn
import torch

"Define fully connected network"


class create_dense(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(create_dense, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.linear_1 = nn.Linear(self.dim_in, 16)
        self.linear_2 = nn.Linear(16, 16)
        self.linear_3 = nn.Linear(16, self.dim_out)

    def forward(self, x):
        x= self.linear_1(x)
        x= self.tanh(x)
        x = self.linear_2(x)
        x= self.relu(x)
        x= self.linear_3(x)
        return x


"Define rnn"


class create_rnn(nn.Module):
    def __init__(self):
        super(create_rnn, self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.linear_1 = nn.Linear(1, 3)
        self.linear_2 = nn.Linear(1, 3)
        self.linear_3 = nn.Linear(3, 1)

    def forward(self, x):
        ts = np.shape(x)[0]
        bs = np.shape(x)[1]
        ft = np.shape(x)[2]

        Ht = torch.zeros(bs, ft)
        H = torch.zeros(ts,bs,ft)

        for t in range(ts):
            Ct = self.tanh(self.linear_1(x[t]) + self.linear_2(Ht))
            Ht= self.linear_3(Ct)
            H[t] = Ht#.clone()

        return H


"Define network"
dense_network = create_dense(1, 1)
rnn_network = create_rnn()

"Define optimizer for neural network"
parameters_rnn = rnn_network.parameters()
parameters_dense = dense_network.parameters()

betas = (0.9, 0.9)
optimizer_rnn = optim.Adam(params=parameters_rnn, lr=0.002)
optimizer_dense = optim.Adam(params=parameters_dense, lr=0.002)

"Source: https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/"
import seaborn as sb

"import and prepare dataset"
flight_data = sb.load_dataset("flights")
data = flight_data['passengers'].values.astype(float)

total_len = len(data)
data = np.reshape(data, (total_len, 1, 1))
max_value = data.max()
data = data / max_value  # Normalize
data = torch.tensor(data, dtype=torch.float32)

ts = 12
ts_batch = total_len - 2 * ts

epochs = 20

"Define loss function"
# loss_function = nn.L1Loss()
loss_function = nn.MSELoss()
loss = []
for i in range(epochs):
    print("epoch:", i)

    for j in range(ts_batch):
        optimizer_rnn.zero_grad()
        optimizer_dense.zero_grad()

        "Load input with target output"
        x = data[j: j + ts]
        y = data[ts + j: 2 * ts + j]

        "Forward"
        output_rnn = rnn_network.forward(x)
        output_dense = dense_network.forward(output_rnn)

        # if j==0:
        #     plt.clf()
        # plt.plot( np.array(output_dense.detach())[:,0] )
        # plt.plot( np.array(y.detach())[:,0] )
        #     plt.pause(0.01)

        "Compute loss"
        pytorch_loss = loss_function(output_dense, y)
        pytorch_loss.backward()
        loss.append(np.array(pytorch_loss.detach()))

        "Update weights"
        optimizer_rnn.step()
        optimizer_dense.step()

    plt.clf()
    output_rnn = rnn_network.forward(data[:-ts])
    output_dense = dense_network.forward(output_rnn)
    plt.plot(max_value * np.array(output_dense.detach())[:-ts, 0])
    plt.plot(max_value * np.array(data.detach())[ts:, 0])
    plt.pause(0.01)

# plt.clf()
plt.figure()
plt.plot(loss)
