# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:55:16 2022

@author: Simon
"""


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn,optim
from torch import nn

import Project_3_functions as p4fcn



"Define fully connected network"
class create_dense(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(create_dense, self).__init__()
        
        self.nr_linear = 3
        
        self.dim_in  = dim_in
        self.dim_out = dim_out
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        h = 100
        self.linear_0 = nn.Linear(self.dim_in , h)
        self.linear_1 = nn.Linear(h, h)
        self.linear_2 = nn.Linear(h, self.dim_out )

    def forward(self, x):
        
        x = self.tanh( self.linear_0(x) )
        x = self.relu( self.linear_1(x) )        
        x = self.linear_2(x)
        
        return x



"Define lstm"
class create_lstm(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(create_lstm, self).__init__()

        "We initialize placeholder values for our internal outputs"
        self.nr_linear = 8
        
        self.Ht = torch.tensor(0)
        self.Ct = torch.tensor(0)

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.linear_0 = nn.Linear(dim_in, dim_out)  #f X
        self.linear_1 = nn.Linear(dim_out, dim_out) #f H

        self.linear_2 = nn.Linear(dim_in, dim_out)  #i X
        self.linear_3 = nn.Linear(dim_out, dim_out) #i H
        
        self.linear_4 = nn.Linear(dim_in, dim_out)  #c X
        self.linear_5 = nn.Linear(dim_out, dim_out) #c H

        self.linear_6 = nn.Linear(dim_in, dim_out)  #o X
        self.linear_7 = nn.Linear(dim_out, dim_out) #o H
        
        

    def forward(self, x):
        
        stepsize = 1
        
        ts = np.shape(x)[0]
        bs = np.shape(x)[1]
        ft = np.shape(x)[2]
        
        H  = torch.zeros( ts, bs, self.dim_out )
        
        "We create placeholder values"
        "If these are zero, then we have no initial value on our snapshot"
        if torch.sum(self.Ht) == 0:
            Ht = torch.zeros( bs, self.dim_out )
        else:
            Ht = self.Ht
            
        if torch.sum(self.Ct) == 0:
            Ct = torch.zeros( bs, self.dim_out )
        else:
            Ct = self.Ct
        
        for t in range(ts):
            
            f = self.sigmoid( self.linear_0(x[t]) + self.linear_1(Ht) )
            i = self.tanh(    self.linear_2(x[t]) + self.linear_3(Ht) )
            c = self.sigmoid( self.linear_4(x[t]) + self.linear_5(Ht) )
            o = self.tanh(    self.linear_6(x[t]) + self.linear_7(Ht) )
            
            fc = f*Ct
            ic = i*c
            Ct = ic + fc
            Ht = o * self.tanh(Ct)
            
            H[t] = Ht
            
            
            "We save the C and H snapshot values to be initial values next snapshot"
            if t == stepsize - 1:    
                self.Ht = Ht.detach().clone()
                self.Ct = Ct.detach().clone()
            
        return H


"""
Build feed_forward with 2x lstm models (stacked) then fully connected:
Implement normalization
    X -> lstm 1 -> normalization -> lstm 2 -> normalization -> dense
"""
def feed_forward( X ):
    
    
    return final_output



"Define network"
dense_final = create_dense(1,1)
lstm_1_network = create_lstm(1,20)
lstm_2_network = create_lstm(20,1)

"We store models in a list"
models = [ dense_final, lstm_1_network, lstm_2_network ]



"Define optimizer for neural network"
parameters_dense_final = dense_final.parameters()
parameters_lstm_1 = lstm_1_network.parameters()
parameters_lstm_2 = lstm_2_network.parameters()

"Define learning rate and apply Adam optimizer"
LR = 0.003
b = (0.9, 0.99)

optimizer_dense_final = optim.Adam( params = parameters_dense_final  , lr = LR, betas = b )
optimizer_lstm_1 = optim.Adam( params = parameters_lstm_1 , lr = LR, betas = b )
optimizer_lstm_2 = optim.Adam( params = parameters_lstm_2 , lr = LR, betas = b )



"import and prepare dataset"
"Load road data"
road_data = np.load("road_data.npy")
road_1 = torch.tensor( road_data[0], dtype=torch.float32)
total_len = len(road_1)


"https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html"
"Apply noise filtering on data"
road_1 = p4fcn.signal_filter(road_1.clone())


stepsize = 1 #snapshot stride
T = 12       #how many timesteps we want to predict
ts = 32      #Our snapshot size
ts_batch = total_len - ts - T

epochs = 20

"Define loss function"
loss_function = nn.L1Loss()
#loss_function = nn.MSELoss()

for i in range(epochs):
    loss = []
    print("epoch:",i)
    
    
    "We must reset snapshot values after every epoch"
    lstm_1_network.Ht = torch.tensor(0)
    lstm_1_network.Ct = torch.tensor(0)
    lstm_2_network.Ht = torch.tensor(0)
    lstm_2_network.Ct = torch.tensor(0)
    
    for j in range(ts_batch):
        
        "We always remove old gradients before training"
        "If not, we may get gradient conflict after a training pause"
        optimizer_lstm_1.zero_grad()
        optimizer_lstm_2.zero_grad()
        optimizer_dense_final.zero_grad()

        "Load input with target output"
        x_0 = road_1[j:j+ts]
        
        "How many timesteps we want our model to train on"
        y_0 = road_1[j+T: j+T+ts ]
        
        "Assign target output"
        nr_roads = 1
        target_output = torch.zeros(ts,nr_roads,1)
        target_output[:,0,:] = y_0[:,0]
        
        "Our model's output"
        final_output = feed_forward( x_0 )
        "Compute loss"
        pytorch_loss = loss_function( final_output, target_output )
        pytorch_loss.backward()
        loss.append( np.array( pytorch_loss.detach() ))
        
        "Remove zero gradients"
        models = p4fcn.gradient_push(models)
        
        dense_final, lstm_1, lstm_2,  \
                = models
        
        "Update weights"
        optimizer_lstm_1.step()
        optimizer_lstm_2.step()
        optimizer_dense_final.step()
        
        "Some plots"
        "Comment this part to run faster"
        # if i >= 0: 
        #     plt.clf()
        #     plt.plot( 70*np.array(final_output.detach())[:,0] )
        #     plt.plot( 70*np.array(target_output.detach())[:,0] )
        #     plt.ylim(50, 70)
        #     plt.pause(0.01)

    
    "For the model to be competitive, we must achieve at least 0.001 or less. Lower is better"
    print( "Avg training loss:", np.average(np.array(loss)) )
    

"Full plot for road 1"
# plt.figure()
# final_output_test = feed_forward( road_1 )
# plt.plot( 70 * np.array(final_output_test.detach())[:,0] )
# plt.plot( 70 * np.array(road_1.detach())[:,0] )    
# plt.pause(0.01)










