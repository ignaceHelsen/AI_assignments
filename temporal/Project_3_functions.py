# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:12:32 2022

@author: Simon
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn,optim
from torch import nn


def signal_filter(X):
    Xn = np.array(X[:,0,0].detach())
    b, a = signal.butter(3, 0.2)

    zi = signal.lfilter_zi(b, a)
    z,_ = signal.lfilter(b, a, Xn, zi = (zi * Xn[0]) )
    z2,_ = signal.lfilter(b, a, z,  zi = (zi * z[0]) )
    
    y = signal.filtfilt(b, a, Xn )
    
    y = np.reshape( y , (np.shape( np.array(X))) )
    y = torch.tensor( y.copy(), dtype=torch.float32 )
    
    return y



def gradient_push(models):
    models_return = []
    for i in range(len(models)):
        current_model = models[i]
        
        for p in current_model.parameters():
            
            g = p.grad
            eps = 0.001
            sign_g = torch.sign( g[torch.abs(g)<eps] )
            sign_g[sign_g==0] = 1
            g[  torch.abs(g) < eps ] = eps * sign_g
            p.grad = g
            
                
        models_return.append(current_model)
    return models_return







