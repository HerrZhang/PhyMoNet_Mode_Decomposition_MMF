#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math 
import numpy as np
    
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r




def mean2_tensor(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2_tensor(a,b):
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r