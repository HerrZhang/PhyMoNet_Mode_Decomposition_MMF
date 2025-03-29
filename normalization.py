# -*- coding: utf-8 -*-

import numpy as np

def normalization(O, minOut, maxOut):
    minO = np.min(O)
    maxO = np.max(O)
    normalized = (O-minO)/(maxO-minO)*(maxOut-minOut)+minOut
    return normalized
