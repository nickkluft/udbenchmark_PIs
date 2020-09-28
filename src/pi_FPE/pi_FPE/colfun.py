#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to easily perform matrix calculations on arrays
@author: Nick Kluft
"""
import numpy as np

def prod_col(A,B):   
    mult_ord = (np.arange(9).reshape(3,3)).T
    C = np.zeros(B.shape)
    for i_col in range(B.shape[1]):
            Ai = mult_ord[int(i_col%3),:]
            Bi = np.array([0,1,2])+ int(3*np.floor(i_col/3))
            C[:,i_col] = np.sum(np.multiply(A[:,Ai],B[:,Bi]),axis=1)
    return C

def norm_col(dat):
    dat = np.sum(dat**2,axis=1)**.5
    dat = np.vstack([dat,dat,dat]).transpose()
    return dat

def transpose_col(mat):
    Tmat = np.c_[mat[:,0::3],mat[:,1::3],mat[:,2::3]]
    return Tmat