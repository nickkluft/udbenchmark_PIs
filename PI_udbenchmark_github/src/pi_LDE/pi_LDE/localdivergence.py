#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:42:06 2020

@author: Nick1
"""
import numpy as np
import pandas as pd
import sys
import os
from .getevents import read_events
from termcolor import colored

def calcStateSpace(sig,events,fs,n_dim,delay):
# =============================================================================
#     Calculate state space using n_dimension with time delay [delay] in samples
# =============================================================================
    from scipy.interpolate import interp1d
    print("creating state space...")
    # make sorted heel strike list 
    lhs = np.array(events.l_heel_strike)*fs
    rhs = np.array(events.r_heel_strike)*fs
    meve = np.array([])
    arr_ieve = np.hstack((rhs,lhs))
    arr_neve = np.hstack((np.zeros(rhs.shape),np.ones(lhs.shape)))
    meve = np.vstack((arr_ieve,arr_neve))
    meve = meve[:,meve[0,:].argsort()]
    # get number of strides and samples 
    n_samples = (meve.shape[1]-1)*100
    # new signal without trailinabs
    idx_truesamp = int(meve[0,0]-1)
    idx_tail = int(meve[0,-1]-1)
    #idx_cut = np.hstack((np.zeros(int(meve[0,0])),np.ones(n_truesamp),np.zeros(n_tail)))
#    sig_new = np.vstack((com.COMx,com.COMy,com.COMz))
    sig_new = sig[idx_truesamp:idx_tail]
    t_new = np.linspace(0,1,sig_new.shape[0])
    t_inter = np.linspace(0,1,n_samples)
    # sligthly different than Sjoerd's method (does not extrapolate data)
    fsig_int = interp1d(t_new,sig_new,kind = 'cubic')
    sig_int = fsig_int(t_inter)
    # create empty statespace array    
    state = np.zeros([n_samples-(n_dim-1)*delay,n_dim])
    #create the state space now:
    for i_dim in range(n_dim):
        idx_start = (i_dim)*delay
        idx_stop = n_samples-(n_dim-i_dim-1)*delay
        state[:,i_dim] = sig_int[idx_start:idx_stop]
    return state


def calcLDE(state,ws,fs,period,nnbs):
# =============================================================================
#     Calculate Local Differgence Exponent
# =============================================================================
    win = np.round(ws*fs)
    [m,n] = state.shape
    state = np.vstack((state,np.zeros([int(win),int(n)])*np.nan))
    divmat = np.zeros([int(m*nnbs),int(win)])*np.nan
    difmat = np.zeros([int(m+win),int(n)])*np.nan
    L1 = .5*period*fs
    print('Calculating local divergence ...')
    sys.stdout.write("[%s]" % ("." * 10))
    sys.stdout.flush()
    sys.stdout.write("\b" * (10+1))
    for i_t in range(m):
        if np.greater_equal((i_t/m)*100,round((100*i_t/m)+.499999)):
               sys.stdout.write("#")
               sys.stdout.flush()
        for i_d in range(n):
            difmat[:,i_d] = (np.subtract(state[:,i_d],state[i_t,i_d]))**2
        arridx1 = np.array([1,i_t-np.round(L1)])
        arridx2 = np.array([m,i_t+np.round(L1)])
        idx_start = int(np.round(np.amax(arridx1))-1)
        idx_stop = int(np.round(np.amin(arridx2))-1)
        difmat[idx_start:idx_stop] = difmat[idx_start:idx_stop]*np.nan
        idx_sort = np.sum(difmat,1).argsort()
        for i_nn in range(nnbs):
            div_tmp = np.subtract(state[i_t:i_t+int(win),:],state[idx_sort[i_nn]:(idx_sort[i_nn]+int(win)),:])
            divmat[i_t*nnbs+i_nn,:] = np.sum(div_tmp**2,1)**.5 # track divergence, and store
    sys.stdout.write("]\n") # end progress bar  
    divergence = np.nanmean(np.log(divmat),0)
    xdiv = np.linspace(1,divergence.shape[0],divergence.shape[0])/fs
    xs = xdiv[1:int(np.floor(L1))]
    Ps = np.polynomial.polynomial.polyfit(xs,divergence[1:int(np.floor(L1))],1)    

    lde = np.ones([1,2])*np.nan
    lde[0] = Ps[1]
    
    L2 = np.round(4*period*fs)    
    if L2 < win:
        idxl = (np.linspace(0,win-L2-1,int(win-L2))+int(L2))
        Pl = np.polynomial.Polynomial.fit(idxl/fs,divergence[int(idxl[0]):int(idxl[-1])],1)
        lde[1] = Pl[1]
    return divergence,lde

def store_result(file_out, value):
    file = open(file_out, 'w')
    file.write('type: \'scalar\'\nvalue: ' + format(round(value,2), '.5f'))
    file.close()
    return True

USAGE = """usage: run_pi file_in_joint file_in_events folder_out
file_in_com: csv file containing the  positonal CoM data
file_in_events: csv file containing the  timing of gait events
folder_out: folder where the PI yaml files will be stored
"""

def main():
    file_in_com = sys.argv[1]
    file_in_events = sys.argv[2]
    folder_out = sys.argv[3]    
    
    if len(sys.argv) != 4:
        print(colored("Wrong input parameters !", "red"))
        print(colored(USAGE, "yellow"))
        return -1
    
    # check input parameters are good
    if not os.path.exists(file_in_com):
        print(colored("Input file {} does not exist".format(file_in_com), "red"))
        return -1
    if not os.path.isfile(file_in_com):
        print(colored("Input path {} is not a file".format(file_in_com), "red"))
        return -1
    if not os.path.exists(file_in_events):
        print(colored("Input file {} does not exist".format(file_in_events), "red"))
        return -1
    if not os.path.isfile(file_in_events):
        print(colored("Input path {} is not a file".format(file_in_events), "red"))
        return -1

    if not os.path.exists(folder_out):
        print(colored(
            "Output folder {} does not exist".format(folder_out),
            "red"))
        return -1
    if not os.path.isfile(file_in_com):
        print(colored("Output path {} is not a folder".format(file_in_com), "red"))
        return -1
    if not os.path.isfile(file_in_events):
        print(colored("Output path {} is not a folder".format(file_in_com), "red"))
        return -1
    
    # load joint data
    com = pd.read_csv(file_in_com)
    # load events structure
    events = read_events(file_in_events)
    # from com data calculate the sampling frequency
    fs = 1000/np.mean(np.diff(np.array(com.time)))
    # calculate the spatiotemporal parameters
    ws= 8
    period = 2.04
    nnbs = 4
    ndim = 4
    delay = 10
    state = calcStateSpace(com.COMy,events,fs,ndim,delay)
    diverg,locdiv = calcLDE(state,ws,fs,period,nnbs)
    print(locdiv[0][0])
    file_out0 = folder_out + "/pi_LDE.yaml"
    if not store_result(file_out0, locdiv[0][0]):
        return -1
    print(colored(
        "local divergence exp.: {} stored in {}".format(round(locdiv[0][0],2), file_out0),
        "green"))
    return 0


    


    