#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@package pi_SampTemp
@file SpatTemp
@author Nick Kluft
@brief Computer the spatiotemporal parameters on the basis of jointTrajectory and
gaitEvents files

Copyright (C) 2020 Vrije Universiteit Amsterdam (FGB-Human Movement Sciences)
Distributed under the pache 2.0 license.
"""
import sys
import os
import numpy as np
import pandas as pd
from .getevents import read_events
from termcolor import colored

def calcSpatTemp(joint,events,fs):
    print("Calculating spatiotemporal parameters...")
    lto = np.array(events.l_toe_off)*fs
    rto = np.array(events.r_toe_off)*fs
    lhs = np.array(events.l_heel_strike)*fs
    rhs = np.array(events.r_heel_strike)*fs
    # get list of all events
    meve = np.array([])
    arr_ieve = np.hstack((lhs,rto,rhs,lto))
    arr_neve = np.hstack((np.ones(lhs.shape[0])*0,np.ones(rto.shape[0]),np.ones(rhs.shape[0])*2,np.ones(lto.shape[0])*3))
    meve = np.vstack((arr_ieve,arr_neve))
    meve = meve[:,meve[0,:].argsort()]
    
    t_lstride = np.diff(meve[:,meve[1,:]==0])*(1/fs)
    t_rstride = np.diff(meve[:,meve[1,:]==2])*(1/fs)
        
    # get walking direction
    l_heel_xyz = np.transpose(np.vstack((joint.l_heel_x,joint.l_heel_y,joint.l_heel_z)))
    r_heel_xyz = np.transpose(np.vstack((joint.r_heel_x,joint.r_heel_y,joint.r_heel_z)))
    walkdir = np.argmax(np.ptp(r_heel_xyz,axis=0))
    temp = [1,0]
    mldir = temp[walkdir]
    
    # Step length
    idx_all = np.linspace(1,joint.shape[0],joint.shape[0])
    idx_lhs = np.isin(idx_all,lhs)
    idx_rhs = np.isin(idx_all,rhs)
    
    l_sl = abs(np.subtract(l_heel_xyz[idx_lhs,walkdir],r_heel_xyz[idx_lhs,walkdir]))
    r_sl = abs(np.subtract(r_heel_xyz[idx_rhs,walkdir],l_heel_xyz[idx_rhs,walkdir]))
    
    l_sw = abs(np.subtract(l_heel_xyz[idx_lhs,mldir],r_heel_xyz[idx_lhs,mldir]))
    r_sw = abs(np.subtract(r_heel_xyz[idx_rhs,mldir],l_heel_xyz[idx_rhs,mldir]))
    
    class strct():
        pass
    out = strct()
    out.tstride = np.mean(np.hstack((t_lstride,t_rstride)))
    out.tstride_sd = np.std(np.hstack((t_lstride,t_rstride)))
    out.sl = np.mean(np.hstack((l_sl,r_sl)))
    out.sl_sd = np.std(np.hstack((l_sl,r_sl)))
    out.sw = np.mean(np.hstack((l_sw,r_sw)))
    out.sw_sd = np.std(np.hstack((l_sw,r_sw)))
    return out

def store_result(file_out, value, sdvalue):
    file = open(file_out, 'w')
    file.write('type: \'scalar\'\nvalue: ' + format(value, '.5f') + '\nSD:'+ format(sdvalue, '.5f'))
    file.close()
    return True

USAGE = """usage: run_pi file_in_joint file_in_events folder_out
file_in_joint: csv file containing the  positonal joint data
file_in_events: csv file containing the  timing of gait events
folder_out: folder where the PI yaml files will be stored
"""

def main():
    file_in_joint = sys.argv[1]
    file_in_events = sys.argv[2]
    folder_out = sys.argv[3]    
    
    if len(sys.argv) != 4:
        print(colored("Wrong input parameters !", "red"))
        print(colored(USAGE, "yellow"))
        return -1
    
    # check input parameters are good
    if not os.path.exists(file_in_joint):
        print(colored("Input file {} does not exist".format(file_in_joint), "red"))
        return -1
    if not os.path.isfile(file_in_joint):
        print(colored("Input path {} is not a file".format(file_in_joint), "red"))
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
    if not os.path.isfile(file_in_joint):
        print(colored("Output path {} is not a folder".format(file_in_joint), "red"))
        return -1
    if not os.path.isfile(file_in_events):
        print(colored("Output path {} is not a folder".format(file_in_joint), "red"))
        return -1
    
    # load events structure
    events = read_events(file_in_events)
    # load joint data
    joint = pd.read_csv(file_in_joint)
    # from joint data calculate the sampling frequency
    fs = 1000/np.mean(np.diff(np.array(joint.time)))
    # calculate the spatiotemporal parameters
    out = calcSpatTemp(joint, events, fs)
    
    file_out0 = folder_out + "/pi_stridetime.yaml"
    if not store_result(file_out0, out.tstride, out.tstride_sd):
        return -1
    print(colored(
        "stride time: {} +- {} stored in {}".format(round(out.tstride,2),round(out.tstride_sd,2), file_out0),
        "green"))
    
    file_out1 = folder_out + "/pi_stepwidth.yaml"
    if not store_result(file_out1, out.sw,out.sw_sd):
        return -1
    print(colored(
        "step width: {} +- {} stored in {}".format(round(out.sw,2),round(out.sw_sd,2), file_out1),
        "green"))
    
    file_out2 = folder_out + "/pi_steplength.yaml"
    if not store_result(file_out2, out.sl, out.sl_sd):
        return -1
    print(colored(
        "step length: {} +- {} stored in {}".format(round(out.sl,2),round(out.sl_sd,2), file_out2),
        "green"))
    return 0
    
