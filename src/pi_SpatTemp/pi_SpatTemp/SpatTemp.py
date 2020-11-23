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
    # get hips
    rhip = np.transpose(np.vstack((joint.l_hip_x,joint.l_hip_y,joint.l_hip_z)))
    lhip = np.transpose(np.vstack((joint.r_hip_x,joint.r_hip_y,joint.r_hip_z)))
    # get walking direction
    walkdir = np.argmax(np.ptp(r_heel_xyz,axis=0))
    mldir = np.argmax(np.nanmean(np.abs(rhip-lhip),axis=0))

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
    out.tstride = np.hstack((t_lstride[0,:],t_rstride[0,:]))
    out.r_sl =r_sl
    out.l_sl =l_sl
    out.sw = np.hstack((l_sw,r_sw))
    return out

def store_result(file_out, value):
    file = open(file_out, 'w')
    file.write('---\ntype: \'vector\'\nvalues:')
    for line in value:
        file.write('\n '+format(line, '.5f'))
    file.write('\n')
    file.close()
    return True


USAGE = """usage: run_pi file_in_joint file_in_events folder_out
file_in_joint: csv file containing the  positonal joint data
file_in_events: csv file containing the  timing of gait events
folder_out: folder where the PI yaml files will be stored
"""

def main():


    if len(sys.argv) != 4:
        print(colored("Wrong input parameters !", "red"))
        print(colored(USAGE, "yellow"))
        return -1

    file_in_joint = sys.argv[1]
    file_in_events = sys.argv[2]
    folder_out = sys.argv[3]

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
    fs = 1/np.mean(np.diff(np.array(joint.time)))
    # calculate the spatiotemporal parameters
    out = calcSpatTemp(joint, events, fs)
    file_out0 = folder_out + "/pi_stridetime.yaml"

    if not store_result(file_out0, out.tstride):
        return -1
    print(colored(
        "stride time: vector with size {}x1 stored in {}".format(np.size(out.tstride,0), file_out0),
        "green"))

    file_out1 = folder_out + "/pi_stepwidth.yaml"
    if not store_result(file_out1, out.sw):
        return -1
    print(colored(
        "step width: vector with size {}x1 stored in {}".format(np.size(out.sw,0), file_out1),
        "green"))

    file_out2 = folder_out + "/pi_r_steplength.yaml"
    if not store_result(file_out2, out.r_sl):
        return -1
    print(colored(
        "right step length: vector with size {}x1 stored in {}".format(np.size(out.r_sl,0), file_out2),
        "green"))

    file_out3 = folder_out + "/pi_l_steplength.yaml"
    if not store_result(file_out3, out.l_sl):
        return -1
    print(colored(
        "left step length: vector with size {}x1 stored in {}".format(np.size(out.l_sl,0), file_out3),
        "green"))
    return 0

