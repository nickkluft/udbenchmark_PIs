#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 19:40:48 2020

@author: Nick1
"""

import os
import sys
import numpy as np
import numpy.matlib as mtlb
import pandas as pd
from termcolor import colored
from .getevents import read_events
from scipy import interpolate
# from sklearn import linear_model#,metrics
import statsmodels.api as sm



def timenormalize_step(signal,start,stop):
# =============================================================================
#     Time normalize signal from start[i] to stop[i] with 51 samples
# =============================================================================
    if start[0]>stop[0]:
        del stop[0]
        print('Delete first sample of stop')
    if start.shape[0]>stop.shape[0]:
        start = start[0:stop.shape[0]]     
        print('remove trailing part start')
    if stop.shape[0]>start.shape[0]:
        stop = start[0:start.shape[0]]
        print('remove trailing part stop')
    # determine number of segments
    nseg = start.shape[0]
    # create empty matrix
    cycle = np.zeros([51,nseg])*np.nan
    # loop over segments
    for i in range(0,nseg):
        # create time array to interpolate
        t = np.linspace(0,50,int(stop[i]-start[i]))
        # select segment in signal
        sigoi = signal[int(start[i]):int(stop[i])]
        # if siganl contain only real values
        if not np.isnan(sigoi).any():
            # interpolate the data
            fsig_int = interpolate.interp1d(t,sigoi,kind='quadratic',axis=0)
            cycle[:,i] = fsig_int(np.linspace(0,50,51))
        sigoi = None
    return cycle
    

def calcFPM(joint,com,events,fs):
# =============================================================================
#     Create Foot Placement Model and calculate estimates and goodness of fit
# =============================================================================
    # get events
    lhs = np.round(np.array(events.l_heel_strike)*fs)
    rto = np.round(np.array(events.r_toe_off)*fs)
    rhs = np.round(np.array(events.r_heel_strike)*fs)
    lto = np.round(np.array(events.l_toe_off)*fs)
    
    meve = np.array([])
    arr_ieve = np.hstack((lhs,rto,rhs,lto))
    arr_neve = np.hstack((np.ones(lhs.shape[0])*0,np.ones(rto.shape[0]),np.ones(rhs.shape[0])*2,np.ones(lto.shape[0])*3))
    meve = np.vstack((arr_ieve,arr_neve))
    meve = meve[:,meve[0,:].argsort()]
    ilhs = np.where(meve[1,:]==0)
    ilto = np.where(meve[1,:]==3)
    meve = meve[:,int(ilhs[0][0]):int(1+ilto[-1][-1])]

    lhs = meve[0,np.where(meve[1,:]==0)]
    rto = meve[0,np.where(meve[1,:]==1)]
    rhs = meve[0,np.where(meve[1,:]==2)]
    lto = meve[0,np.where(meve[1,:]==3)]
    
    # get foot position from joint
    lfoot = np.array(pd.concat([(joint.l_TTII_x+joint.l_heel_x)/2,(joint.l_TTII_y+joint.l_heel_y)/2,(joint.l_TTII_z+joint.l_heel_z)/2],axis=1)/1000)
    rfoot = np.array(pd.concat([(joint.r_TTII_x+joint.r_heel_x)/2,(joint.r_TTII_y+joint.r_heel_y)/2,(joint.r_TTII_z+joint.r_heel_z)/2],axis=1)/1000)
    walkdir = np.argmax(np.ptp(np.array(lfoot),axis=0))
    # walkdir = 0
    # get CoM in np.array structure
    com_arr = np.array(np.vstack([com.COMx,com.COMy,com.COMz])).transpose()/1000
    # calculate the 1st and 2nd derivative of the com displacement
    vcom = np.gradient(com_arr[:,walkdir],axis=0)*fs
    # acom = np.gradient(vcom,axis=0)*fs
    # normalize data to the gait cycle
    com_l  = timenormalize_step(com_arr[:,walkdir],lto[0][0:-1],lhs[0][1:])
    com_r  = timenormalize_step(com_arr[:,walkdir], rto[0][0:-1], rhs[0][0:-1])
    vcom_l = timenormalize_step(vcom, lto[0][0:-1], lhs[0][1:])
    vcom_r = timenormalize_step(vcom, rto[0][0:-1], rhs[0][0:-1])
    foot_l = timenormalize_step(lfoot[:,walkdir], rto[0][1:], rhs[0][1:])
    foot_r = timenormalize_step(rfoot[:,walkdir], lto[0][0:-1], lhs[0][1:])
    orig_l = timenormalize_step(rfoot[:,walkdir], lto[0][0:-1], lhs[0][1:])
    orig_r = timenormalize_step(lfoot[:,walkdir], rto[0][0:-1], rhs[0][0:-1])
    # foot at midstance
    foot_l = foot_l[24,:]
    foot_r = foot_r[24,:]
    orig_l = orig_l[24,:]
    orig_r = orig_r[24,:]
    # subtract the origin of foot (orig_=postion of controlateral foot)
    foot_l = foot_l-orig_l
    foot_r = foot_r-orig_r
    # center data
    foot_l = foot_l-np.nanmean(foot_l)
    foot_r = foot_r-np.nanmean(foot_r)
    # loop over samples of the normalized signal
    rsq_l = np.ones(51)*np.nan
    rsq_r = np.ones(51)*np.nan
    FPMl_est = np.ones([com_l.shape[1],51])*np.nan
    FPMr_est = np.ones([com_r.shape[1],51])*np.nan
    print("fitting foot placement model...")
    for i in range(0,51):
        # samples of left foot
        com_l_samp  = com_l[i,:]
        vcom_l_samp = vcom_l[i,:]
        # acom_l_samp = acom_l[i,:]
        # samples of right foot
        com_r_samp  = com_r[i,:]
        vcom_r_samp = vcom_r[i,:]
        # predictors
        pred_lstance = np.vstack([com_l_samp,vcom_l_samp])
        pred_rstance = np.vstack([com_r_samp,vcom_r_samp])
        # remove origin
        pred_lstance = pred_lstance-np.vstack([orig_l,orig_l])
        pred_rstance = pred_rstance-np.vstack([orig_r,orig_r])
        # remove mean
        pred_lstance = pred_lstance-mtlb.repmat(np.nanmean(pred_lstance,axis=1),com_l.shape[1],1).transpose()
        pred_rstance = pred_rstance-mtlb.repmat(np.nanmean(pred_rstance,axis=1),com_r.shape[1],1).transpose()
# =============================================================================
#         Calculation for left steps
# =============================================================================
        df_l = pd.DataFrame(np.vstack([foot_l,pred_lstance]).transpose(),columns=['foot_pos','pred0','pred1'])
        X_l = sm.add_constant(df_l[['pred0','pred1']])
        reg_l = sm.OLS(df_l[['foot_pos']],X_l).fit()
        # print(reg_l.summary())
        rsq_l[i] = reg_l.rsquared
        FPMl_est[:,i] = reg_l.predict(X_l)

# =============================================================================
#         Calculation for right steps
# =============================================================================
        df_r = pd.DataFrame(np.vstack([foot_r,pred_rstance]).transpose(),columns=['foot_pos','pred0','pred1'])
        X_r = sm.add_constant(df_r[['pred0','pred1']])
        reg_r = sm.OLS(df_r[['foot_pos']],X_r).fit()
        # print(reg_l.summary())
        rsq_r[i] = reg_r.rsquared
        FPMr_est[:,i] = reg_r.predict(X_r)

    return FPMl_est,FPMr_est,rsq_l,rsq_r,com_l


def store_result(file_out, value):
    file = open(file_out, 'w')
    file.write('type: \'scalar\'\nvalue: ' + format(value, '.5f'))
    file.close()
    return True


USAGE = """usage: run_pi file_in_joint file_in_com file_in_events folder_out
file_in_joint: csv file containing the  positonal joint data
file_in_com: csv file containing the positional CoM data
file_in_events: csv file containing the  timing of gait events
folder_out: folder where the PI yaml files will be stored
"""

def main():
    file_in_joint = sys.argv[1]
    file_in_com = sys.argv[2]
    file_in_events = sys.argv[3]
    folder_out = sys.argv[4]    
    
    if len(sys.argv) != 5:
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
    if not os.path.isfile(file_in_joint):
        print(colored("Output path {} is not a folder".format(file_in_joint), "red"))
        return -1
    if not os.path.isfile(file_in_com):
        print(colored("Output path {} is not a folder".format(file_in_joint), "red"))
        return -1
    if not os.path.isfile(file_in_events):
        print(colored("Output path {} is not a folder".format(file_in_joint), "red"))
        return -1
    
    # load events structure
    events = read_events(file_in_events)
    # load joint data
    joint = pd.read_csv(file_in_joint)
    # load CoM data
    com = pd.read_csv(file_in_com)
    # from joint data calculate the sampling frequency
    fs = 1000/np.mean(np.diff(np.array(com.time)))
    # calculate the spatiotemporal parameters
    FPMl_est,FPMr_est,rsq_l,rsq_r,com_l = calcFPM(joint,com,events,fs)
    
    file_out0 = folder_out + "/pi_FPM_rsq_l.yaml"
    if not store_result(file_out0, rsq_l[24]):
        return -1
    print(colored(
        "r-squared left leg: {} stored in {}".format(round(rsq_l[24],2), file_out0),
        "green"))
    
    file_out1 = folder_out + "/pi_FPM_rsql_r.yaml"
    if not store_result(file_out1, rsq_r[24]):
        return -1
    print(colored(
        "r-squared right leg: {} stored in {}".format(round(rsq_r[24],2), file_out1),
        "green"))
    return 0
    










    