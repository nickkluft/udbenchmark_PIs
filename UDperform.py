#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:42:06 2020

@author: Nick1
"""
import numpy as np
import pandas as pd
from scipy import optimize
from scipy import interpolate
import pdb

def calcSpatTemp(joint,events,fs):
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
    out.t_stride = np.mean(np.hstack((t_lstride,t_rstride)))
    out.sl = np.mean(np.hstack((l_sl,r_sl)))
    out.sl_sd = np.std(np.hstack((l_sl,r_sl)))
    out.sw = np.mean(np.hstack((l_sw,r_sw)))
    out.sw_sd = np.std(np.hstack((l_sw,r_sw)))
    return out

def calcStateSpace(sig,events,fs,n_dim,delay):
# =============================================================================
#     Calculate state space using n_dimension with time delay [delay] in samples
# =============================================================================
    from scipy.interpolate import interp1d
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


def calcLDE(state,ws,fs,period,nnbs,visualisation):
# =============================================================================
#     Calculate Local Differgence Exponent
# =============================================================================
    win = np.round(ws*fs)
    [m,n] = state.shape
    state = np.vstack((state,np.zeros([win,n])*np.nan))
    divmat = np.zeros([m*nnbs,win])*np.nan
    difmat = np.zeros([m+win,n])*np.nan
    L1 = .5*period*fs
    for i_t in range(m):
        for i_d in range(n):
            difmat[:,i_d] = (np.subtract(state[:,i_d],state[i_t,i_d]))**2
        arridx1 = np.array([1,i_t-np.round(L1)])
        arridx2 = np.array([m,i_t+np.round(L1)])
        idx_start = int(np.round(np.amax(arridx1))-1)
        idx_stop = int(np.round(np.amin(arridx2))-1)
        difmat[idx_start:idx_stop] = difmat[idx_start:idx_stop]*np.nan
        idx_sort = np.sum(difmat,1).argsort()
        for i_nn in range(nnbs):
            div_tmp = np.subtract(state[i_t:i_t+win,:],state[idx_sort[i_nn]:(idx_sort[i_nn]+win),:])
            divmat[i_t*nnbs+i_nn,:] = np.sum(div_tmp**2,1)**.5 # track divergence, and store
        
    divergence = np.nanmean(np.log(divmat),0)
    xdiv = np.linspace(1,divergence.shape[0],divergence.shape[0])/fs
    xs = xdiv[1:int(np.floor(L1))]
    Ps = np.polynomial.polynomial.polyfit(xs,divergence[1:int(np.floor(L1))],1)    
    sfit = np.polynomial.Polynomial(Ps)

    lde = np.ones([1,2])*np.nan
    lde[0] = Ps[1]
    
    L2 = np.round(4*period*fs)    
    if L2 < win:
        idxl = (np.linspace(0,win-L2-1,int(win-L2))+int(L2))
        Pl = np.polynomial.Polynomial.fit(idxl/fs,divergence[int(idxl[0]):int(idxl[-1])],1)
        lfit = np.polynomial.Polynomial(Pl)
        lde[1] = Pl[1]
    
    if visualisation:
        import matplotlib.pyplot as plt
        plt.plot(xdiv,divergence,label='divergence')
        plt.plot(xs,sfit(xs),label='LD short')
        if L2<win:
            plt.plot(idxl/fs,lfit(idxl/fs),label='fit_long')
        plt.xlabel('time [s]')
        plt.ylabel('ln(divergence)')
        plt.legend()
        plt.show
    return divergence,lde

def calcFPE(joint,com,angMom,comIR,mass,TMspeed,TMangle):
# =============================================================================
#     Calculate Foot Placement Estimator
# =============================================================================
    # get time vector
    t = joint.time/1000
    # get sampling frequency
    fs = 1/np.nanmean(np.diff(t))
    # determine the gravitational acceleration
    g = 9.81 #m/s
    # create new foot position
    # l_foot = pd.concat([(joint.l_TTII_x+joint.l_heel_x)/2,(joint.l_TTII_y+joint.l_heel_y)/2,(joint.l_TTII_z+joint.l_heel_z)/2],axis=1)    
    # determine walking directions
    # walkdir = np.argmax(np.ptp(np.array(l_foot),axis=0))
    # get treadmill BLMs
    #LA = pd.concat([joint.PlatformLA_x, joint.PlatformLA_y,joint.PlatformLA_z],axis=1)
    #LB = pd.concat([joint.PlatformLB_x, joint.PlatformLB_y,joint.PlatformLB_z],axis=1)
    #RA = pd.concat([joint.PlatformRA_x, joint.PlatformRA_y,joint.PlatformRA_z],axis=1)
    #RB = pd.concat([joint.PlatformRB_x, joint.PlatformRB_y,joint.PlatformRB_z],axis=1)
    
    # get CoM position in m
    # com = com/1000;
    # create empty omega
    omega = np.ones([t.shape[0],3])*np.nan
    # get angular momentum matrix
    H = np.vstack([angMom.H_x,angMom.H_y,angMom.H_z]).transpose()
    # rearrange IR and calculate omega   
    for i in range(comIR.shape[0]):
        # get inertial tensor and reshape 
        IR = np.vstack([np.hstack([comIR.IRxx[i],comIR.IRxy[i],comIR.IRxz[i]]),
                        np.hstack([comIR.IRyx[i],comIR.IRyy[i],comIR.IRyz[i]]),
                        np.hstack([comIR.IRzx[i],comIR.IRzy[i],comIR.IRzz[i]])])
        # multiply the angular momentum with the inverse of the inertial tensor
        omega[i,:] = np.matmul(np.transpose(H[i,:]),np.linalg.inv(IR))

    # Bring the data into the direction of the angular momentum
    com_arr = np.array(np.vstack([com.COMx,com.COMy,com.COMz])).transpose()
    com_proj = np.array(np.vstack([com.COMx,com.COMy,com.COMz*0])).transpose()
    # calculate the velocity of the CoM and add the current treadmill speed [m/s]
    vcom = (np.gradient(com_arr,axis=0)*fs)+np.hstack(TMspeed)
    H_y_ax = H+np.cross((com_arr-com_proj)*mass,vcom)
    yaxis = np.vstack([H_y_ax[:,0],H_y_ax[:,1],np.zeros([1,com.shape[0]])]).transpose()
    zaxis = np.divide((com_arr-com_proj),com_arr)
    xaxis = np.cross(yaxis,zaxis)
    yaxis = yaxis/norm_col(yaxis)
    zaxis = zaxis/norm_col(zaxis)    
    xaxis = xaxis/norm_col(xaxis)
    R_proj = np.hstack([xaxis,yaxis,zaxis])
    omega_proj = prod_col(transpose_col(R_proj),omega)
    vcom_proj = prod_col(transpose_col(R_proj),vcom)
    com_proj = prod_col(transpose_col(R_proj),com_arr)
    # treadmill angle expressed in the direction of angular momentum
    angTM=np.tile(np.radians(TMangle),[vcom.shape[0],1])
    theta_proj = prod_col(transpose_col(R_proj),angTM)
    # import matplotlib.pyplot as plt
    # plt.plot(t,theta_proj,label = 'treadmill angle')
    # plt.xlabel('Time')
    # plt.ylabel('Angle in radians')
    # plt.legend()
    
    # create empty matrices
    lFPE = np.zeros([vcom.shape[0],1])*np.nan
    phi =  np.zeros([vcom.shape[0],1])*np.nan
    phi[0]=np.radians(3)
    # fmsval =  np.zeros([vcom.shape[0],1])*np.nan
    Jcom =  np.zeros([vcom.shape[0],1])*np.nan
    # loop over all frames
    for i in range(vcom.shape[0]):
        # Get inertial tensor and reshape
        IR = np.vstack([np.hstack([comIR.IRxx[i],comIR.IRxy[i],comIR.IRxz[i]]),
                np.hstack([comIR.IRyx[i],comIR.IRyy[i],comIR.IRyz[i]]),
                np.hstack([comIR.IRzx[i],comIR.IRzy[i],comIR.IRzz[i]])])
        # calculate the moment of inertia in the new direction
        Jcom[i] = np.dot(np.dot(yaxis[i,:],IR),yaxis[i,:])
        if vcom_proj[i,0] == vcom_proj[i,0]:
            # pdb.set_trace()
            phi[i] = optimize.fmin(fFPE,phi[i-1],args = (Jcom[i,0],com_proj[i,2],g,mass,omega_proj[i,1],theta_proj[i,1],vcom_proj[i,0],vcom_proj[i,1]),disp=0)
            # phi[i] = optimize.brentq(fFPE,0,3,args = (Jcom[i,0],com_proj[i,2],g,mass,omega_proj[i,1],theta_proj[i,1],vcom_proj[i,0],vcom_proj[i,1]),xtol=1e-6,rtol=1e-7)             
            lFPE[i] = ((np.cos(theta_proj[i,1])*(com_arr[i,1]))/(np.cos(phi[i])))*np.sin(theta_proj[i,1]+phi[i])
    return lFPE
        
def fFPE(phi,Jcom,hcom,g,m,omega,theta,vx,vy):
# =============================================================================
#     Foot placement estimator formula to optimze
# =============================================================================
    t2 = np.cos(phi)
    t3 = np.cos(theta)
    t4 = theta+phi
    t5 = t3**2
    t7 = 1/t2
    t8 = t7**2
    t9 = hcom**2
    fpefun = (Jcom/2+(m*t5*t8*t9)/2)*((Jcom*omega+m*t3*t7*hcom*(vx*np.cos(t4)+vy*np.sin(t4)))**2)*1/((Jcom+m*t5*t8*t9)**2)-g*m*t3*t7*hcom+g*m*t3*t7*hcom*np.cos(t4)
    return fpefun**2
    
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
    
    
    
def calcFPM(joint,com,events,fs):
# =============================================================================
#     Create Foot Placement Model and calculate estimates and goodness of fit
# =============================================================================
    # from sklearn import linear_model#,metrics
    import statsmodels.api as sm

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
    # acom_l = timenormalize_step(acom, lto, lhs)    
    # acom_r = timenormalize_step(acom, rto, rhs)
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
    for i in range(0,51):
        print(i)
        # samples of left foot
        com_l_samp  = com_l[i,:]
        vcom_l_samp = vcom_l[i,:]
        # acom_l_samp = acom_l[i,:]
        # samples of right foot
        com_r_samp  = com_r[i,:]
        vcom_r_samp = vcom_r[i,:]
        # acom_r_samp = acom_r[i,:]
        # # predictors
        # pred_lstance = np.vstack([com_l_samp,vcom_l_samp,acom_l_samp])
        # pred_rstance = np.vstack([com_r_samp,vcom_r_samp,acom_r_samp])
        # # remove origin
        # pred_lstance = pred_lstance-np.vstack([orig_l,orig_l,orig_l])
        # pred_rstance = pred_rstance-np.vstack([orig_r,orig_r,orig_r])
        # predictors
        pred_lstance = np.vstack([com_l_samp,vcom_l_samp])
        pred_rstance = np.vstack([com_r_samp,vcom_r_samp])
        # remove origin
        pred_lstance = pred_lstance-np.vstack([orig_l,orig_l])
        pred_rstance = pred_rstance-np.vstack([orig_r,orig_r])
        # remove mean
        pred_lstance = pred_lstance-np.matlib.repmat(np.nanmean(pred_lstance,axis=1),com_l.shape[1],1).transpose()
        pred_rstance = pred_rstance-np.matlib.repmat(np.nanmean(pred_rstance,axis=1),com_r.shape[1],1).transpose()
# =============================================================================
#         Calculation for left steps
# =============================================================================
        # pdb.set_trace()
        df_l = pd.DataFrame(np.vstack([foot_l,pred_lstance]).transpose(),columns=['foot_pos','pred0','pred1'])
        X_l = sm.add_constant(df_l[['pred0','pred1']])
        reg_l = sm.OLS(df_l[['foot_pos']],X_l).fit()
        # print(reg_l.summary())
        rsq_l[i] = reg_l.rsquared
        FPMl_est[:,i] = reg_l.predict(X_l)
        # reg_l = linear_model.LinearRegression()
        # reg_l.fit(pred_lstance.transpose(),foot_l.reshape(-1,1))
        # get coefficient of determination
        # rsq_l[i] = reg_l.score(pred_lstance.transpose(), foot_l.reshape(-1,1))
        # get predictions
        # FPMl_est[:,i] = reg_l.predict(pred_lstance.transpose()).transpose()
        # rsq_l[i] = metrics.r2_score(foot_l.reshape(-1,1),FPMl_est[:,i])
# =============================================================================
#         Calculation for right steps
# =============================================================================
        df_r = pd.DataFrame(np.vstack([foot_r,pred_rstance]).transpose(),columns=['foot_pos','pred0','pred1'])
        X_r = sm.add_constant(df_r[['pred0','pred1']])
        reg_r = sm.OLS(df_r[['foot_pos']],X_r).fit()
        # print(reg_l.summary())
        rsq_r[i] = reg_r.rsquared
        FPMr_est[:,i] = reg_r.predict(X_r)
        # reg_r = linear_model.LinearRegression()
        # reg_r.fit(pred_rstance.transpose(),foot_r.reshape(-1,1))        
        # get coefficient of determination
        # rsq_r[i] = reg_r.score(pred_rstance.transpose(), foot_r.reshape(-1,1))
        # get predictions
        # FPMr_est[:,i] = reg_r.predict(pred_rstance.transpose()).transpose()
        # rsq_l[i] = metrics.r2_score(foot_l.reshape(-1,1),FPMl_est[:,i])
    return FPMl_est,FPMr_est,rsq_l,rsq_r,com_l

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
    