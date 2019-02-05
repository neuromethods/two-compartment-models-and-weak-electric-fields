import numpy as np
import scipy.optimize
import scipy.special
import numba
import multiprocessing
import itertools
import time

'''
Core and utility functions for simulations, semi-analytical calculations,
and for analyses of spike trains / voltage time series
'''

def som_impedance_BS(f_vec, params):
    w_vec = 2*np.pi*f_vec
    alpha = np.sqrt((params['gm'] + np.sqrt(params['gm']**2 + \
                     w_vec**2*params['cm']**2))/(2*params['gi']))
    beta = np.sqrt((-params['gm'] + np.sqrt(params['gm']**2 + \
                    w_vec**2*params['cm']**2))/(2*params['gi']))
    z = alpha + 1j*beta  #z2 = -z (see notes page)
    z[w_vec<0] = alpha[w_vec<0] - 1j*beta[w_vec<0] #!!! in case of neg. freq.!! 
    # see https://en.wikipedia.org/wiki/Square_root#Algebraic_formula
    V1_BS_over_Is1 = 1/((params['Cs']*w_vec*1j + params['Gs']) + \
                        z*params['gi']*np.tanh(z*params['L']))
    V1_BS_over_Id1 = V1_BS_over_Is1 / np.cosh(z*params['L'])                    
    return V1_BS_over_Is1, V1_BS_over_Id1                 

    

def fit_2C_params(freq_vals, pBS): 
    # somatic impedance of BS model
    V1_BS_over_Is1, V1_BS_over_Id1 = som_impedance_BS(freq_vals, pBS)
    w_vec = 2*np.pi*freq_vals
    args = (pBS['Cs'], pBS['Gs'], pBS['cm'], pBS['gm'], pBS['gi'], 
            pBS['L'], w_vec, V1_BS_over_Is1, V1_BS_over_Id1)
    p0 = np.array([1.0,1.0,1.0])
    args = (pBS['Cs'], pBS['Gs'], pBS['cm'], pBS['gm'], pBS['gi'], pBS['L'])
    ydata = np.concatenate([np.real(V1_BS_over_Is1), np.imag(V1_BS_over_Is1), 
                            np.real(V1_BS_over_Id1), np.imag(V1_BS_over_Id1),
                            np.real(V1_BS_over_Id1-V1_BS_over_Is1), 
                            np.imag(V1_BS_over_Id1-V1_BS_over_Is1)])                                                             
    popt, _ = scipy.optimize.curve_fit(make_V1_2C_Isd_normalized(args), w_vec, 
                                       ydata, p0=p0, bounds=(0.0, np.inf), 
                                       xtol=1e-9, ftol=1e-9, verbose=1)
                                      
    Cs, Gs, Cd = popt                                                        
    Cs *= pBS['Cs'];  Gs *= pBS['Gs'];  Cd *= pBS['cm']*pBS['L'];
    Gd = (pBS['Gs'] + pBS['gi']/pBS['lambd']*np.tanh(pBS['L']/pBS['lambd']) - Gs) * \
         np.cosh(pBS['L']/pBS['lambd'])
    Gj = (pBS['Gs'] + pBS['gi']/pBS['lambd']*np.tanh(pBS['L']/pBS['lambd']) - Gs) / \
         (1.0 - 1/np.cosh(pBS['L']/pBS['lambd'])) 
    # determine Vr_2C by fitting BS somatic V trace after reset (spike) 
    # for const. input Id0 that yields (threshold or) close-to-threshold V=V0
    # e.g. Vs_inf 1mV below VT, with dendritic depolarization according to Id0,
    # and determine Vr2C by least-sq. fitting that trace over, say, T=10*taum, with taum=Cs/(Gs+Gj);    
    # V-traces are (semi)analytically obtained using Laplace transform  
    Vs_inf = pBS['VT'] 
    Is0 = Vs_inf*(pBS['Gs'] + np.tanh(pBS['L']/pBS['lambd'])*pBS['gi']/pBS['lambd'])
    Id0 = Vs_inf*(np.cosh(pBS['L']/pBS['lambd'])*pBS['Gs'] + \
                  np.sinh(pBS['L']/pBS['lambd'])*pBS['gi']/pBS['lambd'])                          
    taus = Cs/(Gs+Gj)
    T = 1*taus  #10*pBS['Cs']/(pBS['Gs']+Gj)
    tgrid = np.linspace(0.001,T,10)  # 0 will be excluded by the inverse Laplace method
    Vr2C_vals = np.arange(-0.001,0.006,0.0001)  # use larger range for general purpose
#    DeltaVd_vals = np.arange(-0.005,0.005,0.00025)
    DeltaVd_vals = np.array([0.0])
    errors = np.zeros((len(Vr2C_vals),len(DeltaVd_vals)))
    for i in range(len(Vr2C_vals)):
        for j in range(len(DeltaVd_vals)):
            Vr2C = Vr2C_vals[i];  DeltaVd = DeltaVd_vals[j]; 
            # Is0 from above, Id0 = 0:
            times, VsBS, Vs2C = Vs_traces_laplace(pBS,Vs_inf,Is0,0.0,Cs,Gs,Cd,Gd,Gj,Vr2C,DeltaVd,tgrid)
            error1 = np.sum(np.abs(Vs2C-VsBS))
            # Is0 = 0, Id0 from above:
            times, VsBS, Vs2C = Vs_traces_laplace(pBS,Vs_inf,0.0,Id0,Cs,Gs,Cd,Gd,Gj,Vr2C,DeltaVd,tgrid)
            error2 = np.sum(np.abs(Vs2C-VsBS))            
            errors[i,j] = error1 + error2
    i,j = np.unravel_index(errors.argmin(),errors.shape)    
    Vr2C = Vr2C_vals[i]
    DeltaVd = DeltaVd_vals[j]
    tgrid = np.linspace(0,T,200)  # for plotting; 0 will be excluded by the inverse Laplace method
    t1, VsBS_Is, Vs2C_Is = Vs_traces_laplace(pBS,Vs_inf,Is0,0.0,Cs,Gs,Cd,Gd,Gj,Vr2C,DeltaVd,tgrid)
    t2, VsBS_Id, Vs2C_Id = Vs_traces_laplace(pBS,Vs_inf,0.0,Id0,Cs,Gs,Cd,Gd,Gj,Vr2C,DeltaVd,tgrid)
    params2C = {}       
    params2C['Cs'] = Cs    
    params2C['Gs'] = Gs 
    params2C['Cd'] = Cd
    params2C['Gd'] = Gd
    params2C['Gj'] = Gj
    params2C['Vr'] = Vr2C 
    params2C['DeltaV'] = DeltaVd  # this should be 0
    params2C['VT'] = pBS['VT']    
    params2C['Gexp'] = Cs*pBS['Gs']/pBS['Cs']                         
    params2C['DeltaT'] = pBS['DeltaT']
    params2C['Vth'] = pBS['Vth'] 
    params2C['Delta'] = pBS['gi']/params2C['Gj']  #(m)
    V1_2C_over_Is1 = som_impedance_2C_Is(freq_vals, params2C)
    V1_2C_over_Id1 = som_impedance_2C_Id(freq_vals, params2C)
    # somatic voltage response to field (BS model)
    V1_BS_over_E1 = pBS['gi'] * (V1_BS_over_Id1-V1_BS_over_Is1)
    V1_2C_over_E1 = params2C['Gj']*params2C['Delta']*(V1_2C_over_Id1 - V1_2C_over_Is1)
    fitcurves = {}
    fitcurves['freq_vals'] = freq_vals
    fitcurves['Zhat_Is_BS'] = V1_BS_over_Is1
    fitcurves['Zhat_Id_BS'] = V1_BS_over_Id1
    fitcurves['Zhat_Is_2C'] = V1_2C_over_Is1
    fitcurves['Zhat_Id_2C'] = V1_2C_over_Id1
    fitcurves['Vshat_over_E1_BS'] = V1_BS_over_E1
    fitcurves['Vshat_over_E1_2C'] = V1_2C_over_E1
    fitcurves['postspike_time_vals'] = tgrid
    fitcurves['Vs_postspike_Is_BS'] = np.concatenate(([pBS['Vr']], VsBS_Is))
    fitcurves['Vs_postspike_Id_BS'] = np.concatenate(([pBS['Vr']], VsBS_Id))
    fitcurves['Vs_postspike_Is_2C'] = np.concatenate(([params2C['Vr']], Vs2C_Is))
    fitcurves['Vs_postspike_Id_2C'] = np.concatenate(([params2C['Vr']], Vs2C_Id))
    return params2C, fitcurves


    
def make_V1_2C_Isd_normalized(args):
    def V1_2C_Isd_normalized(w_vec, *p):
        CsBS, GsBS, cm, gm, gi, L = args
        Cs, Gs, Cd = p
        Cs *= CsBS;  Gs *= GsBS;  Cd *= cm*L;
        lambd = np.sqrt(gi/gm);
        Gd = (GsBS + gi/lambd*np.tanh(L/lambd) - Gs) * np.cosh(L/lambd)
        Gj = (GsBS + gi/lambd*np.tanh(L/lambd) - Gs) / (1.0 - 1/np.cosh(L/lambd))
        V1_2C_over_Is1 = (Cd*w_vec*1j + Gd + Gj)/ \
                         ((Cs*w_vec*1j + Gs + Gj)*(Cd*w_vec*1j + Gd + Gj) - Gj**2)
        V1_2C_over_Id1 = Gj/((Cs*w_vec*1j + Gs + Gj)*(Cd*w_vec*1j + Gd + Gj) - Gj**2)
        return np.concatenate([np.real(V1_2C_over_Is1), np.imag(V1_2C_over_Is1), 
                               np.real(V1_2C_over_Id1), np.imag(V1_2C_over_Id1),
                               np.real(V1_2C_over_Id1-V1_2C_over_Is1),
                               np.imag(V1_2C_over_Id1-V1_2C_over_Is1)])
    return V1_2C_Isd_normalized
          
    

def Vs_traces_laplace(pBS,Vs_inf,Is0,Id0,Cs,Gs,Cd,Gd,Gj,Vr2C,DeltaVd,tgrid):
    # see INVLAP.m based on 
    # the method uses default values a=6, ns=20, nd=19
    # it is recommended to preserve a=6
    # increasing ns and nd leads to lower error
    a=6;  ns=20;  nd=19;
    if tgrid[0]==0:  
        tgrid = tgrid[1:]  # t=0 is not allowed
    alfa = 1j*np.zeros(ns+1+nd)
    beta = np.zeros(ns+1+nd)
    for n in range(len(alfa)): #prepare necessary coefficients
       alfa[n] = a + n*np.pi*1j
       beta[n] = -np.exp(a)*(-1)**(n+1);
    n = np.arange(1,nd+1)
    bdif = np.cumsum( scipy.special.gamma(nd+1)/scipy.special.gamma(nd+2-n)/
                      scipy.special.gamma(n) )/2**nd
    bdif = bdif[::-1]                  
    beta[ns+1:ns+1+nd] = beta[ns+1:ns+1+nd]*bdif
    beta[0] /= 2.0   
    dV = 0.002*pBS['L'];  Vgrid = np.arange(0,pBS['L']-dV/2,dV);
    VsBS = np.zeros_like(tgrid)
    Vs2C = np.zeros_like(tgrid)    
    for it in range(len(tgrid)):
        tp = tgrid[it]
        s_vals = alfa/tp  # complex frequencies s
        bt_vals = beta/tp
        btF = bt_vals*Vs_BS_laplace(s_vals,Vs_inf,Is0,Id0,pBS['Cs'],pBS['Gs'],
                                    pBS['cm'],pBS['gm'],pBS['gi'],pBS['lambd'],
                                    pBS['L'],Vgrid)   # Vshat(s)
        VsBS[it] = np.sum(np.real(btF))  #original Vs(t)
        btF = bt_vals*Vs_2C_laplace(s_vals,Vs_inf,Is0,Id0,Cs,Gs,
                                    Cd,Gd,Gj,Vr2C,DeltaVd)   # Vshat(s)
        Vs2C[it] = np.sum(np.real(btF))  #original Vs(t)
    return tgrid, VsBS, Vs2C

#@numba.njit    
def Vs_BS_laplace(s_vec,Vs_inf,Is0,Id0,Cs,Gs,cm,gm,gi,lambd,L,Vgrid):
    alpha = np.sqrt( (gm + cm*np.real(s_vec) + np.sqrt((gm + cm*np.real(s_vec))**2 + \
                     np.imag(s_vec)**2*cm**2))/(2*gi) )
    beta = np.sqrt( (-gm - cm*np.real(s_vec) + np.sqrt((gm + cm*np.real(s_vec))**2 + \
                     np.imag(s_vec)**2*cm**2))/(2*gi) )
    z_vec = alpha + 1j*beta
    dV = Vgrid[1]-Vgrid[0]
    Vr = Vgrid[0]
    if Is0>0 and Id0==0:
        integral = dV*np.sum(np.cosh(np.dot(L-np.array([Vgrid]).T,np.array([z_vec]))) * \
                             np.cosh((L-np.array([Vgrid]).T)/lambd), axis=0)
        integral_summand = cm*Vs_inf/np.cosh(L/lambd) * integral
    elif Is0==0 and Id0>0:
        integral = dV*np.sum(np.cosh(np.dot(L-np.array([Vgrid]).T,np.array([z_vec]))) * \
                             ( np.cosh(np.array([Vgrid]).T/lambd) + \
                              Gs*lambd/gi * np.sinh(np.array([Vgrid]).T/lambd) ), axis=0)        
        integral_summand = cm*Vs_inf * integral

    out = ( (Cs*Vr + Is0/s_vec) * np.cosh(z_vec*L) + integral_summand + Id0/s_vec ) / \
          ( (Cs*s_vec + Gs) * np.cosh(z_vec*L) + gi*z_vec*np.sinh(z_vec*L) )
    return out

#@numba.njit 
def Vs_2C_laplace(s_vec,Vs_inf,Is0,Id0,Cs,Gs,Cd,Gd,Gj,Vr2C,DeltaVd):
    Vd0 = (Gj*Vs_inf + Id0)/(Gd + Gj) + DeltaVd    
    out = ( (Is0/s_vec + Cs*Vr2C) * (Cd*s_vec + Gd + Gj) + Gj*(Cd*Vd0 + Id0/s_vec) ) / \
          ( (Cs*s_vec + Gs + Gj) * (Cd*s_vec + Gd + Gj) - Gj**2 )
    return out
    
    
def som_impedance_2C_Is(f_vec, params):
    w_vec = 2*np.pi*f_vec
    Cs = params['Cs'];  Gs = params['Gs'];  
    Cd = params['Cd']; Gd = params['Gd'];  Gj = params['Gj'];
    V1_2C_over_I1 = (Cd*w_vec*1j + Gd + Gj)/ \
                    ((Cs*w_vec*1j + Gs + Gj)*(Cd*w_vec*1j + Gd + Gj) - Gj**2)
    return V1_2C_over_I1
    
def som_impedance_2C_Id(f_vec, params):
    w_vec = 2*np.pi*f_vec
    Cs = params['Cs'];  Gs = params['Gs'];  
    Cd = params['Cd']; Gd = params['Gd'];  Gj = params['Gj'];
    V1_2C_over_I1 = Gj/((Cs*w_vec*1j + Gs + Gj)*(Cd*w_vec*1j + Gd + Gj) - Gj**2)
    return V1_2C_over_I1    
  
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def sim_model_I0sigma(model, Is0_vals, sigmas_vals, Id0_vals, sigmad_vals, params, N_procs):
    if N_procs <= 1:
        # single processing version, i.e. loop
        pool = False
        if model == 'BS_model':
            result = (simBS_for_given_I0sigma_wrapper(arg_tuple) for arg_tuple in 
                      itertools.product(Is0_vals,sigmas_vals,Id0_vals,sigmad_vals,[params])) 
#        elif model == 'P_model':
#            result = (simP_for_given_I0sigma_wrapper(arg_tuple) for arg_tuple in 
#                      itertools.product(Is0_vals,sigmas_vals,Id0_vals,sigmad_vals,[params]))
        elif model == '2C_model':
            result = (sim2C_for_given_I0sigma_wrapper(arg_tuple) for arg_tuple in 
                      itertools.product(Is0_vals,sigmas_vals,Id0_vals,sigmad_vals,[params]))
    else:
        # multiproc version
        pool = multiprocessing.Pool(N_procs)
        if model == 'BS_model':
            result = pool.imap_unordered(simBS_for_given_I0sigma_wrapper, 
                                         itertools.product(Is0_vals,sigmas_vals,Id0_vals,sigmad_vals,[params]))
#        elif model == 'P_model':
#            result = pool.imap_unordered(simP_for_given_I0sigma_wrapper, 
#                                         itertools.product(Is0_vals,sigmas_vals,Id0_vals,sigmad_vals,[params]))
        elif model == '2C_model':
            result = pool.imap_unordered(sim2C_for_given_I0sigma_wrapper, 
                                         itertools.product(Is0_vals,sigmas_vals,Id0_vals,sigmad_vals,[params]))                                 
    total = len(Is0_vals)*len(sigmas_vals)*len(Id0_vals)*len(sigmad_vals)     
    finished = 0
    Sp_times_dict = {}
    Vs_last_dict = {}
    Vs_last_dict = {}
    Vs_hist_dict = {}
    Vsbin_edges_dict = {}
    # TEMP perhaps:
    if model == '2C_model':  
        Vd_hist_dict = {}
        Vdbin_edges_dict = {}
        for Is0, sigmas, Id0, sigmad, Sp_times, Vs, Vshist, Vsbinedges, Vdhist, Vdbinedges in result:
            finished += 1
            print(('{count} of {tot} ' + model + ' simulations completed').
                    format(count=finished, tot=total))
            Sp_times_dict[Is0,sigmas,Id0,sigmad] = Sp_times
            Vs_last_dict[Is0,sigmas,Id0,sigmad] = Vs 
            Vs_hist_dict[Is0,sigmas,Id0,sigmad] = Vshist
            Vsbin_edges_dict[Is0,sigmas,Id0,sigmad] = Vsbinedges
            Vd_hist_dict[Is0,sigmas,Id0,sigmad] = Vdhist
            Vdbin_edges_dict[Is0,sigmas,Id0,sigmad] = Vdbinedges
        if pool:
            pool.close()
        return Sp_times_dict, Vs_last_dict, Vs_hist_dict, Vsbin_edges_dict, Vd_hist_dict, Vdbin_edges_dict
    else:
        for Is0, sigmas, Id0, sigmad, Sp_times, Vs, Vshist, binedges in result:
            finished += 1
            print(('{count} of {tot} ' + model + ' simulations completed').
                    format(count=finished, tot=total))
            Sp_times_dict[Is0,sigmas,Id0,sigmad] = Sp_times
            Vs_last_dict[Is0,sigmas,Id0,sigmad] = Vs 
            Vs_hist_dict[Is0,sigmas,Id0,sigmad] = Vshist
            Vsbin_edges_dict[Is0,sigmas,Id0,sigmad] = binedges
        if pool:
            pool.close()
        return Sp_times_dict, Vs_last_dict, Vs_hist_dict, Vsbin_edges_dict

def simBS_for_given_I0sigma_wrapper(arg_tuple):
    Is0 = arg_tuple[0]
    sigmas = arg_tuple[1]  
    Id0 = arg_tuple[2]
    sigmad = arg_tuple[3]
    pdict = arg_tuple[4]
    tgrid = np.arange(0,pdict['t_end']+pdict['dt']/2,pdict['dt'])
    dx = pdict['dx']
    Is0_vec = Is0*np.ones(len(tgrid))
    Id0_vec = Id0*np.ones(len(tgrid))
    V_init_vec = pdict['V_init'].copy();  M = pdict['M'];
    Cs = pdict['Cs'];  Gs = pdict['Gs'];  cm = pdict['cm'];  gm = pdict['gm'];  
    gi = pdict['gi'];  Vr = pdict['Vr'];  VT = pdict['VT'];  DeltaT = pdict['DeltaT'];
    Vth = pdict['Vth']
    np.random.seed(pdict['seed'])
    rands_vec = np.random.randn(len(tgrid))
    np.random.seed(pdict['seed']+100)
    randd_vec = np.random.randn(len(tgrid))
    tau_input = 0    

    V_soma, Sp_times = simulate_BS_CraNic_numba(tgrid,dx,V_init_vec,M,Cs,Gs,
                                                cm,gm,gi,DeltaT,Vr,VT,Vth,
                                                Is0_vec,sigmas,rands_vec,
                                                Id0_vec,sigmad,randd_vec,tau_input)                                            
    Sp_times = Sp_times[Sp_times>=pdict['t_min']]
    Vs_last = V_soma[tgrid>=pdict['t_end']-pdict['t_Vs_save']]  
    dVhist = 0.25e-3  #(V)
    Vshist, binedges = np.histogram(V_soma[tgrid>=pdict['t_min']], 
                                    bins=np.arange(-0.05, Vth+dVhist/2,dVhist),
                                    density=True)                                         
    return (Is0, sigmas, Id0, sigmad, Sp_times, Vs_last, Vshist, binedges)
    

def sim2C_for_given_I0sigma_wrapper(arg_tuple):
    Is0 = arg_tuple[0]
    sigmas = arg_tuple[1]  
    Id0 = arg_tuple[2]
    sigmad = arg_tuple[3]
    pdict = arg_tuple[4]
    tgrid = np.arange(0,pdict['t_end']+pdict['dt']/2,pdict['dt'])
    Is0_vec = Is0*np.ones(len(tgrid))
    Id0_vec = Id0*np.ones(len(tgrid))
    V_init_vec = pdict['V_init'].copy();  Vr = pdict['Vr'];  VT = pdict['VT'];
    Cs = pdict['Cs'];  Gs = pdict['Gs'];  Cd = pdict['Cd'];  Gd = pdict['Gd'];
    Gj = pdict['Gj'];  DeltaT = pdict['DeltaT'];  DeltaV = pdict['DeltaV'];
    Gexp = pdict['Gexp'];  Vth = pdict['Vth']
    if 'Ge' in pdict.keys():
        Ge = pdict['Ge']
    else:
        Ge = 0.0
    np.random.seed(pdict['seed'])
    rands_vec = np.random.randn(len(tgrid))
    np.random.seed(pdict['seed']+100)
    randd_vec = np.random.randn(len(tgrid))
    tau_input = 0  
    
    #TEMP:
    if Ge>0:  # TODO: closed-loop only for white noise input so far
        fac2 = Gj/(2*(Ge+Gj));
        Is_tmp = Is0_vec * (1.0-fac2) + fac2*Id0_vec
        Id_tmp = Id0_vec * (1.0-fac2) + fac2*Is0_vec  
        print 'actual mus, mud = '
        print Is_tmp[0]/Cs, Id_tmp[0]/Cd 

    V_soma, V_dend, Sp_times = simulate_2C_numba(tgrid,V_init_vec,Cs,Gs,Cd,Gd,Gj,Ge,
                                                 Gexp,DeltaT,Vr,VT,Vth,DeltaV,
                                                 Is0_vec,sigmas,rands_vec,
                                                 Id0_vec,sigmad,randd_vec,tau_input)
    Sp_times = Sp_times[Sp_times>=pdict['t_min']]
    Vs_last = V_soma[tgrid>=pdict['t_end']-pdict['t_Vs_save']]                                           
    dVhist = 0.25e-3  #(V)
    Vshist, binedges = np.histogram(V_soma[tgrid>=pdict['t_min']], 
                                    bins=np.arange(-0.05, Vth+dVhist/2,dVhist),
                                    density=True)    
    Vdhist, dbinedges = np.histogram(V_dend[tgrid>=pdict['t_min']], 
                                     bins=np.arange(-0.05, Vth+0.05+dVhist/2,dVhist),
                                     density=True)      
#    Vd_var = np.var(V_dend)                                
    return (Is0, sigmas, Id0, sigmad, Sp_times, Vs_last, Vshist, binedges, Vdhist, dbinedges) 
    
@numba.njit
def simulate_BS_CraNic_numba(tgrid,dx,V_init_vec,M,Cs,Gs,cm,gm,gi,DeltaT,Vr,VT,Vth,
                             Is0_vec,sigmas,rands_vec,Id0_vec,sigmad,randd_vec,taus): #,tminSTA,STArange):
    # Crank-Nicolson method, using 
    # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    dt = tgrid[1] - tgrid[0]   
    V_soma = V_init_vec[0]*np.ones(len(tgrid))
    s_input = Is0_vec[0]*np.ones(len(tgrid))
    d_input = Id0_vec[0]*np.ones(len(tgrid))
    Sp_times_dummy = np.zeros(int(len(tgrid)/100)) 
    sp_count = int(0)  
    V = V_init_vec
    n = len(V_init_vec)
    forwardrange = range(n)
    backwardrange = range(n-2,-1,-1)
    trange = range(1,len(tgrid))
    k = int(np.round(M/dx))
    sqrt_dt = np.sqrt(dt)
    if taus>0:
        fs = 1-dt/taus
        Is = dt*Is0_vec/taus 
        Id = dt*Id0_vec/taus
    sigsdtnoise = sigmas*sqrt_dt*rands_vec
    sigddtnoise = sigmad*sqrt_dt*randd_vec
    b1 = Cs/dt+gi/(2*dx)+Gs/2;  c1 = -gi/(2*dx);
    bj = cm/dt+gi/dx**2 +gm/2;  cj = -gi/(2*dx**2);  an = -gi/dx**2;
    f1 = Cs/dt-gi/(2*dx)-Gs/2;  f2 = gi/(2*dx);  f3 = gi/(2*dx**2);
    f4 = gi/dx**2;  f5 = cm/dt-gi/dx**2 -gm/2;  f6 = Gs*DeltaT;
    if not f6>0:
        DeltaT = 1.0  # to make sure we don't get errors below
    cnew = np.ones(n-1)
    d = np.ones(n)
    cnew[0] = c1/b1
    for j in range(1,len(cnew)):
        cnew[j] = cj/(bj-cj*cnew[j-1])  
    
#    STA_input = np.zeros(int(STArange/dt))
#    lenSTA = len(STA_input)
#    sp_countSTA = 0
    for i_t in trange:
        if taus>0:
            s_input[i_t] = fs*s_input[i_t-1] + Is[i_t-1] + sigsdtnoise[i_t-1]
            s_input_CN = 0.5*(s_input[i_t-1] + s_input[i_t]) 
            d_input[i_t] = fs*d_input[i_t-1] + Id[i_t-1] + sigddtnoise[i_t-1]
            d_input_CN = 0.5*(d_input[i_t-1] + d_input[i_t])
        else:
            s_input_CN = 0.5*(Is0_vec[i_t-1] + Is0_vec[i_t]) + sigsdtnoise[i_t-1]/dt    
            s_input[i_t-1] = s_input_CN
            d_input_CN = 0.5*(Id0_vec[i_t-1] + Id0_vec[i_t]) + sigddtnoise[i_t-1]/dt    
            d_input[i_t-1] = d_input_CN
        for i_x in forwardrange:        
            if i_x==0:
                d[i_x] = f1*V[i_x] + f2*V[i_x+1] + f6*np.exp((V[i_x]-VT)/DeltaT) + s_input_CN
                d[i_x] = d[i_x]/b1
            elif i_x==n-1:
                d[i_x] = f4*V[i_x-1] + f5*V[i_x] + 2*d_input_CN/dx
                d[i_x] = (d[i_x] - an*d[i_x-1])/(bj - an*cnew[i_x-1])
                V[i_x] = d[i_x]
            else:    
                d[i_x] = f3*(V[i_x-1] + V[i_x+1]) + f5*V[i_x] 
                d[i_x] = (d[i_x] - cj*d[i_x-1])/(bj - cj*cnew[i_x-1])
        for i_x in backwardrange:               
            V[i_x] = d[i_x] - cnew[i_x]*V[i_x+1]    
            if V[0]>Vth:
                V[0] = Vr
                sp_count += 1
                Sp_times_dummy[sp_count-1] = tgrid[i_t]
                # ramp reset
                #V[i_x] += (Vr-V[i_x])*np.max(k-i_x,0)/k --> numba problem
                if k>0 and k<n: 
                    for j in range(1,k):                    
                        V[j] += (Vr-V[j])*(k-j)/k
                elif k>=n:
                    for j in range(1,n):                    
                        V[j] = Vr           
#                # compute STA
#                if tgrid[i_t]>tminSTA:
#                    sp_countSTA += 1
#                    for j in range(lenSTA):
#                        STA_input[j] += s_input[i_t-j-1]                                     
        V_soma[i_t] = V[0]  
        
    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]
#    STA_input = STA_input/sp_countSTA  
    return V_soma, Sp_times #, s_input, STA_input

    
    
@numba.njit 
def simulate_2C_numba(tgrid,V_init_vec,Cs,Gs,Cd,Gd,Gj,Ge,Gexp,DeltaT,Vr,VT,Vth,DeltaV,
                      Is0_vec,sigmas,rands_vec,Id0_vec,sigmad,randd_vec,taus): #,tminSTA,STArange):
    dt = tgrid[1] - tgrid[0]   
    V_soma = V_init_vec[0]*np.ones(len(tgrid))
    V_dend = V_init_vec[1]*np.ones(len(tgrid))
    s_input_dt = Is0_vec[0]*dt*np.ones(len(tgrid))
    d_input_dt = Id0_vec[0]*dt*np.ones(len(tgrid))
    Sp_times_dummy = np.zeros(int(len(tgrid)/10)) 
    sp_count = int(0)  
    V = V_init_vec
    sqrt_dt = np.sqrt(dt)
    if taus>0:
        fs = 1-dt/taus
        Is_dt = dt**2*Is0_vec/taus
        sigs_dtnoise = dt*sigmas*sqrt_dt*rands_vec
        Id_dt = dt**2*Id0_vec/taus
        sigd_dtnoise = dt*sigmad*sqrt_dt*randd_vec
    else:
        Isdt = dt*Is0_vec
        sigsdtnoise = sigmas*sqrt_dt*rands_vec
        Iddt = dt*Id0_vec
        sigddtnoise = sigmad*sqrt_dt*randd_vec
    if Ge>0:  # TODO: closed-loop only for white noise input so far
        fac1 = Gj**2/(Ge+Gj);  fac2 = Gj/(2*(Ge+Gj));
        f1 = dt*(Gj-fac1)/Cs;  f2 = dt/Cs * (fac1 - Gj - Gs);
        f3 = dt*(Gj-fac1)/Cd;  f4 = dt/Cd * (fac1 - Gj - Gd);
        Is_tmp = Is0_vec * (1.0-fac2) + fac2*Id0_vec
        Isdt = dt*Is_tmp
        sigsdtnoise = sigmas * (1.0-fac2) * sqrt_dt*rands_vec
        sigsdtnoise += sigmad * fac2 * sqrt_dt*randd_vec
        Id_tmp = Id0_vec * (1.0-fac2) + fac2*Is0_vec
        Iddt = dt*Id_tmp
        sigddtnoise = sigmad * (1.0-fac2) * sqrt_dt*randd_vec
        sigddtnoise += sigmas * fac2 * sqrt_dt*rands_vec
    else:        
        f1 = dt*Gj/Cs;  f2 = -dt/Cs * (Gj + Gs);
        f3 = dt*Gj/Cd;  f4 = -dt/Cd * (Gj + Gd);  
    f5 = dt*Gexp*DeltaT/Cs;

    if not f5>0:
        DeltaT = 1.0  # to make sure we don't get errors below
#    muSTA = np.zeros(int(STArange/dt))
#    lenSTA = len(muSTA)
#    sp_countSTA = 0   
    for i_t in range(1,len(tgrid)):
        if taus>0:
            s_input_dt[i_t] = fs*s_input_dt[i_t-1] + Is_dt[i_t-1] + sigs_dtnoise[i_t-1]
            d_input_dt[i_t] = fs*d_input_dt[i_t-1] + Id_dt[i_t-1] + sigd_dtnoise[i_t-1]
        else:
            s_input_dt[i_t-1] = Isdt[i_t-1] + sigsdtnoise[i_t-1]  
            d_input_dt[i_t-1] = Iddt[i_t-1] + sigddtnoise[i_t-1]
            
        V[0] += f1*V[1] + f2*V[0] + f5*np.exp((V[0]-VT)/DeltaT) + s_input_dt[i_t-1]/Cs
        V[1] += f3*V[0] + f4*V[1] + d_input_dt[i_t-1]/Cd
        if V[0]>Vth:
            V[0] = Vr
            if DeltaV<(Vr-Vth):
                V[1] = Vr
            else:
                V[1] += DeltaV # inspired by Ostojic 2015
            sp_count += 1
            Sp_times_dummy[sp_count-1] = tgrid[i_t]
#            # compute STA
#            if tgrid[i_t]>tminSTA:
#                sp_countSTA += 1
#                for j in range(lenSTA):
#                    muSTA[j] += mudt[i_t-j-1]               
                      
        V_soma[i_t] = V[0];  V_dend[i_t] = V[1];

    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]
#    s_input = s_input_dt/dt
#    muSTA = muSTA/(sp_countSTA*dt)
    return V_soma, V_dend, Sp_times #, s_input, muSTA

    
# ------------------------------------------------------------------------------
@numba.njit  
def coincidence_factor(ref, comp, window=5, rate=0):
    r"""
    The coincidence factor :math:`\Gamma` between two spike trains is defined as

    .. math::
        
      see Gerstner and Kistler 2002 book

    where :math:`N_{\mathrm{ref}}` are the number of spikes in the reference train,
    :math:`N_{\mathrm{comp}}` is the number of spikes in the comparing train, 
    :math:`N_{\mathrm{coinc}}` is the number of coincident spikes within a time window 
    :math:`\Delta`, :math:`E \left( N_\mathrm{coinc} \right) = 2 v \Delta N_{\mathrm{ref}}` 
    
    
    :param ref: Spike times of the reference train. It is 
        assumed that the spike times are ordered sequentially.
    
    :param comp: Spike times of the train that shifts in time.
        It is assumed that the spike times are ordered sequentially.
    
    :param window: Time window to say that two spikes are synchronized.
        This has a default value of 5.
        
    :param isi: If supplied, this is the isi of the comparing train. Otherwise,
        the rate of the train is computed by taking the last spike minus the
        first spike and dividing by the number of spikes in between.
    
    :return: Coincidence factor
    """

    idx_ref = 0
    idx_comp = 0 
    mask_ref = np.zeros_like(ref) 
    len_ref = len(ref)
    len_comp = len(comp)
    
    while idx_ref < len_ref and idx_comp < len_comp:
        val_a = ref[idx_ref]
        val_b = comp[idx_comp]

        diff = np.abs(val_a - val_b)
        if diff <= window:
            mask_ref[idx_ref] = 1
        
        if val_a == val_b:
            idx_ref += 1
            idx_comp += 1
        else:
            if val_a < val_b:
                idx_ref += 1
            else:
                idx_comp += 1
    num_coincidences = np.sum(mask_ref)
    
#    # alternative (w/o njit), from
#    # http://pythonhosted.org/fit_neuron/_modules/fit_neuron/evaluate/spkd_lib.html#gamma_factor
#    if (len_comp > 1):
#        bins = .5 * (comp[1:] + comp[:-1])
#        indices = np.digitize(ref, bins)
#        diff = abs(ref - comp[indices])
#        matched_spikes = (diff <= window)
#        num_coincidences = sum(matched_spikes)
#    else:
#        indices = [np.amin(abs(comp - ref[i])) <= window for i in xrange(len_ref)]
#        num_coincidences = sum(indices)
        
    total_spikes = len_ref + len_comp
    if len_ref<1 or len_comp<1:
        Gamma = np.nan
    else:
        # for Gamma as described in Badel 2008 JNeurophys, Jolivet 2008 BiolCybern 
        # (v = rate of ref spike train):
#        if not rate:
#            v = (len_ref - 1)/(ref[-1] - ref[0])
#        else:
#            v = rate
        # for Gamma as described in Gerstner&Kistler 2002, Jolivet 2008 JNMethods
        # (v = rate of comp spike train):
        if not rate:
            v = (len_comp - 1)/(comp[-1] - comp[0])
        else:
            v = rate
        expected_coincidences = 2 * v * window * len_ref
        N = 1.0 - 2.0 * v * window
        Gamma = (num_coincidences - expected_coincidences)/(N/2 * total_spikes)
        percent_correct = np.float(num_coincidences)/len_ref
    return Gamma, percent_correct    
# ------------------------------------------------------------------------------

    

def FPsteady_moclo_2C(Is0_vals, sigmas_vals, Id0_vals, sigmad_vals, pdict, N_procs):
    if N_procs <= 1:
        # single processing version, i.e. loop
        pool = False
        result = (FPss_moclo_2C_for_given_musigma_wrapper(arg_tuple) for 
                  arg_tuple in itertools.product(Is0_vals,sigmas_vals,Id0_vals,sigmad_vals,[pdict]))
    else:
        # multiproc version
        pool = multiprocessing.Pool(N_procs) 
        result = pool.imap_unordered(FPss_moclo_2C_for_given_musigma_wrapper, 
                                     itertools.product(Is0_vals,sigmas_vals,Id0_vals,sigmad_vals,[pdict])) 
    total = len(Is0_vals)*len(sigmas_vals)*len(Id0_vals)*len(sigmad_vals)    
    finished = 0
    rates_dict = {};  ps_dict = {};  psh_dict = {};  pshh_dict = {};
    mud_dict = {};  pshint_dict = {};  Vdmean_dict = {};  error_dict = {};
    mudVT_dict = {};  sig2dVT_dict = {};  sig2dc_dict = {};    
    dpsdV_dict = {};  dpshdV_dict = {};  dpshhdV_dict = {};  Vlb_dict = {};   
    for Is0, sigmas, Id0, sigmad, Dout in result:
        finished += 1
        print(('{count} of {tot} FPss_2C calculations completed').format(count=finished, tot=total))      
        rates_dict[Is0,sigmas,Id0,sigmad] = Dout['r']
        ps_dict[Is0,sigmas,Id0,sigmad] = Dout['ps']
        psh_dict[Is0,sigmas,Id0,sigmad] = Dout['psh'] 
        pshh_dict[Is0,sigmas,Id0,sigmad] = Dout['pshh'] 
        mud_dict[Is0,sigmas,Id0,sigmad] = Dout['mud']  
        pshint_dict[Is0,sigmas,Id0,sigmad] = Dout['pshint']  
        Vdmean_dict[Is0,sigmas,Id0,sigmad] = Dout['Vdmean']
        sig2dc_dict[Is0,sigmas,Id0,sigmad] = Dout['sig2dc']  
        mudVT_dict[Is0,sigmas,Id0,sigmad] = Dout['mud_VT']  
        sig2dVT_dict[Is0,sigmas,Id0,sigmad] = Dout['sig2d_VT']  
        error_dict[Is0,sigmas,Id0,sigmad] = Dout['error']
        dpsdV_dict[Is0,sigmas,Id0,sigmad] = Dout['dpsdV']
        dpshdV_dict[Is0,sigmas,Id0,sigmad] = Dout['dpshdV']
        dpshhdV_dict[Is0,sigmas,Id0,sigmad] = Dout['dpshhdV']         
        Vlb_dict[Is0,sigmas,Id0,sigmad] = Dout['Vlb']
    if pool:
        pool.close()    
    Dout = {'rates_dict':rates_dict, 'ps_dict':ps_dict, 'psh_dict':psh_dict, 
            'pshh_dict':pshh_dict, 'mud_dict':mud_dict, 'pshint_dict':pshint_dict, 
            'Vdmean_dict':Vdmean_dict, 'sig2dc_dict':sig2dc_dict, 
            'mudVT_dict':mudVT_dict, 'sig2dVT_dict':sig2dVT_dict, 
            'error_dict':error_dict, 'dpsdV_dict':dpsdV_dict, 'Vlb_dict':Vlb_dict,
            'dpshdV_dict':dpshdV_dict, 'dpshhdV_dict':dpshhdV_dict}    
    return Dout
        

def FPss_moclo_2C_for_given_musigma_wrapper(arg_tuple):  #TODO: clean this one up!
    Dout = {}    
    Is0, sigmas, Id0, sigmad, pdict = arg_tuple   
    mus = Is0/pdict['Cs']
    sigmas /= pdict['Cs']
    mud = Id0/pdict['Cd']
    sigmad /= pdict['Cd']
    V_vec = np.arange(pdict['V_lb'],pdict['Vth']+pdict['dV']/2,pdict['dV'])
    kre = np.argmin(np.abs(V_vec-pdict['Vr']))

    # k=2.5 moclo: looks promising, get this one ready 
    # (Gaussian assumption, i.e. 3rd central moment of Vd is zero) 
    # rough "slicewise" optimization to determine best starting value
    mud3c = 0.0  # if this is 0 we have "k=2.5", if it is optimized we have k=3
    start = time.time()     
    
    
    # TEMP ############################
    # reroute temporarily here
    sigmas_smallest = 15.0/np.sqrt(1000)*1e-12 / pdict['Cs']
    dsigmas = 0.1/np.sqrt(1000)*1e-12 / pdict['Cs']
#        sigmad_largest = 40.0/np.sqrt(1000)*1e-12 / pdict['Cd']
#        dsigmad = 1.0/np.sqrt(1000)*1e-12 / pdict['Cd']
    iterate = False
    if sigmas < sigmas_smallest:
        print 'sigmas seems quite small'
        print ''
        iterate = True
        sigmas_vals = np.arange(sigmas, sigmas_smallest+dsigmas, dsigmas)
        sigmas = sigmas_vals[-1]
#        if sigmad > sigmad_largest: 
#            iterate = True
#            sigmad_vals = np.arange(sigmad, sigmas_smallest+dsigmas, dsigmas)
#            sigmas = sigmas_vals[-1]    
        
    sig2dVT_vec = np.arange(0.05, 1.0, 0.005)*pdict['Vth']**2
    mud_VT_vec = np.arange(0.05, 1.0, 0.005)*pdict['Vth']
#        sig2dVT_vec = np.arange(4.5e-5, 7e-5, 0.025e-5)  #for sigmad>=35/... if sigmas<=15/...
#        mud_VT_vec = np.arange(0.0065, 0.008, 0.000001)  #for sigmad>=35/... if sigmas<=15/...
    #refined grid for, e.g., Is0=Id0=0.5*1e-11, sigmas=12.0/np.sqrt(1000)*1e-12, sigmad=5.0/np.sqrt(1000)*1e-12
    #sig2dVT_vec = np.arange(0.01, 1.0, 0.001)*pdict['VT']**2
    #mud_VT_vec = np.arange(0.2, 1.0, 0.0025)*pdict['VT']
    error_mat = np.zeros((len(mud_VT_vec),len(sig2dVT_vec)))
    qssh_mat = np.zeros((len(mud_VT_vec),len(sig2dVT_vec)))
    qsshh_mat = np.zeros((len(mud_VT_vec),len(sig2dVT_vec)))
    #prep. for FPss_moclok3_2C_IsId_foroptim (to save time):
    mus_orig = mus;  mud_orig = mud;  sigmas_orig = sigmas;  sigmad_orig = sigmad;
    print 'calculating for mus,sigmas,mud,sigmad = '
    print mus,sigmas,mud,sigmad
    
    a = -(pdict['Gs']+pdict['Gj'])/pdict['Cs'];  b = pdict['Gj']/pdict['Cs'];  
    c = pdict['Gj']/pdict['Cd'];  d = -(pdict['Gd']+pdict['Gj'])/pdict['Cd'];
    sigmasd = 0.0;  sigmads = 0.0;  Ge = 0.0;
    
    e = pdict['Gexp']*pdict['DeltaT']/pdict['Cs'];  
    DeltaT = pdict['DeltaT'];  VT = pdict['VT'];

    if not e>0:
        DeltaT = 1.0  # to make sure we don't get errors below
    f1 = a*V_vec + e*np.exp((V_vec-VT)/DeltaT) + mus;  f2 = c*V_vec + mud;
    dV = V_vec[1]-V_vec[0]
    sig2sfactor = 2.0/(sigmas**2 + sigmasd**2)  # sigmas should not be smaller than around 0.03
    sig2d = sigmad**2 + sigmads**2
    sig2crossfactor = (sigmas*sigmads + sigmasd*sigmad)/(sigmas**2 + sigmasd**2)
    n = len(V_vec)
    for j, sig2d_VT in enumerate(sig2dVT_vec):
        for i, mud_VT in enumerate(mud_VT_vec):
            #mud_VT = 0.75*pdict['VT']
            #sig2d_VT = 0.6*pdict['VT']**2
            qssh, qsshh = \
            FPss_moclok3_2C_IsId_foroptim(a,b,c,d,f1,f2,sig2sfactor,sig2d,sig2crossfactor,
                                          dV,kre,n,mud_VT,sig2d_VT,mud3c,pdict['epsilon'])
            # all output quantities must be scaled by r (in the end)             
            qssh_mat[i,j] = qssh
            qsshh_mat[i,j] = qsshh

    error_mat = np.abs(qssh_mat) + np.abs(qsshh_mat)
    print ''
    i,j = np.unravel_index(error_mat.argmin(),error_mat.shape)  
    mud_VT = mud_VT_vec[i]
    sig2d_VT = sig2dVT_vec[j]
    print 'timing for initial grid search: ', time.time()-start 
    print 'i, j, mud_VT, sig2d_VT = ', i, j, mud_VT, sig2d_VT 
    print 'error = ', error_mat[i,j]
    print ''

   
   # now refine:    
    x0 = np.array([mud_VT, sig2d_VT])    
    args = (a,b,c,d,f1,f2,sig2sfactor,sig2d,sig2crossfactor,dV,kre,n,mud3c,pdict['epsilon']) 
                             
    start = time.time()  
    sol2 = scipy.optimize.root(FPss_moclok3_2C_IsId_foroptim_wrapper2, x0, args=args, 
                               method='hybr', options={'factor': 5, 'maxfev': 500, 'xtol': 1e-9})
  
    print sol2      
    mud_VT, sig2d_VT = sol2.x
    error = np.abs(sol2.fun[0]) + np.abs(sol2.fun[1])
    
    
    # done here, if sigmas is not < sigmas_smallest (and error is satisfactory)
    if iterate:  # gradually from a larger sigmas value
        max_count = 3
        start = time.time() 
        for k in range(len(sigmas_vals)-1):
            # previous optimum
            x0 = np.array([mud_VT, sig2d_VT])
            # new paramter value
            sigmas = sigmas_vals[len(sigmas_vals)-k-2]

            sig2sfactor = 2.0/(sigmas**2 + sigmasd**2)  # sigmas should not be smaller than around 0.03
            sig2d = sigmad**2 + sigmads**2
            sig2crossfactor = (sigmas*sigmads + sigmasd*sigmad)/(sigmas**2 + sigmasd**2)                     
            args = (a,b,c,d,f1,f2,sig2sfactor,sig2d,sig2crossfactor,dV,kre,n,mud3c,pdict['epsilon'])
            sol2 = scipy.optimize.root(FPss_moclok3_2C_IsId_foroptim_wrapper2, x0, args=args, 
                                       method='hybr',options={'factor': 0.5, 'maxfev': 1000, 'xtol': 1e-9})
            error = np.abs(sol2.fun[0]) + np.abs(sol2.fun[1])

            count = 0    
            factor1 = [0.95, 0.97, 0.99]
            factor2 = [1.05, 1.03, 1.01]
            if error > 1e-4 or sol2.x[0]<0 or sol2.x[1]<0:
                while error > 1e-4 and count < max_count:  
                    print 'error = ', error, ' ==> searching neighborhood'
                    print 'consider reducing dsigmas'
                    # search neighborhood (around previous optimum)
                    N = 100 + count*100
                    sig2dVT_vec = np.linspace(factor1[count]*sig2d_VT, factor2[count]*sig2d_VT, N)
                    mud_VT_vec = np.linspace(factor1[count]*mud_VT, factor2[count]*mud_VT, N)
                    error_mat = np.zeros((len(mud_VT_vec),len(sig2dVT_vec)))
                    for j, sig2d_VT in enumerate(sig2dVT_vec):
                        for i, mud_VT in enumerate(mud_VT_vec):
                            qssh, qsshh = \
                            FPss_moclok3_2C_IsId_foroptim(a,b,c,d,f1,f2,sig2sfactor,sig2d,sig2crossfactor,
                                                          dV,kre,n,mud_VT,sig2d_VT,mud3c,pdict['epsilon'])
                            error_mat[i,j] = np.abs(qssh) + np.abs(qsshh)
                    i,j = np.unravel_index(error_mat.argmin(),error_mat.shape)
                    mud_VT = mud_VT_vec[i]
                    sig2d_VT = sig2dVT_vec[j]
                    error = np.min(np.min(error_mat))
                    count += 1
            else:
                mud_VT, sig2d_VT = sol2.x        
            print 'mud_VT, sig2d_VT, error = ', mud_VT, sig2d_VT, error
                                    
        print 'timing for "special" iteration: ', time.time()-start 
    
    # do brute force last, and before rerun the hybrid optimization 
    # from different starting points (as long as sol.status>=4 which indicates 
    # that we are trapped in a local minimum and not at the root)
    print ''
    print 'try iterative optimization...' 
    iter_max = 5
    iter_count = 0
    while error > 1e-6 and iter_count <= iter_max:
        x0 = np.array([mud_VT, sig2d_VT])
        sol3 = scipy.optimize.root(FPss_moclok3_2C_IsId_foroptim_wrapper2, x0, args=args, 
                                   method='hybr', options={'factor': 0.25, 'maxfev': 500, 'xtol': 1e-9})
        error = np.abs(sol3.fun[0]) + np.abs(sol3.fun[1])
        mud_VT, sig2d_VT = sol3.x
        iter_count += 1
        print 'iteration ', iter_count, ', error = ', error                               
    if error <= 1e-6:
        print ''
        print 'convergence, yay :)'
        print ''                                          
        
    if error > 1e-6: # last chance: try to improve by "brute force"
        print '' 
        print 'error not satisfactory; try to improve by brute force:'
        print 'mud_VT, sig2d_VT, error = ', mud_VT, sig2d_VT, error
        print '==> searching neighborhood of candidate optimum'
        print ''
        N = 300
        sig2dVTmin = 0.1*pdict['Vth']**2;  sig2dVTmax = 1.0*pdict['Vth']**2;
        mud_VTmin = 0.25*pdict['Vth'];  mud_VTmax = 1.0*pdict['Vth'];
        dsig2dVT = 0.1*pdict['Vth']**2;  dmud_VT = 0.1*pdict['Vth'];
        sig2dVT_vec = np.linspace(np.max([sig2d_VT-dsig2dVT, sig2dVTmin]), 
                                  np.min([sig2d_VT+dsig2dVT, sig2dVTmax]), N)
        mud_VT_vec = np.linspace(np.max([mud_VT-dmud_VT, mud_VTmin]), 
                                  np.min([mud_VT+dmud_VT, mud_VTmax]), N)
        error_mat = np.zeros((len(mud_VT_vec),len(sig2dVT_vec)))
        for j, sig2d_VTval in enumerate(sig2dVT_vec):
            for i, mud_VTval in enumerate(mud_VT_vec):
                qssh, qsshh = \
                FPss_moclok3_2C_IsId_foroptim(a,b,c,d,f1,f2,sig2sfactor,sig2d,sig2crossfactor,
                                              dV,kre,n,mud_VTval,sig2d_VTval,mud3c,pdict['epsilon'])
                error_mat[i,j] = np.abs(qssh) + np.abs(qsshh)
        i,j = np.unravel_index(error_mat.argmin(),error_mat.shape)
        error_new = np.min(np.min(error_mat))
        if error_new < 1e-6:
            mud_VT = mud_VT_vec[i]
            sig2d_VT = sig2dVT_vec[j]
            error = error_new

                
        
    print 'timing for refinement (hybrid root finding): ', time.time()-start    
    print 'mud_VT, sig2d_VT, error = ', mud_VT, sig2d_VT, error  
    
    V_out, ps, psh, pshh, qssh, qsshh, dpsdV_vec, dpshdV_vec, dpshhdV_vec = \
            FPss_moclok3_2C_IsId_full(V_vec,kre,pdict['Cs'],pdict['Cd'],pdict['Gs'],pdict['Gd'], 
                                      pdict['Gj'], Ge, pdict['Gexp'], pdict['DeltaT'], 
                                      pdict['VT'], mus_orig, sigmas_orig, mud_orig, sigmad_orig, 
                                      mud_VT, sig2d_VT, mud3c, pdict['epsilon'])
    
    psint = (V_out[1]-V_out[0])*np.sum(ps)
    r = 1.0/psint
    ps *= r;  psh *= r;  pshh *= r;  qssh *= r;  qsshh *= r;  
    dpsdV = dpsdV_vec*r;  dpshdV = dpshdV_vec*r;  dpshhdV = dpshhdV_vec*r;
    Vlb = V_out[0]
    Vdmean = -(V_out[1]-V_out[0])*np.sum(ps*V_out)*c/d  # =: mu_d
    print 'timing for refinement plus full output: ', time.time()-start                                  
    
       
    print 'rate = ', r
    
    Dout['psh'] = psh;  Dout['pshh'] = pshh;  Dout['mud_VT'] = mud_VT;
    Dout['sig2d_VT'] = sig2d_VT;  Dout['dpshhdV'] = dpshhdV;                     
    Dout['mud'] = 'calc. psh/ps'
    Dout['pshint'] = (V_out[1]-V_out[0])*np.sum(psh)
    Dout['sig2dc'] = 'calc. from pshh/ps'    
    Dout['r'] = r;  Dout['ps'] = ps;  Dout['error'] = error;  Dout['Vlb'] = Vlb; 
    Dout['Vdmean'] = Vdmean;  Dout['dpsdV'] = dpsdV;  Dout['dpshdV'] = dpshdV;
    

    return (arg_tuple[0], arg_tuple[1], arg_tuple[2], arg_tuple[3], Dout)
    
     

@numba.njit   # Is = Cs*[mus + sigmas*xi_s], Id = Cd*[mud + sigmad*xi_d]
def FPss_moclok3_2C_IsId_full(V, kre, Cs, Cd, Gs, Gd, Gj, Ge, Gexp, DeltaT, VT, 
                              mus, sigmas, mud, sigmad, mud_VT, sig2d_VT, mud3c, epsilon):
    if Ge>0:  # closed-loop version
        fac1 = Gj**2 / (Ge+Gj)
        fac2 = Gj / (2*(Ge+Gj))
        a = (fac1 - Gs - Gj)/Cs
        b = (Gj - fac1)/Cs
        d = (fac1 - Gd - Gj)/Cd
        c = (Gj - fac1)/Cd
        mus_tmp = (1.0 - fac2) * mus + fac2 * mud*Cd/Cs
        mud_tmp = (1.0 - fac2) * mud + fac2 * mus*Cs/Cd
        mus = mus_tmp 
        mud = mud_tmp
        sigmasd = fac2 * sigmad*Cd/Cs
        sigmads = fac2 * sigmas*Cs/Cd
        sigmas *= (1.0 - fac2)
        sigmad *= (1.0 - fac2)
#        print 'actual mus,mud = '  #for debugging
#        print mus, mud
    else:  # open-loop version
        a = -(Gs+Gj)/Cs;  b = Gj/Cs;  c = Gj/Cd;  d = -(Gd+Gj)/Cd;  
        sigmasd = 0.0;  sigmads = 0.0; 
 
    e = Gexp*DeltaT/Cs  
    if not e>0:
        DeltaT = 1.0  # to make sure we don't get errors below
    f1 = a*V + e*np.exp((V-VT)/DeltaT) + mus;  f2 = c*V + mud;
    dV = V[1]-V[0]
    sig2sfactor = 2.0/(sigmas**2 + sigmasd**2)  # sigmas should not be smaller than around 0.03
    sig2dfactor = 0.5 * (sigmad**2 + sigmads**2)
    sig2crossfactor = (sigmas*sigmads + sigmasd*sigmad)/(sigmas**2 + sigmasd**2)
    sig2crossfactor2 = sig2crossfactor/sig2sfactor
    
    n = len(V)
    ps = np.zeros(n);  psh = np.zeros(n);  pshh = np.zeros(n);
#    mud = np.zeros(n);  mud[n-1] = mud_VT;
    qssh = np.zeros(n);  qssh[-1] = mud_VT;  
    qsshh = np.zeros(n);  qsshh[-1] = sig2d_VT;  #qss = 1.0;
    k = n-1;
    dpsdV_vec = np.zeros(n);  dpshdV_vec = np.zeros(n);  dpshhdV_vec = np.zeros(n);
    # save for sig2mod calculations later
       
    while k>kre and ps[k]<1e5 and ps[k]>-1e5: #last 2 cond. optional to save time
    #stop backward integration when ps seems to diverge    
        dpsdV_vec[k] = sig2sfactor * (f1[k]*ps[k] + b*psh[k] - 1.0)
        ps[k-1] = ps[k] - dV*dpsdV_vec[k]
        qssh[k-1] = qssh[k] - dV*(f2[k]*ps[k] + d*psh[k] - sig2crossfactor2*dpsdV_vec[k])
        dpshdV_vec[k] = sig2sfactor * (f1[k]*psh[k] + b*pshh[k] - qssh[k]) + sig2crossfactor*ps[k]
        psh[k-1] = psh[k] - dV*dpshdV_vec[k]
        qsshh[k-1] = qsshh[k] - dV*2.0*(f2[k]*psh[k] + d*pshh[k] + sig2dfactor*ps[k] - \
                     sig2crossfactor2*dpshdV_vec[k])        
        if not k==n-1:
            dummy = psh[k]/ps[k]
            dpshhdV_vec[k] = sig2sfactor * ( f1[k]*pshh[k] - qsshh[k] + \
                             b*(mud3c + 3*dummy*pshh[k] - 2*psh[k]*dummy**2) ) + \
                             2*sig2crossfactor*psh[k]
        else:
            dpshhdV_vec[k] = sig2sfactor * ( b*mud3c - qsshh[k] ) + 2*sig2crossfactor*psh[k]   
        pshh[k-1] = pshh[k] - dV*dpshhdV_vec[k]  
#        mud[k-1] = psh[k-1]/ps[k-1]
        k -= 1
    qssh[k] -= mud_VT;  qsshh[k] -= sig2d_VT;
    
    if k==kre:
        while k>0 and ps[k]>epsilon and ps[k]<1e5 and ps[k]>-1e5:  #last 2 cond. optional to save time
        #stop backward integration when ps comes "close enough" to 0 (can also cross 0)
        #or when ps seems to diverge 
            dpsdV_vec[k] = sig2sfactor * (f1[k]*ps[k] + b*psh[k])      
            ps[k-1] = ps[k] - dV*dpsdV_vec[k]  
            qssh[k-1] = qssh[k] - dV*(f2[k]*ps[k] + d*psh[k] - sig2crossfactor2*dpsdV_vec[k])
            dpshdV_vec[k] = sig2sfactor * (f1[k]*psh[k] + b*pshh[k] - qssh[k]) + sig2crossfactor*ps[k]
            psh[k-1] = psh[k] - dV*dpshdV_vec[k]
            qsshh[k-1] = qsshh[k] - dV*2.0*(f2[k]*psh[k] + d*pshh[k] + sig2dfactor*ps[k] - \
                         sig2crossfactor2*dpshdV_vec[k])  
            dummy = psh[k]/ps[k]            
            dpshhdV_vec[k] = sig2sfactor * ( f1[k]*pshh[k] - qsshh[k] + \
                             b*(mud3c + 3*dummy*pshh[k] - 2*psh[k]*dummy**2) ) + \
                             2*sig2crossfactor*psh[k]
            pshh[k-1] = pshh[k] - dV*dpshhdV_vec[k]  
    #        mud[k-1] = psh[k-1]/ps[k-1] 
            k -= 1
  
    return V[k:], ps[k:], psh[k:], pshh[k:], qssh[k:], qsshh[k:], dpsdV_vec[k:], dpshdV_vec[k:], dpshhdV_vec[k:]  
          # all these quantities must be scaled by r later
    
    
@numba.njit   # Is = Cs*[mus + sigmas*xi_s], Id = Cd*[mud + sigmad*xi_d]
def FPss_moclok3_2C_IsId_foroptim(a,b,c,d,f1,f2,sig2sfactor,sig2d,sig2crossfactor,
                                  dV,kre,n,mud_VT,sig2d_VT,mud3c,epsilon):
#    a = -(Gs+Gj)/Cs;  b = Gj/Cs;  c = Gj/Cd;  d = -(Gd+Gj)/Cd;  
#    f1 = a*V + mus;  f2 = c*V + mud;
#    dV = V[1]-V[0]
#    sig2sfactor = 2.0/sigmas**2  # sigmas should not too small (< around 0.03)
#    sig2d = sigmad**2
#    n = len(V)
    sig2d_factor = 0.5*sig2d
    sig2crossfactor2 = sig2crossfactor/sig2sfactor
    ps = 0.0;  psh = 0.0;  pshh = 0.0;
    qssh = mud_VT; qsshh = sig2d_VT;
    k = n-1
    
    while k>kre and ps<1e5 and ps>-1e5: #last 2 cond. optional to save time
    #stop backward integration when ps seems to diverge    
        dpsdV = sig2sfactor * (f1[k]*ps + b*psh - 1.0)
        dpshdV = sig2sfactor * (f1[k]*psh + b*pshh - qssh) + sig2crossfactor*ps
        if not k==n-1:
            dummy = psh/ps
            dpshhdV = sig2sfactor * ( f1[k]*pshh - qsshh + \
                                     b*(mud3c + 3*dummy*pshh - 2*psh*dummy**2) ) + \
                      2*sig2crossfactor*psh               
        else:
            dpshhdV = sig2sfactor * ( b*mud3c - qsshh ) + 2*sig2crossfactor*psh    
        qssh -= dV*(f2[k]*ps + d*psh - sig2crossfactor2*dpsdV)
        qsshh -= dV*2.0*(f2[k]*psh + d*pshh + sig2d_factor*ps - sig2crossfactor2*dpshdV)
        ps -= dV*dpsdV
        psh -= dV*dpshdV
        pshh -= dV*dpshhdV
        k -= 1
    qssh -= mud_VT;  qsshh -= sig2d_VT;
    
    if k==kre:
        while k>0 and ps>epsilon and ps<1e5:  #last cond. optional to save time
        #stop backward integration when ps comes "close enough" to 0 (can also cross 0)
        #or when ps seems to diverge 
            dpsdV = sig2sfactor * (f1[k]*ps + b*psh) 
            dpshdV = sig2sfactor * (f1[k]*psh + b*pshh - qssh) + sig2crossfactor*ps
            dummy = psh/ps
            dpshhdV = sig2sfactor * ( f1[k]*pshh - qsshh + \
                                     b*(mud3c + 3*dummy*pshh - 2*psh*dummy**2) ) + \
                      2*sig2crossfactor*psh                
            qssh -= dV*(f2[k]*ps + d*psh - sig2crossfactor2*dpsdV)
            qsshh -= dV*2.0*(f2[k]*psh + d*pshh + sig2d_factor*ps - sig2crossfactor2*dpshdV)
            ps -= dV*dpsdV
            psh -= dV*dpshdV
            pshh -= dV*dpshhdV
            k -= 1
            
#    #optional to save time:
#    if ps>=1e6 or ps<=-1e6:
#        qssh = np.nan
#        qsshh = np.nan
    
    return qssh, qsshh                                               


def FPss_moclok3_2C_IsId_foroptim_wrapper2(x, *args):
    mud_VT,sig2d_VT = x
    a,b,c,d,f1,f2,sig2sfactor,sig2d,sig2crossfactor,dV,kre,n,mud3c,epsilon = args
    qssh_end, qsshh_end = FPss_moclok3_2C_IsId_foroptim(a,b,c,d,f1,f2,sig2sfactor,
                                                        sig2d,sig2crossfactor,
                                                        dV,kre,n,mud_VT,
                                                        sig2d_VT,mud3c,epsilon)
#    error = np.abs(qssh_end) + np.abs(qsshh_end)
#    if np.isnan(error):
#        error = 1000.0
    return np.array([qssh_end, qsshh_end])  



def FPmod_moclo_2C(Is0, sigmas, Id0, sigmad, **kwargs):
    pdict = kwargs['pdict']
    ssdict = kwargs['FPssdict']
    method = kwargs['method']
    if kwargs['N_procs'] <= 1:
        # single processing version, i.e. loop
        pool = False
        result = (FPmod_moclo_2C_for_given_freq_wrapper(arg_tuple) for 
                  arg_tuple in itertools.product(pdict['f_FP1_vec'],[Is0],[sigmas],
                                                 [Id0],[sigmad],[pdict],[ssdict],[method]))
    else:
        # multiproc version
        pool = multiprocessing.Pool(kwargs['N_procs']) 
        result = pool.imap_unordered(FPmod_moclo_2C_for_given_freq_wrapper, 
                                     itertools.product(pdict['f_FP1_vec'],[Is0],[sigmas],
                                                 [Id0],[sigmad],[pdict],[ssdict],[method])) 
    total = len(pdict['f_FP1_vec'])    
    finished = 0
    r1_dict = {}
    error_dict = {} 
    for freq, r1, error in result:
        finished += 1
        if kwargs['verbatim']:
            print(('{count} of {tot} FPmod_moclo_2C (' + method + ') calculations completed').format(
                    count=finished, tot=total))      
        r1_dict[freq] = r1 
        error_dict[freq] = error
    if pool:
        pool.close()
    return r1_dict, error_dict


def FPmod_moclo_2C_for_given_freq_wrapper(arg_tuple):  #TODO: clean this one up!
    f_vals, Is0, sigmas, Id0, sigmad, pdict, ssdict, method = arg_tuple
    w = 2*np.pi * f_vals
    mus0 = Is0/pdict['Cs']  
    sigmas0 = sigmas/pdict['Cs']
    mud0 = Id0/pdict['Cd']  
    sigmad0 = sigmad/pdict['Cd']
    r0 = ssdict['rates_dict'][Is0, sigmas, Id0, sigmad]
    V_vec = np.arange(ssdict['Vlb_dict'][Is0, sigmas, Id0, sigmad],pdict['Vth']+pdict['dV']/2,pdict['dV'])    
    ps0 = ssdict['ps_dict'][Is0, sigmas, Id0, sigmad][-len(V_vec):]  # for mumod calculation
    dps0dV = ssdict['dpsdV_dict'][Is0, sigmas, Id0, sigmad][-len(V_vec):]  # for sig2mod calculation
    kre = np.argmin(np.abs(V_vec-pdict['Vr'])) 
    
    psh0 = ssdict['psh_dict'][Is0, sigmas, Id0, sigmad][-len(V_vec):]  # for mumod calculation
    pshh0 = ssdict['pshh_dict'][Is0, sigmas, Id0, sigmad][-len(V_vec):]  # for mumod calculation
    dpsh0dV = ssdict['dpshdV_dict'][Is0, sigmas, Id0, sigmad][-len(V_vec):]  # for sig2mod calculation
    dpshh0dV = ssdict['dpshhdV_dict'][Is0, sigmas, Id0, sigmad][-len(V_vec):]  # for sig2mod calculation
    mud0_VT = ssdict['mudVT_dict'][Is0, sigmas, Id0, sigmad]
    sig2d0_VT = ssdict['sig2dVT_dict'][Is0, sigmas, Id0, sigmad]
    
    Ge = 0.0

    # fast method (exploiting linearity of the system)
    if method == 'musmod': 
        mus1 = 1.0;  sigs21 = 0.0;  mud1 = 0.0;  sigd21 = 0.0;
    elif method == 'mudmod': 
        mus1 = 0.0;  sigs21 = 0.0;  mud1 = 1.0;  sigd21 = 0.0;
    elif method == 'sigs2mod':            
        mus1 = 0.0;  sigs21 = 1.0;  mud1 = 0.0;  sigd21 = 0.0;
    elif method == 'sigd2mod': 
        mus1 = 0.0;  sigs21 = 0.0;  mud1 = 0.0;  sigd21 = 1.0;    
        
    qa, qha, qhha, qb, qhb, qhhb, qc, qhc, qhhc, qd, qhd, qhhd = \
        FPxmod_moclok3_2C_lin_numba(V_vec, kre, r0, ps0, psh0, pshh0,
                                    dps0dV, dpsh0dV, dpshh0dV,
                                    pdict['Cs'], pdict['Cd'], pdict['Gs'], 
                                    pdict['Gd'], pdict['Gj'], Ge, pdict['Gexp'], 
                                    pdict['DeltaT'], pdict['VT'], mus0, sigmas0,
                                    mud0, sigmad0, mus1, sigs21, mud1, sigd21,
                                    mud0_VT, sig2d0_VT, w)                               
    A = np.array([[qa,qb,qc], [qha,qhb,qhc], [qhha,qhhb,qhhc]])
    b = np.array([-qd,-qhd,-qhhd])
    x = np.linalg.solve(A, b)
    r1 = x[0] 
    error = np.nan     
          
    return (f_vals, r1, error)                



@numba.njit  # implementation for r1mu (see notes p.6f)
def FPxmod_moclok3_2C_lin_numba(V, kre, r, ps, psh, pshh, dpsdV, dpshdV, dpshhdV,
                                Cs, Cd, Gs, Gd, Gj, Ge, Gexp, DeltaT, VT,
                                mus0, sigmas0, mud0, sigmad0, mus1, sigs21, 
                                mud1, sigd21, mud0_VT, sig2d0_VT, w):  
    # mus1, ..., sigd21 should be indicators here: 0.0 or 1.0 
    if Ge>0:  # closed-loop version
        fac1 = Gj**2 / (Ge+Gj)
        fac2 = Gj / (2*(Ge+Gj))
        a = (fac1 - Gs - Gj)/Cs
        b = (Gj - fac1)/Cs
        d = (fac1 - Gd - Gj)/Cd
        c = (Gj - fac1)/Cd
        mus_tmp = (1.0 - fac2) * mus0 + fac2 * mud0*Cd/Cs
        mud_tmp = (1.0 - fac2) * mud0 + fac2 * mus0*Cs/Cd
        mus0 = mus_tmp 
        mud0 = mud_tmp
        sigmasd0 = fac2 * sigmad0*Cd/Cs
        sigmads0 = fac2 * sigmas0*Cs/Cd
        sigmas0 *= (1.0 - fac2)
        sigmad0 *= (1.0 - fac2)
    else:  # open-loop version
        a = -(Gs+Gj)/Cs;  b = Gj/Cs;  c = Gj/Cd;  d = -(Gd+Gj)/Cd;  
        sigmasd0 = 0.0;  sigmads0 = 0.0;  
                                
    e = Gexp*DeltaT/Cs
    if not e>0:
        DeltaT = 1.0  # to make sure we don't get errors below
    f = a*V + e*np.exp((V-VT)/DeltaT) + mus0;  g = c*V+mud0;  
    fw1 = 1j*w;  fw2 = 1j*w-d;  fw3 = 1j*w-2.0*d;     
    dV = V[1]-V[0]
#    sigs2factor = 2.0/sigmas0**2
#    sigd2 = sigmad0**2
    sigs2factor = 2.0/(sigmas0**2 + sigmasd0**2)  # sigmas should not be smaller than around 0.03
    sigd2 = sigmad0**2 + sigmads0**2
    sig2crossfactor = (sigmas0*sigmads0 + sigmasd0*sigmad0)/(sigmas0**2 + sigmasd0**2)
    sig2crossfactor2 = sig2crossfactor/sigs2factor
    n = len(V)   
    pa = 0.0 + 0*1j;  pha = 0.0 + 0*1j;  phha = 0.0 + 0*1j; 
    qa = 1.0 + 0*1j;  qha = mud0_VT + 0*1j;  qhha = sig2d0_VT + 0*1j;
    pb = 0.0 + 0*1j;  phb = 0.0 + 0*1j;  phhb = 0.0 + 0*1j; 
    qb = 0.0 + 0*1j;  qhb = r + 0*1j;  qhhb = 0.0 + 0*1j; 
    pc = 0.0 + 0*1j;  phc = 0.0 + 0*1j;  phhc = 0.0 + 0*1j; 
    qc = 0.0 + 0*1j;  qhc = 0.0 + 0*1j;  qhhc = r + 0*1j;
    pd = 0.0 + 0*1j;  phd = 0.0 + 0*1j;  phhd = 0.0 + 0*1j; 
    qd = 0.0 + 0*1j;  qhd = 0.0 + 0*1j;  qhhd = 0.0 + 0*1j;
        
    for k in xrange(n-1, kre, -1):
        qa_new = qa + dV*fw1*pa
        dpadVneg = sigs2factor*(qa - f[k]*pa - b*pha) 
        pa_new = pa + dV*dpadVneg
        qha_new = qha + dV*(fw2*pha - g[k]*pa - sig2crossfactor2*dpadVneg)
        dphadVneg = sigs2factor*(qha - f[k]*pha - b*phha) + sig2crossfactor*pa
        pha_new = pha + dV*dphadVneg 
        qhha_new = qhha + dV*(fw3*phha - 2*g[k]*pha - sigd2*pa - 2*sig2crossfactor2*dphadVneg)
        qb_new = qb + dV*fw1*pb
        dpbdVneg = sigs2factor*(qb - f[k]*pb - b*phb) 
        pb_new = pb + dV*dpbdVneg
        qhb_new = qhb + dV*(fw2*phb - g[k]*pb - sig2crossfactor2*dpbdVneg)
        dphbdVneg = sigs2factor*(qhb - f[k]*phb - b*phhb) + sig2crossfactor*pb
        phb_new = phb + dV*dphbdVneg
        qhhb_new = qhhb + dV*(fw3*phhb - 2*g[k]*phb - sigd2*pb - 2*sig2crossfactor2*dphbdVneg)
        qc_new = qc + dV*fw1*pc
        dpcdVneg = sigs2factor*(qc - f[k]*pc - b*phc) 
        pc_new = pc + dV*dpcdVneg
        qhc_new = qhc + dV*(fw2*phc - g[k]*pc - sig2crossfactor2*dpcdVneg)
        dphcdVneg = sigs2factor*(qhc - f[k]*phc - b*phhc) + sig2crossfactor*pc
        phc_new = phc + dV*dphcdVneg
        qhhc_new = qhhc + dV*(fw3*phhc - 2*g[k]*phc - sigd2*pc - 2*sig2crossfactor2*dphcdVneg)
        qd_new = qd + dV*fw1*pd
        dpddVneg = sigs2factor*(qd - f[k]*pd - b*phd - mus1*ps[k] + \
                                sigs21*dpsdV[k]/2.0)
        pd_new = pd + dV*dpddVneg
        qhd_new = qhd + dV*(fw2*phd - g[k]*pd - mud1*ps[k] - sig2crossfactor2*dpddVneg)
        dphddVneg = sigs2factor*(qhd - f[k]*phd - b*phhd - mus1*psh[k] + \
                                 sigs21*dpshdV[k]/2.0) + sig2crossfactor*pd
        phd_new = phd + dV*dphddVneg
        qhhd_new = qhhd + dV*(fw3*phhd - 2*g[k]*phd - 2*mud1*psh[k] - \
                              sigd2*pd - sigd21*ps[k] - 2*sig2crossfactor2*dphddVneg)
        if not k==n-1:        
            dummy = psh[k]/ps[k]
            fp01 = 4.0*dummy**3 - 3.0*dummy*pshh[k]/ps[k]
            fp02 = 3.0*pshh[k]/ps[k] - 6.0*dummy**2
            fp03 = 3.0*dummy
            expra = fp01*pa + fp02*pha + fp03*phha  
            exprb = fp01*pb + fp02*phb + fp03*phhb
            exprc = fp01*pc + fp02*phc + fp03*phhc
            exprd = fp01*pd + fp02*phd + fp03*phhd
        else:
            expra = 0.0;  exprb = 0.0;  exprc = 0.0;  exprd = 0.0;
        phha_new = phha + dV*(sigs2factor*(qhha - f[k]*phha - b*expra) - 2*sig2crossfactor*pha)
        phhb_new = phhb + dV*(sigs2factor*(qhhb - f[k]*phhb - b*exprb) - 2*sig2crossfactor*phb)
        phhc_new = phhc + dV*(sigs2factor*(qhhc - f[k]*phhc - b*exprc) - 2*sig2crossfactor*phc)
        phhd_new = phhd + dV*(sigs2factor*(qhhd - f[k]*phhd - b*exprd - \
                              mus1*pshh[k] + sigs21*dpshhdV[k]/2.0) - 2*sig2crossfactor*phd)
        qa = qa_new;  qha = qha_new;  qhha = qhha_new;
        pa = pa_new;  pha = pha_new;  phha = phha_new;        
        qb = qb_new;  qhb = qhb_new;  qhhb = qhhb_new;
        pb = pb_new;  phb = phb_new;  phhb = phhb_new;
        qc = qc_new;  qhc = qhc_new;  qhhc = qhhc_new;
        pc = pc_new;  phc = phc_new;  phhc = phhc_new;
        qd = qd_new;  qhd = qhd_new;  qhhd = qhhd_new;
        pd = pd_new;  phd = phd_new;  phhd = phhd_new;
    qa -= 1.0
    qha -= mud0_VT  
    qhha -= sig2d0_VT    
    qhb -= r
    qhhc -= r
    for k in xrange(kre, 0, -1):       
        qa_new = qa + dV*fw1*pa
        dpadVneg = sigs2factor*(qa - f[k]*pa - b*pha) 
        pa_new = pa + dV*dpadVneg
        qha_new = qha + dV*(fw2*pha - g[k]*pa - sig2crossfactor2*dpadVneg)
        dphadVneg = sigs2factor*(qha - f[k]*pha - b*phha) + sig2crossfactor*pa
        pha_new = pha + dV*dphadVneg 
        qhha_new = qhha + dV*(fw3*phha - 2*g[k]*pha - sigd2*pa - 2*sig2crossfactor2*dphadVneg)
        qb_new = qb + dV*fw1*pb
        dpbdVneg = sigs2factor*(qb - f[k]*pb - b*phb) 
        pb_new = pb + dV*dpbdVneg
        qhb_new = qhb + dV*(fw2*phb - g[k]*pb - sig2crossfactor2*dpbdVneg)
        dphbdVneg = sigs2factor*(qhb - f[k]*phb - b*phhb) + sig2crossfactor*pb
        phb_new = phb + dV*dphbdVneg
        qhhb_new = qhhb + dV*(fw3*phhb - 2*g[k]*phb - sigd2*pb - 2*sig2crossfactor2*dphbdVneg)
        qc_new = qc + dV*fw1*pc
        dpcdVneg = sigs2factor*(qc - f[k]*pc - b*phc) 
        pc_new = pc + dV*dpcdVneg
        qhc_new = qhc + dV*(fw2*phc - g[k]*pc - sig2crossfactor2*dpcdVneg)
        dphcdVneg = sigs2factor*(qhc - f[k]*phc - b*phhc) + sig2crossfactor*pc
        phc_new = phc + dV*dphcdVneg
        qhhc_new = qhhc + dV*(fw3*phhc - 2*g[k]*phc - sigd2*pc - 2*sig2crossfactor2*dphcdVneg)
        qd_new = qd + dV*fw1*pd
        dpddVneg = sigs2factor*(qd - f[k]*pd - b*phd - mus1*ps[k] + \
                                sigs21*dpsdV[k]/2.0)
        pd_new = pd + dV*dpddVneg
        qhd_new = qhd + dV*(fw2*phd - g[k]*pd - mud1*ps[k] - sig2crossfactor2*dpddVneg)
        dphddVneg = sigs2factor*(qhd - f[k]*phd - b*phhd - mus1*psh[k] + \
                                 sigs21*dpshdV[k]/2.0) + sig2crossfactor*pd
        phd_new = phd + dV*dphddVneg
        qhhd_new = qhhd + dV*(fw3*phhd - 2*g[k]*phd - 2*mud1*psh[k] - \
                              sigd2*pd - sigd21*ps[k] - 2*sig2crossfactor2*dphddVneg)
        dummy = psh[k]/ps[k]
        fp01 = 4.0*dummy**3 - 3.0*dummy*pshh[k]/ps[k]
        fp02 = 3.0*pshh[k]/ps[k] - 6.0*dummy**2
        fp03 = 3.0*dummy
        expra = fp01*pa + fp02*pha + fp03*phha  
        exprb = fp01*pb + fp02*phb + fp03*phhb
        exprc = fp01*pc + fp02*phc + fp03*phhc
        exprd = fp01*pd + fp02*phd + fp03*phhd
        phha_new = phha + dV*(sigs2factor*(qhha - f[k]*phha - b*expra) - 2*sig2crossfactor*pha)
        phhb_new = phhb + dV*(sigs2factor*(qhhb - f[k]*phhb - b*exprb) - 2*sig2crossfactor*phb)
        phhc_new = phhc + dV*(sigs2factor*(qhhc - f[k]*phhc - b*exprc) - 2*sig2crossfactor*phc)
        phhd_new = phhd + dV*(sigs2factor*(qhhd - f[k]*phhd - b*exprd - \
                              mus1*pshh[k] + sigs21*dpshhdV[k]/2.0) - 2*sig2crossfactor*phd)
        qa = qa_new;  qha = qha_new;  qhha = qhha_new;
        pa = pa_new;  pha = pha_new;  phha = phha_new;        
        qb = qb_new;  qhb = qhb_new;  qhhb = qhhb_new;
        pb = pb_new;  phb = phb_new;  phhb = phhb_new;
        qc = qc_new;  qhc = qhc_new;  qhhc = qhhc_new;
        pc = pc_new;  phc = phc_new;  phhc = phhc_new;
        qd = qd_new;  qhd = qhd_new;  qhhd = qhhd_new;
        pd = pd_new;  phd = phd_new;  phhd = phhd_new;
    return qa, qha, qhha, qb, qhb, qhhb, qc, qhc, qhhc, qd, qhd, qhhd
 


def sim_mod(Is0, sigmas, Id0, sigmad, method, pdict, N_procs):
    if N_procs <= 1:
        # single processing version, i.e. loop
        pool = False
        result = (simmod_for_given_freq_wrapper(arg_tuple) for arg_tuple in 
                  itertools.product(pdict['f_sim_vec'],[Is0],[sigmas],
                                    [Id0],[sigmad],[method],[pdict])) 
    else:
        # multiproc version
        pool = multiprocessing.Pool(N_procs)
        result = pool.imap_unordered(simmod_for_given_freq_wrapper, 
                                     itertools.product(pdict['f_sim_vec'],
                                     [Is0],[sigmas],[Id0],[sigmad],[method],[pdict]))
    total = len(pdict['f_sim_vec'])    
    finished = 0
    r1_fit_dict = {}
    phi_fit_dict = {}
    bin_edges_dict = {}
    sp_hist_dict = {}
    mean_rate_dict = {}
    for freq, r1_fit, phi_fit, hist_x, hist_y, meanrate in result:
        finished += 1
        print(('{count} of {tot} ' + method + ' simulations completed').
                format(count=finished, tot=total))
        r1_fit_dict[freq] = r1_fit
        phi_fit_dict[freq] = phi_fit 
        bin_edges_dict[freq] = hist_x
        sp_hist_dict[freq] = hist_y
        mean_rate_dict[freq] = meanrate
    if pool:
        pool.close()
    return r1_fit_dict, phi_fit_dict, bin_edges_dict, sp_hist_dict, mean_rate_dict


def simmod_for_given_freq_wrapper(arg_tuple):
    freq, Is0, sigmas, Id0, sigmad, method, pdict = arg_tuple  
    mus0 = Is0/pdict['Cs']
    sigmas0 = sigmas/pdict['Cs']
    mud0 = Id0/pdict['Cd']
    sigmad0 = sigmad/pdict['Cd']
    # need small dt for high freq: 
    # 0.05ms for up to ~250Hz (ok, but 0.01ms yields slightly different r1, phi for f=251)
    # 0.005ms for ~400 to 650Hz (ok, but 0.0025ms yields somewhat different r1, phi for f=631)
    # 0.001ms for ~800Hz (higher freqs not tested yet)
    # interestingly, however, sig2mod curves match best when using 0.05ms for high freq
    # (631Hz tested explicitly), while it does not make a big difference for mumod, so 
    # we might just stick to the low dt resolution
    if freq<255:
        dt = 0.05/1000  # (sec)
    elif freq>=255 and freq<650:
        dt = 0.01/1000  # (sec)
    elif freq>=650:
        dt = 0.001/1000  # (sec)
    tstart = np.ceil(pdict['mod_t_min']*freq)/freq                                
    cycles = np.floor(pdict['mod_t_end']*freq)    
    duration = cycles/freq  #(s)   
    tgrid = np.arange(0,tstart+duration+dt/2,dt)   
    
    sqrt_dt = np.sqrt(dt)
    if method=='musmod' or method=='musmod_cl':
#        mus = mus0 + pdict['mod_frac']*mus0*np.cos(2.0*np.pi*freq*tgrid)
        mus = mus0 + pdict['mus1']*np.cos(2.0*np.pi*freq*tgrid)
        sigmas = sigmas0;  mud = mud0 * np.ones(len(tgrid));  sigmad = sigmad0;
    elif method=='mudmod' or method=='mudmod_cl':
#        mud = mud0 + pdict['mod_frac']*mud0*np.cos(2.0*np.pi*freq*tgrid)
        mud = mud0 + pdict['mud1']*np.cos(2.0*np.pi*freq*tgrid)
        mus = mus0 * np.ones(len(tgrid));  sigmas = sigmas0;  sigmad = sigmad0;
    elif method=='field':
        mus = mus0 - pdict['Gj']/pdict['Cs'] * pdict['Delta']*pdict['E1'] * \
              np.cos(2.0*np.pi*freq*tgrid)
        mud = mud0 + pdict['Gj']/pdict['Cd'] * pdict['Delta']*pdict['E1'] * \
              np.cos(2.0*np.pi*freq*tgrid)
        sigmas = sigmas0;  sigmad = sigmad0;
    elif method=='sigs2mod':
        mus = mus0 * np.ones(len(tgrid));  mud = mud0 * np.ones(len(tgrid));
#        sigmas2 = sigmas0**2 + pdict['mod_frac']*sigmas0**2 * \
#                                np.cos(2.0*np.pi*freq*tgrid)
        sigmas2 = sigmas0**2 + pdict['sigs21']*np.cos(2.0*np.pi*freq*tgrid)
        sigmas = np.sqrt(sigmas2);  sigmad = sigmad0; 
    elif method=='sigd2mod':
        mus = mus0 * np.ones(len(tgrid));  mud = mud0 * np.ones(len(tgrid));
#        sigmad2 = sigmad0**2 + pdict['mod_frac']*sigmad0**2 * \
#                                np.cos(2.0*np.pi*freq*tgrid)
        sigmad2 = sigmad0**2 + pdict['sigd21']*np.cos(2.0*np.pi*freq*tgrid)
        sigmad = np.sqrt(sigmad2);  sigmas = sigmas0;
                          
    Sp_times_all = np.array([])    
    
            
    if pdict['model']=='2C':
        Vr = pdict['Vr'];  VT = pdict['VT'];  DeltaV = pdict['DeltaV'];
        Cs = pdict['Cs'];  Gs = pdict['Gs'];  Cd = pdict['Cd'];  Gd = pdict['Gd'];
        Gj = pdict['Gj'];  Vth = pdict['Vth'];  DeltaT = pdict['DeltaT'];
        V_soma = pdict['V_init'][0].copy()
        V_dend = pdict['V_init'][1].copy()
        f5 = dt*pdict['Gexp']*DeltaT/Cs
        if not f5>0:
            DeltaT = 1.0  # to make sure we don't get errors below
            
        if method=='musmod_cl' or method=='mudmod_cl':
            Ge = pdict['Ge']
            fac1 = Gj**2/(Ge+Gj);  fac2 = Gj/(2*(Ge+Gj));
            f1 = dt*(Gj-fac1)/Cs;  f2 = dt/Cs * (fac1 - Gj - Gs);
            f3 = dt*(Gj-fac1)/Cd;  f4 = dt/Cd * (fac1 - Gj - Gd);
            mus_tmp = (1.0 - fac2) * mus + fac2 * mud*Cd/Cs
            mud_tmp = (1.0 - fac2) * mud + fac2 * mus*Cs/Cd
            musdt = dt*mus_tmp 
            muddt = dt*mud_tmp          
        else:         
            f1 = dt*Gj/Cs;  f2 = -dt/Cs * (Gj + Gs)
            f3 = dt*Gj/Cd;  f4 = -dt/Cd * (Gj + Gd)
            musdt = dt*mus;  muddt = dt*mud;   
        
        for i_N in range(pdict['mod_N']):
            np.random.seed(seed=None)
            if method=='musmod_cl' or method=='mudmod_cl':
                rands_vec = np.random.randn(len(tgrid))
                randd_vec = np.random.randn(len(tgrid))
                sigsdtnoise = sigmas * (1.0-fac2) * sqrt_dt*rands_vec
                sigsdtnoise += sigmad*Cd/Cs * fac2 * sqrt_dt*randd_vec    
                sigddtnoise = sigmad * (1.0-fac2) * sqrt_dt*randd_vec
                sigddtnoise += sigmas*Cs/Cd * fac2 * sqrt_dt*rands_vec
            else:
                sigsdtnoise = sigmas*sqrt_dt*np.random.randn(len(tgrid))
                sigddtnoise = sigmad*sqrt_dt*np.random.randn(len(tgrid))
            Sp_times = simulate_2C_spikesonly_numba(tgrid,V_soma,V_dend,f1,f2,f3,f4,f5,
                                                    musdt,muddt,sigsdtnoise,sigddtnoise,
                                                    VT,Vth,Vr,DeltaT,DeltaV)
            Sp_times_all = np.append(Sp_times_all, Sp_times) 
                                
    Sp_phases = Sp_times_all[Sp_times_all>=tstart] % (1.0/freq)  
    meanrate = np.float(len(Sp_times_all[Sp_times_all>=tstart]))/ \
               (pdict['mod_N']*duration)
    period = 1./freq  
    dthist = pdict['mod_dthist_frac']*period
    hist_y, hist_x = np.histogram(Sp_phases, bins=np.arange(0, period+dthist/2,dthist))
    hist_y = hist_y/(pdict['mod_N']*cycles*dthist)
    t_hist = hist_x[:-1] + dthist/2
    rmod_hist = hist_y - meanrate
#    sol = scipy.optimize.leastsq(cosine_lsqfit_fun, np.array([1.0, -0.5]), 
#                                 args=(t_hist,rmod_hist,freq))                            
#    r1_fit, phi_fit = sol[0]
    sol = scipy.optimize.least_squares(cosine_lsqfit_fun, np.array([1.0, -0.5]), 
                                 bounds=([0.0, -np.pi], [100.0, np.pi]), 
                                 args=(t_hist,rmod_hist,freq))                            
    r1_fit, phi_fit = sol.x
    if len(pdict['f_sim_vec'])==1:
        return Sp_times_all
    else:         
        return (freq, r1_fit, phi_fit, hist_x, hist_y, meanrate)
 

        
@numba.njit
def simulate_2C_spikesonly_numba(tgrid,V_soma,V_dend,f1,f2,f3,f4,f5,musdt,muddt,
                                 sigsdtnoise,sigddtnoise,VT,Vth,Vr,DeltaT,DeltaV):
    Sp_times_dummy = np.zeros(int(len(tgrid)/10)) 
    sp_count = int(0)
    for i_t in range(1,len(tgrid)):           
        V_soma += f1*V_dend + f2*V_soma + f5*np.exp((V_soma-VT)/DeltaT) + \
                  musdt[i_t-1] + sigsdtnoise[i_t-1]
        V_dend += f3*V_soma + f4*V_dend + muddt[i_t-1] + sigddtnoise[i_t-1]
        if V_soma>Vth:
            V_soma = Vr
            if DeltaV<(Vr-Vth): #TEMP
                V_dend = Vr
            else:
                V_dend += DeltaV # inspired by Ostojic 2015
            sp_count += 1
            Sp_times_dummy[sp_count-1] = tgrid[i_t]             
    
    Sp_times = np.zeros(sp_count)
    if sp_count>0:
        for i in xrange(sp_count):
            Sp_times[i] = Sp_times_dummy[i]
    return Sp_times   
    
def cosine_lsqfit_fun(p, *args):
    t_hist = args[0];  rmod_hist = args[1];  freq = args[2]; 
    out = rmod_hist - p[0]*np.cos(2*np.pi*freq*t_hist + p[1])
    return out   
    