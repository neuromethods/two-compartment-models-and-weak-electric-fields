# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:43:54 2018

@author: jl
"""

import numpy as np
import time
import methods as me
#import locale
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

#locale.setlocale(locale.LC_NUMERIC, 'C')  # to avoid numba llvm problem

EXP = True 

plot_fitting_results = True

simulate_spikerate_BS = False #TODO
simulate_spikerate_2C = True
analytic_spikerate_2C = True
 
analytic_spikerate_modulation_2C = True #modulations of Is, Id, sinusoidal field
simulate_spikerate_modulation_2C = True #corresponding simulations, this takes time...
        
N_procs = 1


paramsBS = {}
# BALL AND STICK NEURON MODEL params (SEALED END, LUMPED SOMA)
paramsBS['c_spec'] = 1.0e-2  # specific membrane capacitance (F/m2)
paramsBS['g_spec'] = 1/3.0  # specific membrane (leak) conductance (S/m2)
paramsBS['gi_spec'] = 1/2.0  # specific internal (axial) conductance (S/m)
paramsBS['L'] = 7e-4 #def.: 7e-4 [4e-4, 10e-4] cable length (m)
paramsBS['M'] = 0  #0 or 3e-4 or >L for full reset #TODO: remove
                    # distance from soma at which dendritic hyperpolarization becomes 
                    # negligible (if <L) --> no reset beyond this point (m)
paramsBS['d_dend'] = 1.0e-6 #def.: 1.0e-6 [0.5e-6, 1.5e-6] cable diameter (m) 
paramsBS['d_soma'] = 15e-6 #def.: 15e-6 [10e-6, 20e-6] soma diameter (m) 
# for (dendritic) cable:
paramsBS['cm'] = paramsBS['c_spec'] * (paramsBS['d_dend'] * np.pi)  # (F/m)
paramsBS['gm'] = paramsBS['g_spec'] * (paramsBS['d_dend'] * np.pi)  # (S/m)
paramsBS['gi'] = paramsBS['gi_spec'] * (paramsBS['d_dend']**2/4 * np.pi)  # (S*m)
paramsBS['lambd'] = np.sqrt(paramsBS['gi']/paramsBS['gm'])
# for (lumped) soma:
paramsBS['Cs'] = paramsBS['c_spec'] * (paramsBS['d_soma']**2 * np.pi)  # (F)
paramsBS['Gs'] = paramsBS['g_spec'] * (paramsBS['d_soma']**2 * np.pi)  # (S)
paramsBS['Vr'] = 0.0  # (V) 
paramsBS['VT'] = 10e-3  # (V)

## BS SIMULATION params
#paramsBS['dt'] = 0.005/1000  # (sec)
#paramsBS['n'] = 200  # number of compartments
#paramsBS['dx'] = paramsBS['L']/paramsBS['n']  # (m)
#paramsBS['V_init'] = 0.0*np.ones(paramsBS['n'])  # (V)

#try this for EXP=False:
#Is0 = 0.2*1e-11  #(A)  
#Id0 = 0.2*1e-11  #(A)
Is0 = 0.4*1e-11  #(A)  
Id0 = 0.4*1e-11  #(A)
sigmas = 20.0/np.sqrt(1000) *1e-12  #(A*sqrt(s))
sigmad = 20.0/np.sqrt(1000) *1e-12 #(A*sqrt(s))

Is1 = 0.05*1e-11  # (A)
Id1 = 0.05*1e-11  # (A)
E1 = 1.0  #(V/m)


if __name__ == '__main__':
           
    if EXP:
        paramsBS['DeltaT'] = 0.0015  # (V)
        paramsBS['Vth'] = 0.02  # (V) the cutoff voltage
    else:
        paramsBS['DeltaT'] = 0.0
        paramsBS['Vth'] = paramsBS['VT']
        
    freq_vals = np.arange(0,1e4,0.5)  # (Hz) 
    params2C, fitcurves = me.fit_2C_params(freq_vals, paramsBS)          
    #taus = Cs/(Gs+Gj)  #s
    #taud = Cd/(Gd+Gj)  #s
    
        
    if plot_fitting_results:
        
        fig1 = plt.figure()
        ax = plt.subplot(211)
        plt.semilogx(fitcurves['freq_vals'], np.abs(fitcurves['Zhat_Is_BS'])/1e9, 
                     'k', label='BS', linewidth=1.5)
        plt.semilogx(fitcurves['freq_vals'], np.abs(fitcurves['Zhat_Id_BS'])/1e9, 
                     'k--', linewidth=1.5)
        plt.semilogx(fitcurves['freq_vals'], np.abs(fitcurves['Zhat_Is_2C'])/1e9, 
                     'g', label='2C', linewidth=1.5)
        plt.semilogx(fitcurves['freq_vals'], np.abs(fitcurves['Zhat_Id_2C'])/1e9, 
                     'g--', linewidth=1.5) 
        plt.title('Somatic impedances for input at soma/dendrite', fontsize=16)
        plt.ylabel('Amplitude  (G$\Omega$)', fontsize=14)   
        plt.xlabel(r'Frequency $\omega / (2 \pi)$  (Hz)', fontsize=14)        
        plt.legend(loc='upper right', fontsize=14)  
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)     

        ax = plt.subplot(212)
        plt.semilogx(fitcurves['freq_vals'], 1e3*np.abs(fitcurves['Vshat_over_E1_BS']), 
                     'k', linewidth=1.5, label='BS')
        plt.semilogx(fitcurves['freq_vals'], 1e3*np.abs(fitcurves['Vshat_over_E1_2C']), 
                     'g', linewidth=1.5, label='2C') 
        plt.title('Subthreshold somatic responses to electric field', fontsize=16)
        plt.ylabel('Amplitude (mV)', fontsize=14)
        plt.xlabel(r'Frequency $ \omega / (2 \pi)$  (Hz)', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 
        plt.legend(loc='lower left', fontsize=14)

        fig2 = plt.figure()
        ax = plt.subplot(121)
        plt.plot([0,0], [1e3*paramsBS['Vth'],1e3*paramsBS['Vr']], 'k')
        plt.plot([0,0], [1e3*paramsBS['Vth'],1e3*params2C['Vr']], 'g')
        plt.plot([0], [1e3*paramsBS['Vth']], 'go')
        plt.plot(fitcurves['postspike_time_vals']*1e3,fitcurves['Vs_postspike_Is_BS']*1e3,'k',
                 fitcurves['postspike_time_vals']*1e3,fitcurves['Vs_postspike_Is_2C']*1e3,'g') 
        plt.title('Post-spike transient for \n threshold input at the soma', fontsize=16)
        plt.ylabel('Membrane voltage (mV)', fontsize=14)
        plt.xlabel('Time $t$ (ms)', fontsize=14)
        plt.xticks([0,1,1e3*fitcurves['postspike_time_vals'][-1]], 
                   [0,1,r'$\tau_{\mathrm{s}}$'])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 

        ax = plt.subplot(122)
        plt.plot([0,0], [1e3*paramsBS['Vth'],1e3*paramsBS['Vr']], 'k')
        plt.plot([0,0], [1e3*paramsBS['Vth'],1e3*params2C['Vr']], 'g')
        plt.plot([0], [1e3*paramsBS['Vth']], 'go')
        plt.plot(fitcurves['postspike_time_vals']*1e3,fitcurves['Vs_postspike_Id_BS']*1e3,'k',
                 fitcurves['postspike_time_vals']*1e3,fitcurves['Vs_postspike_Id_2C']*1e3,'g')
        plt.title('Post-spike transient for \n threshold input at the dendrite', fontsize=16)
        plt.ylabel('Membrane voltage (mV)', fontsize=14)
        plt.xlabel('Time $t$ (ms)', fontsize=14)
        plt.xticks([0,1,1e3*fitcurves['postspike_time_vals'][-1]], 
                   [0,1,r'$\tau_{\mathrm{s}}$'])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 

     
    if simulate_spikerate_2C:  
        # 2C simulation
        params = params2C.copy()
        params['V_init'] = np.array([0.,0.])
        params['dt'] = 0.005/1000 #s
        params['t_end'] = 60.0 #s
        params['t_min'] = 0.25  #s; initial transient to ignore
        params['t_Vs_save'] = 0.5
        params['seed'] = 11
        # simulate 2C model for each mu and sigma value (same frozen noise)            
        start = time.time()  # for timing the computation 
        Sp_times_2C, Vs_2C, Vshist_2C, Vsbins_2C, Vdhist_2C, Vdbins_2C = \
            me.sim_model_I0sigma('2C_model',[Is0], [sigmas], [Id0], [sigmad], 
                                 params, N_procs)
        print('2C simulation took {dur}s'.format(dur=np.round(time.time() - start,2)))
        # calculate rate and Gamma value #TODO: gamma
        Sp_array = Sp_times_2C[Is0,sigmas,Id0,sigmad]                       
        if len(Sp_array)>0:
            rate_2C = np.float(len(Sp_array))/(params['t_end']-params['t_min'])
        else:
            rate_2C = np.nan
        print('')
        print(rate_2C)
            
    
    if analytic_spikerate_2C:
        #Fokker-Planck method
        params = params2C.copy()
        params['dV'] = 1e-6  # (V) 1e-6, for final calculation perhaps: 1e-7 (V) 
        params['V_lb'] = -0.05  # (V), but will be changed by FPss_moclok2_2C
        params['epsilon'] = 1e-8 #1e-10  # for criterion in FPss_moclok2_2C   
        #N_procs = 1 #TEMP!!!  # multiprocessing.cpu_count() # no of parallel processes

        start = time.time()  # for timing the computation 
        # for legacy reasons (necessary)
        Dout = me.FPsteady_moclo_2C([Is0], [sigmas], [Id0], [sigmad], params, N_procs) 
        # Dout is a dictionary of dictionaries
        print('2C FP steady-state took {dur}s'.format(dur=np.round(time.time() - start,2)))

        rate_FP_2C = Dout['rates_dict'][Is0,sigmas,Id0,sigmad]
        errorval = Dout['error_dict'][Is0,sigmas,Id0,sigmad]                        
        # for quick inspection:
        print '2C FP steady-state rate (spikes/s): ' 
        print rate_FP_2C
        print 'error values: '
        print errorval      
        

    if analytic_spikerate_modulation_2C:   
        params = params2C.copy()                       
        params['dV'] = 1e-6  # (V) 1e-6, for final calculation perhaps: 1e-7 (V) 
        params['V_lb'] = -0.05  # (V), but will be changed by FPss_moclok2_2C
        params['epsilon'] = 1e-8 #1e-10  # for criterion in FPss_moclok2_2C
        params['f_FP1_vec'] = 10**np.arange(0.0,3.0,0.025)  # f in (Hz)
                
        # (re-)calculate steady-state
        Dout_ex = me.FPsteady_moclo_2C([Is0], [sigmas], [Id0], [sigmad], params, 1)
        print('r0 = ', np.round(Dout_ex['rates_dict'][Is0,sigmas,Id0,sigmad],2))
        # mod FP calculatiosn            
        start = time.time()  # for timing the computation   
        r1mus_2CFP_dict, _ = me.FPmod_moclo_2C(Is0, sigmas, Id0, sigmad, 
                                pdict=params, FPssdict=Dout_ex, method='musmod', 
                                verbatim=False, N_procs=N_procs) 
        r1mud_2CFP_dict, _ = me.FPmod_moclo_2C(Is0, sigmas, Id0, sigmad, 
                                pdict=params, FPssdict=Dout_ex, method='mudmod', 
                                verbatim=False, N_procs=N_procs)       
        print('2C FP mod (2x) took {dur}s'.format(
               dur=np.round(time.time() - start,2)))  
        r1mus_2CFP = [r1mus_2CFP_dict[fv] for j,fv in enumerate(params['f_FP1_vec'])]
        r1mud_2CFP = [r1mud_2CFP_dict[fv] for j,fv in enumerate(params['f_FP1_vec'])]
           
        fig3 = plt.figure()
        ax = plt.subplot(311)
        plt.semilogx(params['f_FP1_vec'], Is1/params['Cs']*np.abs(r1mus_2CFP),'g')
        plt.title('Response to somatic input modulation', fontsize=14)
        plt.ylabel('Amplitude of $r$ (Hz)', fontsize=14)
        plt.xlabel('Frequency $\omega / (2 \pi)$ (Hz)', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax = plt.subplot(312)
        plt.semilogx(params['f_FP1_vec'], Id1/params['Cd']*np.abs(r1mud_2CFP),'g')
        plt.title('Response to dendritic input modulation', fontsize=14)
        plt.ylabel('Amplitude of $r$ (Hz)', fontsize=14)
        plt.xlabel('Frequency $\omega / (2 \pi)$ (Hz)', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        factor1 = params['Gj']*params['Delta']*E1/params['Cd']
        factor2 = params['Gj']*params['Delta']*E1/params['Cs']
        r1_Emod_2CFP = factor1*np.array(r1mud_2CFP) - factor2*np.array(r1mus_2CFP)
        ax = plt.subplot(313)
        plt.semilogx(params['f_FP1_vec'], np.abs(r1_Emod_2CFP),'g')
        plt.title('Response to sinusoidal field', fontsize=14)
        plt.ylabel('Amplitude of $r$ (Hz)', fontsize=14)
        plt.xlabel('Frequency $\omega / (2 \pi)$ (Hz)', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        
    if simulate_spikerate_modulation_2C:
        params = params2C.copy()   
        params['f_sim_vec'] = 10**np.arange(0.3,3.0,0.4) # f in (Hz)
        params['mod_t_end'] = 100.0  #>=100s
        params['mod_t_min'] = 0.25  #s
        params['mus1'] = Is1/params['Cs'] #V/s
        params['mud1'] = Id1/params['Cd'] #V/s
        params['E1'] = E1  #V/m
        params['mod_N'] = 100  #>=100
        params['mod_dthist_frac'] = 0.025
        params['V_init'] = np.array([0.,0.])
        params['model'] = '2C'

        abs_r1mus_2Csim = np.zeros_like(params['f_sim_vec'])
        phase_r1mus_2Csim = np.zeros_like(params['f_sim_vec'])
        abs_r1mud_2Csim = np.zeros_like(params['f_sim_vec'])
        phase_r1mud_2Csim = np.zeros_like(params['f_sim_vec'])
        abs_r1field_2Csim = np.zeros_like(params['f_sim_vec'])
        phase_r1field_2Csim = np.zeros_like(params['f_sim_vec'])
    
        abs_r1mus_2Csim_dict, phase_r1mus_2Csim_dict, \
        bin_edges_mus_2Csim_dict, sp_hist_mus_2Csim_dict, \
        mean_rate_mus_2Csim_dict = me.sim_mod(Is0, sigmas, Id0, sigmad, 
                                              'musmod', params, N_procs)                                      
        plt.figure()
        plt.suptitle('2C mus-mod simulation results')
        for j,f in enumerate(params['f_sim_vec']):
            abs_r1mus_2Csim[j] = abs_r1mus_2Csim_dict[f]                       
            phase_r1mus_2Csim[j] = phase_r1mus_2Csim_dict[f]           
            plt.subplot(2,np.ceil(len(params['f_sim_vec'])/2.0),j+1)
            width = bin_edges_mus_2Csim_dict[f][1]-bin_edges_mus_2Csim_dict[f][0]
            plt.bar(bin_edges_mus_2Csim_dict[f][:-1], sp_hist_mus_2Csim_dict[f], 
                    width=width, color='red', alpha=0.5)
            period = 1./f;  dthist = params['mod_dthist_frac']*period;        
            tplot = np.arange(0,period+dthist/10,dthist/5)
            plt.plot(tplot, mean_rate_mus_2Csim_dict[f] + \
                            abs_r1mus_2Csim_dict[f] * \
                            np.cos(2*np.pi*f*tplot + phase_r1mus_2Csim_dict[f])) 
        
        abs_r1mud_2Csim_dict, phase_r1mud_2Csim_dict, \
        bin_edges_mud_2Csim_dict, sp_hist_mud_2Csim_dict, \
        mean_rate_mud_2Csim_dict = me.sim_mod(Is0, sigmas, Id0, sigmad, 
                                              'mudmod', params, N_procs)                                      
        plt.figure()
        plt.suptitle('2C mud-mod simulation results')
        for j,f in enumerate(params['f_sim_vec']):
            abs_r1mud_2Csim[j] = abs_r1mud_2Csim_dict[f]          
            phase_r1mud_2Csim[j] = phase_r1mud_2Csim_dict[f]            
            plt.subplot(2,np.ceil(len(params['f_sim_vec'])/2.0),j+1)
            width = bin_edges_mud_2Csim_dict[f][1]-bin_edges_mud_2Csim_dict[f][0]
            plt.bar(bin_edges_mud_2Csim_dict[f][:-1], sp_hist_mud_2Csim_dict[f], 
                    width=width, color='red', alpha=0.5)
            period = 1./f;  dthist = params['mod_dthist_frac']*period;        
            tplot = np.arange(0,period+dthist/10,dthist/5)
            plt.plot(tplot, mean_rate_mud_2Csim_dict[f] + \
                            abs_r1mud_2Csim_dict[f] * \
                            np.cos(2*np.pi*f*tplot + phase_r1mud_2Csim_dict[f]))
            
        abs_r1field_2Csim_dict, phase_r1field_2Csim_dict, \
        bin_edges_field_2Csim_dict, sp_hist_field_2Csim_dict, \
        mean_rate_field_2Csim_dict = me.sim_mod(Is0, sigmas, Id0, sigmad, 
                                                'field', params, N_procs)                                      
        plt.figure()
        plt.suptitle('2C field simulation results')
        for j,f in enumerate(params['f_sim_vec']):
            abs_r1field_2Csim[j] = abs_r1field_2Csim_dict[f]  
            phase_r1field_2Csim[j] = phase_r1field_2Csim_dict[f]           
            plt.subplot(2,np.ceil(len(params['f_sim_vec'])/2.0),j+1)
            width = bin_edges_field_2Csim_dict[f][1]-bin_edges_field_2Csim_dict[f][0]
            plt.bar(bin_edges_field_2Csim_dict[f][:-1], sp_hist_field_2Csim_dict[f], 
                    width=width, color='red', alpha=0.5)
            period = 1./f;  dthist = params['mod_dthist_frac']*period;        
            tplot = np.arange(0,period+dthist/10,dthist/5)
            plt.plot(tplot, mean_rate_field_2Csim_dict[f] + \
                            abs_r1field_2Csim_dict[f] * \
                            np.cos(2*np.pi*f*tplot + phase_r1field_2Csim_dict[f]))
                    
        if analytic_spikerate_modulation_2C:
           plt.figure(fig3.number)
        else:
           plt.figure()
        ax = plt.subplot(311)
        plt.semilogx(params['f_sim_vec'], abs_r1mus_2Csim, 'go')
        plt.title('Response to somatic input modulation', fontsize=14)
        plt.ylabel('Amplitude of $r$ (Hz)', fontsize=14)
        plt.xlabel('Frequency $\omega / (2 \pi)$ (Hz)', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax = plt.subplot(312)
        plt.semilogx(params['f_sim_vec'], abs_r1mud_2Csim, 'go')
        plt.title('Response to dendritic input modulation', fontsize=14)
        plt.ylabel('Amplitude of $r$ (Hz)', fontsize=14)
        plt.xlabel('Frequency $\omega / (2 \pi)$ (Hz)', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax = plt.subplot(313)
        plt.semilogx(params['f_sim_vec'], abs_r1field_2Csim, 'go')
        plt.title('Response to sinusoidal field', fontsize=14)
        plt.ylabel('Amplitude of $r$ (Hz)', fontsize=14)
        plt.xlabel('Frequency $\omega / (2 \pi)$ (Hz)', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
