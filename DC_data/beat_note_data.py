"""
@author: Chris Phillips
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import basics

def phase_range_shifted(phi):
    N = len(phi)
    phi_shift = phi - phi[0] - (phi[-1]-phi[0])/(N-1) * np.array(range(N))
    return max(abs(phi_shift))
    
class DataProcess:
    def __init__(self, 
                 dir_base,
                 BN_data,
                 frep_data = None, 
                 
                 flip_BN_order = False,

                 plot_on = False, 
                 fignum_plot = 11,

                 f_lowpass = None, 
                 f_prefilter_lowpass = None):
                
        self.coherence_metric = phase_range_shifted
        self.N_cw = 2
        
        C1_signals = [None]*self.N_cw
        C2_signals = [None]*self.N_cw
        
        if (flip_BN_order):
            C2_signals[0] = BN_data["ch1_cw1"]
            C2_signals[1] = BN_data["ch1_cw2"]
            C1_signals[0] = BN_data["ch2_cw1"]
            C1_signals[1] = BN_data["ch2_cw2"]
        else:
            C1_signals[0] = BN_data["ch1_cw1"]
            C1_signals[1] = BN_data["ch1_cw2"]
            C2_signals[0] = BN_data["ch2_cw1"]
            C2_signals[1] = BN_data["ch2_cw2"]
        
        # %% Define parameters
        self.dt = C1_signals[0]['dt']
        self.nt = int(C1_signals[0]['nt'])
        self.df = 1/self.dt / self.nt
        self.t_grid = np.fft.fftfreq(self.nt, 1/self.nt/self.dt)
        self.t_grid -= np.min(self.t_grid)
        self.f_grid = np.fft.fftfreq(self.nt, self.dt)
        t_grid_shift = np.fft.fftshift(self.t_grid)

        # %% Low-pass filter of signals (if necessary)
        if(f_lowpass is None):
            self.filter_func = lambda xx: xx
        else:
            self.filter_func = lambda xx: np.fft.ifft(np.fft.fft(xx) * (abs(self.f_grid) < f_lowpass))
        
        if(f_prefilter_lowpass is None):
            self.prefilter_func = lambda xx: xx
        else:
            self.prefilter_func  = lambda xx: np.fft.ifft(np.fft.fft(xx) * basics.functions.tanh_filter(self.f_grid/f_prefilter_lowpass, -1, 1, .1, .1))
            for ind in range(self.N_cw):
                C1_signals[ind]['y_downsampled'] = self.prefilter_func(C1_signals[ind]['y_downsampled'])
                C2_signals[ind]['y_downsampled'] = self.prefilter_func(C2_signals[ind]['y_downsampled'])
        
        ###################################################
        # Need to take first product (to get rid of f_cw) #
        C_prod_array = [None]*self.N_cw
        arg_opt_array = [None] * self.N_cw 
        for ind_cw_C1 in range(self.N_cw): # Iterate through cw-lasers of first channel
            if(plot_on):
                fignum_curr = fignum_plot + 10 + ind_cw_C1
                plt.figure(fignum_curr)
                fig, ax = plt.subplots(self.N_cw, 2, num = fignum_curr, clear=True)
            
            C_prod_array[ind_cw_C1] = np.zeros([self.N_cw,2])
            C1_curr = C1_signals[ind_cw_C1]['y_downsampled']
            

            for ind_C2_signal in range(self.N_cw): # Iterate through cw-lasers of second channel
                for ind_conj in range(2): # Compute both options (normal and complex-conjugate)
                    if(ind_conj == 0):
                        C2_curr = C2_signals[ind_C2_signal]['y_downsampled']
                    elif(ind_conj == 1):
                        C2_curr = np.conj(C2_signals[ind_C2_signal]['y_downsampled'])
                    
                    y_prod = C1_curr * C2_curr # Compute product of beat notes
                    phi_vec_temp = np.unwrap(np.angle(y_prod)) # Extract phase
     
                    # Remove fit to be less-susceptible to drifts when estimating correlation properties
                    phi_vec = phi_vec_temp - np.polynomial.polynomial.polyval(
                        np.arange(len(phi_vec_temp)),
                        np.polynomial.polynomial.polyfit(np.arange(len(phi_vec_temp)), phi_vec_temp, 2)
                        )
                    
                    # Calculate coherence for the selected cw-laser and conjugation-setting
                    C_prod_array[ind_cw_C1][ind_C2_signal, ind_conj] = self.coherence_metric( phi_vec )
                    
                    if(plot_on):
                        if(self.N_cw == 1):
                            ax_curr = ax[ind_conj]
                        else:
                            ax_curr = ax[ind_C2_signal, ind_conj]
                        ax_curr.plot(t_grid_shift, phi_vec)
                        ax_curr.plot(t_grid_shift, np.unwrap(np.angle(C1_curr)))
                        ax_curr.plot(t_grid_shift, np.unwrap(np.angle(C2_curr)))
            
            # Find options which are correlated with first cw-laser (yields corresponding second cw laser)
            arg_opt_array[ind_cw_C1]  = basics.functions.argmin_array(C_prod_array[ind_cw_C1])

        # Store in self
        self.C_prod_array = C_prod_array
        self.arg_opt_array = arg_opt_array
        
        ##################################################
        # ececute the first product (to get rid of f_cw) #
        BN_prod = [None] * self.N_cw
        for ind_cw_C1 in range(self.N_cw): # Iterate through both cw-lasers of first_channels
            BN_curr = {}

            # Store signal of i-th cw-laser on first channel
            BN_curr['y_C1'] = C1_signals[ind_cw_C1]['y_downsampled'] 
            BN_curr['ind_cw_C1'] = ind_cw_C1
            
            # Check which is corresponing cw-laser on second channel
            ind_C2_signal = arg_opt_array[ind_cw_C1][0]
            ind_conj = arg_opt_array[ind_cw_C1][1]

            # Store corresponding cw-laser of second channel
            BN_curr['ind_cw_C2'] = ind_C2_signal
            if(ind_conj == 0):
                BN_curr['y_C2'] = C2_signals[BN_curr['ind_cw_C2']]['y_downsampled']
                BN_curr['conj_sign'] = [1, 1]
            elif(ind_conj == 1):
                BN_curr['y_C2'] = np.conj(C2_signals[BN_curr['ind_cw_C2']]['y_downsampled'])
                BN_curr['conj_sign'] = [1,-1]
            
            # Compute product between corresponding cw-lasers of both channels
            BN_curr['f0_original_C1'] = BN_curr['conj_sign'][0] * C1_signals[BN_curr['ind_cw_C1']]['f0_original']
            BN_curr['f0_original_C2'] = BN_curr['conj_sign'][1] * C2_signals[BN_curr['ind_cw_C2']]['f0_original']
            BN_curr['f0_original_sum'] = BN_curr['f0_original_C1'] + BN_curr['f0_original_C2']
            BN_curr['y_prod'] = self.filter_func(BN_curr['y_C1'] * BN_curr['y_C2'])

            # Extract phase of product
            BN_curr['phi_vec_array'] = np.zeros([2,self.nt])
            BN_curr['phi_vec_array'][0,:] = np.unwrap(np.angle(BN_curr['y_C1']))
            BN_curr['phi_vec_array'][1,:] = np.unwrap(np.angle(BN_curr['y_C2']))
            BN_curr['phi_vec_prod'] = np.unwrap(np.angle(BN_curr['y_prod']))
            
            BN_prod[ind_cw_C1] = BN_curr
        
        self.BN_prod = BN_prod
        
        if(plot_on):
            C_signals = [C1_signals, C2_signals]
            fig, ax = plt.subplots(self.N_cw, 2, num = fignum_plot, clear=True)
            for ind_cw in range(self.N_cw):
                for ind_comb in range(2):
                
                    if(self.N_cw == 1):
                        ax_curr = ax[ind_comb]
                    else:
                        ax_curr = ax[ind_cw, ind_comb]
                    phi_curr = np.unwrap(np.angle(C_signals[ind_comb][ind_cw]['y_downsampled']))
                    ax_curr.plot(t_grid_shift, phi_curr)
                    ax_curr.plot(t_grid_shift,BN_prod[ind_cw]['phi_vec_array'][ind_comb,:],'--')

        if(frep_data is not None):
            self.analyze_frep_data(frep_data)
        
        
    def analyze_frep_data(self, 
                          frep_data, 
                          n_points_scan = 100):
        
        self.frep_data = [None]*2

        # Iterate through the two channels with different f_rep
        print(frep_data)
        for ind in range(2):
            # Load data
            if (ind == 0):
                self.frep_data[ind] = frep_data['ch1']
            else:
                self.frep_data[ind] = frep_data['ch2']

            # FFT
            yf_curr = np.fft.fft( self.frep_data[ind]['y_downsampled'])

            # Coarse estimate of f_rep
            idx_f_max = np.argmax(abs(yf_curr))

            # Refined estimate of f_rep
            idx_scan = idx_f_max + list(range(-n_points_scan , n_points_scan ))
            self.frep_data[ind]['idx_f_eff'] = sum(abs(yf_curr[idx_scan])**2 * np.array(idx_scan)) / sum(abs(yf_curr[idx_scan])**2)
            
            self.frep_data[ind]['f_rep_estimate'] = (self.frep_data[ind]['f0_original'] + 
                                                  self.f_grid[idx_f_max] + 
                                                  (self.frep_data[ind]['idx_f_eff'] - idx_f_max) * self.df)
    
    def beatnote_diff(self, 
                      plot_on = False):
        
        if(len(self.BN_prod) != 2):
            raise Exception('beatnote_diff function only valid for 2 beatnote inputs (2x cw lasers)')
        
        BN_prod = self.BN_prod
        t_grid_shift = np.fft.fftshift(self.t_grid)

        #####################################################
        # Need to take second product (to get rid of f_ceo) #
        D_prod_array = np.zeros(2)

        # Check the two configurations depending on complex-conjugate
        for ind_conj in range(2):
            y_BN1 = BN_prod[0]['y_prod']
            if(ind_conj == 0):
                y_BN2 = BN_prod[1]['y_prod']
            else:
                y_BN2 = np.conj(BN_prod[1]['y_prod'])
            
            y_double_prod = y_BN1 * y_BN2
            
            phi_vec = np.unwrap(np.angle(y_double_prod))
            D_prod_array[ind_conj] = self.coherence_metric( phi_vec )
        
        # Get the configuration which minimizes coherence metric
        argmax_D = np.argmin(D_prod_array)
        
        # Compute product
        BN_double = {}
        if(argmax_D == 0):
            BN_double['conj_sign'] = 1
            BN_double['y_prod'] = self.filter_func(BN_prod[0]['y_prod'] * BN_prod[1]['y_prod'])
        else:
            BN_double['conj_sign'] = -1
            BN_double['y_prod'] = self.filter_func(BN_prod[0]['y_prod'] * np.conj(BN_prod[1]['y_prod']))
        
        # Unwrap the phase
        BN_double['phi_vec_prod'] = np.unwrap(np.angle(BN_double['y_prod']))
        BN_double['phi_vec_shift'] = BN_double['phi_vec_prod'] - (t_grid_shift - t_grid_shift[0]) * (BN_double['phi_vec_prod'][-1] - BN_double['phi_vec_prod'][0]) / (t_grid_shift[-1] - t_grid_shift[0])
        
        self.BN_double = BN_double
        
        if(plot_on):
            fig, ax = plt.subplots(1, 1, num = 15, clear=True)        
            ax.plot(t_grid_shift, BN_prod[0]['phi_vec_shift'])
            ax.plot(t_grid_shift, BN_prod[1]['phi_vec_shift'])
            ax.plot(t_grid_shift, BN_double['phi_vec_shift'])

    def calculate_Df_DN(self, tpk_vec, frep_data, f_shift_if_negative = False):
        f0_orig_sum_vec = np.array([x['f0_original_sum'] for x in self.BN_prod])
        f0_BN_diff = f0_orig_sum_vec[0] + f0_orig_sum_vec[1] * self.BN_double['conj_sign']
        
        # Careful: For this f_rep needs to be exact
        if f0_BN_diff < 0:
            if(f_shift_if_negative and (frep_data is not None)):
                f0_BN_diff += self.frep_data[0]['f_rep_estimate']
            else:
                raise('Error - negative f0_BN_diff measured; select beat-notes above f_rep/2 or flip channels)')

        # Get beat-note phase at IGM peaks
        phi_BN_diff_downsampled = np.interp(tpk_vec, np.fft.fftshift(self.t_grid), self.BN_double['phi_vec_prod'])
        phi_BN_diff_abs = phi_BN_diff_downsampled + f0_BN_diff * 2*np.pi*tpk_vec
        
        # Calculate number of comb-lines between cw lasers
        dphi_BN_diff_average = (phi_BN_diff_abs[-1] - phi_BN_diff_abs[0]) / (len(tpk_vec) - 1)
        N_cw_lines_calculated = (dphi_BN_diff_average) / (2*np.pi)
        
        if(abs(N_cw_lines_calculated - np.round(N_cw_lines_calculated)) > 1e-2):
            print('ERROR - integer number of comb lines not found')
        
        self.BN_double['N_cw_lines'] = int(np.round(N_cw_lines_calculated))
        
        print('Delta f (CW lasers): {} MHz'.format(f0_BN_diff*1e-6))
        print('Delta N (CW lasers): {}'.format(N_cw_lines_calculated))
        
        