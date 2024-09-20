import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import basics

class NoiseSpectrum:
    def __init__(self, 
                 y, 
                 dt, 
                 smooth_method = None, 
                 **kwargs):        
        n = len(y)
        if(n > 10**7):
            print('Warning, {} length data'.format(n))
        
        self.orig = {'y': y, 'dt': dt, 'nt': len(y), 'df': 1/dt/n} # original data as inputted

        if(smooth_method is None):
            core_algorithm = 'fft'

            if ('N_smooth_fft' in kwargs):
                self.N_smooth_fft = kwargs['N_smooth_fft']
            else:
                self.N_smooth_fft = None
            
        elif(smooth_method == 'welch'):
            core_algorithm = 'welch'
            self.N_segment_scale = kwargs['N_segment_scale']
            
        if('f_max_integration' in kwargs):
            f_max_integration = kwargs['f_max_integration']
        else:
            f_max_integration = np.inf

        self.method = core_algorithm
        
        self.calculate_PSD_1sided()
        self.integrate_PSD(f_max_integration = f_max_integration)
        
    def calculate_PSD_1sided(self):
        
        #y_orig = self.orig['y']
        if(self.method == 'fft'):
            self.f_PSD, self.PSD = self.PSD_1sided_func(self.orig['y'], N_smooth_fft = self.N_smooth_fft)

        if(self.method == 'welch'):
            nt_inner = 2 * int(np.round(self.orig['nt'] / self.N_segment_scale / 2))
            num_segments = self.N_segment_scale*2-1
            f_segment = np.fft.fftfreq(nt_inner, self.orig['dt'])
            
            idx_offset_vec = list(range(-int(nt_inner/2),int(nt_inner/2)))
            idx_start_vec = np.linspace(-idx_offset_vec[0], self.orig['nt']-1-idx_offset_vec[-1], num_segments).astype(int)
            
            PSD_full = np.zeros(nt_inner)
            for idx_seg in range(num_segments):
                idx_curr = idx_offset_vec + idx_start_vec[idx_seg]
                f_temp, PSD_curr = self.PSD_2sided_func(self.orig['y'][idx_curr])
                
                PSD_full += PSD_curr / num_segments
            
            self.f_PSD, self.PSD = self.PSD_1sided_conversion(f_segment, PSD_full)
            self.PSD_units = self.base_unit + '^2/Hz'
    
    def integrate_PSD(self, 
                      custom_PSD = None,
                      direction = 'backwards',
                      store_value = True, 
                      cumulative = True, 
                      f_max_integration = np.inf):
        
        if( (direction == 'backwards') or (direction == 'back')):
            int_sign_func = np.flip
        elif((direction == 'forwards') or (direction == 'forw')):
            int_sign_func  = lambda x: x
        else:
            raise Exception('integration direction not defined')
        
        if(custom_PSD is not None):
            PSD = custom_PSD
            store_value = False
        else:
            PSD = self.PSD
        
        
        if(cumulative == True):
            noise_integrated = np.sqrt(int_sign_func(np.cumsum(int_sign_func(PSD * 
                                                                             abs(np.gradient(self.f_PSD)) * 
                                                                             (self.f_PSD < f_max_integration)
                                                                             ))))
        else:
            noise_integrated = np.sqrt(int_sign_func(np.sum(int_sign_func(PSD * 
                                                                          abs(np.gradient(self.f_PSD)) * 
                                                                          (self.f_PSD < f_max_integration)
                                                                          ))))
            store_value = False
            
        if(store_value):
            self.PSD_integrated = noise_integrated
        else:
            return noise_integrated
        
       
    def PSD_2sided_func(self, y, N_smooth_fft = None):
        """
        input units ARB
        fft_ESD yields units of [ARB][t]
        sum(|fft_ESD|^2 * df ) = sum(|y|^2 * dt)
        so LHS and RHS ~ energy, or [arb]^2 [t]
        power: divide by T
        sum(|fft_ESD|^2 * df / T) = sum(|y|^2 * dt) / T = sum(|y|^2) / nt ~ <|y|^2>
        
        sum(|fft_ESD|^2 * df / T) = sum(|fft_ESD|^2 * (1/nt/dt) / (dt*nt))
                                  = sum(|fft_ESD|^2 / (dt*nt)^2)
                                  = sum(|fft|^2 / (nt)^2) 
                                  = sum(PSD * df)
        """
        
        # prepare signal, e.g. including apodization
        y_prep = self.prep_data(y)
        n = len(y_prep)
        dt = self.orig['dt']
        df = 1/dt/n
        f_full = np.fft.fftfreq(n, dt)
        
        yf2_prep = abs(np.fft.fft(y_prep) )**2
        if(N_smooth_fft is not None):
            PSD_full = basics.functions.movingaverage(yf2_prep , N_smooth_fft) / (n**2 * df)
        else:
            PSD_full = yf2_prep / (n**2 * df)
        return f_full, PSD_full
    
    def PSD_1sided_func(self, y, N_smooth_fft = None):
        n = len(y)
        f_full, PSD_full = self.PSD_2sided_func(y, N_smooth_fft)
        
        idx_pos = range(0,int(n/2))
        PSD_out = 2*PSD_full[idx_pos]
        PSD_out[0] = PSD_full[0]
        
        f_PSD = f_full[idx_pos]
        
        return f_PSD, PSD_out
    
    def PSD_1sided_conversion(self, f_full, PSD_full):
        n = len(PSD_full)
        idx_pos = range(0,int(n/2))
        PSD_out = 2*PSD_full[idx_pos]
        PSD_out[0] = PSD_full[0]
        
        f_PSD = f_full[idx_pos]
        
        return f_PSD, PSD_out
        
    def prep_data_shift_apodize(self, y):
        """
        prepare phase data so there is no slope or offset at zero
        and a suitably scaled apodization window is applied
        """
        y -= (y[-1] - y[0]) / (len(y)-1) * np.array(range(len(y)))
        y -= y[0]
        filter_func_unscaled = lambda N: np.cos(np.pi/2 * np.linspace(-1,1,N))
        filter_func = lambda N: filter_func_unscaled(N) / np.sqrt(sum(filter_func_unscaled(N)**2)/N)
        return y * filter_func(len(y)) 
        
class TJ_PSD(NoiseSpectrum):
    base_unit = 's'
    
    def __init__(self, y_in,dt, **kwargs):
        self.prep_data = super().prep_data_shift_apodize        
        
        if 'f0' not in kwargs.keys():
            raise Exception('For TJ_PSD, y must be phase and kwarg input f0 must be carrier frequency')
        else:
            self.f0 = kwargs['f0']
            y = y_in / (2*np.pi *self.f0)
            
        super().__init__(y, dt, **kwargs)
        
        #self.PSD = PN_PSD.PSD / (2*np.pi*self.f0)**2
        
    def plot_PSD(self, fignum = 112, xlims = [10, 1000000]):
        
        fig, ax = plt.subplots(2, 1, num = fignum, clear=True)        
        ax[0].plot(self.f_PSD, self.PSD  * 1e30,'-')
        ax[0].set_ylabel(r'TJ-PSD ($\mathrm{fs}^2$/Hz)')
        
        ax[1].plot(self.f_PSD, self.PSD_integrated * 1e15,'-')
        ax[1].set_ylabel('Integrated TJ (fs)')
        
        for ax_ in ax:
            ax_.grid()
            ax_.set_yscale('log')
            ax_.set_xscale('log')
            ax_.set_xlim(xlims)
            ax_.set_xlabel('Frequency (Hz)')
        return fig, ax
   