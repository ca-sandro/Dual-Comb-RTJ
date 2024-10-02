import copy
import json

import numpy as np
import scipy.interpolate
import scipy.signal

import numpy.fft as fft_pack
import basics

class SimParameters(basics.functions.ParametersInterface):
    def __init__(self, **kwargs):
        REQUIRED_ARGS = ['f_rep',                   # f_rep, used to define scan window (should be accurate to get spectroscopic/lidar etc info)
                         'Df_rep_approx',           # approx Df_rep, used to assist with peak search
                         'file_stem_IGM',
                         'file_stem_ch1',
                         'file_stem_ch2',
                         'dt']
        
        DEFAULT_VALS = {'N_PREPROCESS_MAX'      : 6,     # approximate number of interferograms to analyze to infer IGM properties
                        'N_PROCESS_MAX'         : np.inf, # how many periods to process
                        'PREPROCESS_PEAK_TOL'   : 0.1,    # How much are peaks separated (in x * Dfrep), input to scipy.signal.find_peaks(distance = )
                        'PREPROCESS_PEAK_HEIGHT': 0.5,    # How high are peaks (in x * max_height), input to scipy.signal.find_peaks(height = )
                        'PREPROCESS_PEAK_FLUC'  : 1e-2,   # Maximum amont of fluctuation in estimated peaks (used as sanity-check)
                        'PROCESS_PEAK_POW'      : 2,      # Power in peak search using center-of-mass
                        'N_PERIODS_DF_AV_MAX'   : 10,     # How many periods to average to estimate Df_rep

                        'PEAK_SEARCH_TOL'       : 1e-2,   # Region (x * 1/Dfrep) to serach for next peak with coarse peak search (based on scipy.signal.find_peaks())
                        'TPK_REFINE_SCALE_TIME' : 20,     # Region (x * t_FWHM) to serach for next peak with refined peak search (based on moment-integral)
                        'TRIGGER_ZOOM_SCALE'    : 8,      # Region (x * t_FWHM) to zoom on peak to extract its properties (f_center and FWHM in frequency domain)
                        'FREQ_CENTER_OVERRIDE'  : None,   # Can select new f_center of RF dual-comb (if algorithm does not provide accurate estimate)
                        }
        
        self.init_vars(DEFAULT_VALS, REQUIRED_ARGS, kwargs)

# %%
class DataProcessCore:
    def __init__(self, sp):
        self.sp = copy.deepcopy(sp)

        self.preprocess_trigger()   # pre-process trigger signal (find temporal/spectral properties of trigger input data)
        self.find_tpk_array()       # find trigger peaks (and phases) through whole trace
        
    def preprocess_trigger(self):
        """
        to determine signal structure:
            pre-search with some flexible conditioning
            estimate duration and center-frequeny of signal
            estimate Dfrep from several periods (used to populate the first Df_rep values and to initiate the main loop)
        """
        # place characteristic info about trigger into self.trig
        self.trigger_properties = {}

        self.trigger_data_file = self.sp.file_stem_IGM.parent / (self.sp.file_stem_IGM.name + '.npy')
        y_mmap = np.load(self.trigger_data_file, mmap_mode='r+')
        self.sp.nt = len(y_mmap)

        # determine number of points to load for preprocessing routine
        t_range_preproc = np.abs(self.sp.N_PREPROCESS_MAX / self.sp.Df_rep_approx)
        n_temp = np.min([np.round(t_range_preproc / self.sp.dt), self.sp.nt])
        nt_load = int(2*np.floor(n_temp/2))
        
        if(nt_load > 1e7):
            print('Warning: {} M data points in preprocessing step'.format(nt_load*1e-6))
        
        # load data from mmap to avoid loading whole trace
        y = y_mmap[:nt_load] + 0 

        # define frequency grid for the data segment loaded
        f_grid = fft_pack.fftfreq(nt_load, self.sp.dt)
        
        # create envelope on which to do do the peak pre-search
        # Note, f_rep here is for filtering out signal replicas, it doesn't need to be very precise
        filter_profile = ((f_grid > 0) * (f_grid < self.sp.f_rep/2))
        y_onesided = fft_pack.ifft(fft_pack.fft(y) * filter_profile)
        y_temp = np.abs(y_onesided) ** self.sp.PROCESS_PEAK_POW
        y_scaled = (y_temp - np.min(y_temp)) / (np.max(y_temp) - np.min(y_temp))

        # find trigger properties and initial set of (coarse) peaks
        trig_prop = find_signal_properties(self.sp.dt,
                                         self.sp,
                                         y_scaled          = y_scaled, # intensity envelope
                                         y_onesided        = y_onesided, # assumes Fourier spectrum is continuous
                                         f_spectrum_bound  = self.sp.f_rep/2,
                                         period_approx     = 1/self.sp.Df_rep_approx)
        
        trig_prop['nt_refine']          = int(2 * np.floor(trig_prop['time_fwhm'] / self.sp.dt * self.sp.TPK_REFINE_SCALE_TIME/2))
        trig_prop['t_refine']           = fft_pack.fftshift(fft_pack.fftfreq(trig_prop['nt_refine'], 1/trig_prop['nt_refine']/self.sp.dt))
        trig_prop['idx_range_refine']   = list(range(-int(trig_prop['nt_refine']/2),int(trig_prop['nt_refine']/2),1))
        
        self.trigger_properties.update(trig_prop)
            
        tpk_norm_refined = []
        for idx_outer, pk_idx in enumerate(self.trigger_properties['tpk_coarse']):
            idx_vec_curr = pk_idx + self.trigger_properties['idx_range_refine'] # indices on which to perform the refined search
            if ((min(idx_vec_curr) > 0) and (max(idx_vec_curr) < nt_load) ): # ignore first & last trigger peak                 
                
                # find t0 with precise search routine using coarsely-determined peaks    
                tpk_norm_curr = pk_idx + self.find_tpk_refined_IGM(y[idx_vec_curr])
                
                # append to vector of *true* arrival times
                tpk_norm_refined.append(tpk_norm_curr)   
        
        self.trigger_presearch = {}
        self.trigger_presearch['tpk_norm'] = tpk_norm_refined
        self.trigger_presearch['Df_rep'] = np.mean(1/np.diff(np.array(tpk_norm_refined)*self.sp.dt))       
        
    def find_tpk_refined_IGM(self, y_curr):
        t_grid_curr = fft_pack.fftshift(fft_pack.fftfreq(len(y_curr), 1/len(y_curr)/self.sp.dt))
        
        y_time = (y_curr-np.mean(y_curr))

        # mixing with a sine wave followed by filtering around DC   
        f_correction = self.trigger_properties['freq_center'] 
        y_freq_filt = fft_pack.ifftshift(basics.functions.phase_correction(fft_pack.fftshift(y_time),
                                                                           f_correction, 
                                                                           np.diff(self.trigger_properties['t_refine'])[0]) )
        
        tpk_local = sum(abs(y_freq_filt)**self.sp.PROCESS_PEAK_POW * t_grid_curr) / sum(abs(y_freq_filt)**self.sp.PROCESS_PEAK_POW) 
                        
        return (tpk_local/self.sp.dt)
 

    def find_tpk_array(self):
        """
        process trigger data to find peaks (and phases, if applicable) through the whole traces
        
        for each period:
            coarse peak search
            refine to find exact delay
        
        output lists:
            t_pk
            period_valid
        """
                    
        continue_pk_search = True

        # Load data
        y_mmap = np.load(self.trigger_data_file, mmap_mode='r+')        
        
        # Define region where we search for peak
        n_search_offset = np.max([self.trigger_properties['nt_refine'], 
                                  2*int(round(1/self.trigger_presearch['Df_rep']/self.sp.dt * self.sp.PEAK_SEARCH_TOL/2))])
        
        t_grid_coarse = fft_pack.fftfreq(n_search_offset, 1/n_search_offset/self.sp.dt)
        
        # Define parameters
        self.N_periods_average = int(np.min([self.sp.N_PREPROCESS_MAX, self.sp.N_PERIODS_DF_AV_MAX]))
        self.trigger_peaks = {}
        self.trigger_peaks['t_norm'] = []
        
        while(continue_pk_search):
            if(self.sp.N_PROCESS_MAX < len(self.trigger_peaks['t_norm'])):
                continue_pk_search = False
                
            if(np.mod(len(self.trigger_peaks['t_norm']), 100) == 0):
                print("calculating t_pk, period {}".format(len(self.trigger_peaks['t_norm'])))
            
            # keep track of moving Df_rep
            if(len(self.trigger_peaks['t_norm']) <= self.N_periods_average):
                Df_moving_av = self.trigger_presearch['Df_rep']
            else:
                Df_moving_av = 1/(np.mean(np.diff(np.array(self.trigger_peaks['t_norm'][-self.N_periods_average:]))) * self.sp.dt)
            idx_offset_Df = int(1/Df_moving_av/self.sp.dt)
            
            # estimation of next peak position
            if(len(self.trigger_peaks['t_norm']) == 0):
                idx_search_center = self.trigger_presearch['tpk_norm'][0]
            else:
                idx_search_center = int(self.trigger_peaks['t_norm'][-1]) + idx_offset_Df
            

            idx_search_min = int(max([0, idx_search_center - n_search_offset/2]))
            idx_search_max = idx_search_min + n_search_offset 
            idx_search_range = range(idx_search_min, idx_search_max) 

            # check if we have passed the last peak
            if(max(idx_search_range) > self.sp.nt):
                continue_pk_search = False
            else:
                y_coarse = y_mmap[idx_search_range] + 0 
                    
                # Frequency shift
                y_temp = abs(fft_pack.ifftshift(basics.functions.phase_correction(fft_pack.fftshift(y_coarse),
                                                                                self.trigger_properties['freq_center'], 
                                                                                np.diff(t_grid_coarse)[0])))

                # Coarse peak search
                y_scaled = (y_temp - np.min(y_temp)) / (np.max(y_temp) - np.min(y_temp))
        
                tpk_temp = scipy.signal.find_peaks(
                    y_scaled, 
                    height = self.sp.PREPROCESS_PEAK_HEIGHT,
                    distance = int(round(self.sp.PREPROCESS_PEAK_TOL * 1/Df_moving_av/ self.sp.dt)))
                
                tpk_idx_local = int(tpk_temp[0][0]) 
                idx_pk_center = idx_search_range[tpk_idx_local]
            
                # refine coarse estimate with refined peak search
                idx_range_curr = list(map(int, idx_pk_center + np.array(self.trigger_properties['idx_range_refine'])))
                y_curr = y_mmap[idx_range_curr]

                tpk_norm_offset =  self.find_tpk_refined_IGM(y_curr)

                tpk_norm_curr = idx_pk_center + tpk_norm_offset

                self.trigger_peaks['t_norm'].append(tpk_norm_curr)
            
        # Store peaks
        self.t_norm_vec = np.array(self.trigger_peaks['t_norm']) * self.sp.dt

# %% find temporal and spectral properties of a periodic signal
def find_signal_properties(dt,
                         sp,
                         y_scaled = None, 
                         y_onesided = None,
                         period_approx = None,
                         f_spectrum_bound = None):
    
    PREPROCESS_PEAK_TOL     = sp.PREPROCESS_PEAK_TOL 
    PREPROCESS_PEAK_HEIGHT  = sp.PREPROCESS_PEAK_HEIGHT
    PREPROCESS_PEAK_FLUC    = sp.PREPROCESS_PEAK_FLUC
    TRIGGER_ZOOM_SCALE      = sp.TRIGGER_ZOOM_SCALE
    
    peak_presearch_distance = int(round(PREPROCESS_PEAK_TOL * period_approx/ dt)) # used to guide the distance between peaks in the search
            
    # Rough estimation of peaks
    tpk_temp = scipy.signal.find_peaks(y_scaled, 
                                       height = PREPROCESS_PEAK_HEIGHT * np.max(y_scaled),
                                       distance = peak_presearch_distance)
    tpk_diff = np.diff(tpk_temp[0])
    
    # Check that peaks are not fluctuating too much (which would be indicator for problems with peak finding)
    tpk_fluctuation_check = np.max(np.abs( (tpk_diff/np.mean(tpk_diff) - 1)))
    if(tpk_fluctuation_check > PREPROCESS_PEAK_FLUC):
        raise Exception('Error - tpk estimations fluctuating above PREPROCESS_PEAK_FLUC')
    
    # peaks to analyze in pre-processing
    peaks_analysis_vec = list(range(0,len(tpk_temp[0]))) 

    # indices of a single period
    n_indices_period_pow2 = 2**np.floor(np.log2(np.min(np.abs(tpk_diff)))) 
    
    time_width_fwhm_vec = []
    freq_center_vec = []

    sig_prop = {}

    for ind_peaks in peaks_analysis_vec:
        # dont perform analysis on first or last entries
        if((ind_peaks == 0) or (ind_peaks == (len(tpk_temp[0])-1))):
            pass

        else:
            # index of the approximately-determined peak from the presearch
            idx_tpk = tpk_temp[0][ind_peaks]
            
            # select period
            idx_range_curr = idx_tpk + list(range(-int(n_indices_period_pow2/2),int(n_indices_period_pow2/2),1))
            
            # find fwhm in time-domain (in 'number of indices' units (dt grid point spacing))
            width_fwhm_curr = basics.functions.fwhm((abs(y_scaled[idx_range_curr ]) ))
            time_width_fwhm_vec.append(width_fwhm_curr)
            
            # find indices to zoom in around the peak in time
            nt_zoom = 2*int(TRIGGER_ZOOM_SCALE/2 * width_fwhm_curr)
            if(nt_zoom >= n_indices_period_pow2):
                raise Exception('Error - fwhm width of trigger above analysis threshold')
            
            idx_zoom_curr = idx_tpk + list(range(-int(nt_zoom/2), int(nt_zoom/2),1))
                        
            # frequency grid corresponding to zoom-in around peak
            f_zoom = fft_pack.fftshift(fft_pack.fftfreq(nt_zoom, dt))

            # determine frequency range of interest:                
            num_idx_freq_offset = 2; # offset by 2 to avoid DC 
            idx_f_search_min = int(nt_zoom/2) + num_idx_freq_offset # int(nt_zoom/2) corresponds to "0" frequency
            idx_f_search_max = np.argmin(abs(f_zoom - f_spectrum_bound)) - num_idx_freq_offset # go nearly up to f_rep/2
                
            # zoom in Fourier space
            y_onesided_zoom_curr = y_onesided[idx_zoom_curr]

            yf_onesided_zoom_curr = fft_pack.fftshift(fft_pack.fft(fft_pack.fftshift(y_onesided_zoom_curr - np.mean(y_onesided_zoom_curr))))
            idx_freq_zoom = list(range(idx_f_search_min,idx_f_search_max,1))
               
            # center in frequency (units of Hz)
            yf_abs_width_calc = abs(yf_onesided_zoom_curr[idx_freq_zoom])**2
            center_freq_idx_zoom = sum(yf_abs_width_calc * idx_freq_zoom) / sum(yf_abs_width_calc)
            
            freq_center_vec.append(np.interp(center_freq_idx_zoom, list(range(0,len(f_zoom))), f_zoom) )
    
    # temporal width
    sig_prop['time_fwhm']                = dt * np.mean(time_width_fwhm_vec)
    
    # center frequency
    if(sp.FREQ_CENTER_OVERRIDE is None):
        sig_prop['freq_center']   = np.mean(np.array(freq_center_vec))
    else:
        sig_prop['freq_center']   = sp.FREQ_CENTER_OVERRIDE

    # First peak
    sig_prop['tpk_coarse']              = tpk_temp[0]
   
    return sig_prop