import copy
import numpy as np
from scipy.interpolate import interp1d

# -------------------------------------------- #
class ParametersInterface:
    def __init__(self,**kwargs):
        REQUIRED_ARGS = []
        DEFAULT_VALS = {}
        
        self.init_vars(DEFAULT_VALS, REQUIRED_ARGS, kwargs)
    
    def init_vars(self, DEFAULT_VALS, REQUIRED_ARGS, kwargs):
        for arg in REQUIRED_ARGS:
            if(arg in kwargs):
                setattr(self, arg, kwargs[arg])
            else:
                raise Exception('Error: must input ' + arg)
        
        for arg in DEFAULT_VALS:
            if(arg in kwargs):
                setattr(self, arg, kwargs[arg])
            else:
                setattr(self, arg, DEFAULT_VALS[arg])
                

def tanh_filter(x, xi, xf, wi, wf):
    """
    generic filter of user defined width and extent based on tanh functions
    tanh functions are infinitely differentiable and very smooth. 
    This makes them convenient in many situations where one wishes to maintain "clean" 
    spectral and temporal characteristics of complex fields
    """
    y = (
        (1 + np.tanh((x - xi)/wi)) * 
        (1 + np.tanh(-(x - xf)/wf)) ) / 4
    return y

def fwhm(vec, 
         y_thresh = 0.5, 
         offset_input = False):
    """
    full width at half maximum calculation
    the does not look for mltiple-pulse situations. It treats everything in the field 
    as one pulse and looks for the outer boundaries where the intensity is half the maximum
    
    offset_input can be used to put the function between 0 and 1
    y_threshold can be used to calculate something other than half the maximum
    """
    
    x = np.array(range(len(vec)))
    if(offset_input):
        v_offset = min(vec)
    else:
        v_offset = 0 
    vec_off = vec - v_offset
    if(max(vec_off) <= 0):
        return -1
    else:
        y = vec_off / max(vec_off)
        #y = (vec - v_offset) / (max(vec) - v_offset)

          
    x_thresh = [ii for ii, vv in enumerate(y) if vv >= y_thresh]
    if(len(x_thresh) == 0):
        return -1
    else:
        #raise Exception('x_thresh length zero')
        idx_lims = [min(x_thresh), max(x_thresh)]
        
        
        idx_lims_adj = [None,None]
        for ii,vv in enumerate(idx_lims):
            if(vv == 0):
                idx_lims_adj[ii] = x[0]
            elif(vv == len(vec)-1):
                idx_lims_adj[ii] = x[-1]
            else:
                idx_interp = [vv-1, vv, vv+1]
                x_interp_curr = interp1d(y[idx_interp], x[idx_interp])
                idx_lims_adj[ii] = x_interp_curr(y_thresh)
            
        return max(idx_lims_adj) - min(idx_lims_adj)


def phase_correction(v_in, f0, dt):
    """
    function to perform a frequency shift on an oscillating signal 
    usually to bring it around 0 frequency
    includes spectral filter to apply AFTER the frequency shift operation
    """
    nt = len(v_in)
    t_grid = np.fft.fftfreq(nt, 1/nt/dt)
    f_grid = np.fft.fftfreq(nt, dt)
        
    v_prod = (v_in - np.mean(v_in)) * np.exp(-1j*2*np.pi*f0*t_grid)

    spectral_filter_total = (abs(f_grid) < f0)
    
    v_prod_filt = np.fft.fft(v_prod) * spectral_filter_total
    out_vec = np.fft.ifft(v_prod_filt)
    
    return out_vec

def argmin_array(arr_in):
    arr = np.array(arr_in)
    if(len(arr.shape) != 2):
        raise Exception('Error, function defined for 2D array')
    rowcol_idx = np.argmin(np.array(arr))
    
    col = np.mod(rowcol_idx, arr.shape[1])
    row = int(np.floor(rowcol_idx/arr.shape[1]))

    return [row, col]
