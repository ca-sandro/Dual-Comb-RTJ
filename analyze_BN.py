from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

import basics
import DC_data.dual_comb_data as dcd
import DC_data.beat_note_data as bnd
import DC_data.spectrum_downsample as DS

config = 1
if(config == 1):
    # Path
    dir_data = Path(r'Desktop/Noise_Manual/2024_09_11_Beatnote_Evaluation/data/proc') # Path
    filename_core = '--trace20khz--00003' # Filename stem

    # Processing paramters:
    f_rep = 1041e6 # Repetition rate of the laser
    D_frep_approx = 20e3 # Repetition rate difference of the laser
    dt = 2.000000026702864 * 1e-10 # Sampling time of oscilloscope
    nt = 50000002 # Number of points in trace
    
    downsample_BN = True # Select if we have to downsample the data (needs to be done once in the begining)

    # If does not work due to issues with difference of beatnotes:
    flip_BN_order = True # (A): Select if order of beat-notes should be flipped
    select_higher_BN = True # (B): Select if higher-frequency beat-notes can be used, in this case need to select downsample_BN = True


# %% Construct the full file names
ch1_filename_stem =  r'C1' + filename_core
ch2_filename_stem =  r'C2' + filename_core
IGM_filename_stem =  r'C3' + filename_core
BN_filename_stem = r'BN' + filename_core 

# Store measurement parameters
sp = dcd.SimParameters(f_rep                    = f_rep,
                       Df_rep_approx            = D_frep_approx,  
                       file_stem_IGM            = dir_data / IGM_filename_stem, 
                       file_stem_ch1            = dir_data / ch1_filename_stem, 
                       file_stem_ch2            = dir_data / ch2_filename_stem, 
                       dt                       = dt,
                       nt                       = nt
                       )

model = dcd.DataProcessCore(sp)

# %% Downsample the beatnotes
BN_filename = BN_filename_stem + r'.pickle'
if (downsample_BN == True):
    BN_data = DS.downsample_beatnotes(sp, 
                                      select_higher_BN = select_higher_BN)
    
    with open(dir_data / BN_filename, 'wb') as f:
        pickle.dump(BN_data, f)
else:
    BN_data = pickle.load(open(dir_data / BN_filename, 'rb'))

"""
# Experimental: Get frep data
meas_frep = True # Enable experimental determination of f_rep
frep_filename_stem = r'frep' + filename_core 
frep_filename = frep_filename_stem + r'.pickle'

if (downsample_BN == True):
    BN_data, frep_data = DS.downsample_beatnotes(sp, n_plot=1000000, meas_frep=meas_frep)
    
    with open(dir_data / frep_filename, 'wb') as f:
        pickle.dump(frep_data, f)
else:
    BN_data = pickle.load(open(dir_data / BN_filename, 'rb'))
    
    if meas_frep:
        frep_data = pickle.load(open(dir_data / frep_filename, 'rb'))
    else:
        frep_data = None
"""

# %% Calculate BN phase
BN_analysis = bnd.DataProcess(dir_data, 
                   BN_data, 
                   flip_BN_order = flip_BN_order,
                   plot_on=False)

BN_analysis.beatnote_diff(plot_on = False)
BN_analysis.calculate_Df_DN(np.array(model.t_norm_vec))

# %% plot TJ-PSD
phi_scaled = BN_analysis.BN_double['phi_vec_prod']/BN_analysis.BN_double['N_cw_lines']

TJ_PSD = basics.noise_analysis.TJ_PSD(phi_scaled, 
                                      BN_analysis.dt, 
                                      N_segment_scale = 1, 
                                      f0 = model.sp.f_rep,
                                      smooth_method = 'welch')
TJ_PSD.plot_PSD(fignum = 201)
plt.show()