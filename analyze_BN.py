# -*- coding: utf-8 -*-
"""
Parameters:



"""

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
    dir_data = Path(r'Desktop/Noise_Manual/2024_09_11_Beatnote_Evaluation/data/proc') # Path
    filename_core = '--trace20khz--00002' # Filename

    # Processing settings
    downsample_BN = False # Select if we have to downsample the data (needs to be done only once)
    flip_BN_order = False # Select if order of beat-notes should be flipped

    # Measurement parameters

# %% interferogram analysis
ch1_filename_stem =  r'C1' + filename_core
ch2_filename_stem =  r'C2' + filename_core
IGM_filename_stem =  r'C3' + filename_core
BN_filename_stem = r'BN' + filename_core 

sp = dcd.SimParameters(f_rep                    = 1041e6,
                       Df_rep_approx            = 20e3,   # needed to do preprocessing steps where Df is found more accurately
                       N_PREPROCESS_MAX         = 6,       # approximate number of interferograms to analyze to infer IGM properties
                       PEAK_SEARCH_TOL          = 1e-2,     # how far to search around expected peak region
                       PROCESS_PEAK_POW         = 4,
                       TPK_REFINE_SCALE_FREQ    = 4, # 2
                       TPK_REFINE_SCALE_TIME    = 20, # 10,
                       file_stem_IGM            = dir_data / IGM_filename_stem, 
                       file_stem_ch1            = dir_data / ch1_filename_stem, 
                       file_stem_ch2            = dir_data / ch2_filename_stem, 
                       dt                       = 2.000000026702864 * 1e-10,
                       nt                       = 50000002
                       )

model = dcd.DataProcessCore(sp)

# %%
BN_filename = BN_filename_stem + r'.pickle'

if (downsample_BN == True):
    BN_data = DS.downsample_beatnotes(sp, n_plot=1000000)
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

# %% apply IGM data to calculate Df phase
BN_analysis = bnd.DataProcess(dir_data, 
                   BN_data, 
                   flip_BN_order = flip_BN_order,
                   plot_on=False)

BN_analysis.beatnote_diff(plot_on = False)
BN_analysis.calculate_Df_DN(np.array(model.t_norm_vec))

# %% plot PN-PSD and TJ-PSD
phi_scaled = BN_analysis.BN_double['phi_vec_prod']/BN_analysis.BN_double['N_cw_lines']

TJ_PSD = basics.noise_analysis.TJ_PSD(phi_scaled, 
                                      BN_analysis.dt, 
                                      N_segment_scale = 1, 
                                      f0 = model.sp.f_rep,
                                      smooth_method = 'welch')
TJ_PSD.plot_PSD(fignum = 201)
plt.show()