To execute the code:

1) set the path to where the data is stored
2) Select the filename_core:

The files need to be named according to the following system:

- First channel with beatnotes:  'C1' + filename_core + '.npy'
- Second channel with beatnotes: 'C2' + filename_core + '.npy'
- Interferograms:                'C3' + filename_core + '.npy'

the filename_core can be any string.

Example:
filename_core = '_trace':

- First channel:  'C1_trace.npy'
- Second channel: 'C2_trace.npy'
- Interferograms: 'C3_trace.npy'

3) Provide the processing parameters:
 
- f_rep_approx: repetition rate of the laser (approximately)
- Df_rep_approx: repetition rate difference (approximately)
- dt: sampling time-step of the oscilloscope (inverse of sampling rate)
- nt: number of samples in the trace

4) Select if beat-notes have to be downsampled
This needs to be set to True once in the beginning, or if difference beat-notes have to be selected (see below in step (6))

Note: This involves an FFT of the full trace, meaning that it is a slow operation. It should thus only be executed if necessary

5) if you reciece the error: 'ERROR - integer number of comb lines not found', the selected beat-notes did not correspond to the same which yields an offset by the resulting signal by N*frep (N is an integer)

In this case, we can flip the order of the beat-notes to potentially account for this

--> select flip_BN_order = True

6) if you still receive the error, we need to select higher beat-notes above f_rep to select the corresponding beat-notes

--> select select_higher_BN = True


