import numpy as np
import matplotlib.pyplot as plt

def downsample_function(yf_full, sp, f0, fBW):
    nt_orig = sp.nt
    dt_orig = sp.dt

    df = 1/nt_orig/dt_orig    
    
    f0_grid_norm = 2*(round(f0 / df / 2))
    idx_range = (2*round(fBW / df /2))
    f0_original = f0_grid_norm * df

    nt = idx_range
    dt = 1/idx_range/df

    idx_f0 = f0_grid_norm
    idx_downsample = np.array(idx_f0 + np.fft.fftshift(np.arange(-idx_range/2, idx_range/2))).astype(int)

    yf_downsampled = yf_full[idx_downsample]
    y_downsampled = np.fft.ifft(yf_downsampled)

    DS_data = {"nt_orig": nt_orig,
               "dt_orig": dt_orig,
               "dt": dt,
               "nt": nt,
               "f0_original": f0_original,
               "y_downsampled": y_downsampled}

    return DS_data

def downsample_beatnotes(sp, select_higher_BN, n_plot = 5000, meas_frep = False):  
    #sp.nt = 1000000
    
    y_mmap = np.load(sp.file_stem_ch1.parent / (sp.file_stem_ch1.name + '.npy'), mmap_mode='r+')
    sp.nt = len(y_mmap)
    
    y_data_ch1 = y_mmap[:sp.nt] + 0 

    y_mmap = np.load(sp.file_stem_ch2.parent / (sp.file_stem_ch2.name + '.npy'), mmap_mode='r+')
    y_data_ch2 = y_mmap[:sp.nt] + 0 

    dt_orig = sp.dt

    f_grid_plot = np.fft.fftfreq(n_plot, dt_orig)
    yf_full_ch1_plot = np.fft.fft(y_data_ch1[0:n_plot]) 
    yf_full_ch2_plot = np.fft.fft(y_data_ch2[0:n_plot]) 
        
    if (select_higher_BN):
        f_lim = sp.f_rep
    else:
        f_lim = sp.f_rep/2
        
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.fft.fftshift(f_grid_plot) * 1e-6, abs(np.fft.fftshift(yf_full_ch1_plot)), color='red')
    ax[0].set_xlim(0, f_lim * 1e-6)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('RF Frequency (MHz)')
    ax[0].set_ylabel('Amplitude (a.u.)')
    ax[0].set_title('Channel 1')

    ax[1].plot(np.fft.fftshift(f_grid_plot) * 1e-6, abs(np.fft.fftshift(yf_full_ch2_plot)), color='blue')
    ax[1].set_xlim(0, f_lim * 1e-6)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('RF Frequency (MHz)')
    ax[1].set_ylabel('Amplitude (a.u.)')
    ax[1].set_title('Channel 2')
    plt.show()

    f0_ch1_cw1 = float(input("Center frequency (channel 1, beat-note 1) (in MHz): \n"))
    f0_ch1_cw2 = float(input("Center frequency (channel 1, beat-note 2) (in MHz): \n"))
    f0_ch2_cw1 = float(input("Center frequency (channel 2, beat-note 1) (in MHz): \n"))
    f0_ch2_cw2 = float(input("Center frequency (channel 2, beat-note 2) (in MHz): \n"))
    fBW = float(input("RF_bandwidth of beat-notes (in MHz): \n"))
    
    print("perform fft (slow for long traces)")
       
    yf_full_ch1 = np.fft.fft(y_data_ch1)
    yf_full_ch2 = np.fft.fft(y_data_ch2)
    # Iterate through the cw lasers and beat-notes
    BN_data = {"ch1_cw1": downsample_function(yf_full_ch1, sp, f0=f0_ch1_cw1 * 1e6, fBW=fBW * 1e6),
            "ch1_cw2": downsample_function(yf_full_ch1, sp, f0=f0_ch1_cw2 * 1e6, fBW=fBW * 1e6),
            "ch2_cw1": downsample_function(yf_full_ch2, sp, f0=f0_ch2_cw1 * 1e6, fBW=fBW * 1e6),
            "ch2_cw2": downsample_function(yf_full_ch2, sp, f0=f0_ch2_cw2 * 1e6, fBW=fBW * 1e6)}        
    
    if (meas_frep == True):
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(np.fft.fftshift(f_grid_plot) * 1e-6, abs(np.fft.fftshift(yf_full_ch1_plot)))
        ax[0].set_xlim(0, 1.1 * sp.f_rep * 1e-6)
        ax[0].set_yscale('log')
        ax[0].set_xlabel('RF Frequency (MHz)')
        ax[0].set_ylabel('Amplitude (a.u.)')
        ax[0].set_title('Channel 1')

        ax[1].plot(np.fft.fftshift(f_grid_plot) * 1e-6, abs(np.fft.fftshift(yf_full_ch2_plot)))
        ax[1].set_xlim(0, 1.1 * sp.f_rep * 1e-6)
        ax[1].set_yscale('log')
        ax[1].set_xlabel('RF Frequency (MHz)')
        ax[1].set_ylabel('Amplitude (a.u.)')
        ax[0].set_title('Channel 2')
        plt.show()
        
        frep_ch1 = float(input("Center frequency frep (channel 1) (in MHz): \n"))
        frep_ch2 = float(input("Center frequency frep (channel 2) (in MHz): \n"))
        frep_BW = float(input("RF_bandwidth of frep (in MHz): \n"))

        # Iterate through the cw lasers and beat-notes
        frep_data = {"ch1": downsample_function(yf_full_ch1, sp, f0=frep_ch1 * 1e6, fBW=frep_BW * 1e6),
                    "ch2": downsample_function(yf_full_ch2, sp, f0=frep_ch2 * 1e6, fBW=frep_BW * 1e6)}
    else:
        frep_data = None
        
    # Plot downsampled data
    t_grid = np.arange(0, BN_data['ch1_cw1']['nt']) * BN_data['ch1_cw1']['dt']
    fig, ax = plt.subplots(2, 2)
    ax[0,0].plot(t_grid * 1e3, np.gradient(np.unwrap(np.angle(BN_data['ch1_cw1']["y_downsampled"]))), color = 'red')
    ax[1,0].plot(t_grid * 1e3, np.gradient(np.unwrap(np.angle(BN_data['ch1_cw2']["y_downsampled"]))), color = 'red')
    ax[0,1].plot(t_grid * 1e3, np.gradient(np.unwrap(np.angle(BN_data['ch2_cw1']["y_downsampled"]))), color = 'blue')
    ax[1,1].plot(t_grid * 1e3, np.gradient(np.unwrap(np.angle(BN_data['ch2_cw2']["y_downsampled"]))), color = 'blue')
    
    ax[0,0].set_xlabel('Time (ms)')
    ax[0,1].set_xlabel('Time (ms)')
    ax[1,0].set_xlabel('Time (ms)')
    ax[1,1].set_xlabel('Time (ms)')
    
    ax[0,0].set_ylabel(r'Gradient[$\phi$(t)]')
    ax[0,1].set_ylabel(r'Gradient[$\phi$(t)]')
    ax[1,0].set_ylabel(r'Gradient[$\phi$(t)]')
    ax[1,1].set_ylabel(r'Gradient[$\phi$(t)]')

    ax[0,0].set_title('Channel 1')
    ax[0,1].set_title('Channel 2')
    plt.show()
    
    f_grid = np.fft.fftfreq(BN_data['ch1_cw1']['nt'], BN_data['ch1_cw1']['dt'])
    fig2, ax2 = plt.subplots(2, 2) 
    ax2[0,0].plot(np.fft.fftshift(f_grid) * 1e-6, np.fft.fftshift(np.abs(np.fft.fft(BN_data['ch1_cw1']["y_downsampled"]))), color= 'red')
    ax2[1,0].plot(np.fft.fftshift(f_grid) * 1e-6, np.fft.fftshift(np.abs(np.fft.fft(BN_data['ch1_cw2']["y_downsampled"]))), color= 'red')
    ax2[0,1].plot(np.fft.fftshift(f_grid) * 1e-6, np.fft.fftshift(np.abs(np.fft.fft(BN_data['ch2_cw1']["y_downsampled"]))), color= 'blue')
    ax2[1,1].plot(np.fft.fftshift(f_grid) * 1e-6, np.fft.fftshift(np.abs(np.fft.fft(BN_data['ch2_cw2']["y_downsampled"]))), color= 'blue')
    
    ax2[0,0].set_yscale('log')
    ax2[1,0].set_yscale('log')
    ax2[0,1].set_yscale('log')
    ax2[1,1].set_yscale('log')
    
    ax2[0,0].set_xlabel('Frequency (MHz)')
    ax2[0,1].set_xlabel('Frequency (MHz)')
    ax2[1,0].set_xlabel('Frequency (MHz)')
    ax2[1,1].set_xlabel('Frequency (MHz)')
    
    ax2[0,0].set_ylabel('Amplitude (a.u.)')
    ax2[0,1].set_ylabel('Amplitude (a.u.)')
    ax2[1,0].set_ylabel('Amplitude (a.u.)')
    ax2[1,1].set_ylabel('Amplitude (a.u.)')

    ax2[0,0].set_title('Channel 1')
    ax2[0,1].set_title('Channel 2')
    plt.show()  
    
    # return BN_data, frep_data
    return BN_data