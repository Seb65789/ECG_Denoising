import numpy as np
import itertools
import random
import wfdb
import os
import matplotlib.pyplot as plt

os.chdir('src/')

def get_PLI_noise(x,noise_power):
    '''
    Generates PLI noise with a frequency of 50 Hz to be added to the input signal x.

    Parameters:
        x (numpy array): Input signal.

    Returns:
        numpy array: PLI noise with the same length as x.
    '''
    # Generate a time vector from 0 to 10 seconds with len(x) points
    t = np.linspace(0, 10, len(x))  
    
    # Generate a random phase shift for the noise
    phase_noise = np.random.uniform(0, 2 * np.pi)
    
    # Generate the PLI noise signal
    pli_noise = np.sin(2 * np.pi * 50 * t + phase_noise)

    return pli_noise * np.sqrt(noise_power / np.mean(pli_noise ** 2))

def get_BW_noise(x,noise_power):
    '''
    Generates Baseline Wander (BW) noise to be added to the input signal x.

    Parameters:
        x (numpy array): Input signal.

    Returns:
        numpy array: BW noise with the same length as x.
    '''
    # Generate a time vector from 0 to 10 seconds with len(x) points
    t = np.linspace(0, 10, len(x)) 

    # Determine the sampling frequency based on the length of x
    freq_signal = 100 if len(x) // 12 == 1000 else 500  # Frequency of the signal
    Num_samples = 12000 if freq_signal == 100 else 60000  # Number of points in the signal
    delta_f = freq_signal / Num_samples  # Frequency resolution
    Num_sinusoidal = int(0.5 / delta_f)  # Number of sinusoidal components (K)

    # Generate random coefficients and phases for each sinusoidal component
    a_k = np.random.uniform(0, 1, Num_sinusoidal)  # Random amplitudes
    phase_noise = np.random.uniform(0, 2 * np.pi, Num_sinusoidal)  # Random phases

    # Initialize the BW noise as a zero array
    bw_noise = np.zeros(len(x))

    # Add the contribution of each sinusoidal component to the BW noise
    for k in range(Num_sinusoidal):
        bw_noise += a_k[k] * np.cos(2 * np.pi * k * delta_f * t + phase_noise[k])

    return bw_noise * np.sqrt(noise_power / np.mean(bw_noise ** 2))

def get_EMG_noise(x, noise_power):
    '''
    Generates Electromyographic (EMG) noise to be added to the input signal x.

    Parameters:
        x (numpy array): Input signal.
        SNR_dB (float): Desired Signal-to-Noise Ratio in decibels.

    Returns:
        numpy array: EMG noise with the same length as x.
    '''
    # Generate white Gaussian noise with zero mean and unit variance
    emg_noise = np.random.normal(0, 1, len(x))

    # Scale the noise to match the desired noise power
    emg_noise = emg_noise * np.sqrt(noise_power / np.mean(emg_noise ** 2))

    return emg_noise

def get_white_gaussian_noise(x, noise_power):

    noise = np.random.normal(0, np.sqrt(noise_power), len(x))

    return noise

def get_EMA_noise(x, noise_power):
    '''
    Adds random artifacts to the input signal x.

    Parameters:
        x (numpy array): Input signal (e.g., signal with BW noise).    

    Returns:
        numpy array: Signal with added random artifacts.
    '''

    ema_noise = np.zeros_like(x)

    num_artifacts = np.random.randint(1,10)

    total_duration = 0

    # Add random artifacts
    for _ in range(num_artifacts):

        duration = np.random.randint(10,50) if len(x) == 12000 else np.random.randint(50,250)
        total_duration += duration

        # Random start index for the artifact
        start_idx = np.random.randint(0, len(x) - duration)
        
        # Add the artifact (random Gaussian noise) to the signal
        ema_noise[start_idx:start_idx + duration] +=  np.random.normal(0, 1, duration) #over 50-250 samples

    power_per_sample = noise_power / total_duration

    if np.mean(ema_noise ** 2) == 0:
        return ema_noise  # return as is (all zeros)

    return ema_noise * np.sqrt(power_per_sample / np.mean(ema_noise** 2))

def get_random_combination_weights(n=5):
    weights = np.random.rand(n)
    return weights / np.sum(weights)

def plot_signal(signal,signal2 = None,name = "signal") :

    num_channels = 12

    fig, axes = plt.subplots(12, 1, figsize=(10, 2 * 12))  

    for i in range(num_channels):
        axes[i].plot(signal[:, i], label='Signal 1', color='blue',linestyle='--')
        if signal2 is not None:
            axes[i].plot(signal2[:, i], label='Signal 2', color='orange')
        axes[i].set_title(f'Canal {i + 1}')
        axes[i].set_xlabel('Temps')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend(loc='upper right')

    # Ajuster l'espacement entre les subplots
    plt.tight_layout()
    plt.savefig(f'{name}_test.pdf')

def apply_noises(x, snr_dB):
    '''
    Applies multiple types of noise to the input signal x and adjusts the total noise power 
    to match the target SNR.

    Parameters:
        x (numpy array): Input signal.
        snr_dB (float): Desired Signal-to-Noise Ratio in decibels.

    Returns:
        tuple: Noisy signal and total noise.
    '''
    # Weights
    weights = get_random_combination_weights()

    snr_dB -= 1

    # Calculate the power of the input signal
    signal_power = np.mean(x ** 2)

    # Calculate the total noise power based on the target SNR
    total_noise_power = signal_power / (10**(snr_dB / 10))

    # Distribute the total noise power according to the weights
    P_individual_noises = [w * total_noise_power for w in weights]

    # Generate noises with their specific power
    pli = get_PLI_noise(x, P_individual_noises[0])
    bw = get_BW_noise(x, P_individual_noises[1])
    emg = get_EMG_noise(x, P_individual_noises[2])
    wgn = get_white_gaussian_noise(x, P_individual_noises[3])
    ema = get_EMA_noise(x, P_individual_noises[4])

    # Combine all noises
    total_noise = pli + bw + emg + wgn + ema
    noise_snr = calculate_snr(x,total_noise)
    #print("Target Noise SNR level : {} | Noise SNR level : {:.4f}".format(snr_dB+1,noise_snr))

    # Add the total noise to the signal
    x_noisy = x + total_noise


    return x_noisy

def calculate_snr(signal, noise):
    power_noise = np.mean(noise**2)
    
    power_signal = np.mean(signal**2)
    
    snr = 10 * np.log10(power_signal / power_noise)
    
    return snr





