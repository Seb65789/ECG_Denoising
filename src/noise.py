import numpy as np
import itertools
import random
import wfdb
import os
import matplotlib.pyplot as plt

os.chdir('src/')

def get_PLI_noise(x,SNR_dB):
    '''
    Generates PLI noise with a frequency of 50 Hz to be added to the input signal x.

    Parameters:
        x (numpy array): Input signal.

    Returns:
        numpy array: PLI noise with the same length as x.
    '''
    # Calculate the power of the input signal
    P_signal = np.mean(x ** 2)

    # Calculate the noise power based on the desired SNR
    P_noise = P_signal / (10 ** (SNR_dB / 10))

    # Generate a time vector from 0 to 10 seconds with len(x) points
    t = np.linspace(0, 10, len(x))  
    
    # Generate a random phase shift for the noise
    phase_noise = np.random.uniform(0, 2 * np.pi)
    
    # Generate the PLI noise signal
    pli_noise = np.sin(2 * np.pi * 50 * t + phase_noise)

    return pli_noise * np.sqrt(P_noise / np.mean(pli_noise ** 2))

def get_BW_noise(x,SNR_dB):
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

    # Calculate the power of the input signal
    P_signal = np.mean(x ** 2)

    # Calculate the noise power based on the desired SNR
    P_noise = P_signal / (10 ** (SNR_dB / 10))
    
    return bw_noise * np.sqrt(P_noise / np.mean(bw_noise ** 2))

def get_EMG_noise(x, SNR_dB):
    '''
    Generates Electromyographic (EMG) noise to be added to the input signal x.

    Parameters:
        x (numpy array): Input signal.
        SNR_dB (float): Desired Signal-to-Noise Ratio in decibels.

    Returns:
        numpy array: EMG noise with the same length as x.
    '''
    # Calculate the power of the input signal
    P_signal = np.mean(x ** 2)

    # Calculate the noise power based on the desired SNR
    P_noise = P_signal / (10 ** (SNR_dB / 10))

    # Generate white Gaussian noise with zero mean and unit variance
    emg_noise = np.random.normal(0, 1, len(x))

    # Scale the noise to match the desired noise power
    emg_noise = emg_noise * np.sqrt(P_noise / np.mean(emg_noise ** 2))

    return emg_noise

def get_white_gaussian_noise(x, SNR_dB):

    signal_power = np.mean(x**2)
    SNR_linear = 10**(SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(x))

    return noise

def get_EMA_noise(x, SNR_dB):
    '''
    Adds random artifacts to the input signal x.

    Parameters:
        x (numpy array): Input signal (e.g., signal with BW noise).    

    Returns:
        numpy array: Signal with added random artifacts.
    '''
    
    P_signal = np.mean(x ** 2)
    P_noise_total = P_signal / (10 ** (SNR_dB / 10))

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

    P_per_sample = P_noise_total / total_duration

    if np.mean(ema_noise ** 2) == 0:
        return ema_noise  # return as is (all zeros)

    return ema_noise * np.sqrt(P_per_sample / np.mean(ema_noise** 2))

def get_random_combination_dB(target_dB):


    # Génère toutes les combinaisons possibles avec remplacement dans l'intervalle donné
    all_combinations = [
        comb for comb in itertools.product(range(2, target_dB + 1), repeat=5)
        if np.mean(comb) == target_dB
    ]

    if not all_combinations:
        raise ValueError("No combinaison found")

    return list(random.choice(all_combinations))


def apply_noises(x, snr_list):
    
    # List of noise types
    noise_functions = [
        get_PLI_noise,  # PLI noise (50 Hz)
        get_BW_noise,   # Baseline Wander (BW) noise
        get_EMG_noise,  # EMG noise
        get_white_gaussian_noise,  # White Gaussian noise
        get_EMA_noise   # EMA noise
    ]
    
    # Loop through each noise type and apply it to the signal
    for i, snr_db in enumerate(snr_list):
        if i < len(noise_functions):  # Make sure the index is within the noise function list
            noise_func = noise_functions[i]
            x += noise_func(x, snr_db)
    
    return x


ecg = wfdb.rdrecord('data/records500/00000/00001_hr')
signal = ecg.p_signal
flat_signal = signal.flatten('F')


comb_dB = get_random_combination_dB(6)
print(comb_dB)

noisy_signal = flat_signal + apply_noises(flat_signal,comb_dB) 
print(noisy_signal.shape)
noisy_signal = noisy_signal.reshape((5000,12),order = 'F')




# Nombre de canaux
num_channels = ecg.p_signal.shape[1]

# Créer une figure avec un sous-graphe pour chaque canal
fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels))  # Ajuste la taille de la figure

# Pour chaque canal, afficher son signal
for i in range(num_channels):
    axes[i].plot(ecg.p_signal[:, i], label=f'Canal {i + 1}')
    axes[i].set_title(f'Canal {i + 1}')
    axes[i].set_xlabel('Temps')
    axes[i].set_ylabel('Amplitude')
    axes[i].legend()

# Ajuster l'espacement entre les subplots
plt.tight_layout()

# Afficher les signaux
plt.savefig('clear_signal_test.pdf')


# Nombre de canaux
num_channels = ecg.p_signal.shape[1]

# Créer une figure avec un sous-graphe pour chaque canal
fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2 * num_channels))  # Ajuste la taille de la figure

# Pour chaque canal, afficher son signal
for i in range(num_channels):
    axes[i].plot(noisy_signal[:, i], label=f'Canal {i + 1}')
    axes[i].set_title(f'Canal {i + 1}')
    axes[i].set_xlabel('Temps')
    axes[i].set_ylabel('Amplitude')
    axes[i].legend()

# Ajuster l'espacement entre les subplots
plt.tight_layout()

# Afficher les signaux
plt.savefig('noisy_signal_test.pdf')