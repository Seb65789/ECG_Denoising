import wfdb
import numpy as np
import matplotlib.pyplot as plt
import wfdb.plot
import os
import noise 
from collections import Counter




def get_snr_distribution(total_samples):
    snr_values = np.arange(0,32,2)
    print(snr_values)
    # Séparer les valeurs par intervalle
    snr_low = snr_values[snr_values <= 10]       # [0, 2, 4, 6, 8, 10]
    snr_mid = snr_values[(snr_values > 10) & (snr_values <= 20)]  # [12, 14, 16, 18, 20]
    snr_high = snr_values[snr_values > 20]       # [22, 24, 26, 28, 30]

    # Définir le nombre d'échantillons
    n_noisy = int(0.7 * total_samples)
    n_low = int(0.25 * n_noisy)
    n_mid = int(0.5 * n_noisy)
    n_high = n_noisy - n_low - n_mid  # to get 100% of the 70%

    # Générer les SNRs aléatoirement dans chaque plage
    snr_list = []
    snr_list += np.zeros((total_samples-n_noisy)).tolist()
    snr_list += list(np.random.choice(snr_low, n_low, replace=True))
    snr_list += list(np.random.choice(snr_mid, n_mid, replace=True))
    snr_list += list(np.random.choice(snr_high, n_high, replace=True))

    # Shuffle la liste pour la rendre bien répartie
    np.random.shuffle(snr_list)

    return snr_list


def main():

    # Create the datasets if they don't exists
    if not(os.path.exists('data/records100/ecg100.npz') and os.path.exists('data/records500/ecg500.npz')) :
        data_100 = [
        wfdb.rdrecord(f'data/records100/{str(folder).zfill(5)}/{str(num).zfill(5)}_lr').p_signal.flatten('F')
        for folder in range(0, 22000, 1000)
        for num in range(folder + 1, folder + 1000)
        if os.path.exists(f'data/records100/{str(folder).zfill(5)}/{str(num).zfill(5)}_lr.dat')
        ]

        data_500 = [
        wfdb.rdrecord(f'data/records500/{str(folder).zfill(5)}/{str(num).zfill(5)}_hr').p_signal.flatten('F')
        for folder in range(0, 22000, 1000)
        for num in range(folder + 1, folder + 1000)
        if os.path.exists(f'data/records500/{str(folder).zfill(5)}/{str(num).zfill(5)}_hr.dat')
        ]

        X_100 = np.array(data_100)
        X_500 = np.array(data_500)

        print(X_100.shape)
        print(X_500.shape)

        np.savez_compressed('data/records100/ecg100.npz',data = X_100)
        np.savez_compressed('data/records500/ecg500.npz',data = X_500)

        print("Data saved.")
    
    X_100 = np.load('data/records100/ecg100.npz')
    X_500 = np.load('data/records500/ecg500.npz')

    data_100 = X_100['data']
    data_500 = X_500['data']
    
    snr_list_100 = get_snr_distribution(data_100.shape[0])
    snr_list_500 = get_snr_distribution(data_500.shape[0])

    print(len(snr_list_100))
    print(len(snr_list_500))
    snr_counts = Counter(snr_list_100)

    # Afficher proprement
    print("Répartition des niveaux SNR :")
    for snr_level in sorted(snr_counts):
        print(f"SNR {snr_level} dB : {snr_counts[snr_level]} occurrences")



if __name__ == '__main__' :
    main()
