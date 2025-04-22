import wfdb
import numpy as np
import os
import src.noise 
from collections import Counter
from sklearn.model_selection import train_test_split

import torch

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self,vrs = 100) :
        
        data = np.load(f'data/{vrs}/clear.npz')
        data_noisy = np.load(f'data/{vrs}/noisy.npz')

        train_X = data_noisy["train"]
        train_y = data["train"]
        val_X = data_noisy["val"]
        val_y = data["val"]
        test_X = data_noisy["test"]
        test_y = data["test"]

        self.X = torch.tensor(train_X).float()
        self.y = torch.tensor(train_y).float()

        self.val_X = torch.tensor(val_X).float()
        self.val_y = torch.tensor(val_y).float()

        self.test_X = torch.tensor(test_X).float()
        self.test_y = torch.tensor(test_y).float()

        print("Dataset created !")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx].view(12,1000),self.y[idx].view(12,1000)


def get_snr_distribution(total_samples):
    snr_values = np.arange(0, 32, 2)
    # Séparer les valeurs par intervalle
    snr_low = snr_values[snr_values <= 10]       # [0, 2, 4, 6, 8, 10]
    snr_mid = snr_values[(snr_values > 10) & (snr_values <= 20)]  # [12, 14, 16, 18, 20]
    snr_high = snr_values[snr_values > 20]       # [22, 24, 26, 28, 30]

    # Définir le nombre d'échantillons
    n_noisy = total_samples
    n_low = int(0.25 * n_noisy)
    n_mid = int(0.5 * n_noisy)
    n_high = n_noisy - n_low - n_mid  # pour obtenir 100% des 70%

    # Générer les SNRs aléatoirement dans chaque plage
    snr_low_samples = np.random.choice(snr_low, n_low, replace=True)
    snr_mid_samples = np.random.choice(snr_mid, n_mid, replace=True)
    snr_high_samples = np.random.choice(snr_high, n_high, replace=True)

    # Créer un tableau numpy avec les SNRs

    snr_array = np.concatenate([snr_low_samples, snr_mid_samples, snr_high_samples])

    # Mélanger la liste pour la rendre bien répartie
    np.random.shuffle(snr_array)

    return snr_array

def main():
    
    if not(os.path.exists('data/100/clear.npz') and os.path.exists('data/100/noisy.npz')) :
        data_100 = [
        wfdb.rdrecord(f'data/records100/{str(folder).zfill(5)}/{str(num).zfill(5)}_lr').p_signal.flatten('F')
        for folder in range(0, 22000, 1000)
        for num in range(folder + 1, folder + 1000)
        if os.path.exists(f'data/records100/{str(folder).zfill(5)}/{str(num).zfill(5)}_lr.dat')
        ]

        data_100 = np.array(data_100)

        data_100_noisy = data_100.copy()

        snr_list_100 = get_snr_distribution(data_100.shape[0])

        snr_counts_100 = Counter(snr_list_100)
        print(len(snr_counts_100))
        

        print("SNR levels repartition for datasets:")
        print("\nFor dataset 100:")
        for snr_level, count in sorted(snr_counts_100.items()):
            print(f"SNR {snr_level} dB : {count} occurrences")

        for i in range(len(snr_list_100)) :
            if i % 1000 == 0 : print(i)
            data_100_noisy[i] += src.noise.apply_noises(data_100_noisy[i],snr_dB=snr_list_100[i])
            
        data_100_noisy = data_100_noisy.astype(np.float32)

        # Splitting into train test val datasets
        train_100_noisy, test_100_noisy, train_100_clear, test_100_clear = train_test_split(data_100_noisy, data_100, test_size=0.10, shuffle=True)
        train_100_noisy, val_100_noisy, train_100_clear, val_100_clear= train_test_split(train_100_noisy,train_100_clear, test_size=len(test_100_clear), shuffle=True) 
        
        print(f"Clear : \nTrain 100: {train_100_clear.shape}, Val 100: {val_100_clear.shape}, Test 100: {test_100_clear.shape}")
        print(f"Noisy : \nTrain 100: {train_100_noisy.shape}, Val 100: {val_100_noisy.shape}, Test 100: {test_100_noisy.shape}")

        
        # Save
        os.makedirs("data/100", exist_ok=True)
        os.makedirs("data/500", exist_ok=True)

        np.savez_compressed("data/100/noisy.npz", train=train_100_noisy,val=val_100_noisy,test=test_100_noisy)
        np.savez_compressed("data/100/clear.npz", train=train_100_clear,val=val_100_clear,test=test_100_clear)

        print("Data saved.")

        ''' Higher model data --use this version to create the 500 dataset
        if not(os.path.exists('data/records500/ecg500.npz') :
        
            data_500 = [
            wfdb.rdrecord(f'data/records500/{str(folder).zfill(5)}/{str(num).zfill(5)}_hr').p_signal.flatten('F')
            for folder in range(0, 22000, 1000)
            for num in range(folder + 1, folder + 1000)
            if os.path.exists(f'data/records500/{str(folder).zfill(5)}/{str(num).zfill(5)}_hr.dat')
            ]

            X_500 = np.array(data_500)
            np.savez_compressed('data/records500/ecg500.npz',data = X_500)

        X_500 = np.load('data/records500/ecg500.npz')
        data_500 = X_500['data']
        data_500_noisy = data_500.copy()

        snr_list_500 = get_snr_distribution(data_500.shape[0])
        snr_counts_500 = Counter(snr_list_500)
        print(len(snr_counts_500))

        print("\nFor dataset 500:")
        for snr_level, count in sorted(snr_counts_500.items()):
            print(f"SNR {snr_level} dB : {count} occurrences")

        for i in range(len(snr_list_500)) :
            if i % 1000 == 0 : print(i)
            data_500_noisy[i] += noise.apply_noises(data_500_noisy[i],snr_dB=snr_list_500[i])


        # Splitting into train test val datasets
        train_500_noisy, test_500_noisy, train_500_clear, test_500_clear = train_test_split(data_500_noisy, data_500, test_size=0.10, shuffle=True)
        train_500_noisy, val_500_noisy, train_500_clear, val_500_clear= train_test_split(train_500_noisy,train_500_clear, test_size=0.10, shuffle=True) 
        
        print(f"Clear : \nTrain 500: {train_500_clear.shape}, Val 500: {val_500_clear.shape}, Test 500: {test_500_clear.shape}")
        print(f"Noisy : \nTrain 500: {train_500_noisy.shape}, Val 500: {val_500_noisy.shape}, Test 500: {test_500_noisy.shape}")

        np.savez_compressed("data/500/noisy.npz", train=train_500_noisy,val=val_500_noisy,test=test_500_noisy)
        np.savez_compressed("data/500/clear.npz", train=train_500_clear,val=val_500_clear,test=test_500_clear)
        '''
    else : 
        print("Dataset 100 exists.")

        
if __name__ == '__main__' :
    main()
