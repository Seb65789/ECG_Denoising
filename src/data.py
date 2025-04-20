import wfdb
import numpy as np
import matplotlib.pyplot as plt
import wfdb.plot
import os

# Set path 
os.chdir('src/')

# Extract data

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

np.savez_compressed('data/ecg100.npz',X_100)
np.savez_compressed('data/ecg500.npz',X_500)

print("Data saved.")