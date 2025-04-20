import wfdb
import numpy as np
import matplotlib as plt

# We have to concatenate the files together

test_ecg = wfdb.rdrecord('data/records100/00000/00001_lr')

signal_test = test_ecg.p_signal


data_100 = []
data_500 = []

for folder in range(00000,22000,1000):
    folder = str(folder).zfill(5) # to ensure that their are 5 digits 
    path_100 = f'src/data/records_100/{folder}'
    for num in range(int(folder),int(folder)+1000) :
        pass
