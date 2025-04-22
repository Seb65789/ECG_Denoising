import argparse as parse
from torch.utils.data import DataLoader
import os
import src.data
from src.data import ECGDataset
import torch
import torch.nn as nn
import time
from src.utils import train
import matplotlib.pyplot as plt
from src.noise import plot_signal
from itertools import product
import numpy as np
import pandas as pd




def main():

    # On cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create arguments
    arguments = parse.ArgumentParser() # Creating the arguments

    arguments.add_argument('-vrs',default=100,type=int,help="100 for small data, 500 for bigger data")
    arguments.add_argument('-epochs',default=20,type=int) # How many epochs
    arguments.add_argument("-lr",default=0.001,type=float)
    arguments.add_argument("-optimizer",choices=['sgd','adam'],type=str,default='sgd')
    arguments.add_argument("-activation",choices=['tanh','relu'],type=str,default='tanh')
    arguments.add_argument("-pooling",choices=['avg','max'],type=str,default='avg')
    arguments.add_argument("-layers",default=2,type=int)
    arguments.add_argument("-batch_size",default=256,type=int)
    arguments.add_argument('-mode',choices=['search','try'],default='search')

    opt = arguments.parse_args()

    # Build dataset
    #src.data.main()

    # Loading Dataset
    dataset = ECGDataset(vrs=opt.vrs)
    X_val, y_val = dataset.val_X, dataset.val_y
    X_test,y_test = dataset.test_X, dataset.test_y
    X_val = X_val.view(2178, 12, 1000).to(device)
    y_val = y_val.view(2178, 12, 1000).to(device)
    X_test = X_test.view(2178, 12, 1000).to(device)
    y_test = y_test.view(2178, 12, 1000).to(device)
    print(opt.mode)

    if opt.mode == 'try' : # Don't shuffle for reproductibility
    
        train(dataset=dataset,X_val=X_val,y_val=y_val,X_test=X_test,y_test=y_test,
          nb_epochs=opt.epochs,batch_size=opt.batch_size,optimizer=opt.optimizer,lr=opt.lr,
          layers=opt.layers,activation=opt.activation,pooling=opt.pooling,vrs='100',device=device,mode = opt.mode)
    
    # Cutsom gridsearch mode
    if opt.mode == 'search' :

        print('GridSearch Mode !')
        # Custom GridSearch
        param_grid = {
            'lr': np.linspace(0.001,0.1,5).tolist(),
            'epochs' : [20,50,100,200],
            'batch_size': [ 64, 128, 256, 512],
            'optimizer': ['adam', 'sgd'],
            'nb_layers': [2, 4, 6],
            'activation' : ['relu','tanh'],
            'pooling':['avg']
        }

        #results
        results = []

        # Creates the combinaisons
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]

        for i, params in enumerate(param_combinations):
            print(f"\n>>> Try {i+1}/{len(param_combinations)} with : {params}")

            loss,snr = train(dataset=dataset,X_val=X_val,y_val=y_val,X_test=X_test,y_test=y_test,
                nb_epochs=params['epochs'],
                lr=params['lr'],
                batch_size=params['batch_size'],
                optimizer=params['optimizer'],
                layers=params['nb_layers'],
                activation=params['activation'],
                pooling=params['pooling'],
                vrs=opt.vrs,
                device=device,
                mode= opt.mode)
            

            # adding parameters and scores to list to create df
            result_entry = params.copy()  
            result_entry['MSE'] = loss
            result_entry['SNR_dB'] = snr
            results.append(result_entry)
            print('Test loss : {:.8f} | Denoisy SNR level : {:.8f}'.format(loss,snr) )

        # Convertit en DataFrame
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'results/{opt.vrs}/GridSearch_Results.csv', index=False)
        print("GridSearch finished !")
        
    if opt.model == 'best' :

        # Upload dataframe
        df_performances = pd.read_csv(f'results/{opt.vrs}/GridSearch_Results.csv')

        best_configuration = df_performances.loc[df_performances['SNR_dB'].idxmin()]
        best_lr = best_configuration['lr']
        best_epochs = best_configuration['epochs']
        best_batch_size = best_configuration['batch_size']
        best_optimizer = best_configuration['optimizer']
        best_nb_layers = best_configuration['nb_layers']
        best_activation = best_configuration['activation']
        best_pooling = best_configuration['pooling']

        train(dataset=dataset,X_val=X_val,y_val=y_val,X_test=X_test,y_test=y_test,
              nb_epochs=best_epochs,batch_size=best_batch_size,optimizer=best_optimizer,
              lr=best_lr,layers=best_nb_layers,activation=best_activation,pooling=best_pooling,
              vrs=opt.vrs,device=device,mode='try')


if __name__ == '__main__':
    main()