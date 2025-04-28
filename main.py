import argparse as parse
from src.data import ECGDataset
import torch
import src.noise 
import src.data 
from src.DAE import ECG_DAE
from src.utils import train
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
    arguments.add_argument('-mode',choices=['search','try','best'],default='search')

    opt = arguments.parse_args()

    # Build dataset
    src.data.main()

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
            'epochs' : [20,50,100],
            'batch_size': [ 64, 128, 256],
            'optimizer': ['adam'],
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

        best_configuration = df_results.loc[df_results['SNR_dB'].idxmin()]        
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
              vrs=opt.vrs,device=device,mode='best')

        print("Best Model Saved !")
        
    if opt.mode == 'best' :

        df_results = pd.read_csv(f'results/{opt.vrs}/GridSearch_Results.csv')

        best_configuration = df_results.loc[df_results['MSE'].idxmin()]        
        best_lr = best_configuration['lr']
        best_epochs = best_configuration['epochs']
        best_batch_size = best_configuration['batch_size']
        best_optimizer = best_configuration['optimizer']
        best_nb_layers = best_configuration['nb_layers']
        best_activation = best_configuration['activation']
        best_pooling = best_configuration['pooling']
        best_mse = best_configuration['MSE']
        print("Best hyperparameters:")
        print(f"Learning Rate: {best_lr}")
        print(f"Epochs: {best_epochs}")
        print(f"Batch size: {best_batch_size}")
        print(f"Optimizer: {best_optimizer}")
        print(f"Nb Layers: {best_nb_layers}")
        print(f"Activation: {best_activation}")
        print(f"Pooling: {best_pooling}")

        print("Best MSE : ",best_mse)
        print("Best SNR : ", best_configuration['SNR_dB'])

        # Load the model
        model = ECG_DAE(input=(best_batch_size,12,1000 if opt.vrs == '100' else 5000),
                        nb_layers=best_nb_layers,
                        activation_type=best_activation,
                        pooling_type=best_pooling,mode='try').to(device)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(f"results/{opt.vrs}/best_modelweights.pth", map_location=device,weights_only=True))
        model.eval()  # Met le modèle en mode évaluation
        

        # Copie des données de test
        y_test_bw = y_test[0].clone().detach().cpu().numpy().flatten()
        y_test_pli = y_test[0].clone().detach().cpu().numpy().flatten()
        y_test_ema = y_test[0].clone().detach().cpu().numpy().flatten()
        y_test_emg = y_test[0].clone().detach().cpu().numpy().flatten()
        y_test_gauss = y_test[0].clone().detach().cpu().numpy().flatten()


        # Calcul de la puissance du signal
        signal_power = np.mean(y_test[0].detach().cpu().numpy().flatten() ** 2)


        # Calcul de la puissance du bruit
        noise_power = signal_power / (10**(5 / 10))
        

        # Ajout de bruit aux signaux
        y_test_bw += src.noise.get_BW_noise(y_test_bw, noise_power)
        y_test_pli += src.noise.get_PLI_noise(y_test_pli, noise_power)
        y_test_emg += src.noise.get_EMG_noise(y_test_emg, noise_power)
        y_test_ema += src.noise.get_EMA_noise(y_test_ema, noise_power)
        y_test_gauss += src.noise.get_white_gaussian_noise(y_test_gauss, noise_power)
        
        y_test_bw = torch.tensor(y_test_bw).view(1,12,1000).to(device)
        y_test_pli = torch.tensor(y_test_pli).view(1,12,1000).to(device)
        y_test_emg = torch.tensor(y_test_emg).view(1,12,1000).to(device)
        y_test_ema = torch.tensor(y_test_ema).view(1,12,1000).to(device)
        y_test_gauss = torch.tensor(y_test_gauss).view(1,12,1000).to(device)
        
        
        y_pred_bw = model(y_test_bw)
        y_pred_ema = model(y_test_ema)
        y_pred_gauss = model(y_test_gauss)
        y_pred_pli = model(y_test_pli)  
        y_pred_emg = model(y_test_emg)


        snr_bw = src.noise.calculate_snr(y_test[0].view(12,1000).detach().cpu().numpy().flatten(),y_pred_bw.view(12,1000).detach().cpu().numpy().flatten())
        snr_pli = src.noise.calculate_snr(y_test[0].view(12,1000).detach().cpu().numpy().flatten(),y_pred_pli.view(12,1000).detach().cpu().numpy().flatten())
        snr_ema = src.noise.calculate_snr(y_test[0].view(12,1000).detach().cpu().numpy().flatten(),y_pred_ema.view(12,1000).detach().cpu().numpy().flatten())
        snr_emg = src.noise.calculate_snr(y_test[0].view(12,1000).detach().cpu().numpy().flatten(),y_pred_emg.view(12,1000).detach().cpu().numpy().flatten())
        snr_gauss = src.noise.calculate_snr(y_test[0].view(12,1000).detach().cpu().numpy().flatten(),y_pred_gauss.view(12,1000).detach().cpu().numpy().flatten())

        src.noise.plot_signal(y_test[0].detach().cpu().numpy().T,
                    y_pred_bw.detach().cpu().numpy().T,
                    name = f'results/{opt.vrs}/clear_denoisy_BW_SNR_{snr_bw}')
        src.noise.plot_signal(y_test[0].detach().cpu().numpy().T,
                    y_pred_pli.detach().cpu().numpy().T,
                    name = f'results/{opt.vrs}/clear_denoisy_PLI_SNR_{snr_pli}')
        src.noise.plot_signal(y_test[0].detach().cpu().numpy().T,
                    y_pred_ema.detach().cpu().numpy().T,
                    name = f'results/{opt.vrs}/clear_denoisy_EMA_SNR_{snr_ema}')
        src.noise.plot_signal(y_test[0].detach().cpu().numpy().T,
                    y_pred_emg.detach().cpu().numpy().T,
                    name = f'results/{opt.vrs}/clear_denoisy_EMG_SNR_{snr_emg}')
        src.noise.plot_signal(y_test[0].detach().cpu().numpy().T,
                    y_pred_gauss.detach().cpu().numpy().T,
                    name = f'results/{opt.vrs}/clear_denoisy_Gaussian_SNR_{snr_gauss}')
if __name__ == '__main__':
    main()