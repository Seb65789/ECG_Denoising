import argparse as parse
from torch.utils.data import DataLoader
import os
import src.data
from src.data import ECGDataset
from src.DAE import ECG_DAE
import torch
import torch.nn as nn
import time
from src.utils import evaluate,train_batch
import matplotlib.pyplot as plt
from src.noise import plot_signal


def main():

    # On cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    opt = arguments.parse_args()

    # Build dataset
    src.data.main()

    # Loading Dataset
    dataset = ECGDataset(vrs=opt.vrs)

    # Creating batches
    train_dataloader = DataLoader(dataset=dataset,batch_size=opt.batch_size,shuffle=False) # Don't shuffle for reproductibility
    X_val, y_val = dataset.val_X, dataset.val_y
    X_test,y_test = dataset.test_X, dataset.test_y
    X_val = X_val.view(2178, 12, 1000).to(device)
    y_val = y_val.view(2178, 12, 1000).to(device)
    X_test = X_test.view(2178, 12, 1000).to(device)
    y_test = y_test.view(2178, 12, 1000).to(device)

    print(X_val.shape,y_val.shape)

    # Reshape Val and test to (samples,channels,values)

    model = ECG_DAE(input=(opt.batch_size,12,1000 if opt.vrs == 100 else 5000),
                    nb_layers=opt.layers,
                    activation_type=opt.activation,
                    pooling_type=opt.pooling).to(device)
    
    # Criterion
    criterion = nn.MSELoss(reduction='sum')

    # Optimizer 
    optims = {"sgd":torch.optim.SGD,
              "adam": torch.optim.Adam}
    
    optimizer = optims[opt.optimizer]
    optimizer = optimizer(model.parameters(),lr = opt.lr)

    # Training 
    epochs = torch.arange(1,opt.epochs+1)

    init_loss,init_snr_x,init_snr_y = evaluate(model,criterion,X_val,y_val)

    print("Initial validation loss : {:.4f} | Noisy SNR level : {:.4f} | DeNoisy SNR level : {:.4f} ".format(init_loss,init_snr_x,init_snr_y))

    # Performances
    train_losses = []
    val_losses = []
    val_snr_x = []
    val_snr_y = []

    # Training time measure
    start = time.time()
    
    for epoch in epochs:
        print(f"Training epoch {epoch}")
        epoch_loss_train = [] # train loss of epoch
        for X_batch,y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss = train_batch(model=model, # loss of training on batch
                               X=X_batch,
                               y=y_batch,
                               optimizer=optimizer
                               ,criterion=criterion)
            epoch_loss_train.append(loss) # add loss
        
        epoch_loss_train = torch.tensor(epoch_loss_train).mean().item() # Mean loss
        
        epoch_loss_val, epoch_val_snr_x, epoch_val_snr_y = evaluate(model,criterion,X_val,y_val)
        print("Train loss: {:.4f} | Validation loss: {:.4f} | Validation Noisy SNR: {:.4f} | Validation DeNoisy SNR: {:.4f}".format(epoch_loss_train,epoch_loss_val,epoch_val_snr_x,epoch_val_snr_y))

        train_losses.append(epoch_loss_train)
        val_losses.append(epoch_loss_val)
        val_snr_x.append(epoch_val_snr_x)
        val_snr_y.append(epoch_val_snr_y)    

    end = time.time() - start
    minutes = int(end//60)
    seconds = int(end%60)
    print("Training took {} minutes and {} seconds".format(minutes,seconds))
    
    test_loss,test_snr_x,test_snr_y = evaluate(model,criterion,X_test,y_test)
    print(f"Test loss : {test_loss} | Test Noisy SNR : {test_snr_x} | Test DeNoisy SNR : {test_snr_y}")

    # Saving the results
    if not(os.path.isdir(f"results/{opt.vrs}/")) :
        os.makedirs(f"results/{opt.vrs}/")  
        print(f"Results {opt.vrs} directory created")

    with open(f"results/{opt.vrs}/results.txt","a") as f:
        f.write("\nTraining took {}:{} - Final test loss {:.4f}" \
        "\n Parameters : \n\t - layers : {} \n\t --batch_size : {}\n\t -epochs :{} \n\t -learning_rate : {} "
            " \n\t -Optimizer : {}  ".format(minutes,seconds,test_loss,opt.layers,opt.batch_size,opt.epochs,opt.lr,opt.optimizer))
    
    plt.plot(epochs,train_losses,label='Training loss')
    plt.plot(epochs,val_losses,label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/{opt.vrs}/Training_Validation_loss_layer-{opt.layers}-_epochs-{opt.epochs}-_batchsize-{opt.batch_size}-'
                '_optimizer-{optimizer}-_lr-{opt.lr}-_activation-{opt.activation}-_pooling-{opt.pooling}.pdf')
    
    plt.plot(epochs,val_snr_x,label='Noisy SNR')
    plt.plot(epochs,val_snr_y,label='DeNoisy SNR')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/{opt.vrs}/Training_Validation_loss_layer-{opt.layers}-_epochs-{opt.epochs}-_batchsize-{opt.batch_size}-'
                '_optimizer-{optimizer}-_lr-{opt.lr}-_activation-{opt.activation}-_pooling-{opt.pooling}.pdf')
   

    test_x = X_test[0].view(1,12,1000)
    test_y = y_test[0].view(1,12,1000).detach().cpu().numpy()
    test_y_pred = model(test_x).detach().cpu().numpy()
    test_x = test_x.detach().cpu().numpy()

    plot_signal(test_y.reshape(12,1000).T,test_y_pred.reshape(12,1000).T,name=f'results/{opt.vrs}/original_denoisy')
    plot_signal(test_x.reshape(12,1000).T,test_y_pred.reshape(12,1000).T,name=f'results/{opt.vrs}/noisy_denoisy')

if __name__ == '__main__':
    main()