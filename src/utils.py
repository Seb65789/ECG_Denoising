from src.noise import calculate_snr
import torch
import torch.nn as nn
import time 
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from src.noise import plot_signal
from src.DAE import ECG_DAE

def evaluate(model,criterion,X,y):
    model.eval() # Evaluation mode

    y_pred = model(X) # Predictions

    loss = criterion(y_pred,y) # Compute the loss mse
    loss = loss.item() 

    # To numpy
    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    batch_size = X.shape[0]
    snr_x_total = 0
    snr_y_total = 0

    for sample in range(batch_size):

        y_signal = y_np[sample].flatten()
        x_signal = X_np[sample].flatten()
        y_pred_signal = y_pred_np[sample].flatten() 

        snr_x_total += calculate_snr(y_signal.T,x_signal.T)
        snr_y_total += calculate_snr(y_signal.T,y_pred_signal.T)


    snr_x = snr_x_total / batch_size
    snr_y = snr_y_total / batch_size

    return loss, snr_x, snr_y # Compare snr of x noisy with denoisy x

def train_batch(model,X,y,optimizer,criterion,**kwargs):
    optimizer.zero_grad() # reset optimizer

    y_pred = model(X) # prediction
    loss = criterion(y_pred,y) # loss
    loss.backward() # Compute gradients

    # Gradient update
    optimizer.step()
    return loss.item()

def train(dataset,X_val,y_val,X_test,y_test,nb_epochs,batch_size,optimizer,lr,layers,activation,pooling,vrs, device,mode) :

    model = ECG_DAE(input=(batch_size,12,1000 if vrs == '100' else 5000),
                        nb_layers=layers,
                        activation_type=activation,
                        pooling_type=pooling,mode=mode).to(device)

    train_dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)

    # Criterion
    criterion = nn.MSELoss()

    # Optimizer 
    optims = {"sgd":torch.optim.SGD,
            "adam": torch.optim.Adam}
    
    optim = optims[optimizer]
    optim = optim(model.parameters(),lr = lr)

    # Training 
    epochs = torch.arange(1,nb_epochs+1)


    if not(mode == "search") :
        init_loss,init_snr_x,init_snr_y = evaluate(model,criterion,X_val,y_val)        
        print("Initial validation loss : {:.4f} | Noisy SNR level : {:.4f} | DeNoisy SNR level : {:.4f} ".format(init_loss,init_snr_x,init_snr_y)) 
    if not(mode == "search") :
    # Performances
        train_losses = []
        val_losses = []
        val_snr_x = []
        val_snr_y = []

    # Training time measure
    start = time.time()
    
    for epoch in epochs:
        print(f"Training epoch {epoch}/{nb_epochs}")
        epoch_loss_train = [] # train loss of epoch
        for X_batch,y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss = train_batch(model=model, # loss of training on batch
                            X=X_batch,
                            y=y_batch,
                            optimizer=optim
                            ,criterion=criterion)
            if not(mode == "search") : epoch_loss_train.append(loss) # add loss
        if not(mode == "search") :
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
    if not(mode == "search") : 
        print(f"Test loss : {test_loss} | Test Noisy SNR : {test_snr_x} | Test DeNoisy SNR : {test_snr_y}")

        # Saving the results
        if not(os.path.isdir(f"results/{vrs}/")) :
            os.makedirs(f"results/{vrs}/")  
            print(f"Results {vrs} directory created")

        with open(f"results/{vrs}/results.txt","a") as f:
            f.write("\nTraining took {}:{} - Final test loss {:.4f}" \
            "\n Parameters : \n\t - layers : {} \n\t --batch_size : {}\n\t -epochs :{} \n\t -learning_rate : {} "
                " \n\t -Optimizer : {}  ".format(minutes,seconds,test_loss,layers,batch_size,nb_epochs,lr,optimizer))
        
        plt.plot(epochs,train_losses,label='Training loss')
        plt.plot(epochs,val_losses,label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/{vrs}/Training_Validation_loss_layer-{layers}-_epochs-{nb_epochs}-_batchsize-{batch_size}.pdf')
        plt.close()

        plt.plot(epochs,val_snr_x,label='Noisy SNR')
        plt.plot(epochs,val_snr_y,label='DeNoisy SNR')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/{vrs}/Training_Validation_SNR_layer-{layers}-_epochs-{nb_epochs}-_batchsize-{batch_size}.pdf')   
        plt.close()

        x_test = X_test[0]
        y_pred = model(y_test[0].view(1,12,10*int(vrs)))
        y_test = y_test[0]

        plot_signal(y_test.detach().cpu().numpy().T,
                    y_pred.detach().cpu().numpy().T,
                    name = f'results/{vrs}/clear_denoisy')
        
        plot_signal(y_test.detach().cpu().numpy().T,
                    x_test.detach().cpu().numpy().T,
                    name = f'results/{vrs}/clear_noisy')
        
        if mode == 'best' :
            torch.save(model.state_dict(), "results/{vrs}/best_model_weights.pth")
    
    return test_loss,test_snr_y


