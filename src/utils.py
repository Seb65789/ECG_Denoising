from src.noise import calculate_snr
import torch

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

        snr_x_total += calculate_snr(y_signal,x_signal)
        snr_y_total += calculate_snr(y_signal,y_pred_signal)


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

