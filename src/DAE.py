import torch
import torch.nn as nn
import numpy as np

class ECG_DAE(nn.Module):
    def __init__(self,input,nb_layers=2,activation_type='tanh',pooling_type='avg',**kwargs):

        # The input is already in (batch_size,n_channels,len_signal)

        super().__init__()

        if len(input) != 3 :
            print(f"The input dimension should be (batch_size,n_channels,signal_length) but it is not : {len(input)} dimensions")
            raise("Wrong input dimensions")

        activations = {'tanh':nn.Tanh(),
                       'relu':nn.ReLU()}
        activation = activations[activation_type]

        poolings = {'avg': nn.AvgPool1d(2),
                    'max': nn.MaxPool1d(2)} # Divide by two each layer
        pooling = poolings[pooling_type]

        
        in_out_sizes = [input[1]*2**layer_i for layer_i in range(nb_layers+1)] # Multiply output channels by 2

        kernel_sizes = np.arange(3,3+2*(nb_layers),2) 
        kernel_sizes = (kernel_sizes[::-1]).tolist()# The kernels decreasing with the number of layers

        # Let's create the encoder
        encoder_layers = [layer for layer_i in range(0,nb_layers-1) # To divide the input by 4 and perform the loop 2 times
                            for layer in (nn.Conv1d(in_channels=in_out_sizes[layer_i],
                                                    out_channels=in_out_sizes[layer_i+1],
                                                    kernel_size=kernel_sizes[layer_i],
                                                    padding=(kernel_sizes[layer_i]-1)//2),
                                        activation,
                                        pooling) # Apply formula pad = (kernel-1)/2
                        ]
        encoder_layers.extend([nn.Conv1d(in_channels=in_out_sizes[-2]
                                        ,out_channels=in_out_sizes[-1]
                                        ,kernel_size=kernel_sizes[-1]
                                        ,padding=(kernel_sizes[-1]-1)//2),
                            activation,
                            pooling])
        
        self.encoder = nn.Sequential(*encoder_layers) # * to unpack layers
        print("\n----------------------------------------------------------Encoder----------------------------------------------------------\n",self.encoder)
        print("---------------------------------------------------------------------------------------------------------------------------")

        #bottleneck
        self.bottleneck = nn.Sequential(nn.Conv1d(in_channels=in_out_sizes[-1],
                                                  out_channels=in_out_sizes[-1],
                                                  kernel_size=kernel_sizes[-1],
                                                  padding=(kernel_sizes[-1]-1)//2)
                                        ,activation,
                                        nn.Upsample(2))

        print("\n-----------------------------------------------------------Bottleneck--------------------------------------------------------\n",self.bottleneck)
        print("---------------------------------------------------------------------------------------------------------------------------")


        # reverse the in_out_sizes
        in_out_sizes = in_out_sizes[::-1]
        kernel_sizes = kernel_sizes[::-1]

        # Now the decoder
        decoder_layers = [layer for layer_i in range(nb_layers-1)
                          for layer in (nn.Conv1d(in_channels=in_out_sizes[layer_i],
                                                  out_channels = in_out_sizes[layer_i+1],
                                                  kernel_size=kernel_sizes[layer_i],
                                                  padding=(kernel_sizes[layer_i]-1)//2),
                                        activation,
                                        nn.Upsample(2))] # 
        
        decoder_layers.append(nn.Conv1d(in_channels = in_out_sizes[-2], # Last reconstructing layer
                                         out_channels = in_out_sizes[-1],
                                         kernel_size=3,
                                         padding=1))
        
        self.decoder = nn.Sequential(*decoder_layers)
        print("\n----------------------------------------------------------Decoder----------------------------------------------------------\n",self.decoder)
        print("---------------------------------------------------------------------------------------------------------------------------")


    def forward(self,x):
        # Check the input dimensions
        batch_size, n_channels, len_signal = x.shape  # This decomposes the shape into 3 dimensions
        assert n_channels == self.encoder[0].in_channels, f"The expected number of channels is {self.encoder[0].in_channels}, but the input has {n_channels} channels."
        assert len_signal == self.encoder[0].kernel_size[0], f"The expected signal length is {self.encoder[0].kernel_size[0]}, but the input has {len_signal} values per signal."

        
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

def main():
    input = (1,12,1000)
    ECG_DAE(input=input,nb_layers=4)

if __name__ == '__main__'  :

    main()