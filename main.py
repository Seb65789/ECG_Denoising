import argparse as parse

def main():

    # Create arguments
    arguments = parse.ArgumentParser() # Creating the arguments

    arguments.add_argument('-epochs',default=20,type=int) # How many epochs

    arguments.add_argument("-lr",default=0.001,type=float)

    arguments.add_argument("-optimizer",choices=['sgd','adam'],type=str)

    arguments.add_argument("-activation",choices=['tanh','relu'],type=str,default='tanh')

    arguments.add_argument("-pooling",choices=['avg','max'],type=str,default='tanh')
    
    arguments.add_argument("-layers",default=2,type=int)
    
    arguments.add_argument("-batch_size",default=64,type=int)

    opt = arguments.parse_args()

if __name__ == 'main':
    main()