# Imports python modules
import argparse

def train_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_directory',help='Path to the directory containing the data to be used as training and validation.')
    parser.add_argument('--save_dir',default='checkpoint.pth',help='Path to the directory containing the used model checkpoint.')
    parser.add_argument('--arch',default='vgg13',help='Name or alias of the CNN model architecture to be applied. Currently only resnet, densenet and vgg are supported')
    parser.add_argument('--optim',default='SGD',help='Name or alias of the optimizing method to be applied during optimization')
    parser.add_argument('--learning_rate',default=0.01, type=float ,help='Learning rate used for optimizing the training of the model')
    parser.add_argument('--hidden_nodes',default=[2048,512], type=int, nargs='*', help='A list of the number of nodes in each respective hidden layer.')
    parser.add_argument('--drop_rate',default=0.2, type=float ,help='Dropout rate used within the classifier network')
    parser.add_argument('--epochs',default=8, type=int ,help='Number of epochs over which your model is trained.')
    parser.add_argument('--gpu',action='store_true',help='Boolean to define whether or not to use the GPU for training the model.')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()