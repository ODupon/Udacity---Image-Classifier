# Imports here
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from predict_input_args import predict_input_args
import argparse
from load_checkpoint import load_checkpoint
import json

def predict(model, image_path, modelpath, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load in the image from its path and process it
    im = Image.open(image_path)
    im = process_image(im)

    # Load in the model from its path
    
    # Set the device the model works on
    
    model.to(device)

    # Calculate the class probabilities for im
    model.eval()
    with torch.no_grad():
        input_image = np.expand_dims(im, 0)
        i_img = torch.from_numpy(input_image)
        
        i_img = i_img.to(device)
        output = model.forward(i_img)

    ps = torch.exp(output)

    top_p, top_class = ps.topk(topk,dim=1)

    top_p = top_p.type(torch.FloatTensor)
    top_class = top_class.type(torch.FloatTensor)
    
    top_p = top_p.numpy()[0]
    top_class = top_class.numpy()[0]
    
    idx_to_class = { v : k for k,v in model.class_to_idx.items()}
    
    for i in range(len(top_class)):
        top_class[i] = idx_to_class[(int(top_class[i]))]

    return top_p, top_class

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Resize to 256
    image = image.resize((256,256))
    
    # Crop sizes: L,B = 128-112; R,T = 128+112
    image = image.crop((16,16,240,240))
    
    # Array to floats
    image = np.array(image)/255

    # Normalise image
    mu = np.array([0.485, 0.456, 0.406])
    sigma = np.array([0.229, 0.224, 0.225])
    image = (image - mu)/sigma

    # Transpose color channel to first dimension
    image = image.transpose((2, 0, 1))

    return torch.FloatTensor(image)

# Main program function defined below
def main():
    args = predict_input_args()
    # Function that checks command line arguments using in_arg  
    # check_command_line_arguments(in_arg)
    checkpoint = torch.load(args.checkpoint, map_location=('cuda' if (torch.cuda.is_available() and (args.gpu==True)) else 'cpu'))
    model = load_checkpoint(checkpoint)
    # utils = predict_utility.predict_utils(model)
    # image_path = 'flowers/test/20/image_04910.jpg'
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    device = torch.device('cuda' if (torch.cuda.is_available() and (args.gpu==True)) else 'cpu')
    
    probs, classes = predict(model, args.input, args.checkpoint, device, args.top_k)
    
    c = []
    for x in classes:
        c.append(cat_to_name[str(int(x))])
    
    print('Top {} classes and their associated probabilities'.format(args.top_k))
    for i in range(len(classes)):
        print('Class : {}, probability: {:.3f}'.format(c[i],probs[i]))
    
    
# Call to main function to run the program

if __name__ == "__main__":
    main()