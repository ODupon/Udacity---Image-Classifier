from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
import Classifier_model

def load_checkpoint(checkpoint):
    model = getattr(models, checkpoint['model'])(pretrained=True)
    
    for params in model.parameters():
        params.requires_grad = False

    
    classifier = Classifier_model.Network(checkpoint['input_size'],checkpoint['output_size'],checkpoint['hidden_layers'])
    
    # Check which model architecture is used and assign the correct corresponding attribute
    # Currently only resnet, densenet and vgg are supported
    if checkpoint['model'].startswith('resnet') == True:
        model.fc = classifier
    else:
        model.classifier = classifier

    # grab map of labels to indicies dictionary
    model.class_to_idx = checkpoint['class_to_idx']

    model.load_state_dict(checkpoint['state_dict'])

    return model