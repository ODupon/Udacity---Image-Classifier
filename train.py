# Imports here
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from train_input_args import train_input_args
import Classifier_model
import argparse

def check_arch(arch):
    # This function checks the used achitecture and returns the corresponding input size of the classifier
    # To add another architecture, just add anoter elif conditional and make sure it return the correct input size
    
    if arch.startswith('vgg') == True:
        return 25088
    
    elif arch.startswith('densenet') == True:
        return 2208
        
    elif arch.startswith('resnet') == True:
        return 512

# Main program function defined below
def main():
    args = train_input_args()
    # Function that checks command line arguments using in_arg  
    # check_command_line_arguments(in_arg)

    train_dir = args.data_directory + '/train'
    valid_dir = args.data_directory + '/valid'

    # The transforms for the training, validation, and testing sets
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    image_train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)

    # Define the dataloaders
    valid_dataloaders = torch.utils.data.DataLoader(image_valid_datasets, batch_size=64)
    train_dataloaders = torch.utils.data.DataLoader(image_train_datasets, batch_size=64, shuffle=True)
    
    # Check whether or not the device runs on the GPU or not
    
    device = torch.device('cuda' if (torch.cuda.is_available() and (args.gpu==True)) else 'cpu')
    # Load in a pre-trained model and freeze their parameters. In this case we used the VGG-19 model.
    
    model = getattr(models, args.arch)(pretrained=True)
    input_size = check_arch(args.arch)

    for params in model.parameters():
        params.requires_grad = False

    classifier = Classifier_model.Network(input_size, len(image_train_datasets.class_to_idx), args.hidden_nodes, drop_p=args.drop_rate)
    
    # Check which model architecture is used and assign the correct corresponding attribute
    # Currently only resnet, densenet and vgg are supported
    if args.arch.startswith('resnet') == True:
        model.fc = classifier
        model_classifier = model.fc
    else:
        model.classifier = classifier
        model_classifier = model.classifier
    
    criterion = nn.NLLLoss()
    optimizer = getattr(optim, args.optim)(model_classifier.parameters(), lr=args.learning_rate)
    
    model.to(device)
    
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 103
    train_losses, val_losses = [], []
    
    print("start")
    for e in range(epochs):
        epoch_prog = 0
        for images,labels in train_dataloaders:
            steps += 1
            epoch_prog += 1
            
            # Move the tensors to the proper device
            images, labels = images.to(device), labels.to(device)

            # Clear the gradients such that they don't accumulate
            optimizer.zero_grad()

            # Run the images through the model
            output = model.forward(images)

            # Calculate the loss and backpropagate it
            loss = criterion(output,labels)
            loss.backward()

            # Take an update step and readjust the weights
            optimizer.step()
            print("Epoch training progress: {}/{}".format(epoch_prog,len(train_dataloaders)))
            running_loss += loss.item()
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                epoch_test_prog = 0
                model.eval()

                # Turn of gradients to save memory and computation
                with torch.no_grad():
                    for images,labels in valid_dataloaders:
                        epoch_test_prog += 1
                        
                        # Move the tensors to the proper device
                        images, labels = images.to(device), labels.to(device)
                        
                        # Run the validation images through the model and calculate their loss
                        output = model(images)
                        batch_loss = criterion(output,labels)
                        val_loss += batch_loss

                        # Calculate the accuracy of the classification
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                        
                        print("Epoch validation progress: {}/{}".format(epoch_test_prog,len(valid_dataloaders)))

                train_losses.append(running_loss/len(train_dataloaders))
                val_losses.append(val_loss/len(valid_dataloaders))

                print("Epoch: {}/{}".format(e+1,epochs),
                      "Training Loss: {:.3f}".format(running_loss/len(train_dataloaders)),
                      "Validation Loss: {:.3f}".format(val_loss/len(valid_dataloaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_dataloaders)))
                running_loss = 0
                model.train()
    
    # Save the checkpoint 
    model.class_to_idx = image_train_datasets.class_to_idx
    checkpoint = {'input_size': input_size,
                  'output_size': 102,
                  'hidden_layers': args.hidden_nodes,
                  'class_to_idx': model.class_to_idx,
                  'model': args.arch,
                  'epochs': args.epochs,
                  'optimizer_dict':optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    
    
    torch.save(checkpoint, args.save_dir)    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()