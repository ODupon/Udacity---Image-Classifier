# Imports python modules
import argparse

def predict_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input',help='Path to the directory containing the image to be predicted.')
    parser.add_argument('checkpoint',help='Path to the directory containing the used model checkpoint.')
    parser.add_argument('--category_names',default='cat_to_name.json', type=str,help='Path to the directory containing the dictionary for mapping the categories to real names.')
    parser.add_argument('--top_k',default=5, type=int ,help='Number of top class prediction that are provided.')
    parser.add_argument('--gpu',action='store_true',help='Boolean to define whether or not to use the GPU for training the model.')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()