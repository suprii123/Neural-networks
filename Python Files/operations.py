import numpy as np
import os

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def fpass(inputs, weights, biases):
    return np.dot(inputs, weights) + biases

def d_sigmoid(x): # derivative of sigmoid
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def save_results(results):
    os.chdir(r'C:\Users\samar\OneDrive\Desktop\Neural Networks\Results')
    with open(results[0] + '.txt', 'a') as file:
        line = f'Hidden Layers: {results[1]} | Batch Size: {results[2]} | Learning Rate: {results[3]} | Epoch: {results[4]} | Time Taken (in s): {results[5]} | Accuracy: {results[6]}\n\n'
        file.write(line)