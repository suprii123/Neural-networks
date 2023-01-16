import sys
sys.path.append(r'.\Python Files')
import numpy as np
import training


# defining variables
l_hiddenLayers = [96]
batch_size = 1
lrate = 0.01
epoch = 10000

title = 'XOR'

inputs = [[0,0], [0,1], [1,0], [1,1]]
outputs = [[1,0], [0,1], [0,1], [1,0]]

train_inputs = np.array(inputs)
train_outputs = np.array(outputs)

test_inputs = train_inputs
test_outputs = train_outputs

variables = (l_hiddenLayers, batch_size, lrate, epoch)
train, test = (train_inputs, train_outputs), (test_inputs, test_outputs)
training.main(variables, train, test, title)