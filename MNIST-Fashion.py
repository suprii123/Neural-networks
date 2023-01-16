import sys
sys.path.append(r'.\Python Files')
import numpy as np
from mnist import MNIST
import training


# defining variables
l_hiddenLayers = [784]
batch_size = 1
lrate = 0.1
epoch = 2

title = 'MNIST-Fashion'

# importing database
mndata = MNIST(r'.\Datasets\MNIST-Fashion')
x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()

# converting training database to usable format
temp = y_train
y_train = []
for i in temp:
    temp_list = []
    for j in range(10):
        if i == j:
            temp_list.append(1)
        else:
            temp_list.append(0)
    y_train.append(temp_list)

train_inputs = np.array(x_train)
train_inputs = np.float_(train_inputs)
train_inputs /= 255
train_outputs = np.array(y_train)


# converting testing database to usable format
temp = y_test
y_test = []
for i in temp:
    temp_list = []
    for j in range(10):
        if i == j:
            temp_list.append(1)
        else:
            temp_list.append(0)
    y_test.append(temp_list)

test_inputs = np.array(x_test)
test_inputs = np.float_(test_inputs)
test_inputs /= 255
test_outputs = np.array(y_test)

variables = (l_hiddenLayers, batch_size, lrate, epoch)
train, test = (train_inputs, train_outputs), (test_inputs, test_outputs)
training.main(variables, train, test, title)