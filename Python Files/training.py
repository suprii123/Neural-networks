import numpy as np
from operations import *
import time
import testing


def main(variables, train, test, title):

    l_hiddenLayers, batch_size, lrate, epoch = variables[0], variables[1], variables[2], variables[3]
    train_inputs = train[0]
    train_outputs = train[1]

    # calculating time taken
    start = time.time()

    # initializing weights and biases
    l_weightMatrices = []
    l_biasMatrices = []
    l_weightMatrices.append(np.random.uniform(-1, 1, size=(train_inputs.shape[1], l_hiddenLayers[0])))
    l_biasMatrices.append(np.random.uniform(-1, 1, size=(1, l_hiddenLayers[0])))
    for i in range(len(l_hiddenLayers)-1):
        l_weightMatrices.append(np.random.uniform(-1, 1, size=(l_hiddenLayers[i], l_hiddenLayers[i+1])))
        l_biasMatrices.append(np.random.uniform(-1, 1, size=(1, l_hiddenLayers[i+1])))
    l_weightMatrices.append(np.random.uniform(-1, 1, size=(l_hiddenLayers[-1], train_outputs.shape[1])))
    l_biasMatrices.append(np.random.uniform(-1, 1, size=(1, train_outputs.shape[1])))

    # calculating derivate of cost functions w.r.t all the outputs
    def dMatrix(index):
        returnMatrix = np.zeros((batch_size, l_hiddenLayers[index]))
        for nrow in range(returnMatrix.shape[0]):
            d = l_dWithOutputs[0]
            d = d[nrow:nrow+1]
            a = l_outputLayers[index+1]
            a = a[nrow:nrow+1]
            w = l_weightMatrices[index+1]
            
            m = np.multiply(d_sigmoid(a), d)
            m = np.dot(m, w.T)
            returnMatrix[nrow] = m[0]
        return returnMatrix

    # main loop
    for _ in range(epoch):
        n_batches = train_inputs.shape[0]//batch_size
        if train_inputs.shape[0] % batch_size:
            n_batches += 1

        for i in range(n_batches):
            input_matrix = train_inputs[i*batch_size:(i+1)*batch_size]
            output_matrix = train_outputs[i*batch_size:(i+1)*batch_size]
            if input_matrix.shape[0] != batch_size:
                batch_size = input_matrix.shape[0]
            l_activationLayers = []
            l_outputLayers = []
            l_dWithOutputs = []

            # forward pass
            l_activationLayers.append(fpass(input_matrix, l_weightMatrices[0], l_biasMatrices[0]))
            l_outputLayers.append(sigmoid(l_activationLayers[0]))
            for j in range(len(l_hiddenLayers)):
                l_activationLayers.append(fpass(l_outputLayers[j], l_weightMatrices[j+1], l_biasMatrices[j+1]))
                l_outputLayers.append(sigmoid(l_activationLayers[j+1]))
            
            # calculating derivates 
            l_dWithOutputs.append(l_outputLayers[-1]-output_matrix)
            for j in range(len(l_hiddenLayers)-1, -1, -1):
                l_dWithOutputs.insert(0, dMatrix(j))
            
            # updating weights
            m1 = np.multiply(l_dWithOutputs[0], d_sigmoid(l_outputLayers[0]))
            wchange = np.dot(input_matrix.T, m1)
            wchange /= batch_size
            l_weightMatrices[0] -= lrate * wchange
            for j in range(len(l_hiddenLayers)):
                m1 = np.multiply(l_dWithOutputs[j+1], d_sigmoid(l_outputLayers[j+1]))
                wchange = np.dot(l_outputLayers[j].T, m1)
                wchange /= batch_size
                l_weightMatrices[j+1] -= lrate * wchange

            # updating biases
            for j in range(len(l_biasMatrices)):
                m1 = np.multiply(l_dWithOutputs[j], d_sigmoid(l_outputLayers[j]))
                bchange = np.sum(m1, axis=0)
                bchange = np.array([bchange])
                bchange /= batch_size
                l_biasMatrices[j] -= lrate * bchange

    end = time.time()
    time_taken_in_s = end - start


    # testing
    results =[title, l_hiddenLayers, batch_size, lrate, epoch, time_taken_in_s]
    model = (l_weightMatrices, l_biasMatrices)
    testing.test(model, test, results)