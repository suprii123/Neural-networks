from operations import *


def test(model, test, results):
    l_weightMatrices = model[0]
    l_biasMatrices = model[1]
    test_inputs = test[0]
    test_outputs = test[1]

    correct = 0
    for i in range(test_inputs.shape[0]):
        a = fpass(test_inputs[i:i+1], l_weightMatrices[0], l_biasMatrices[0])
        a = sigmoid(a)
        for j in range(len(l_weightMatrices)-1):
            a = fpass(a, l_weightMatrices[j+1], l_biasMatrices[j+1])
            a = sigmoid(a)
        
        if list(a[0]).index(max(list(a[0]))) == list(test_outputs[i]).index(max(list(test_outputs[i]))):
            correct += 1
    
    accuracy = '{:.5f} %'.format((correct/test_inputs.shape[0])*100)
    print(accuracy)
    results.append(accuracy)
    save_results(results)