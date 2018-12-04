'''
Python version - 3.6
'''
import numpy as np
from load_mnist import fashion_mnist
import matplotlib.pyplot as plt
import pdb
import random

def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):

    Z = cache["Z"]
    A, cache = sigmoid(Z)
    dZ = dA * A * (1 - A)

    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):

    W1 = np.random.randn(n_h, n_in) * 0.01
    b1 = np.random.randn(n_h, 1) * 0.01
    W2 = np.random.randn(n_fin, n_h) * 0.01
    b2 = np.random.randn(n_fin, 1) * 0.01

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters

def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):

    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache

def cost_estimate(A2, Y):
    m = Y.shape[1]
    cost = -1 * (1/m) * ( np.sum( np.multiply(np.log(A2),Y) ) + np.sum( np.multiply(np.log(1-A2),(1-Y)) ) )

    return cost

def linear_backward(dZ, cache, W, b):

    dW = np.dot(dZ, cache["A"].T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):

    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def salt_and_pepper_noise(image,prob):
    output = np.copy(image)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 1
    return output

def softmax_cross_entropy_loss(Z, Y=np.array([])):

    maximumZ = np.max(Z, axis = 0, keepdims = True)
    e = np.exp(Z - maximumZ)
    A = e / np.sum(e, axis = 0, keepdims = True)
    cache = {}
    cache["A"] = A
    
    loss = 0
    for i in range(Y.shape[1]):
        x = int(Y[0][i])
        loss += np.log(A[x, i])

    loss = -loss/Y.shape[1]
    return A,cache,loss

def softmax_cross_entropy_loss_der(Y, cache):
    n,m = Y.shape
    dZ = cache['A'].copy()
    for index, x in np.ndenumerate(Y):
        dZ[int(x), index[1]] -= 1
    return dZ/m

def two_layer_network(X, Y, test_X, test_Y, net_dims, num_iterations=2000, learning_rate=0.1):

    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    
    A0 = X
    train_costs = []
    validation_costs = []
    test_costs = []
    final_data = X
    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE
        A1, cache1 = layer_forward(A0, parameters["W1"], parameters["b1"], "sigmoid")
        A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")

        # cost estimation
        train_cost = cost_estimate(A2, X)

        # test_A0 = test_X
        # test_A1, test_cache1 = layer_forward(test_A0, parameters["W1"], parameters["b1"], "sigmoid")
        # test_A2, test_cache2 = layer_forward(test_A1, parameters["W2"], parameters["b2"], "sigmoid")
        # test_cost = cost_estimate(test_A2, test_Y)

        m = Y.shape[1]
        
        # Backward Propagation
        ### CODE HERE
        dA2 = (1.0/m) * (np.divide(-1*X, A2) + np.divide(1-X, 1-A2))
        dA1, dW2, db2 = layer_backward(dA2, cache2, parameters["W2"], parameters["b2"], "sigmoid")
        dA0, dW1, db1 = layer_backward(dA1, cache1, parameters["W1"], parameters["b1"], "sigmoid")
        # dZ2 = sigmoid_der(dA2, cache2)
        # dW2 = np.dot(dZ2, A1.T)
        # db2 = np.sum(dZ2, axis=1, keepdims=True)

        #update parameters
        ### CODE HERE
        parameters["W2"] = parameters["W2"] - learning_rate * dW2 
        parameters["b2"] = parameters["b2"] - learning_rate * db2 
        parameters["W1"] = parameters["W1"] - learning_rate * dW1
        parameters["b1"] = parameters["b1"] - learning_rate * db1 

        if ii % 10 == 0:
            train_costs.append(train_cost)
        if ii % 10 == 0:
            print ("Train Cost at iteration %i is: %f" %(ii, train_cost))
            # print(cache1)
            # print(cache2)
        final_data = A2
    
    return train_costs, test_costs, validation_costs, parameters, final_data

def relu(Z):
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    dZ = np.array(dA, copy=True)
    n,m = dZ.shape
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):

    maximumZ = np.max(Z, axis = 0, keepdims = True)
    e = np.exp(Z - maximumZ)
    A = e / np.sum(e, axis = 0, keepdims = True)
    cache = {}
    cache["A"] = A
    
    loss = 0
    for i in range(Y.shape[1]):
        x = int(Y[0][i])
        loss += np.log(A[x, i])

    loss = -loss/Y.shape[1]
    return A,cache,loss

def softmax_cross_entropy_loss_der(Y, cache):
    n,m = Y.shape
    dZ = cache['A'].copy()
    for index, x in np.ndenumerate(Y):
        dZ[int(x), index[1]] -= 1
    return dZ/m

def linear_forward_mult(A, W, b):
    cache = {}
    cache["A"] = A
    Z = np.dot(W, A) + b
    return Z, cache

def layer_forward_mult(A_prev, W, b, activation):

    Z, lin_cache = linear_forward_mult(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    L = len(parameters)//2  
    A = X
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward_mult(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward_mult(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward_mult(dZ, cache, W, b):
  
    A_prev = cache["A"]
    ## CODE HERE
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    return dA_prev, dW, db

def layer_backward_mult(dA, cache, W, b, activation):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward_mult(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    L = len(caches) 
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward_mult(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients

def classify(X, parameters):
    A1, cache = multi_layer_forward(X, parameters)
    Ypred = []
    Z=A1
    iter1 = Z.shape[1]
    Z = Z.T
    e_z = Z
    for i in range(iter1):
        e_z[i] = np.exp(Z[i] - np.max(Z[i]))
        e_z[i] = e_z[i]/np.sum(e_z[i])

    A = e_z
    for i in range(A.shape[0]):
        #print(A2[i])
        Ypred.append(np.argmax(A[i]))
    return Ypred
    AL, _ = multi_layer_forward(X, parameters)
    A, _, _ = softmax_cross_entropy_loss(AL)
    labels = np.argmax(A, axis=0)
    Ypred = labels.reshape(1, len(labels))
    return Ypred

def update_parameters(parameters, gradients, epoch, learning_rate1, learning_rate2):
    L = len(parameters)//2
    for l in range(1,L+1):
        if l == L:
            parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate2 * gradients["dW"+str(l)]
            parameters["b"+str(l)] = parameters["b"+str(l)] - learning_rate2 * gradients["db"+str(l)]
        else:
            parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate1 * gradients["dW"+str(l)]
            parameters["b"+str(l)] = parameters["b"+str(l)] - learning_rate1 * gradients["db"+str(l)]

    ### CODE HERE 
    return parameters

def multi_layer_network(parameters, X, Y, net_dims, num_iterations=500, learning_rate=0.5, decay_rate=0.0):
    A0 = X
    costs = []
    alpha = learning_rate
    for ii in range(num_iterations):
        ### CODE HERE
        # Forward Prop 
        ## call to multi_layer_forward to get activations
        A1, cache = multi_layer_forward(A0, parameters)
        ## call to softmax cross entropy loss
        A2, cache1, cost = softmax_cross_entropy_loss(A1, Y)

        ## call to softmax cross entropy loss der
        dZ = softmax_cross_entropy_loss_der(Y, cache1)
        gradients = multi_layer_backward(dZ, cache, parameters)
        parameters = update_parameters(parameters, gradients, ii, learning_rate, 0.1)
        if ii % 10 == 0:
            costs.append(cost)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, learning_rate))
    
    return costs, parameters
def accuracy(train_Pred,train_label):
    count_train_errors = 0
    for i in range(len(train_Pred)):
        # print(train_Pred[0][i])
        # print(train_label[0][i])
        if train_Pred[i] != train_label[0][i] :
            count_train_errors = count_train_errors + 1
    
    #print(count_train_errors)
    trAcc = (len(train_Pred) - count_train_errors) * 100/ len(train_Pred)
    return trAcc

def main():
    # getting the subset dataset from Fashion MNIST
    class_range = [0,1,2,3,4,5,6,7,8,9]
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_data, train_label, test_data, test_label = \
            fashion_mnist(noTrSamples=5000,noTsSamples=600,\
            class_range=class_range,\
            noTrPerClass=500, noTsPerClass=60)

    print("train data", train_data.shape)
    print("test data",test_data.shape)
    print(train_label)

    plt.figure(figsize=(10,10))

    n_in, m = train_data.shape
    n_fin = 784
    n_h = 300


    net_dims = [n_in, n_h, n_fin]
    print(train_data.shape)
    num_iterations = 1000
    learning_rate = 0.01
    train_costs, val_costs, test_costs, parameters1, final_data = two_layer_network(train_data, train_label, test_data, test_label, net_dims, num_iterations=num_iterations, learning_rate=learning_rate)

    print("train costs = ", train_costs)

    
    layer1_w1 = parameters1["W1"]
    layer1_b1 = parameters1["b1"]
    print(layer1_w1.shape)

    net_dims = [layer1_w1.shape[0], 100, layer1_w1.shape[0]]
    num_iterations = 1000
    learning_rate = 0.005
    train_costs, val_costs, test_costs, parameters2, final_data = two_layer_network(layer1_w1, train_label, test_data, test_label, net_dims, num_iterations=num_iterations, learning_rate=learning_rate)

    layer2_w1 = parameters2["W1"]
    layer2_b1 = parameters2["b1"]
    print(layer2_w1.shape)

    print("train costs = ", train_costs)


    # net_dims = [layer2_w1.shape[0], 100, layer2_w1.shape[0]]
    # num_iterations = 500
    # learning_rate = 0.02
    # train_costs, val_costs, test_costs, parameters3, final_data = two_layer_network(layer2_w1, train_label, test_data, test_label, net_dims, num_iterations=num_iterations, learning_rate=learning_rate)

    # layer3_w1 = parameters3["W1"]
    # layer3_b1 = parameters3["b1"]
    # print(layer3_w1.shape)
    # print("train costs = ", train_costs)

    final_parameters = {}
    final_parameters["W1"] = layer1_w1
    final_parameters["b1"] = layer1_b1
    final_parameters["W2"] = layer2_w1
    final_parameters["b2"] = layer2_b1
    # final_parameters["W3"] = layer3_w1
    # final_parameters["b3"] = layer3_b1

    # final_parameters["W3"] = np.random.randn(100, layer2_w1.shape[0]) * 0.01
    # final_parameters["b3"] = np.random.randn(100, 1) * 0.01

    print("X shape ", train_data.shape)
    print("W1 shape ", final_parameters["W1"].shape)
    print("b1 shape ", final_parameters["b1"].shape)
    # print("W2 shape ", final_parameters["W2"].shape)
    # print("b2 shape ", final_parameters["b2"].shape)
    # print("W3 shape ", final_parameters["W3"].shape)
    # print("b3 shape ", final_parameters["b3"].shape)

    final_net_dims = [784, 100]
    final_net_dims.append(10) # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(final_net_dims))

    finetune_train_data, finetune_train_label, test_data, test_label = \
            fashion_mnist(noTrSamples=1000,noTsSamples=600,\
            class_range=class_range,\
            noTrPerClass=100, noTsPerClass=60)

    num_iterations = 2000
    costs, final_params = multi_layer_network(final_parameters, finetune_train_data, finetune_train_label, final_net_dims, \
            num_iterations=num_iterations, learning_rate=0.00001)

    print("Fine tune costs = ",costs)

    test_Pred = classify(test_data, final_params)
    train_Pred = classify(train_data, final_params)
    print("Accuracy for training set is {0:0.3f} %".format(accuracy(train_Pred,train_label)))
    print("Accuracy for test set is {0:0.3f} %".format(accuracy(test_Pred,test_label)))


if __name__ == "__main__":
    main()