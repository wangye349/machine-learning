#this project is used to implement back propagation to
#tell machine to learning how to read the numbles. The ultimated
#goal is to get the accuracy up to 95%.

#so this project is done by some steps as the follows.
# 1.get the data
# 2.create a nerual network
# 3.compute the cost and the gradient
# 4.run random initialization
# 5.run forward propagation
# 6.run backward propagation
# 7.use gradient descent to get to the smallest point
# 8.use gradient check
# 9.predict the result
# 10.understand the picture's numble

import numpy as np
import scipy as scy
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import random
from numpy import *
import scipy.optimize as opt

def sigmoid(input):
    output = np.ones((np.size(input,0),np.size(input,1))) / (np.ones((np.size(input,0),np.size(input,1))) + np.exp(-input))
    return output

def random_numbles_initialization(height, width):
    random_matrix = []

    for i in range(height * width):
        random_matrix.append(random.random() + 0.1)

    random_matrix = np.array(random_matrix)
    random_matrix = random_matrix.reshape(height, width)
    return random_matrix

def change_y_into_10_label_size(input_y):
    output_y = np.zeros((np.size(input_y),10))
    for i in range(np.size(input_y)):
        output_y[i][input_y[i][0]-1] = 1
    return output_y

def compute_the_cost_function(Theta1,Theta2,input_x,output_y,hidden_layer_size,lambda_input):
    x_and_y_numbles = np.size(input_x,0)
    #print x_and_y_numbles
    x_labels = np.size(input_x,1)
    y_labels = np.size(output_y,1)
    #print x_labels, y_labels
    a1 = np.c_[np.ones(x_and_y_numbles),input_x]
    #print a1
    z2 = np.dot(a1,Theta1)
    a2_temp = sigmoid(z2)
    a2 = np.c_[np.ones(x_and_y_numbles),a2_temp]

    z3 = np.dot(a2,Theta2)
    a3 = sigmoid(z3)
    #print np.size(a3)
    x =  sum(np.multiply(output_y,np.log(a3)))
    y =  sum(np.multiply((np.ones((x_and_y_numbles,y_labels))-output_y),(np.log(np.ones((x_and_y_numbles,y_labels))-a3))))

    J = (x + y)/x_and_y_numbles

    delta_3 = a3 - output_y

    delta_2 = np.multiply(sigmoid_gradient(z2),(np.dot(delta_3,Theta2.T))[:,1:])
    #delta_2 = delta_2_temp[:,1:]

    Theta1_grad = np.dot(a1.T,delta_2)/x_and_y_numbles
    Theta2_grad = np.dot(a2.T,delta_3)/x_and_y_numbles

    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (lambda_input/x_and_y_numbles * Theta1)[:,1:]
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (lambda_input/x_and_y_numbles * Theta2)[:,1:]
    # J = -(1/x_and_y_numbles) * sum(np.multiply(output_y,np.log(a3)) + \
    #     np.multiply((np.ones((x_and_y_numbles,y_labels))-output_y),(np.log(np.ones((x_and_y_numbles,y_labels))-a3))))\
    #     +(lambda_input/(2*x_and_y_numbles)) * (sum((np.multiply(Theta2[:,1:],Theta2[:,1:])) + sum(np.multiply(Theta1[:,1:],Theta1[:,1:]))))

    return J, Theta1_grad, Theta2_grad

def sigmoid_gradient(z):
        g_z = sigmoid(z)
        return np.multiply(g_z,(np.ones((np.size(g_z,0),np.size(g_z,1)))-g_z))

def gradient_descent(Theta1,Theta2,Theta1_gradient,Theta2_gradient,alpha):
    for i in range(50):
        Theta1_temp = Theta1 - alpha * Theta1_gradient
        Theta2 = Theta2 - alpha * Theta2_gradient
        Theta1 = Theta1_temp
    return Theta1,Theta2

def convert_to_numble(input):
    for i in range(np.size(input)):
        if(input[i] > 0.5):
            output = i
            break
    return output

def backpropagation_main_function(a):
    data = sio.loadmat('ex4data1.mat')
    Y = data['y']
    y = change_y_into_10_label_size(Y)
    X = data['X']
    x_and_y_numbles = np.size(X,0)
    x_labels = np.size(X,1)
    y_labels = np.size(y,1)
    hidden_layer_size = 25
    Theta = sio.loadmat('ex4weights.mat')
    Theta1_initialization = Theta['Theta1'].T
    Theta2_initialization = Theta['Theta2'].T

    theta = sio.loadmat('savedata.mat')
    Theta1 = theta['Theta1']
    Theta2 = theta['Theta2']


    a1 = np.c_[np.ones(x_and_y_numbles),X]
    #print a1
    z2 = np.dot(a1,Theta1)
    a2_temp = sigmoid(z2)
    a2 = np.c_[np.ones(x_and_y_numbles),a2_temp]

    z3 = np.dot(a2,Theta2)
    a3 = sigmoid(z3)

    a3_new = []
    for i in range(np.size(a3,0)):
        a3_new.append(convert_to_numble(a3[i]))

    a3_new_new = np.array(a3_new).T
    print a3_new_new
    print Y
    count = 0
    for i in range(5000):
        if(a3_new_new[i] == Y[i][0]-1):
            count = count + 1
            #print count
    print count/5000


    #Theta1_initialization = random_numbles_initialization(x_labels+1,hidden_layer_size)
    #Theta2_initialization = random_numbles_initialization(hidden_layer_size+1,y_labels)
    #compute_the_cost_function(Theta1,Theta2,input_x,output_y,hidden_layer_size,lambda_input):
    # lambda_input = 0
    # (J_cost_function, Theta1_gradient, Theta2_gradient) = compute_the_cost_function\
    # (Theta1_initialization,Theta2_initialization,X,y,hidden_layer_size,lambda_input)
    #
    # alpha = 3.2
    # J_cost_function_collection = []
    # Theta1 = Theta1_initialization
    # Theta2 = Theta2_initialization
    # for i in range(8000):
    #     Theta1_temp = Theta1 - alpha * Theta1_gradient
    #     Theta2 = Theta2 - alpha * Theta2_gradient
    #     Theta1 = Theta1_temp
    #     (J_cost_function, Theta1_gradient, Theta2_gradient) = compute_the_cost_function\
    #     (Theta1,Theta2,X,y,hidden_layer_size,lambda_input)
    #     J_cost_function_collection.append(J_cost_function)
    #
    # x_labels_temp = []
    # for i in range(8000):
    #     x_labels_temp.append(i)
    # plt.plot(x_labels_temp,J_cost_function_collection,linewidth = 1)
    # plt.show()
    # sio.savemat('savedata.mat',{"Theta1":Theta1,"Theta2":Theta2})


# def function_this(z):
#     (theta1,theta2) = z
#
#     data = sio.loadmat('ex4data1.mat')
#     Y = data['y']
#     y = change_y_into_10_label_size(Y)
#     X = data['X']
#     x_and_y_numbles = np.size(X,0)
#     x_labels = np.size(X,1)
#     y_labels = np.size(y,1)
#     hidden_layer_size = 25
#     lambda_input = 0
#
#     (fun,a,b) = compute_the_cost_function\
#     (theta1,theta2,X,y,hidden_layer_size,lambda_input)
#
#     return fun
#
# def function_that(z):
#     (theta1,theta2) = z
#
#     data = sio.loadmat('ex4data1.mat')
#     Y = data['y']
#     y = change_y_into_10_label_size(Y)
#     X = data['X']
#     x_and_y_numbles = np.size(X,0)
#     x_labels = np.size(X,1)
#     y_labels = np.size(y,1)
#     hidden_layer_size = 25
#     lambda_input = 0
#
#     (fun,a,b) = compute_the_cost_function\
#     (theta1,theta2,X,y,hidden_layer_size,lambda_input)
#
#     return (a,b)
#
# points = []
#
# def function_i_dont_know(k):
#
#     Theta = sio.loadmat('ex4weights.mat')
#     Theta1_initialization = Theta['Theta1'].T
#     Theta2_initialization = Theta['Theta2'].T
#
#     init_point =(Theta1_initialization,Theta2_initialization)
#
#     result = opt.fmin_cg(function_this,init_point,function_that)
#     print result
