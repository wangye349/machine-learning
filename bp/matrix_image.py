from PIL import Image
#import numpy as np
# import scipy
import matplotlib.pyplot as plt
import numpy as np
import scipy as scy
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import random
#from numpy import *
import scipy.optimize as opt

def sigmoid(input):
    #output = np.ones((np.size(input,0),np.size(input,1))) / (np.ones((np.size(input,0),np.size(input,1))) + np.exp(-input))
    output = []
    for i in range(np.size(input)):
        output.append(1/(1+np.exp(-input[i])))
    return output

def convert_to_numble(input):
    for i in range(np.size(input)):
        if(input[i] > 0.5):
            output = i
            break
    return output

def ImageToMatrix(filename):
    im = Image.open(filename)
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    new_data = np.reshape(data, (height, width))
    return new_data

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

filename = '25.bmp'
data = ImageToMatrix(filename)
print type(data)
data = data.tolist()
new_data = [1]
for i in range(20):
    for j in range(20):
        # print i,j
        # print data[i][j]
        new_data.append(data[i][j])

theta = sio.loadmat('savedata.mat')
Theta1 = theta['Theta1']
Theta2 = theta['Theta2']

a2_temp = sigmoid(np.dot(Theta1.T,new_data))
a2 = [1]

for i in range(np.size(a2_temp)):
    a2.append(a2_temp[i])

a3 = sigmoid(np.dot(Theta2.T,a2))

print a3
print convert_to_numble(a3)+1


# new_im = MatrixToImage(data)
# plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
# new_im.show()
# new_im.save('lena_1.bmp')
