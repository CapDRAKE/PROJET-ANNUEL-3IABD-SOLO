import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit # numerically stable sigmoid function
import VisualizeNN as VisNN
from celluloid import Camera # getting the camera
from IPython.display import HTML
from time import localtime, strftime



# Activation functions
def linear(x):
    return x

def linear_der(x):
    return 1

def tanh(x):
    return np.tanh(x)

def tanh_der(x):
    return 1-np.tanh(x)**2


def sigmoid(x):
    return expit(x)

def sigmoid_der(x):
    u = sigmoid(x)
    return u*(1-u)

def relu(x):
    return x*(x>0)

def relu_der(x):
    return (x>0)

# Weight initialization
def kaiming(network_config, l):
    return np.random.normal(size=(network_config[l+1], network_config[l])) * np.sqrt(2./network_config[l])

# Multilayer Perceptron Class
class NeuralNetwork(object):

    def __init__(self, network_config,Classifier=True,activations=None):

        if activations is None: activations = (len(network_config)-1)*("relu",)+(  "sigmoid" if Classifier else "relu",   ) 
        self.activations = activations
        self.Classifier = Classifier
        self.n_layers = len(network_config)

        # Weights
        self.W = [kaiming(network_config, l) for l in range(self.n_layers-1)]
        # Bias
        self.b = [np.zeros((network_config[l], 1)) for l in range(1, self.n_layers)]

        # Pre-activation
        self.z = [None for l in range(1, self.n_layers)]
        # Activations
        self.a = [None for l in range(self.n_layers)]
        # Gradients
        self.dW = [None for l in range(self.n_layers-1)] 
        self.db = [None for l in range(1, self.n_layers)]

    def grouped_rand_idx(self, n_total, batch_size):
        idx = np.random.permutation(n_total)
        return [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]
    

    def train(self, x_train, y_train, x_valid, y_valid, x_test, y_test, epochs, batch_size, learning_rate):

        eta = learning_rate / batch_size
        visualize_each = int(epochs/30)
        valide_each = int(epochs/50)
        for epoch in range(epochs):

            if epoch % valide_each == 0:
                self.prediction(x_valid, y_valid, epoch, mode="valid")

            if epoch % visualize_each == 0:
                self.visualize_weights(epoch)

            idx_list = self.grouped_rand_idx(len(x_train), batch_size)
            for idx in idx_list:
                # Get batch of random training samples
                x_batch, y_batch = x_train[idx], y_train[idx]
                self.feedforward(x_batch) 
                self.backprop_gradient_descent(y_batch, eta)

        self.visualize_weights(epoch+1)
        self.prediction(x_valid, y_valid, epoch+1, mode="valid")
        # Compute test accuracy and loss
        self.prediction(x_test, y_test, epoch+1, mode="test")

    def backprop_gradient_descent(self, Y, eta):
        # Backpropagation
        if self.activations[-1]=="sigmoid":
            delta = (self.a[-1] - Y) * sigmoid_der(self.z[self.n_layers-2]) 
        elif self.activations[-1]=="linear":
            delta = (self.a[-1] - Y) * linear_der(self.z[self.n_layers-2]) 
        elif self.activations[-1]=="tanh":
            delta = (self.a[-1] - Y) * tanh_der(self.z[self.n_layers-2]) 
        else:
            delta = (self.a[-1] - Y) * relu_der(self.z[self.n_layers-2])
            
        self.dW[self.n_layers-2] = np.matmul(delta.T, self.a[self.n_layers-2])
        self.db[self.n_layers-2] = np.sum(delta.T, axis=1, keepdims=True)

        for l in reversed(range(self.n_layers-2)):
            if self.activations[l]=="sigmoid":
                delta = np.matmul(delta, self.W[l+1]) * sigmoid_der(self.z[l])
            elif self.activations[l]=="linear":
                delta = np.matmul(delta, self.W[l+1]) * linear_der(self.z[l])
            elif self.activations[l]=="tanh":
                delta = np.matmul(delta, self.W[l+1]) * tanh_der(self.z[l])
            else:
                delta = np.matmul(delta, self.W[l+1]) * relu_der(self.z[l])
            self.dW[l] = np.matmul(self.a[l].T, delta).T
            self.db[l] = np.sum(delta.T, axis=1, keepdims=True)

        # Gradient descent: Update Weights and Biases
        for l in range(self.n_layers-1):
            self.W[l] -= eta * self.dW[l]
            self.b[l] -= eta * self.db[l]

        # Reset gradients
        self.dW = [None for l in range(self.n_layers-1)]
        self.db = [None for l in range(self.n_layers-1)]

    def feedforward(self, X):
        self.a[0] = X 
        for l in range(self.n_layers-1):
            self.z[l] = np.matmul(self.a[l], self.W[l].T) + self.b[l].T     # Pre-activation hidden layer
            if self.activations[l]=="sigmoid":
                self.a[l+1] = sigmoid(self.z[l])
            elif self.activations[l]=="linear":
                self.a[l+1] = linear(self.z[l])
            elif self.activations[l]=="tanh":
                self.a[l+1] = tanh(self.z[l])
            else:
                self.a[l+1] = relu(self.z[l])
        # Activation hidden layer
    
    def pred(self, X, Y):
        neurons = X
        for l in range(self.n_layers-1):
            if self.activations[l]=="sigmoid":
                neurons = sigmoid(np.matmul(neurons, self.W[l].T) + self.b[l].T)
            elif self.activations[l]=="linear":
                neurons = linear(np.matmul(neurons, self.W[l].T) + self.b[l].T)
            elif self.activations[l]=="tanh":
                neurons = tanh(np.matmul(neurons, self.W[l].T) + self.b[l].T)
            else:
                neurons = relu(np.matmul(neurons, self.W[l].T) + self.b[l].T)
        
        logits =neurons
        
        if self.activations[-1]=="sigmoid":
            output =logits
            if self.Classifier:
                accuracy = (np.argmax(logits, axis=1) == np.argmax(Y, axis=1)).sum() / len(X)
            else:
                accuracy = (np.argmax(logits, axis=1) == np.argmax(Y, axis=1)).sum() / len(X)
            loss = np.sum((Y - sigmoid(logits))**2) / len(X)
        else:
            output =logits
            accuracy = 0
            loss = np.sum((Y - logits)**2) / len(X)
        return output,loss, accuracy

    def prediction(self, X, Y, epoch, mode):
        output,loss, accuracy = self.pred(X, Y)
        print('epoch {1} {0}_loss {2:.6f} {0}_accuracy {3:.4f}'.format(mode, epoch, loss, accuracy), flush=True)

    def predict(self, X):
        neurons = X
        for l in range(self.n_layers-1):
            if self.activations[l]=="sigmoid":
                neurons = sigmoid(np.matmul(neurons, self.W[l].T) + self.b[l].T)
            elif self.activations[l]=="linear":
                neurons = linear(np.matmul(neurons, self.W[l].T) + self.b[l].T)
            elif self.activations[l]=="tanh":
                neurons = tanh(np.matmul(neurons, self.W[l].T) + self.b[l].T)
            else:
                neurons = relu(np.matmul(neurons, self.W[l].T) + self.b[l].T)
        return neurons

    def visualize_weights(self, epoch):
        return

    def visualize(self,fig=None,epoch=0):
        
        network=VisNN.DrawNN(self.network_structure(),self._coefs(),self._bias())
        return network.draw( fig= fig,epoch=epoch)
    
    def  network_structure(self):
        structure=[len(self.W[0][0])]
        for W in self.W:
            structure.append(len(W))
        return np.array(structure)
    
    def _coefs(self):
        weights=[]
       
        for W in self.W:
            _weights=[[] for i in range(len( W[0] ))] 
            for Wx in W:
                for i,weight in enumerate(Wx):
                    _weights[i].append(weight)
            weights.append(np.array(_weights))
        return weights
    
    def _bias(self):
        bias=[]            
        for b in self.b:
            bias.append([_b[0] for _b in  b ])
        return bias