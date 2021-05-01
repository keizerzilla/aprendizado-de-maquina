# arquivo: neuralnet.py
# autor: Artur Rodrigues Rocha Neto (adaptado de Michael Nielsen)

import random
import numpy as np

class NeuralNet:
    
    def __init__(self, sizes, verbose=True):
        """
        Parâmetros
        ----------
            sizes:
            verbose:
        
        Retorno
        -------
            Nenhum.
        """
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.verbose = verbose
    
    def sigmoid(self, z):
        """
        Parâmetros
        ----------
            z:
        
        Retorno
        -------
            sigma:
        """
        
        sigma = 1.0 / (1.0 + np.exp(-z))
        return sigma
    
    def sigmoid_prime(self, z):
        """
        Parâmetros
        ----------
            z:
        
        Retorno
        -------
            sigma_p:
        """
        sigma_p = self.sigmoid(z) * (1.0 - self.sigmoid(z))
        return sigma_p
    
    def predict(self, a):
        """
        Parâmetros
        ----------
            a:
        
        Retorno
        -------
            u:
        """
        
        u = a
        for b, w in zip(self.biases, self.weights):
            u = self.sigmoid(np.dot(w, u) + b)
        
        return u
    
    def score(self, X_test, y_test):
        """
        Parâmetros
        ----------
            X_test:
            y_test:
        
        Retorno
        -------
            matches:
        """
        
        matches = 0
        for a, u in zip(X_test, y_test):
            a = np.reshape(a, (-1, 1))
            u = np.reshape(u, (-1, 1))
            
            out1 = np.argmax(self.predict(a))
            out2 = np.argmax(u)
            
            if out1 == out2:
                matches = matches + 1
        
        return matches
    
    def cost_derivative(self, output_activations, u):
        """
        Parâmetros
        ----------
            output_activations:
            u:
        
        Retorno
        -------
            d:
        """
        
        d = output_activations - u
        return d
    
    def backpropagation(self, a, u):
        """
        Parâmetros
        ----------
            a:
            u:
        
        Retorno
        -------
            nabla_b:
            nabla_w:
        """
        
        a = np.reshape(a, (-1, 1))
        u = np.reshape(u, (-1, 1))
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = a
        activations = [a]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], u) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        return nabla_b, nabla_w
    
    def update(self, X, y, eta):
        """
        Parâmetros
        ----------
            X:
            y:
            eta:
        
        Retorno
        -------
            Nenhum.
        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for a, u in zip(X, y):
            delta_nabla_b, delta_nabla_w = self.backpropagation(a, u)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w - (eta / X.shape[0]) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / X.shape[0]) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def fit(self, X_train, y_train, epochs, batch_size, eta):
        """
        Parâmetros
        ----------
            X_train: matriz de dimensão (n_amostras, n_atributos)
            y_train: matriz de dimensão (n_amostras, n_saidas)
            epochs: número de interações de treinamento.
            batch_size: tamanho das subdivisões do conjunto de treinamendo para o GDE.
            eta: taxa de aprendizado.
        
        Retorno
        -------
            Nenhum.
        """
        
        if X_train.shape[0] != y_train.shape[0]:
            raise Exception("Número de amostras não batem!")
        
        n_samples = X_train.shape[0]
        
        for i in range(epochs):
            p = np.random.permutation(n_samples)
            X_train = X_train[p]
            y_train = y_train[p]
            
            for j in range(0, n_samples, batch_size):
                self.update(X_train[j:j+batch_size], y_train[j:j+batch_size], eta)
            
            if self.verbose:
                print(f"Epoch {i+1}: complete")
