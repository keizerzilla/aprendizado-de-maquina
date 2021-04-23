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
        
        test_results = [(np.argmax(self.predict(x)), np.argmax(y)) for x, y in zip(X_test, y_test)]
        matches = sum(int(x == y) for x, y in test_results)
        return matches
    
    def cost_derivative(self, output_activations, y):
        """
        Parâmetros
        ----------
            output_activations:
            y:
        
        Retorno
        -------
            d:
        """
        
        d = output_activations - y
        return d
    
    def backpropagation(self, x, y):
        """
        Parâmetros
        ----------
            x:
            y:
        
        Retorno
        -------
            nabla_b:
            nabla_w:
        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        return nabla_b, nabla_w
    
    def update(self, batch, eta):
        """
        Parâmetros
        ----------
            batch:
            eta:
        
        Retorno
        -------
            Nenhum.
        """
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def fit(self, X_train, y_train, epochs, batch_size, eta):
        """
        Parâmetros
        ----------
            X_train: lista de arrays numpy com os dados de entrada.
            y_train: vetor coluna de mesma dimensão da camada de saída com as saídas esperadas.
            epochs: número de interações de treinamento.
            batch_size: tamanho das subdivisões do conjunto de treinamendo para o GDE.
            eta: taxa de aprendizado.
        
        Retorno
        -------
            Nenhum.
        """
        
        training_data = list(zip(X_train, y_train))
        n = len(training_data)
        
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[i:i+batch_size] for i in range(0, n, batch_size)]
            for batch in batches:
                self.update(batch, eta)
            
            if self.verbose:
                print(f"Epoch {j+1}: complete")
