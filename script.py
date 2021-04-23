# arquivo: script.py
# autor: Artur Rodrigues Rocha Neto

import time
from mnist import *
from neuralnet import *

start = time.time()
X_train, y_train, X_test, y_test = load_mnist()

#print(type(X_train), X_train.shape)
#print(type(y_train), y_train.shape)
#print(type(X_test), X_test.shape)
#print(type(y_test), y_test.shape)

net = NeuralNet([784, 30, 10])
net.fit(X_train, y_train, 30, 10, 3.0)
matches = net.score(X_test, y_test)
acc = round(100 * matches / len(X_test), 2)

delta = time.time() - start
print(f"Delta: {round(delta, 2)}s, ACC: {acc}")
