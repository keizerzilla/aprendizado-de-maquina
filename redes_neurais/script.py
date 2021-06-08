# arquivo: script.py
# autor: Artur Rodrigues Rocha Neto
# benchmark atual: ~120s, 97.31 ACC

import time
from mnist import *
from neuralnet import *

X_train, y_train, X_test, y_test = load_mnist()

start = time.time()

net = NeuralNet([784, 30, 10])
net.fit(X_train, y_train, 30, 10, 3.0)
matches = net.score(X_test, y_test)
acc = round(100 * matches / X_test.shape[0], 2)

delta = time.time() - start

print(f"Delta: {round(delta, 2)}s, ACC: {acc} (matches: {matches})")
