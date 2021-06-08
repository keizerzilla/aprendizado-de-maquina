# arquivo: mnist.py
# autor: Artur Rodrigues Rocha Neto

import os
import numpy as np
import pandas as pd

def load_mnist(data_folder="data/"):
    """
    Parâmetros
    ----------
        data_folder:
    
    Retorno:
        X_train:
        y_train:
        X_test:
        y_test:
    """
    
    train = pd.read_csv(os.path.join(data_folder, "train.csv"))
    #test = pd.read_csv(os.path.join(data_folder, "test.csv"))
    
    X = np.array(train.drop(["label"], axis=1)) / 255.0
    
    y = []
    for row in train["label"]:
        e = np.zeros((10,))
        e[row] = 1.0
        y.append(e)
    y = np.array(y)
    
    p = np.random.permutation(X.shape[0])
    X = X[p]
    y = y[p]
    
    n_train = int(X.shape[0] * 0.7)
    n_test = int(X.shape[0] - n_train)
    X_train = X[0:n_train]
    y_train = y[0:n_train]
    X_test = X[0:n_test]
    y_test = y[0:n_test]
    
    return X_train, y_train, X_test, y_test
