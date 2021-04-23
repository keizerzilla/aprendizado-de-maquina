# arquivo: mnist.py
# autor: Artur Rodrigues Rocha Neto

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def matrix_tolist(m):
    """
    O destino desta função e se tornar obsoleta.
    
    Parâmetros
    ----------
        m:
    
    Retorno:
        l:
    """
    
    l = [row.reshape(-1, 1) for row in m]
    return l
    
def load_mnist(data_folder="data/"):
    """
    Esta função, bem como a arquitetura da rede, ainda respondem aos formatos de dados
    definidos pelo autor original do código. É de suma importância que estes sejam
    modificados o quanto antes, visando melhor compatibilidade com outras bibliotecas
    e APIs (como scikit-learn e tensorflow).
    
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
    test = pd.read_csv(os.path.join(data_folder, "test.csv"))
    
    X = train.drop(["label"], axis=1)
    y = []
    for row in train["label"]:
        e = np.zeros((10, 1))
        e[row] = 1.0
        y.append(e)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    return matrix_tolist(np.array(X_train) / 255.0), y_train, matrix_tolist(np.array(X_test) / 255.0), y_test
