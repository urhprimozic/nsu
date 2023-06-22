import numpy as np
import torch
from sklearn.utils import resample

# koda za pripravo podatkov
def transform_data(data, N, M, obcine_output, n_samples=None):
    '''
    Pripravi vektor oblike (X, y) , kjer je 
    -  element x_i iz X
    matrika Nx |obcine|, vzeta iz data
    - y je vektor Mx|obcine| z napovedi za naprej

    Če n_samples is not none, potem še sampla le n_sample samplov 

    Parameters
    ----------
    - data - cvs file
    - obcine_output - obcine, ki jih das v output
    '''
    X = []
    y = []
    for i in range(len(data) - (N + M)):
        X.append(data.iloc[i:N+i].to_numpy())

        future = data.iloc[N+i:N+i+M]
        y.append(future[obcine_output].to_numpy())
    X = np.array(X)
    y = np.array(y)
    # subsample
    if n_samples is not None:
        X, y = resample(X, y, n_samples=n_samples, replace=False)
    

    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    return X, y
