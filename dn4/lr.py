import numpy as np
# funkcije iz vaj

from scipy.optimize import minimize
def lasso_regresija(X, y, lam=1, meja=1e-2):
    '''
    Vrne par izraz, napaka
    '''
    imena = X.columns

    def f(beta):
        yhat = X.dot(beta)
        return np.sum((yhat-y)**2) + lam*np.sum(np.abs(beta))
    beta = minimize(f, np.random.random(X.shape[1]))["x"]
    
    yhat = X.dot(beta)
    print(X.shape[0])
    napaka = np.sum((yhat-y)**2)/float(X.shape[0])


    izraz = ""
    for i,b in enumerate(beta):
        if b > meja:
            if len(izraz) > 0:
                izraz += " + "
            izraz +=  f"{b:.3f}*{imena[i]}"
    return izraz, napaka


def ridge_regresija(X, y, lam=1, meja=1e-2):
    imena = X.columns
    beta = np.linalg.pinv(X.T.dot(X) + lam*np.identity(X.shape[1])).dot(X.T).dot(y)

    izraz = ""
    for i,b in enumerate(beta):
        if b > meja:
            if len(izraz) > 0:
                izraz += " + "
            izraz +=  f"{b:.3f}*{imena[i]}"
    return izraz

