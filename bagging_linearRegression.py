# Author : Guillermo López
# Date : February 1st, 2021
# Description : Train several linear regression models to avoid overfitting and describe non-linear relations.

def bagging_linearRegression(X_train, y_train, n_batch=50):
    n_total, m = X_train.shape
    W = np.zeros((n_batch,m))
    B = np.zeros(n_batch)
    batch = np.around(np.random.rand(n_total) * n_batch)
    for i in np.arange(0, n_batch):
        X = X_train[batch == i].copy()
        X.reset_index(drop=True, inplace=True)
        n, m = X.shape
        y = y_train[batch == i].copy()
        y.reset_index(drop=True, inplace=True)
        # using analytical solution
        # calculating coefficients
        w_a = np.linalg.inv(np.dot(X.T,X))
        w_b = np.dot(X.T,y).T
        w_c = X.sum()
        w = np.dot(w_a, (w_b - np.array(w_c)).T)
        # calculating bias
        n = len(X)
        b_a = y.sum()
        b_b = np.dot(w.T,w_c)
        b = (1/n) * (np.array(b_a) - b_b)
        W[i] = w.T
        B[i] = b
    return(W,B)
