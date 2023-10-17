import numpy as np

def compute_mse(y, tx, w):
    """compute the loss with MSE.
    Args:
        y: numpy array of shape=(N, ). The labels.
        tx: numpy array of shape=(N, D). The input data.
        w: numpy array of shape=(D, ). The weights"""
    e = y - tx.dot(w)
    mse = e.dot(e)/(2*len(y))
    return mse

def compute_gradient_mse(y, tx, w):
    """compute the gradient of loss.
    Args:
        y: numpy array of shape=(N, ). The labels.
        tx: numpy array of shape=(N, D). The input data.
        w: numpy array of shape=(D, ). The weights."""

    e = y - tx.dot(w)
    grad =-1/(len(y)) * tx.T.dot(e)
    return grad

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """ gradient descent algorithm using mean squared error as the loss function 
        y = output vector
        tx = input matrix
        initial_w = initial weights
        max_iters = maximum number of iterations
        gamma = learning rate"""
        
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * gradient
    print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w

def compute_y_test(x_test, w):
    """compute the output vector y_test"""
    x_test_std = standardize(x_test)[0]
    y_test = x_test_std.dot(w)
    #y_test_abs_rounded = np.where(np.abs(y_test) > 0.5, 1, 0)
    print('the number of heart attack predicted in the test sample are :', np.sum(y_test), 'out of', len(y_test), 'samples', 'which is', np.sum(y_test)/len(y_test)*100, '%')
    return y_test

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset."""
    data_size = len(y)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            batch_indices = indices[start_index:end_index]
            yield y[batch_indices], tx[batch_indices]

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.
    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(y)
    return grad

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """ stochastic gradient descent algorithm using mean squared error as the loss function 
        y = output vector
        tx = input matrix
        initial_w = initial weights
        max_iters = maximum number of iterations
        gamma = learning rate"""
    
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # compute gradient and loss
            loss = compute_mse(y_batch, tx_batch, w)
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w by gradient
            w = w - gamma * grad
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return loss, w

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return loss, w

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return loss, w

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    poly_x = np.zeros((len(x), degree+1))
    for i, z in enumerate(x):
        print(z)
        for j in range(degree+1):
            poly_x[i,j] = z**j

    return poly_x

def poly_ridge_ls(y,tx,lambda_=0,ridge_y=False,ls_y = False, degree = 1):
    if ridge_y == False and ls_y == False:
        print('no algorithm has been chosen')
        return
    elif ridge_y == True and ls_y == True:
        print('both algorithm have been chosen')
        return
        
    elif ls_y == True and ridge_y == False:
        tx_poly = build_poly(tx, degree)
        weights, mse = least_squares(y, tx_poly)
        rmse = np.sqrt(2*mse)
        return rmse, weights
    
    elif ls_y == False and ridge_y == True:
        tx_poly = build_poly(tx, degree)
        weights, mse = ridge_regression(y, tx_poly, lambda_)
        rmse = np.sqrt(2*mse)
        return rmse, weights
    
def compute_loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(pred))) - y.T.dot(pred)
    return loss

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = tx.dot(w)
    gradient = tx.T.dot(sigmoid(pred) - y)
    return gradient
    
def logistic_regression(y, x, max_iter, gamma, initial_w):
    """calculate the loss and the weights using logistic regression.
        Args : 
        x = input matrix of the training set (N,D) where N is the number of samples and D the number of features
        y = output vector of the training set(N,) where N is the number of samples
        max_iter = maximum number of iterations
        gamma = learning rate
        initial_w = initial weights
        return :
        loss = loss of the logistic regression
        w = weights of the logistic regression"""
    w = initial_w
    for n_iter in range(max_iter):
        loss = compute_loss_logistic(y, x, w)
        gradient = compute_gradient_logistic(y, x, w)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iter - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w
    
