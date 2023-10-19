import numpy as np

def compute_mse(y, tx, w):
    """compute the loss with MSE.
    Args:
        y: numpy array of shape=(N, ). The labels.
        tx: numpy array of shape=(N, D). The input data.
        w: numpy array of shape=(D, ). The weights"""
    e = y - tx.dot(w)
    mse = 1/(2*len(y)) * e.dot(e)
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
        w = w - gamma * gradient
        loss = compute_mse(y, tx, w)
    print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters, l=loss))
    return loss, w

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

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
    if max_iters < 1:
        raise ValueError("max_iters must be greater than 1.")
    losses = []
    w = initial_w

    for n_iter in range(max_iters+1):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # compute gradient and loss
            loss = compute_mse(y_batch, tx_batch, w)
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w by gradient
            w = w - gamma * grad
            # store w and loss
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

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/ (1 + np.exp(t))

def compute_loss_logistic(y, tx, w):
    """compute the loss for y in [-1, 1]: negative log likelihood."""
    pred = tx.dot(w)
    loss = 1/len(y)*np.sum(np.log(1 + np.exp(-y*pred)))
    return loss

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = tx.dot(w)
    gradient = tx.T.dot(sigmoid(pred))/len(y)
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
    for n_iter in range(max_iter+1):
        loss = compute_loss_logistic(y, x, w)
        gradient = compute_gradient_logistic(y, x, w)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iter - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w

def reg_logistic_regression(y, x, lambda_, max_iter, gamma, initial_w):
    """calculate the loss and the weights using regularized logistic regression.
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
        loss = compute_loss_logistic(y, x, w) + lambda_/2*np.linalg.norm(w)**2
        gradient = compute_gradient_logistic(y, x, w) + lambda_*w
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iter - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w
    