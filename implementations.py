import numpy as np
import matplotlib.pyplot as plt

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
    losses = np.zeros(max_iters)
    weights = np.zeros((max_iters, tx.shape[1]))
    
    w = initial_w
    if max_iters < 1:
        loss = compute_mse(y, tx, w)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=0, ti=0, l=loss))
    else:
        for n_iter in range(max_iters):
            gradient = compute_gradient_mse(y, tx, w)
            w = w - gamma * gradient
            weights[n_iter, :] = w
            loss = compute_mse(y, tx, w)
            losses[n_iter] = loss
        print("Gradient Descent({bi}/{ti}): Final loss={l}".format(
                bi=n_iter, ti=max_iters, l=loss))
    return weights, losses

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def compute_y_test(x_test, w):
    """compute the output vector y_test"""
    y_test = x_test.dot(w)
    y_test_abs_rounded = np.where(np.abs(y_test) > 0, 1, -1)
    print("the number of ones in y_test is", 100*np.sum(y_test_abs_rounded == 1)/len(y_test_abs_rounded), ' %')
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

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # compute gradient and loss
            grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_mse(y_batch, tx_batch, w)
            # update w by gradient
            # store w and loss
            losses.append(loss)

    print(
        "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        )
    )
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/ (1 + np.exp(t))

def compute_loss_logistic(y, tx, w):
    """compute the loss for y in [-1, 1]: negative log likelihood."""
    pred = tx.dot(w)
    loss = np.sum(np.log(1 + np.exp(pred)) - y * pred)/len(y)
    return loss

def compute_gradient_logistic(y, tx, w):
    """compute the gradient of loss."""
    pred = tx.dot(w)
    gradient = tx.T.dot(sigmoid(pred) - y)/len(y)
    return gradient
    
def logistic_regression(y, x, initial_w, max_iter, gamma):
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
        gradient = compute_gradient_logistic(y, x, w)
        w = w - gamma * gradient
        loss = compute_loss_logistic(y, x, w)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iter - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def reg_logistic_regression(y, x, lambda_, initial_w, max_iter, gamma):
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
    for n_iter in range(max_iter+1):
        gradient = compute_gradient_logistic(y, x, w) + lambda_*w
        w = w - gamma * gradient
        loss = compute_loss_logistic(y, x, w)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iter - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_gradient_descent(y, x, max_iters, k_indices, k):
    """find the best initial_w for gradient descent over a k-fold cross-validation"""
    # get k'th subgroup in test, others in train
    test_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)].flatten()
    x_test = x[test_indices, :]
    y_test = y[test_indices]
    x_train = x[train_indices, :]
    y_train = y[train_indices]
    # form data with polynomial degree
    #x_train = build_poly(x_train, degree)
    #x_test = build_poly(x_test, degree)
    # ridge regression
    initial_w = np.random.choice([-1,1], size=(22,))
    w, loss = mean_squared_error_gd(y_train, x_train, initial_w, max_iters=max_iters, gamma=0.01)
    # calculate the loss for train and test data
    loss_tr = compute_mse(y_train, x_train, w)
    loss_te = compute_mse(y_test, x_test, w)
    return loss_tr, loss_te, w

#I want ot test the initial weights on the accuracy of the prediction using cross validation, based on the proportion between ones and minus ones in the init weights 
# and plot the accuracy as a function of this proportion

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_accuracy(y, x, initial_w, k_indices, k):
    """find the best initial_w for gradient descent over a k-fold cross-validation"""
    # get k'th subgroup in test, others in train
    test_indices = k_indices[k]
    train_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)].flatten()
    x_test = x[test_indices, :]
    y_test = y[test_indices]
    x_train = x[train_indices, :]
    y_train = y[train_indices]
    # form data with polynomial degree
    #x_train = build_poly(x_train, degree)
    #x_test = build_poly(x_test, degree)
    # ridge regression
    w, loss = mean_squared_error_gd(y_train, x_train, initial_w, max_iters=50, gamma=0.01)
    # calculate the loss for train and test data
    y_test_pred = compute_y_test(x_test, w)
    accuracy = np.sum(y_test_pred == y_test)/len(y_test)
    return accuracy, w


def confusion_matrix(y_test, y_pred):
    """compute the confusion matrix"""
    TP = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    TN = np.sum(np.logical_and(y_pred == -1, y_test == -1))
    FP = np.sum(np.logical_and(y_pred == 1, y_test == -1))
    FN = np.sum(np.logical_and(y_pred == -1, y_test == 1))

    f1_score = 2*TP/(2*TP + FP + FN)
    return TP, TN, FP, FN, f1_score



