import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

random.seed(400)


def softmax(x: np.ndarray) -> np.ndarray:
    """ Standard definition of the softmax function """
    exps = np.exp(x)
    return exps / np.sum(exps, axis=0)


def one_hot_encode(values: np.ndarray) -> np.ndarray:
    n_values = len(values)
    one_hot_encoded = np.zeros((n_values, 10))
    one_hot_encoded[np.arange(n_values), values] = 1
    return one_hot_encoded


def load_batch(filename) -> (np.ndarray, np.ndarray, np.ndarray):
    """ Copied from the dataset website """
    import pickle
    with open('cifar-10-python/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    X = dict[b'data'].astype('float64').T
    y = np.asarray(dict[b'labels'])  # .astype('float64')
    Y = one_hot_encode(y).T

    return X, Y, y


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    X -= mean
    X /= std
    return X


def EvaluateCLF(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = W @ X + b
    P = softmax(s)
    assert P.shape == (W.shape[0], X.shape[1])
    return P


def ComputeCost(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.ndarray, lmbda: float) -> float:
    P = EvaluateCLF(X, W, b)
    lcross = -Y.T @ np.log(P)
    # cost = np.sum(lcross) / X.shape[1] + lmbda * np.sum(W ** 2)
    cost = np.mean(lcross) + lmbda * np.sum(W ** 2)
    loss = log_loss(np.argmax(Y, axis=0).flatten(), P.T)
    return loss


def ComputeAccuracy(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    P = EvaluateCLF(X, W, b)
    pred = np.argmax(P, axis=0)
    acc = np.mean(y == pred)
    return acc


def ComputeGradsNum(X, Y, P, W, b, lamda, h=1e-6):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = ComputeCost(X, Y, W, b, lamda)

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i, j] = (c2 - c) / h

    return grad_W, grad_b


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h=1e-6):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return [grad_W, grad_b]


def ComputeGradients(X: np.ndarray, Y: np.ndarray, P: np.ndarray, W: np.ndarray, b: np.ndarray, lmbda: float) -> (
np.ndarray, np.ndarray):
    N = X.shape[1]
    # error gradient of the cross-entropy loss function for the softmax function
    Gb = P - Y

    # multiply and sum intermediate error with previous layer (here input)
    grad_W = (1 / N) * Gb @ X.T

    # same, but for bias we always assume 1s
    grad_b = (1 / N) * np.sum(Gb, axis=1, keepdims=True)

    # add regularization derivative
    if lmbda != 0:
        grad_W += 2 * lmbda * W

    assert grad_W.shape == W.shape
    assert grad_b.shape == b.shape

    return grad_W, grad_b


def getBatchDumb(X, Y, y, batchSize=100):
    X_batch = X[:, :batchSize]
    Y_batch = Y[:, :batchSize]
    y_batch = y[:batchSize]
    return X_batch, Y_batch, y_batch


def YieldMiniBatch(_X, _Y, _y, batchSize=100):
    X = _X.T
    Y = _Y.T
    y = _y

    permutation = np.random.permutation(len(X))
    X = X[permutation]
    Y = Y[permutation]
    y = y[permutation]

    amountOfBatches = np.ceil(len(X) / batchSize).astype(int)
    for i in range(amountOfBatches):
        start = i * batchSize
        yield i, X[start:start + batchSize].T, Y[start:start + batchSize].T, y[start:start + batchSize]


def plot_cost_and_accuracy(costs, accuracies):
    fig, ax1 = plt.subplots()

    # plot the cost
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(range(1, len(costs)+1), costs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # plot the accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(1, len(accuracies)+1), accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # set the limits for y-axis scales
    ax1.set_ylim([0, max(costs) + 0.5])
    ax2.set_ylim([min(accuracies) - 0.02, 1])

    # add a title and display the plot
    plt.title('Cost and Accuracy Plot')
    plt.show()



D = 3072  # dimensionality
N = 10000  # samples
K = 10  # classes

X_train, Y_train, y_train = load_batch("data_batch_1")
# helper.montage(X_train)

X_val, Y_val, y_val = load_batch("data_batch_2")

X_test, Y_test, y_test = load_batch("test_batch")

assert X_train.shape == (D, N)
assert Y_train.shape == (K, N)
assert y_test.shape == (N,)

Xmean = X_train.mean(axis=1).reshape(-1, 1)
Xstd = X_train.std(axis=1).reshape(-1, 1)

assert Xmean.shape == (D, 1)
assert Xstd.shape == (D, 1)

X_train = normalize(X_train, Xmean, Xstd)
X_val = normalize(X_val, Xmean, Xstd)
X_test = normalize(X_test, Xmean, Xstd)

Weights = np.random.normal(0, 0.01, (K, D))
bias = np.random.normal(0, 0.01, (K, 1))

params = {
    "epochs": 50,
    "batchSize": 100,
    "l2": 0,
    "lr": 0.001,
}


# X_batch, Y_batch, y_batch = getBatchDumb(X_train, Y_train, y_train, params["batchSize"])

def TrainCLF(Weights, bias, params, X_train, Y_train, y_train):
    epochs = params["epochs"]
    batchSize = params["batchSize"]
    batchesPerEpoch = np.ceil(X_train.shape[1] / batchSize).astype(int)
    costs = []
    accs = []

    for j in range(epochs):
        for i, X_batch, Y_batch, y_batch in YieldMiniBatch(X_train, Y_train, y_train, batchSize):
            Probs = EvaluateCLF(X_batch, Weights, bias)

            grad_W, grad_b = ComputeGradients(X_batch, Y_batch, Probs, Weights, bias, params["l2"])

            # grad_W_num, grad_b_num = ComputeGradsNum(X_batch, Y_batch, Probs, Weights, bias, params["l2"])
            # assert np.allclose(grad_b, grad_b_num, atol=1e-4)
            # assert np.allclose(grad_W, grad_W_num, atol=1e-4)

            Weights -= params["lr"] * grad_W / batchSize
            bias -= params["lr"] * grad_b / batchSize

        cost = ComputeCost(X_train, Y_train, Weights, bias, params["l2"])
        acc = ComputeAccuracy(X_train, y_train, Weights, bias)
        costs.append(cost)
        accs.append(acc)

        print(f"""Epoch {j + 1}/{epochs}: Batch {i + 1}/{batchesPerEpoch}: Cost={cost} ; Acc={acc}""")

    return Weights, bias, costs, accs


Weights, bias, costs, accs = TrainCLF(Weights, bias, params, X_train, Y_train, y_train)

plot_cost_and_accuracy(costs, accs)

print()
