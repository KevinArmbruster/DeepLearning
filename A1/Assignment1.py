import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def one_hot_encode(values: np.ndarray) -> np.ndarray:
    one_hot_encoded = np.zeros((len(values), 10))
    for i, val in enumerate(values):
        for j in np.arange(one_hot_encoded.shape[1]):
            one_hot_encoded[i, j] = 1 if val == j else 0
    return one_hot_encoded


def load_batch(filename) -> (np.ndarray, np.ndarray, np.ndarray):
    """ Copied from the dataset website """
    import pickle
    with open('cifar-10-python/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    X = dict[b'data'].astype('float64').T
    y = np.asarray(dict[b'labels']).astype('float64')
    Y = one_hot_encode(y).T

    return X, Y, y


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    X -= mean
    X /= std
    return X


def evaluateCLF(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = W @ X + b
    P = softmax(s)
    assert P.shape == (W.shape[0], X.shape[1])
    return P


def getCost(X: np.ndarray, Y: np.ndarray, W: np.ndarray, b: np.ndarray, lmbda: float) -> float:
    P = evaluateCLF(X, W, b)
    lcross = Y.T @ np.log(P)
    cost = 1/X.shape[1] * np.sum(lcross) + lmbda * np.sum(W**2)
    return cost

def getAccuracy(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    P = evaluateCLF(X, W, b)
    pred = np.argmax(P, axis=0)
    res = y == pred
    correct = np.sum(res)
    acc = correct / len(res)
    return acc


def getGradients(X: np.ndarray, Y: np.ndarray, P: np.ndarray, W: np.ndarray, b: np.ndarray, lmbda: float) -> (np.ndarray, np.ndarray):
    grad_W = 0
    grad_b = 0

    assert grad_W.shape == W.shape
    assert grad_b.shape == b.shape

    return grad_W, grad_b


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

lmbda = 1.
P = evaluateCLF(X_train, Weights, bias)
cost = getCost(X_train, Y_train, Weights, bias, lmbda)
acc = getAccuracy(X_train, y_train, Weights, bias)
grad_W, grad_b = getGradients(X_train, Y_train, P, Weights, lmbda)

print()
