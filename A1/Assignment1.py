from collections import defaultdict

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
    N = P.shape[1]
    ce_loss = -np.sum(Y * np.log(P)) / N
    cost = ce_loss + 2 * lmbda * np.sum(W)
    return cost, ce_loss


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
        c2, _ = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2, _ = ComputeCost(X, Y, W_try, b, lamda)
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
        c1, _ = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2, _ = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1, _ = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2, _ = ComputeCost(X, Y, W_try, b, lamda)

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


def plot_cost_and_accuracy(history):
    fig, ax1 = plt.subplots()

    # plot the cost
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cost / Loss', color=color)
    ax1.plot(range(1, len(history["costs"])+1), history["costs"], color=color, label=f"""Cost (final: {history["costs"][-1]:.2f})""")
    ax1.plot(range(1, len(history["val_costs"])+1), history["val_costs"], color="magenta", label=f"""Val Cost (final: {history["val_costs"][-1]:.2f})""")

    ax1.plot(range(1, len(history["losses"])+1), history["losses"], color="yellow", label=f"""Loss (final: {history["losses"][-1]:.2f})""")
    ax1.plot(range(1, len(history["val_losses"])+1), history["val_losses"], color="orange", label=f"""Val Loss (final: {history["val_losses"][-1]:.2f})""")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale("log")
    #ax1.set_yscale('symlog', linthresh=0.01)

    # plot the accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(1, len(history["accuracies"])+1), history["accuracies"], color=color, label=f"""Accuracy (final: {history["val_accuracies"][-1]:.2f})""")
    ax2.plot(range(1, len(history["val_accuracies"])+1), history["val_accuracies"], color="cyan", label=f"""Val Accuracy (final: {history["val_accuracies"][-1]:.2f})""")
    ax2.tick_params(axis='y', labelcolor=color)

    # set the limits for y-axis scales
    #ax1.set_ylim([0.1, max(history["losses"]) + 0.5])
    ax2.set_ylim([min(history["accuracies"]) - 0.02, 1])

    # add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    plt.title(f'Cost and Accuracy Plot (l2={params["l2"]}) (eta={params["lr"]})')
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
    "epochs": 40,
    "batchSize": 100,
    "l2": 1,
    "lr": .001,
}


# X_batch, Y_batch, y_batch = getBatchDumb(X_train, Y_train, y_train, params["batchSize"])

def TrainCLF(Weights, bias, params, X_train, Y_train, y_train, X_val, Y_val, y_val):
    epochs = params["epochs"]
    batchesPerEpoch = np.ceil(X_train.shape[1] / params["batchSize"]).astype(int)
    history = defaultdict(list)

    for j in range(epochs):
        for i, X_batch, Y_batch, y_batch in YieldMiniBatch(X_train, Y_train, y_train, params["batchSize"]):
            Probs = EvaluateCLF(X_batch, Weights, bias)

            grad_W, grad_b = ComputeGradients(X_batch, Y_batch, Probs, Weights, bias, params["l2"])

            # grad_W_num, grad_b_num = ComputeGradsNum(X_batch, Y_batch, Probs, Weights, bias, params["l2"])
            # assert np.allclose(grad_b, grad_b_num, atol=1e-4)
            # assert np.allclose(grad_W, grad_W_num, atol=1e-4)

            Weights -= params["lr"] * grad_W
            bias -= params["lr"] * grad_b

        # Train data
        cost, loss = ComputeCost(X_train, Y_train, Weights, bias, params["l2"])
        acc = ComputeAccuracy(X_train, y_train, Weights, bias)
        history["costs"].append(cost)
        history["losses"].append(loss)
        history["accuracies"].append(acc)

        # Val data
        cost_val, loss_val = ComputeCost(X_val, Y_val, Weights, bias, params["l2"])
        acc_val = ComputeAccuracy(X_val, y_val, Weights, bias)
        history["val_costs"].append(cost_val)
        history["val_losses"].append(loss_val)
        history["val_accuracies"].append(acc_val)

        print(f"""Epoch {j + 1}/{epochs}: Batch {i + 1}/{batchesPerEpoch}: Cost={cost} ; Acc={acc} ; Val Cost={cost_val} ; Val Acc={acc_val}""")

    return Weights, bias, history


Weights, bias, history = TrainCLF(Weights, bias, params, X_train, Y_train, y_train, X_val, Y_val, y_val)

plot_cost_and_accuracy(history)

# Test data
acc_test = ComputeAccuracy(X_test, y_test, Weights, bias)


def plot_weights(W):
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))

    for i in range(len(W)):
        im = np.reshape(W[i, :], (32, 32, 3))
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        im = np.transpose(im, (1, 0, 2))  # permute

        axs[i // 5, i % 5].imshow(im)
        axs[i // 5, i % 5].axis('off')
        axs[i // 5, i % 5].set_title(f'Weight {i + 1}')

    plt.show()

plot_weights(Weights)


print(f"Final test acc {acc_test}")
