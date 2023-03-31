from collections import defaultdict
from typing import Tuple, Any

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import copy

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
    with open('../A1/cifar-10-python/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    X = dict[b'data'].astype('float64').T
    y = np.asarray(dict[b'labels'])  # .astype('float64')
    Y = one_hot_encode(y).T

    return X, Y, y


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    X -= mean
    X /= std
    return X


def ForwardPass(X: np.ndarray, params) -> dict:
    s1 = params["W1"] @ X + params["b1"]
    h = np.maximum(0, s1)
    s2 = params["W2"] @ h + params["b2"]
    pred = softmax(s2)

    assert h.shape == (params["W1"].shape[0], X.shape[1])
    assert pred.shape == (params["W2"].shape[0], X.shape[1])

    activation = {"h": h, "p": pred}
    return activation


def ComputeCost(X: np.ndarray, Y: np.ndarray, params: dict, lmbda: float) -> tuple[float, float]:
    act = ForwardPass(X, params)
    N = act["p"].shape[1]
    ce_loss = -np.sum(Y * np.log(act["p"])) / N
    cost = ce_loss + 2 * lmbda * np.sum(params["W1"]) \
                   + 2 * lmbda * np.sum(params["W2"])
    return cost, ce_loss


def ComputeAccuracy(X: np.ndarray, y: np.ndarray, params: dict) -> float:
    act = ForwardPass(X, params)
    pred = np.argmax(act["p"], axis=0)
    acc = np.mean(y == pred)
    return acc


def ComputeGradsNum(X, Y, params, lmbda, h=1e-5):
    W = [params["W1"], params["W2"]]
    b = [params["b1"], params["b2"]]
    # Initialize gradients
    grad_W = [np.zeros_like(Wi) for Wi in W]
    grad_b = [np.zeros_like(bi) for bi in b]

    # Compute cost with current parameters
    c, _ = ComputeCost(X, Y, params, lmbda)

    # Compute gradients numerically for each parameter
    for j in range(len(b)):
        grad_b[j] = np.zeros_like(b[j])

        for i in range(len(b[j])):
            b_try = b.copy()
            b_try[j][i] += h
            tmp = buildParamsDict(params["W1"], b_try[0], params["W2"], b_try[1])
            c2, _ = ComputeCost(X, Y, tmp, lmbda)
            grad_b[j][i] = (c2 - c) / h

    for j in range(len(W)):
        grad_W[j] = np.zeros_like(W[j])

        for i in range(W[j].size):
            W_try = W.copy()
            W_try[j].flat[i] += h
            tmp = buildParamsDict(W_try[0], params["b1"], W_try[1], params["b2"])
            c2, _ = ComputeCost(X, Y, tmp, lmbda)
            grad_W[j].flat[i] = (c2 - c) / h

    return grad_b, grad_W


def ComputeGradients(X: np.ndarray, Y: np.ndarray, activation: dict, params: dict, lmbda: float) -> dict:
    N = X.shape[1]

    error_out = activation["p"] - Y

    grad_W2 = (1 / N) * error_out @ activation["h"].T
    grad_b2 = (1 / N) * np.sum(error_out, axis=1, keepdims=True)

    error_hidden = params["W2"].T @ error_out
    hidden_activation_derivative = activation["h"] * (1 - activation["h"])
    error_hidden = np.multiply(error_hidden, hidden_activation_derivative)

    grad_W1 = (1 / N) * error_hidden @ X.T
    grad_b1 = (1 / N) * np.sum(error_hidden, axis=1, keepdims=True)

    # add regularization derivative
    if lmbda != 0:
        grad_W1 += 2 * lmbda * params["W1"]
        grad_W2 += 2 * lmbda * params["W2"]

    assert grad_W1.shape == params["W1"].shape
    assert grad_W2.shape == params["W2"].shape
    assert grad_b1.shape == params["b1"].shape
    assert grad_b2.shape == params["b2"].shape

    grads = {"grad_W2": grad_W2, "grad_b2": grad_b2, "grad_W1": grad_W1, "grad_b1": grad_b1}
    return grads


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

    plt.title(f'Cost and Accuracy Plot (l2={hyper["l2"]}) (eta={hyper["lr"]})')
    plt.show()


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


def initNNParams(hidden_nodes):
    params = {}
    params["W1"] = np.random.normal(0, 1 / np.sqrt(D), (hidden_nodes, D))
    params["b1"] = np.random.normal(0, 1 / np.sqrt(hidden_nodes), (hidden_nodes, 1))

    params["W2"] = np.zeros((K, hidden_nodes))
    params["b2"] = np.zeros((K, 1))
    return params


def buildParamsDict(W1, b1, W2, b2):
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def updateNNParams(params, grads, hyper):
    params["W1"] -= hyper["lr"] * grads["grad_W1"]
    params["b1"] -= hyper["lr"] * grads["grad_b1"]

    params["W2"] -= hyper["lr"] * grads["grad_W2"]
    params["b2"] -= hyper["lr"] * grads["grad_b2"]
    return params


def TrainCLF(params, hyper, X_train, Y_train, y_train, X_val, Y_val, y_val):
    epochs = hyper["epochs"]
    batchesPerEpoch = np.ceil(X_train.shape[1] / hyper["batchSize"]).astype(int)
    history = defaultdict(list)

    for j in range(epochs):
        for i, X_batch, Y_batch, y_batch in YieldMiniBatch(X_train, Y_train, y_train, hyper["batchSize"]):
            activation = ForwardPass(X_batch, params)

            grads = ComputeGradients(X_batch, Y_batch, activation, params, hyper["l2"])

            grad_b_num, grad_W_num = ComputeGradsNum(X_batch, Y_batch, params, hyper["l2"])

            assert np.allclose(grads["grad_b1"], grad_b_num[0], atol=1e-4)
            assert np.allclose(grads["grad_W1"], grad_W_num[0], atol=1e-4)

            assert np.allclose(grads["grad_b2"], grad_b_num[1], atol=1e-4)
            assert np.allclose(grads["grad_W2"], grad_W_num[1], atol=1e-4)

            updateNNParams(params, grads, hyper)

            # Train data
            cost, loss = ComputeCost(X_train, Y_train, params, hyper["l2"])
            acc = ComputeAccuracy(X_train, y_train, params)
            history["costs"].append(cost)
            history["losses"].append(loss)
            history["accuracies"].append(acc)

            # Val data
            cost_val, loss_val = ComputeCost(X_val, Y_val, params, hyper["l2"])
            acc_val = ComputeAccuracy(X_val, y_val, params)
            history["val_costs"].append(cost_val)
            history["val_losses"].append(loss_val)
            history["val_accuracies"].append(acc_val)

            print(f"""Epoch {j + 1}/{epochs}: Batch {i + 1}/{batchesPerEpoch}: Cost={cost} ; Acc={acc} ; Val Cost={cost_val} ; Val Acc={acc_val}""")

    return params, history


D = 3072  # dimensionality
N = 10000  # samples
K = 10  # classes
hidden_nodes = 50

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

params = initNNParams(hidden_nodes)

hyper = {
    "epochs": 40,
    "batchSize": 5,
    "l2": 0,
    "lr": .001,
}


params, history = TrainCLF(params, hyper, X_train, Y_train, y_train, X_val, Y_val, y_val)

plot_cost_and_accuracy(history)

# Test data
acc_test = ComputeAccuracy(X_test, y_test, params)

plot_weights(params["W1"])
plot_weights(params["W2"])


print(f"Final test acc {acc_test}")
