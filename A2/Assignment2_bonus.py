from collections import defaultdict

import numpy as np
import random
import matplotlib.pyplot as plt
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


def load_all_batches():
    X, Y, y = load_batch("data_batch_1")
    for i in range(2,6):
        X_, Y_, y_ = load_batch("data_batch_" + i.__str__())
        X = np.concatenate((X, X_), axis=1)
        Y = np.concatenate((Y, Y_), axis=1)
        y = np.concatenate((y, y_))
    return X, Y, y


def train_test_split(X, Y, y, test_size):
    X_train = X[:, test_size:]
    X_test = X[:, :test_size]

    Y_train = Y[:, test_size:]
    Y_test = Y[:, :test_size]

    y_train = y[test_size:]
    y_test = y[:test_size]
    return X_train, Y_train, y_train, X_test, Y_test, y_test


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    X -= mean
    X /= std
    return X


def ForwardPass(X: np.ndarray, layers: list, cache=False) -> np.ndarray:
    s1 = layers[0]["W"] @ X + layers[0]["b"]
    h = relu(s1)

    s2 = layers[1]["W"] @ h + layers[1]["b"]
    pred = softmax(s2)

    assert h.shape == (layers[0]["W"].shape[0], X.shape[1])
    assert pred.shape == (layers[1]["W"].shape[0], X.shape[1])

    if cache:
        layers[0]["activation"] = h
        layers[1]["activation"] = pred

    return pred


def relu(X, d=False):
    if not d:
        return np.maximum(0, X)
        # np.where(X > 0, X, 0)
    else:
        return np.where(X > 0, 1, 0)


def ComputeCost(X: np.ndarray, Y: np.ndarray, layers: list, lmbda: float) -> tuple[float, float]:
    pred = ForwardPass(X, layers)
    N = pred.shape[1]
    loss_cross = -np.sum(Y * np.log(pred)) / N
    # loss_cross222 = sum(-np.log((Y * pred).sum(axis=0))) / N
    # assert np.allclose(loss_cross222, loss_cross)

    loss_regularization = sum([lmbda * (layer["W"] ** 2).sum() for layer in layers])
    cost = loss_cross + loss_regularization
    return cost, loss_cross


def ComputeAccuracy(X: np.ndarray, y: np.ndarray, layers: list) -> float:
    pred = ForwardPass(X, layers)
    p = np.argmax(pred, axis=0)
    acc = np.mean(y == p)
    return acc


def ComputeGradsNum(X, Y, layers, lambda_, h=1e-5):
    Ws = [layers[0]["W"], layers[1]["W"]]
    bs = [layers[0]["b"], layers[1]["b"]]

    grad_W = [np.zeros_like(Wl) for Wl in Ws]
    grad_b = [np.zeros_like(bl) for bl in bs]

    c, _ = ComputeCost(X, Y, layers, lambda_)

    for j in range(len(bs)):
        for i in range(bs[j].size):
            b_try = copy.deepcopy(bs)
            b_try[j][i] += h
            tmp = BuildTmpLayers([layers[0]["W"], layers[1]["W"]], b_try)
            c2, _ = ComputeCost(X, Y, tmp, lambda_)
            grad_b[j][i] = (c2 - c) / h

    for j in range(len(Ws)):
        for i in np.ndindex(Ws[j].shape):
            W_try = copy.deepcopy(Ws)
            W_try[j][i] += h
            tmp = BuildTmpLayers(W_try, [layers[0]["b"], layers[1]["b"]])
            c2, _ = ComputeCost(X, Y, tmp, lambda_)
            grad_W[j][i] = (c2 - c) / h

    return grad_W, grad_b


def ComputeGradsNumGit(X, Y, layers, lmbda, h=1e-5):
    W2 = layers[1]["W"]
    b2 = layers[1]["b"]
    W1 = layers[0]["W"]
    b1 = layers[0]["b"]
    grad_W2 = np.zeros(shape=W2.shape)
    grad_b2 = np.zeros(shape=b2.shape)
    grad_W1 = np.zeros(shape=W1.shape)
    grad_b1 = np.zeros(shape=b1.shape)
    c, _ = ComputeCost(X, Y, layers, lmbda)

    for i in range(b1.shape[0]):
        b1_try = b1.copy()
        b1_try[i, 0] += h
        tmp = BuildTmpLayers([layers[0]["W"], layers[1]["W"]], [b1_try, b2])
        c2, _ = ComputeCost(X, Y, tmp, lmbda)
        grad_b1[i, 0] = (c2 - c) / h

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = W1.copy()
            W1_try[i, j] += h
            tmp = BuildTmpLayers([W1_try, W2], [layers[0]["b"], layers[1]["b"]])
            c2, _ = ComputeCost(X, Y, tmp, lmbda)
            grad_W1[i, j] = (c2 - c) / h

    for i in range(b2.shape[0]):
        b2_try = b2.copy()
        b2_try[i, 0] += h
        tmp = BuildTmpLayers([layers[0]["W"], layers[1]["W"]], [b1, b2_try])
        c2, _ = ComputeCost(X, Y, tmp, lmbda)
        grad_b2[i, 0] = (c2 - c) / h

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = W2.copy()
            W2_try[i, j] += h
            tmp = BuildTmpLayers([W1, W2_try], [layers[0]["b"], layers[1]["b"]])
            c2, _ = ComputeCost(X, Y, tmp, lmbda)
            grad_W2[i, j] = (c2 - c) / h

    return grad_W2, grad_b2, grad_W1, grad_b1


def ComputeGradients(X: np.ndarray, Y: np.ndarray, layers: list, lmbda: float):
    error = layers[1]["activation"] - Y

    N = layers[1]["activation"].shape[1]
    layers[1]["dW"] = (1 / N) * error @ layers[0]["activation"].T + 2 * lmbda * layers[1]["W"]
    layers[1]["db"] = np.mean(error, axis=1, keepdims=True)

    error = layers[1]["W"].T @ error
    dHidden = error * relu(layers[0]["activation"], d=True)

    layers[0]["dW"] = (1 / N) * dHidden @ X.T + 2 * lmbda * layers[0]["W"]
    layers[0]["db"] = np.mean(dHidden, axis=1, keepdims=True)

    assert layers[0]["dW"].shape == layers[0]["W"].shape
    assert layers[1]["dW"].shape == layers[1]["W"].shape
    assert layers[0]["db"].shape == layers[0]["b"].shape
    assert layers[1]["db"].shape == layers[1]["b"].shape
    return


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
    ax1.plot(range(1, len(history["costs"]) + 1), history["costs"], color=color,
             label=f"""Cost (final: {history["costs"][-1]:.2f})""")
    ax1.plot(range(1, len(history["val_costs"]) + 1), history["val_costs"], color="magenta",
             label=f"""Val Cost (final: {history["val_costs"][-1]:.2f})""")

    ax1.plot(range(1, len(history["losses"]) + 1), history["losses"], color="yellow",
             label=f"""Loss (final: {history["losses"][-1]:.2f})""")
    ax1.plot(range(1, len(history["val_losses"]) + 1), history["val_losses"], color="orange",
             label=f"""Val Loss (final: {history["val_losses"][-1]:.2f})""")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale("log")
    # ax1.set_yscale('symlog', linthresh=0.01)

    # plot the accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(1, len(history["accuracies"]) + 1), history["accuracies"], color=color,
             label=f"""Accuracy (final: {history["accuracies"][-1]:.2f})""")
    ax2.plot(range(1, len(history["val_accuracies"]) + 1), history["val_accuracies"], color="cyan",
             label=f"""Val Accuracy (final: {history["val_accuracies"][-1]:.2f})""")
    ax2.tick_params(axis='y', labelcolor=color)

    # set the limits for y-axis scales
    # ax1.set_ylim([0.1, max(history["losses"]) + 0.5])
    ax2.set_ylim([min(history["accuracies"]) - 0.02, 1])

    # add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    plt.title(f'Cost and Accuracy Plot (l2={hyper["l2"]})')
    plt.show()


def plot_cyclic_lr_and_cost_and_accuracy(history):
    fig, ax1 = plt.subplots()

    # plot the cost
    color = 'tab:red'
    ax1.set_xlabel('Update Steps')
    ax1.set_ylabel('Cost / Loss', color=color)
    ax1.plot(range(1, len(history["costs"]) + 1), history["costs"], color=color,
             label=f"""Cost (final: {history["costs"][-1]:.2f})""")
    ax1.plot(range(1, len(history["val_costs"]) + 1), history["val_costs"], color="magenta",
             label=f"""Val Cost (final: {history["val_costs"][-1]:.2f})""")

    ax1.plot(range(1, len(history["losses"]) + 1), history["losses"], color="yellow",
             label=f"""Loss (final: {history["losses"][-1]:.2f})""")
    ax1.plot(range(1, len(history["val_losses"]) + 1), history["val_losses"], color="orange",
             label=f"""Val Loss (final: {history["val_losses"][-1]:.2f})""")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale("log")
    # ax1.set_yscale('symlog', linthresh=0.01)

    # plot the accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(1, len(history["accuracies"]) + 1), history["accuracies"], color=color,
             label=f"""Accuracy (final: {history["accuracies"][-1]:.2f})""")
    ax2.plot(range(1, len(history["val_accuracies"]) + 1), history["val_accuracies"], color="cyan",
             label=f"""Val Accuracy (final: {history["val_accuracies"][-1]:.2f})""")
    ax2.tick_params(axis='y', labelcolor=color)

    # plot the cyclic learning rate
    ax3 = ax1.twinx()
    color = 'gray'
    # ax3.set_ylabel('Learning Rate', color=color)
    ax3.plot(range(1, len(history["lr"]) + 1), history["lr"], color=color, label=f"""Learning Rate""")
    # ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_yticks([])


    # set the limits for y-axis scales
    # ax1.set_ylim([0.1, max(history["losses"]) + 0.5])
    ax2.set_ylim([min(history["accuracies"]) - 0.02, 1])

    # add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2 + lines3, labels + labels2 + labels3)

    plt.title(f'Cost and Accuracy Plot (l2={hyper["l2"]})')
    plt.show()


def plot_weights(W, max_images=10):
    fig, axs = plt.subplots(2, 5, figsize=(max_images, 4))

    for i in range(len(W)):
        im = np.reshape(W[i, :], (32, 32, 3))
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        im = np.transpose(im, (1, 0, 2))  # permute

        axs[i // 5, i % 5].imshow(im)
        axs[i // 5, i % 5].axis('off')
        axs[i // 5, i % 5].set_title(f'Weight {i + 1}')

        if i >= max_images - 1:
            break

    plt.show()


def InitializeLayers(hidden_nodes: list) -> list:
    layers = []
    for i in range(1, len(hidden_nodes)):
        fan_in = hidden_nodes[i - 1]
        fan_out = hidden_nodes[i]

        W = np.random.normal(0, 1 / np.sqrt(fan_in), (fan_out, fan_in))
        b = np.zeros((fan_out, 1))

        layers.append({"W": W, "b": b})

    return layers


def BuildTmpLayers(Ws: list, bs: list) -> list:
    f = {"W": Ws[0], "b": bs[0]}
    s = {"W": Ws[1], "b": bs[1]}
    return [f, s]


def SGDStep(layers, lr):
    for layer in layers:
        layer["W"] = layer["W"] - lr * layer["dW"]
        layer["b"] = layer["b"] - lr * layer["db"]
    return layers


def CyclicLearningRate(step, cycle, hyper):
    min_peak = 2 * cycle * hyper["eta_step_size"]
    max_peak = (2 * cycle + 1) * hyper["eta_step_size"]

    if min_peak <= step <= max_peak:
        return hyper["eta_min"] + (step - min_peak) / hyper["eta_step_size"] * (hyper["eta_max"] - hyper["eta_min"])
    else:
        return hyper["eta_max"] - (step - max_peak) / hyper["eta_step_size"] * (hyper["eta_max"] - hyper["eta_min"])


def TrainCLF(layers, hyper, X_train, Y_train, y_train, X_val, Y_val, y_val):
    epochs = hyper["epochs"]
    batchesPerEpoch = np.ceil(X_train.shape[1] / hyper["batchSize"]).astype(int)
    history = defaultdict(list)
    update_step = 0

    for j in range(epochs):
        for i, X_batch, Y_batch, y_batch in YieldMiniBatch(X_train, Y_train, y_train, hyper["batchSize"]):
            # dim = 1000
            # X_batch = X_batch[:dim,]
            # layers = InitializeLayers([dim, 50, 10])

            pred = ForwardPass(X_batch, layers, cache=True)

            ComputeGradients(X_batch, Y_batch, layers, hyper["l2"])

            # grad_W, grad_b = ComputeGradsNum(X_batch, Y_batch, layers, hyper["l2"])
            #
            # assert np.allclose(grad_b[1], layers[1]["db"], atol=1e-5)
            # assert np.allclose(grad_W[1], layers[1]["dW"], atol=1e-5)
            #
            # assert np.allclose(grad_b[0], layers[0]["db"], atol=1e-5)
            # assert np.allclose(grad_W[0], layers[0]["dW"], atol=1e-5)

            SGDStep(layers, hyper["lr"])

            update_step += 1


        # Train data
        cost, loss = ComputeCost(X_train, Y_train, layers, hyper["l2"])
        acc = ComputeAccuracy(X_train, y_train, layers)
        history["costs"].append(cost)
        history["losses"].append(loss)
        history["accuracies"].append(acc)

        # Val data
        cost_val, loss_val = ComputeCost(X_val, Y_val, layers, hyper["l2"])
        acc_val = ComputeAccuracy(X_val, y_val, layers)
        history["val_costs"].append(cost_val)
        history["val_losses"].append(loss_val)
        history["val_accuracies"].append(acc_val)

        print(
            f"""Epoch {j + 1}/{epochs}: Update Steps={update_step}: Cost={cost:8.4f} ; Acc={acc:8.4f} ; Val Cost={cost_val:8.4f} ; Val Acc={acc_val:8.4f}""")

    return layers, history


def CyclicTrainCLF(layers, hyper, X_train, Y_train, y_train, X_val, Y_val, y_val):
    history = defaultdict(list)
    update_step = 0

    for cycle in range(hyper["cycles"]):
        nextCycle = False
        while not nextCycle:
            for i, X_batch, Y_batch, y_batch in YieldMiniBatch(X_train, Y_train, y_train, hyper["batchSize"]):
                pred = ForwardPass(X_batch, layers, cache=True)

                ComputeGradients(X_batch, Y_batch, layers, hyper["l2"])

                update_step += 1
                lr = CyclicLearningRate(update_step, cycle, hyper)
                SGDStep(layers, lr)

                if update_step % (2 * hyper["eta_step_size"]) == 0:
                    nextCycle = True
                    break

                if update_step % hyper["summary_step_size"] == 0:
                    history["lr"].append(lr)

                    # Train data
                    cost, loss = ComputeCost(X_train, Y_train, layers, hyper["l2"])
                    acc = ComputeAccuracy(X_train, y_train, layers)
                    history["costs"].append(cost)
                    history["losses"].append(loss)
                    history["accuracies"].append(acc)

                    # Val data
                    cost_val, loss_val = ComputeCost(X_val, Y_val, layers, hyper["l2"])
                    acc_val = ComputeAccuracy(X_val, y_val, layers)
                    history["val_costs"].append(cost_val)
                    history["val_losses"].append(loss_val)
                    history["val_accuracies"].append(acc_val)

                    print(f"""Epoch {cycle + 1}/{hyper["cycles"]}: Update Steps={update_step}: Cost={cost:8.4f} ; Acc={acc:8.4f} ; Val Cost={cost_val:8.4f} ; Val Acc={acc_val:8.4f}""")

    return layers, history


def param_search(hyper):
    l2s = []
    for l2 in np.linspace(hyper["search_l2_min"], hyper["search_l2_max"], hyper["search_steps"]):
        layers = InitializeLayers(hidden_nodes)
        hyper["l2"] = l2
        layers, history = CyclicTrainCLF(layers, hyper, X_train, Y_train, y_train, X_val, Y_val, y_val)
        l2s.append((l2, history["val_accuracies"][-1]))

    l2s.sort(key=lambda x: x[1], reverse=True)

    for (l2, acc) in l2s:
        print(f"{l2:.8f}: {acc}")

    hyper["l2"] = l2s[0][0]
    return hyper, l2s


D = 3072  # dimensionality
K = 10  # classes
hidden_nodes = [D, 50, K]

X, Y, y = load_all_batches()

X_train, Y_train, y_train, X_val, Y_val, y_val = train_test_split(X, Y, y, 1000)

X_test, Y_test, y_test = load_batch("test_batch")

N = X_train.shape[1]  # samples
assert X_train.shape == (D, N)
assert Y_train.shape == (K, N)
assert y_train.shape == (N,)

Xmean = X_train.mean(axis=1).reshape(-1, 1)
Xstd = X_train.std(axis=1).reshape(-1, 1)

assert Xmean.shape == (D, 1)
assert Xstd.shape == (D, 1)

X_train = normalize(X_train, Xmean, Xstd)
X_val = normalize(X_val, Xmean, Xstd)
X_test = normalize(X_test, Xmean, Xstd)


hyper = {
    # "epochs": 40,
    "batchSize": 100,
    # "l2": 0.00005560,
    "l2": 0.01,
    # "lr": 0.003,
    "cycles": 1,
    "eta_min": 1e-5,
    "eta_max": 1e-1,
    "eta_step_size": 500,  # n_s = 2 floor(n / n batch)
    "search_l2_min": 1e-7,
    "search_l2_max": 1e-4,
    "search_steps": 10,
    "summary_step_size": 100,
}


# hyper, search_results = param_search(hyper)

layers = InitializeLayers(hidden_nodes)
layers, history = CyclicTrainCLF(layers, hyper, X_train, Y_train, y_train, X_val, Y_val, y_val)

# plot cyclic lr
# plt.plot(range(len(history["lr"])), history["lr"])
# # plt.yscale("symlog")
# plt.show()

plot_cyclic_lr_and_cost_and_accuracy(history)
# plot_cost_and_accuracy(history)

# Test data
acc_test = ComputeAccuracy(X_test, y_test, layers)

# plot_weights(layers[0]["W"])
# plot_weights(layers[1]["W"])


print(f"Final test acc {acc_test}")
