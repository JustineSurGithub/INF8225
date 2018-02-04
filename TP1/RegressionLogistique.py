import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

__author__ = "Justine Pepin, d'apres le code fourni du TP1 listing 2"

digits = datasets.load_digits()

X = digits.data

y = digits.target

y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1  # one hot target or shape NxK
Y = np.identity(len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]))  # Weights of shape KxL

best_W = None
best_accuracy = 0
lr = 0.001
nb_epochs = 500
minibatch_size = len(y) // 20

losses_train = []
losses_validation = []
accuracies = []


def softmax(x):
    # assurez-vous que la fonction est numeriquement stable
    # e.g. softmax(np.array([1000, 10000, 100000], ndim=2))
    return np.exp(x - max(x)) / np.sum(np.exp(x - max(x)))


def get_accuracy(X, y, W):
    sum = 0
    for v in range(0, len(X)):
        y_pred = softmax(np.dot(W, X[v]))
        sum = sum + np.vdot(y[v], y_pred)
    return sum / len(X) * 100  # Pourcentage


def get_grads(y, y_pred, X):
    return np.outer(y, X) - np.outer(y_pred, X)


def get_loss(y, y_pred):
    return np.log(np.vdot(y, y_pred))


def shuffle_arrays(X, y):
    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y


for epoch in range(nb_epochs):
    loss_train = 0
    loss_validation = 0
    accuracy = 0
    for i in range(0, X_train.shape[0], minibatch_size):
        if X_train.shape[0] - i < minibatch_size:
            break
        g = 0
        for mb in range(i, minibatch_size + i):
            xi = np.append(X_train[mb], [1])
            theta = np.append(W, np.ones((10, 1)), axis=1)
            y_pred = softmax(np.dot(theta, xi))
            g = g - get_grads(y_train[mb], y_pred, X_train[mb])
        g = g / minibatch_size
        W = W - lr * g
    X_train, y_train = shuffle_arrays(X_train, y_train)
    # compute the loss on the train set
    for i in range(0, X_train.shape[0]):
        y_pred = softmax(np.dot(W, X_train[i]))
        loss_train = loss_train - get_loss(y_train[i], y_pred)
    loss_train = loss_train / X_train.shape[0]
    losses_train.append(loss_train)
    # compute the accuracy on the validation set
    accuracy = get_accuracy(X_validation, y_validation, W)
    accuracies.append(accuracy)
    if accuracy > best_accuracy:
        # select the best parameters based on the validation accuracy
        best_accuracy = accuracy
        best_W = W
    for i in range(0, X_validation.shape[0]):
        y_pred = softmax(np.dot(W, X_validation[i]))
        loss_validation = loss_validation - get_loss(y_validation[i], y_pred)
    loss_validation = loss_validation / X_validation.shape[0]
    losses_validation.append(loss_validation)

# OUTPUTS
accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)
print('\nLa précision obtenue = ', accuracy_on_unseen_data)  # 0.897506925208
print('\nPour un lr = ', lr)
print('\nAvec une taille de mini-batch = ', minibatch_size)

plt.figure(1)
ax1 = plt.subplot(221)
ax1.plot(losses_train, 'b', losses_validation, 'm')
ax1.set_title('Courbes d\'apprentissage')
ax1.set_ylabel('Log négatif de vraisemblance moyenne')
ax1.set_xlabel('Epoch')
ax1.text(30, .55, 'Entraînement', color='blue')
ax2 = plt.subplot(222)
ax2.plot(accuracies)
ax2.set_title('Précision sur l\'ensemble de validation')
ax2.set_ylabel('Pourcentage')
ax2.set_xlabel('Epoch')
ax1.text(30, .65, 'Validation', color='magenta')
ax3 = plt.subplot(223)
ax3.imshow(best_W[4, :].reshape(8, 8))
ax3.set_title('Poids du chiffre 4')
ax4 = plt.subplot(224)
ax4.imshow(best_W[7, :].reshape(8, 8))
ax4.set_title('Poids du chiffre 7')
plt.show()
