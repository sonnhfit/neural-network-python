import sys
#sys.path.append('..')

from neural_network.network import Network
from layers.fc_layer import FCLayer
from layers.activation_layer import ActivationLayer
import numpy as np

# activation function and its derivative
def act(x):
    return np.tanh(x);


def act_prime(x):
    return 1-np.tanh(x)**2;


# loss function and its derivative
def loss(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));


def loss_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]]);
y_train = np.array([[[0]], [[1]], [[1]], [[0]]]);

# network
net = Network();
net.add(FCLayer((1,2), (1,3)));
net.add(ActivationLayer((1,3), act, act_prime));
net.add(FCLayer((1,3), (1,1)));
net.add(ActivationLayer((1,1), act, act_prime));

# train
net.use(loss, loss_prime);
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1);

# test
out = net.predict(x_train);
print(out);