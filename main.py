import numpy as np
from NeuralNetwork import Layers, utils, Model

class NeuralNetwork(Model):
    def __init__(self):
        self.layers = [
            Layers.Dense(784, 128),
            Layers.Relu(),
            Layers.Dense(128, 64),
            Layers.Relu(),
            Layers.Dense(64, 10)
        ]
    
    def predict(self, x):
        return np.argmax(self.forward(x.reshape(1, 784)).flatten())
    
    def test(self, x_train, y_train):
        batch_size = 150

        batches_x = utils.makeBatches(x_train, batch_size)
        batches_y = utils.makeBatches(np.argmax(y_train, axis=1), batch_size)

        batch_num = batches_x.shape[0]

        correct = 0
        
        for batchIndex in range(batch_num):
            batch_x = batches_x[batchIndex]
            batch_y = batches_y[batchIndex]
            y_predicted = np.argmax(self.forward(batch_x), axis=1)
            correct += batch_size - np.count_nonzero(batch_y - y_predicted)
        return correct / x_train.shape[0]

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

data = np.array(mnist['data']).reshape(-1, 784)
target = np.array(mnist['target']).astype(int)

x_train = data[:60000] / 255
y_train = utils.batchOneHotEncode(target[:60000])

x_test = data[60000:] / 255
y_test = utils.batchOneHotEncode(target[60000:])


network = NeuralNetwork()

network.train(x_train, y_train, 10)

result = network.test(x_test, y_test)
print(result)