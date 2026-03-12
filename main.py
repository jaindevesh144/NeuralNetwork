import numpy as np
from NeuralNetwork import Layers, utils, Model

class NeuralNetwork(Model):
    def __init__(self):
        self.layers = [
            Layers.Dense(784, 128),
            Layers.Relu(),
            Layers.Dense(128, 64),
            Layers.Relu(),
            Layers.Dense(64, 10),
            Layers.Softmax()
        ]
    
    def predict(self, x):
        return np.argmax(self.forward(x.reshape(1, 784)).flatten())
    
    def evaluate(self, x_test, y_test):
        batch_size = 150

        batches_x = utils.makeBatches(x_test, batch_size)
        batches_y = utils.makeBatches(np.argmax(y_test, axis=1), batch_size)

        batch_num = batches_x.shape[0]

        correct = 0
        
        for batchIndex in range(batch_num):
            batch_x = batches_x[batchIndex]
            batch_y = batches_y[batchIndex]
            y_predicted = np.argmax(self.forward(batch_x), axis=1)
            correct += np.sum(y_predicted == batch_y)

        return correct / (batch_num * batch_size)

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

data = np.array(mnist['data']).reshape(-1, 784)
target = np.array(mnist['target']).astype(int)

x_train = data[:60000] / 255
y_train = utils.batchOneHotEncode(target[:60000])

x_test = data[60000:] / 255
y_test = utils.batchOneHotEncode(target[60000:])


network = NeuralNetwork()

network.train(x_train, y_train, 10, 64, 0.1)

result = network.evaluate(x_test, y_test)
print(f"Test Accuracy: {result * 100:.2f}%")