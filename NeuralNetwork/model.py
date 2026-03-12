import NeuralNetwork.utils as utils

class Model:
    def forward(self, batch_x):
        y_predicted = batch_x
        for layer in self.layers:
            y_predicted = layer.forward(y_predicted)
        return y_predicted
    
    def backprop(self, batch_x, batch_y, lr):
        y_predicted = self.forward(batch_x)

        grad = utils.CrossEntropyDerivative(y_predicted, batch_y)

        for layer in self.layers[::-1]:
            grad = layer.backward(grad, lr)

    def train(self, x_train, y_train, epoch=5, batch_size=128, learningRate=0.1):
        # Aranging the data according to batch size
        batches_x = utils.makeBatches(x_train, batch_size)
        batches_y = utils.makeBatches(y_train, batch_size)

        batch_num = batches_x.shape[0]

        # Running backprop for every batch
        for currentEpoch in range(epoch):
            for batchIndex in range(batch_num):
                batch_x = batches_x[batchIndex]
                batch_y = batches_y[batchIndex]
                effectiveLearingRate = learningRate * (0.95 ** currentEpoch)
                self.backprop(batch_x, batch_y, effectiveLearingRate)
            print(f"Epoch {currentEpoch + 1}")

