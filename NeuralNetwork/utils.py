import numpy as np

def CrossEntropyDerivative(softmaxOutput, target):
    eps = 1e-9
    softmaxOutput = np.clip(softmaxOutput, eps, 1 - eps)
    return -target / softmaxOutput

def batchOneHotEncode(batch_y):
    batch_size = batch_y.shape[0]
    return np.eye(10)[batch_y].reshape(batch_size, 10)

def makeBatches(data, batchSize):
    batchNum = data.shape[0] // batchSize
    dataDimensions = data.shape[1:]
    return data[:batchNum*batchSize].reshape(-1, batchSize, *dataDimensions)
