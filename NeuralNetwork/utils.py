import numpy as np

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def batchOneHotEncode(batch_y):
    batch_size = batch_y.shape[0]
    return np.eye(10)[batch_y].reshape(batch_size, 10)

def makeBatches(data, batchSize):
    batchNum = data.shape[0] // batchSize
    dataDimensions = data.shape[1:]
    return data[:batchNum*batchSize].reshape(-1, batchSize, *dataDimensions)

def getPatches(img, kernelShape):
    kh, kw = kernelShape
    paddedImg = np.pad(img, ((kh//2, kh//2), (kw//2, kw//2)))
    return np.lib.stride_tricks.as_strided(paddedImg, img.shape + kernelShape, 2 * paddedImg.strides)

def convolve2d(img, kernels):
    patches = getPatches(img, kernels.shape[1:])
    output = np.tensordot(patches, kernels, axes=((2, 3), (1, 2)))
    return np.moveaxis(output, -1, 0)