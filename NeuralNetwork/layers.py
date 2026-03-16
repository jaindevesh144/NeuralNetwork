import numpy as np
import NeuralNetwork.utils as utils




class Dense:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.B = np.zeros((1, out_features))

    def forward(self, x):
        # x.shape = (batch, in_features)
        self.x = x
        y = x @ self.W + self.B
        return y

    def backward(self, grad, lr=0.01):
        # grad.shape = (batch, out_features)
        # dW.shape = (in_features, out_features)
        batchSize = grad.shape[0]
        dW = self.x.T @ grad / batchSize
        dB = np.sum(grad, axis=0, keepdims=True) / batchSize
        dx = grad @ self.W.T

        self.W -= dW * lr
        self.B -= dB * lr

        return dx




class Conv2D:
    def __init__(self, kernelNum, kernalShape):
        self.kernelShape = kernalShape
        self.kernels = np.random.randn(kernelNum, *kernalShape)

    def forward(self, imgs):
        self.x = imgs
        kh, kw = self.kernelShape

        paddedImg = np.pad(imgs, (
            (0,0),
            (kh // 2, kh//2),
            (kw // 2, kw // 2)
        ), mode="constant")
        patches = np.lib.stride_tricks.as_strided(
            paddedImg,
            imgs.shape + self.kernelShape,
            paddedImg.strides + paddedImg.strides[1:]
        )

        convolvedImgs = np.moveaxis(np.tensordot(patches, self.kernels, ((3, 4), (1, 2))), 3, 1)
        return convolvedImgs

    def backward(self, grad, lr):
        pass




class MaxPool2D:
    def __init__(self, kernelShape):
        self.kernelShape = kernelShape

    def forward(self, images):
        flattenedImages = images.reshape(-1, *images.shape[2:])
        kernelShape = self.kernelShape
        kh, kw = kernelShape
        outImageShape = (
            flattenedImages.shape[1] - (kh // 2) * 2,
            flattenedImages.shape[2] - (kw // 2) * 2
        )

        patches = np.lib.stride_tricks.as_strided(
            flattenedImages, (
                flattenedImages.shape[0],
                *outImageShape
            ) + kernelShape,
            flattenedImages.strides + flattenedImages.strides[1:]
        )

        outImages = np.max(patches, axis=(3,4), keepdims=True)
        self.mask = (outImages == patches).astype(int)
        # self.mask = np.lib.stride_tricks.as_strided(
        #     self.mask,
        #     images.shape,
        #     self.mask.strides[:3]
        # )
        outImages = outImages.reshape(*images.shape[:-2], *outImageShape)

        return outImages




# Leaky Relu implementation. Standard relu kills the neurons after multiple epochs
class Relu:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, x * self.alpha)

    def backward(self, grad, lr):
        # Gradient is 1 where x > 0, and alpha (0.01) where x <= 0
        dx = np.ones_like(self.x)
        dx[self.x <= 0] = self.alpha
        return grad * dx




class Softmax:
    def forward(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out

    def backward(self, grad, lr):
        s = self.out
        dot = np.sum(grad * s, axis=1, keepdims=True)
        return s * (grad - dot)
