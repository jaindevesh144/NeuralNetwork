import numpy as np

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