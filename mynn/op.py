from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True # whether this layer has parameters to be optimized
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass

class Linear(Layer):
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        assert self.input.shape[0] == grad.shape[0], "The batch size of input and grad should be the same."
        assert self.input.shape[1] == self.W.shape[0], "The input dimension should be equal to the weight dimension."

        self.grads['W'] = np.dot(self.input.T, grad) / grad.shape[0]
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / grad.shape[0] 
        return np.dot(grad, self.W.T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = initialize_method(size=(out_channels, in_channels, *kernel_size))
        self.b = initialize_method(size=(out_channels,))

        self.grads = {'W': np.zeros_like(self.W), 'b': np.zeros_like(self.b)}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch_size, _, H, W = X.shape
        kh, kw = self.kernel_size
        oh = (H + 2 * self.padding - kh) // self.stride + 1
        ow = (W + 2 * self.padding - kw) // self.stride + 1

        if self.padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output = np.zeros((batch_size, self.out_channels, oh, ow))
        for i in range(oh):
            for j in range(ow):
                h, w = i * self.stride, j * self.stride
                window = X[:, :, h:h+kh, w:w+kw]  # (batch, in_ch, kh, kw)
                output[:, :, i, j] = np.einsum('nchw,ochw->no', window, self.W) + self.b
        return output

    def backward(self, grad):
        batch_size, _, oh, ow = grad.shape
        _, _, H, W = self.input.shape
        kh, kw = self.kernel_size

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.input)

        X = self.input
        if self.padding > 0:
            X = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            dX_padded = np.zeros_like(X)
        else:
            dX_padded = dX

        for i in range(oh):
            for j in range(ow):
                h, w = i * self.stride, j * self.stride
                window = X[:, :, h:h+kh, w:w+kw]  # shape: (batch, in_ch, kh, kw)
                grad_slice = grad[:, :, i, j]    # shape: (batch, out_ch)
                dW += np.einsum('no,nchw->ochw', grad_slice, window)
                dX_padded[:, :, h:h+kh, w:w+kw] += np.einsum('no,ochw->nchw', grad_slice, self.W)
                db += np.sum(grad_slice, axis=0)

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        self.grads['W'] = dW / batch_size
        self.grads['b'] = db / batch_size
        return dX

    def clear_grad(self):
        self.grads['W'].fill(0)
        self.grads['b'].fill(0)
      
class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.output = None
        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
        return self.output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = grads * self.output * (1 - self.output)
        return output

class MultiCrossEntropyLoss(Layer):
    def __init__(self, model=None, max_classes=10, has_softmax=True) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = has_softmax
        self.probs = None        # Softmax probabilities
        self.labels = None       # Ground truth labels
        self.predicts = None     # Raw predictions (logits)
        self.optimizable = False
        
    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        self.labels = labels
        self.predicts = predicts
        
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            logits = predicts - np.min(predicts, axis=1, keepdims=True)
            self.probs = logits / np.sum(logits, axis=1, keepdims=True)
        
        batch_size = predicts.shape[0]
        true_class_probs = self.probs[np.arange(batch_size), labels]
        
        true_class_probs = np.clip(true_class_probs, 1e-12, 1.0)
        
        return -np.mean(np.log(true_class_probs))
    
    def backward(self):
        batch_size, num_classes = self.probs.shape
        
        one_hot = np.zeros_like(self.predicts)
        one_hot[np.arange(batch_size), self.labels] = 1
        
        if self.has_softmax:
            self.grads = (self.probs - one_hot) / batch_size
        else:    
            sum_logits = np.sum(self.predicts, axis=1, keepdims=True) - \
                         np.sum(np.min(self.predicts, axis=1, keepdims=True), axis=1, keepdims=True)
            p_true = self.probs[np.arange(batch_size), self.labels]
            
            term1 = (np.eye(num_classes)[self.labels] - self.probs) / sum_logits
            term2 = 1.0 / np.clip(p_true, 1e-12, 1.0)
            self.grads = -term1 * term2[:, None] / batch_size
        
        if self.model is not None:
            self.model.backward(self.grads)

    def cancel_softmax(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_
        self.weights = None
        self.optimizable = False
    
    def forward(self, weights):
        self.weights = weights
        return 0.5 * self.lambda_ * np.sum(weights ** 2)
    
    def backward(self):
        return self.lambda_ * self.weights
    
    def __call__(self, weights):
        return self.forward(weights)

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

class MaxPool2D(Layer):
    def __init__(self, pool_size, stride=None) -> None:
        super().__init__()
        self.optimizable = False
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = stride if stride is not None else pool_size
        self.input = None

    def forward(self, X):
        self.input = X
        batch_size, channels, height, width = X.shape
        kh, kw = self.pool_size
        oh = (height - kh) // self.stride + 1
        ow = (width - kw) // self.stride + 1

        output = np.empty((batch_size, channels, oh, ow))

        for i in range(oh):
            for j in range(ow):
                h, w = i * self.stride, j * self.stride
                window = X[:, :, h:h+kh, w:w+kw]
                output[:, :, i, j] = np.max(window, axis=(2, 3))

        return output

    def backward(self, grad):
        batch_size, channels, oh, ow = grad.shape
        _, _, height, width = self.input.shape
        kh, kw = self.pool_size

        dX = np.zeros_like(self.input)

        for i in range(oh):
            for j in range(ow):
                h, w = i * self.stride, j * self.stride
                window = self.input[:, :, h:h+kh, w:w+kw]
                mask = (window == np.max(window, axis=(2, 3), keepdims=True))
                dX[:, :, h:h+kh, w:w+kw] += mask * grad[:, :, i:i+1, j:j+1]

        return dX

class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.optimizable = False

    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.p).astype(X.dtype) / (1.0 - self.p)
            return X * self.mask
        else:
            return X

    def backward(self, grad):
        return grad * self.mask

def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition