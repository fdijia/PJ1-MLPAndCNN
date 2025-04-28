from .op import *
import pickle

class Model_MLP(Layer):
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    layer_f = Sigmoid()
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X, training=True):
        return self.forward(X, training=training)

    def forward(self, X, training=True):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    layer_f = Sigmoid()
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    def __init__(self, layers_config=None):
        self.layers = []
        self.layers_config = layers_config
        
        if layers_config is not None:
            self._build_model(layers_config)
    
    def _build_model(self, layers_config):
        for config in layers_config:
            layer_type = config['type']
            
            if layer_type == 'Conv2D':
                layer = Conv2D(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    kernel_size=config['kernel_size'],
                    stride=config.get('stride', 1),
                    padding=config.get('padding', 0),
                    weight_decay=config.get('weight_decay', False),
                    weight_decay_lambda=config.get('weight_decay_lambda', 1e-8)
                )
            elif layer_type == 'MaxPool2D':
                layer = MaxPool2D(
                    pool_size=config['pool_size'],
                    stride=config.get('stride', None)
                )
            elif layer_type == 'Flatten':
                layer = Flatten()
            elif layer_type == 'Dropout':
                layer = Dropout(p=config['p'])
            elif layer_type == 'Linear':
                layer = Linear(
                    in_dim=config['in_dim'],
                    out_dim=config['out_dim'],
                    weight_decay=config.get('weight_decay', False),
                    weight_decay_lambda=config.get('weight_decay_lambda', 0.0)
                )
            elif layer_type == 'ReLU':
                layer = ReLU()
            elif layer_type == 'Sigmoid':
                layer = Sigmoid()
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            self.layers.append(layer)
    
    def __call__(self, X, training=True):
        return self.forward(X, training=training)
    
    def forward(self, X, training=True):
        if not self.layers:
            raise ValueError('Model has not been initialized. Provide layers_config or load a model.')
        
        self.input_shape = X.shape
        output = X
        
        for layer in self.layers:
            if isinstance(layer, Dropout):
                output = layer.forward(output, training=training)
            else:
                output = layer.forward(output)
        
        return output
    
    def backward(self, loss_grad):
        grads = loss_grad
        
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.layers_config = saved_data['config']
        self._build_model(self.layers_config)
        
        # Load parameters into each layer
        param_idx = 0
        for layer in self.layers:
            if layer.optimizable:
                layer.W = saved_data['params'][param_idx]['W']
                layer.b = saved_data['params'][param_idx]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = saved_data['params'][param_idx]['weight_decay']
                layer.weight_decay_lambda = saved_data['params'][param_idx]['weight_decay_lambda']
                param_idx += 1
    
    def save_model(self, save_path):
        saved_data = {
            'config': self.layers_config,
            'params': []
        }
        
        for layer in self.layers:
            if layer.optimizable:
                saved_data['params'].append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'weight_decay_lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(saved_data, f)