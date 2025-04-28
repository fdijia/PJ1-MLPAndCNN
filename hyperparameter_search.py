class MLP:
    def simple_MLP():
        layers_config = [
            {'type': 'Flatten'},
            {
                'type': 'Linear',
                'in_dim': 28 * 28,
                'out_dim': 600,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'ReLU'},
            {
                'type': 'Linear',
                'in_dim': 600,
                'out_dim': 10,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            }
        ]
        return layers_config

    def best_MLP():
        layers_config = [
            {'type': 'Flatten'},
            {
                'type': 'Linear',
                'in_dim': 28 * 28,
                'out_dim': 650,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'ReLU'},
            {'type': 'Dropout', 'p': 0.4},
            {
                'type': 'Linear',
                'in_dim': 650,
                'out_dim': 10,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
        ]
        return layers_config
    
    
class CNN:
    def simplist_CNN():
        layers_config = [
            {
                'type': 'Conv2D',
                'in_channels': 1,
                'out_channels': 6,
                'kernel_size': 3,
                'stride': 1,
                'padding': 0,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'MaxPool2D', 'pool_size': 2, 'stride': 2},

            {'type': 'Flatten'},
            {'type': 'Dropout', 'p': 0.5},
            {
                'type': 'Linear',
                'in_dim': 6 * 13 * 13,
                'out_dim': 10,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },  
        ]
        return layers_config
    
    def simplist_CNN_ReLU():
        layers_config = [
            {
                'type': 'Conv2D',
                'in_channels': 1,
                'out_channels': 6,
                'kernel_size': 3,
                'stride': 1,
                'padding': 0,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'ReLU'},
            {'type': 'MaxPool2D', 'pool_size': 2, 'stride': 2},
            {'type': 'Flatten'},
            {'type': 'Dropout', 'p': 0.5},
            {
                'type': 'Linear',
                'in_dim': 6 * 13 * 13,
                'out_dim': 10,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },  
        ]
        return layers_config
        
    def doubleConv():
        layers_config = [
            {
                'type': 'Conv2D',
                'in_channels': 1,
                'out_channels': 6,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'MaxPool2D', 'pool_size': 2, 'stride': 2},
            
            {
                'type': 'Conv2D',
                'in_channels': 6,
                'out_channels': 16,
                'kernel_size': 3,
                'stride': 1,
                'padding': 0,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'MaxPool2D', 'pool_size': 2, 'stride': 2},

            {'type': 'Flatten'},
            
            {
                'type': 'Linear',
                'in_dim': 16 * 6 * 6,
                'out_dim': 10,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },  
        ]
        return layers_config

    def doubleConvReLU():
        layers_config = [
            {
                'type': 'Conv2D',
                'in_channels': 1,
                'out_channels': 6,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'ReLU'},
            {'type': 'MaxPool2D', 'pool_size': 2, 'stride': 2},

            {
                'type': 'Conv2D',
                'in_channels': 6,
                'out_channels': 16,
                'kernel_size': 3,
                'stride': 1,
                'padding': 1,
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'ReLU'},
            {'type': 'MaxPool2D', 'pool_size': 2, 'stride': 2},

            {'type': 'Flatten'},

            {
                'type': 'Linear',
                'in_dim': 16 * 7 * 7,  # 16x7x7=784
                'out_dim': 64,  # 小的隐层，64维
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
            {'type': 'ReLU'},

            {
                'type': 'Linear',
                'in_dim': 64,
                'out_dim': 10,  # 最后输出10类
                'weight_decay': True,
                'weight_decay_lambda': 1e-4,
            },
        ]

        return layers_config