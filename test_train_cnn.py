import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import hyperparameter_search as paraSearch
import os

def load_mnist_data():
    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)
    if os.path.exists('idx.pickle'):
        with open('idx.pickle', 'rb') as f:
            idx = pickle.load(f)
    else:
        np.random.seed(309)
        idx = np.random.permutation(np.arange(num))
        with open('idx.pickle', 'wb') as f:
            pickle.dump(idx, f)
    # normalize from [0, 255] to [0, 1]
    train_imgs = train_imgs / train_imgs.max()
    valid_imgs = valid_imgs / valid_imgs.max()
    return train_imgs[idx], train_labs[idx]
        
        
def load_augmented_mnist_data():
        train_images_path = r'.\dataset\augmented_mnist_images.npy'
        train_labels_path = r'.\dataset\augmented_mnist_labels.npy'
        train_imgs = np.load(train_images_path)
        train_labs = np.load(train_labels_path)
        if not os.path.exists('myidx.pickle'):
            with open('myidx.pickle', 'rb') as f:
                idx = pickle.load(f)
        else:
            np.random.seed(309)
            idx = np.random.permutation(np.arange(train_labs.shape[0]))
            with open('myidx.pickle', 'wb') as f:
                pickle.dump(idx, f)
        return train_imgs[idx], train_labs[idx]      

train_imgs, train_labs = load_mnist_data()
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

cnn_model = nn.models.Model_CNN(layers_config=paraSearch.CNN.simplist_CNN())

optimizer = nn.optimizer.MomentGD(init_lr=3.0, model=cnn_model, beta=0.9)
scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, batch_size=50)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=8, log_iters=10, save_dir=r'./best_models', save_name='bestCNN1.pickle')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.tight_layout()
plot(runner, axes)
plt.savefig('./figs/bestCNN1.png', dpi=300, bbox_inches='tight')