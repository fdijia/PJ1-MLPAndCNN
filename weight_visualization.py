import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

## 画线性层
def draw_linears(mat, save_name):
    plt.figure()
    plt.matshow(mat)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('figs/weights/' + save_name + '.png')
    plt.show()


# ## 画卷积层
def draw_convs(W, save_name):

    out_channels, in_channels, kh, kw = W.shape
    fig, axes = plt.subplots(in_channels, out_channels, figsize=(2*out_channels, 2*in_channels))

    for idy in range(in_channels):
        for idx in range(out_channels):
            weight = W[idx, idy, :, :]  # (3,3)
            ax = axes[idy*out_channels+idx]
            im = ax.matshow(weight, cmap='gray')
            ax.set_title(f'Filter in{idy}-out{idx}')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig('figs/weights/' + save_name + '.png')
        plt.show()
        
model = nn.models.Model_CNN()
model.load_model(r'.\best_models\bestCNN2.pickle')
mat = model.layers[0].params['W']
draw_convs(mat, save_name='bestCNN2')
