import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP()
model.load_model(r'.\best_models\model_7_2.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

mat = model.layers[0].params['W']

# _, axes = plt.subplots(30, 20)
# _.set_tight_layout(1)
# axes = axes.reshape(-1)
# for i in range(600):
#         axes[i].matshow(mats[0].T[i].reshape(28,28))
#         axes[i].set_xticks([])
#         axes[i].set_yticks([])
## 画线性层
plt.figure()
plt.matshow(mat)
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks([])
plt.yticks([])
plt.savefig('figs/weights/model7_2weight1.png')

# ## 画卷积层
# W = mats[0]

# out_channels, in_channels, kh, kw = W.shape
# # 创建画布
# fig, axes = plt.subplots(in_channels, out_channels, figsize=(2*out_channels, 2*in_channels))

# for idy in range(in_channels):
#     for idx in range(out_channels):
#         weight = W[idx, idy, :, :]  # (3,3)
#         ax = axes[idy*out_channels+idx]
#         im = ax.matshow(weight, cmap='gray')
#         ax.set_title(f'Filter in{idy}-out{idx}')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.savefig('figs/weights/cnnmodel3weight.png')