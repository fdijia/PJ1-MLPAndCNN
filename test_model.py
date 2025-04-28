import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle


def load_mnist_data():
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)        
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)
    test_imgs = test_imgs / test_imgs.max()
    return test_imgs, test_labs

def load_augmented_mnist_data():
    test_images_path = r'.\dataset\augmented_mnist_images.npy'
    test_labels_path = r'.\dataset\augmented_mnist_labels.npy'
    test_imgs = np.load(test_images_path)
    test_labs = np.load(test_labels_path)
    return test_imgs, test_labs

model = nn.models.Model_CNN()
model.load_model(r'.\best_models\myBestMLP.pickle')
test_imgs, test_labs = load_augmented_mnist_data()
logits = model(test_imgs, training=False)
print(nn.metric.accuracy(logits, test_labs))