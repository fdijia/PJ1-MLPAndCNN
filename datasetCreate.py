import numpy as np
import gzip
from struct import unpack
from scipy.ndimage import rotate, shift, zoom

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)


train_imgs = train_imgs.astype(np.float32) / 255.0


def augment_image(image):
    img = image[0]
    
    # 1. 随机旋转
    angle = np.random.uniform(-5, 5)
    img = rotate(img, angle, reshape=False, mode='nearest')
    
    # 2. 随机平移
    shift_x = np.random.uniform(-2, 2)
    shift_y = np.random.uniform(-2, 2)
    img = shift(img, shift=(shift_x, shift_y), mode='nearest')

    # 3. 随机缩放
    zoom_factor = np.random.uniform(0.9, 1.1)
    img_zoomed = zoom(img, zoom=zoom_factor)

    # zoom后大小不是28x28了，需要裁剪或者填充回28x28
    if img_zoomed.shape[0] > 28:
        # 如果放大了，就中心裁剪
        start = (img_zoomed.shape[0] - 28) // 2
        img = img_zoomed[start:start+28, start:start+28]
    else:
        # 如果缩小了，就在周围补0
        pad_total = 28 - img_zoomed.shape[0]
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        img = np.pad(img_zoomed, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')

    return img[np.newaxis, :, :]


target_num_images = 200000
new_images = []
new_labels = []

new_images.extend(train_imgs)
new_labels.extend(train_labs)

while len(new_images) < target_num_images:
    idx = np.random.randint(0, train_imgs.shape[0])
    img = train_imgs[idx]
    label = train_labs[idx]

    aug_img = augment_image(img)

    new_images.append(aug_img)
    new_labels.append(label)

    if len(new_images) % 10000 == 0:
        print(f"当前样本数: {len(new_images)}")

# 转成 NumPy数组
new_images = np.stack(new_images, axis=0)  # shape: (数量, 1, 28, 28)
new_labels = np.array(new_labels)

print(f"最终数据形状: {new_images.shape}, {new_labels.shape}")

# 保存到本地
np.save(r'.\dataset\augmented_mnist_images.npy', new_images)
np.save(r'.\dataset\augmented_mnist_labels.npy', new_labels)
