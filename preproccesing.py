import os
import numpy as np
from PIL import Image

def padding(arr, target_size):
    pad_width_rows = (target_size[0] - arr.shape[0])
    pad_width_cols = (target_size[1] - arr.shape[1])
    pad_width = [
        (pad_width_rows // 2, pad_width_rows - (pad_width_rows // 2)),
        (pad_width_cols // 2, pad_width_cols - (pad_width_cols // 2)),
    ]
    if len(target_size) == 3:
        pad_width += [(0, 0)]
    return np.pad(arr, pad_width, 'constant')


def dataset_max_image_size(image_dir, mode='train'):
    max_height, max_width = 0, 0
    dataset = 'Pascal-part/' + mode + '_id.txt'
    with open(dataset, 'r') as file:
        for filename in file.readlines():
            image_path = os.path.join(image_dir, filename.strip() + '.jpg')
            with Image.open(image_path) as image:
                max_height, max_width = max(max_height, image.height), max(max_width, image.width)
    return max_height, max_width


def dataset_padding(image_dir, mask_dir, mode='train'):
    h, w = dataset_max_image_size(image_dir, mode)
    dataset = 'Pascal-part/' + mode + '_id.txt'
    X, y = [], []
    with open(dataset, 'r') as file:
        for filename in file.readlines():
            image_path = os.path.join(image_dir, filename.strip() + '.jpg')
            mask_path = os.path.join(mask_dir, filename.strip() + '.npy')
            with Image.open(image_path) as image:
                img = np.array(image)
                padded_img = padding(img, (h, w, 3))
                X.append(padded_img)
                with open(mask_path, 'rb') as filemask:
                    mask = np.load(filemask)
                    padded_mask = padding(mask, (h, w))
                    y.append(padded_mask)

    X = np.array(X)
    y = np.array(y)
    # y = to_categorical(y, num_classes=7)
    return X, y

def without_padding(image_dir, mask_dir, mode='train'):
    dataset = 'Pascal-part/' + mode + '_id.txt'
    X, y = [], []
    with open(dataset, 'r') as file:
        for filename in file.readlines():
            image_path = os.path.join(image_dir, filename.strip() + '.jpg')
            mask_path = os.path.join(mask_dir, filename.strip() + '.npy')
            with Image.open(image_path) as image:
                img = np.array(image)
                X.append(img)
                with open(mask_path, 'rb') as filemask:
                    mask = np.load(filemask)
                    y.append(mask)

    X = np.array(X)
    y = np.array(y)
    # y = to_categorical(y, num_classes=7)
    return X, y

def dataset_frequency(image_dir, mode='train'):
    dataset = 'Pascal-part/' + mode + '_id.txt'
    size_count = {}
    with open(dataset, 'r') as file:
        for filename in file.readlines():
            image_path = os.path.join(image_dir, filename.strip() + '.jpg')
            with Image.open(image_path) as img:
                size = img.size 
                if size in size_count:
                    size_count[size] += 1
                else:
                    size_count[size] = 1
                    
    sorted_size_count = dict(sorted(size_count.items(), key=lambda item: item[1], reverse=True))
    return sorted_size_count


# Assuming your images and masks are in separate folders
image_dir = 'Pascal-part/JPEGImages'
mask_dir = 'Pascal-part/gt_masks'
X_train, y_train = dataset_padding(image_dir, mask_dir, mode='train')
X_val, y_val = dataset_padding(image_dir, mask_dir, mode='val')