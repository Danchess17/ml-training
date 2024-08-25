import os 
import shutil
import random
import pathlib

image_dir = 'Pascal-part/JPEGImages'
mask_dir = 'Pascal-part/gt_masks'

def get_lens(image_dir, mask_dir):
    return len(os.listdir(image_dir)), len(os.listdir(mask_dir))

def check(image_dir, mask_dir):
    i, m = len(os.listdir(image_dir)), len(os.listdir(mask_dir))
    with open( 'Pascal-part/train_id.txt', 'r') as trainfile:
        with open( 'Pascal-part/val_id.txt', 'r') as valfile:
            t, v = len(trainfile.readlines()), len(valfile.readlines())
    return i, m, t, v, (i == m == t + v)

print(get_lens(image_dir, mask_dir))
print(check(image_dir, mask_dir))
# i need to split train into train/test and do it with masks too

# i took first 707 test images 

def GenerateDatasetFolder():
    pathlib.Path('dataset/images').mkdir(parents=True, exist_ok=True) 
    pathlib.Path('dataset/masks').mkdir(parents=True, exist_ok=True) 

    with open('Pascal-part/train_id.txt', 'r') as t:
        lines = t.readlines()
        # random.shuffle(lines)
        test_ids = lines[:707]
        train_ids = lines[707:]
        print(len(test_ids), len(train_ids))
        with open('dataset/images/test.txt', 'w') as test_file:
            test_file.writelines(test_ids)
        with open('dataset/images/train.txt', 'w') as train_file:
            train_file.writelines(train_ids)
    
    with open('Pascal-part/val_id.txt', 'r') as v:
        lines = v.readlines()
        with open('dataset/images/validation.txt', 'w') as test_file:
            test_file.writelines(lines)

def DeleteDataset():
    shutil.rmtree('dataset')

GenerateDatasetFolder()
# DeleteDataset()
