import glob
import os
import numpy as np
train_data_path = './data/train/data1500/*.bmp'
train_label_path = './data/train/label1500/*.bmp'
val_path = './data/val/'
train_data = glob.glob(train_data_path)
train_label = glob.glob(train_label_path)
val_data_path = './data/val/data1500/*.bmp'
val_label_path = './data/val/label1500/*.bmp'

no_of_images = len(train_data)
shuffle = np.random.permutation(no_of_images)
for i in shuffle[:100]:
    image = train_data[i].split('/')[-1]
    os.rename(train_data[i],os.path.join(val_path,image))
for i in shuffle[:100]:
    image = train_label[i].split('/')[-1]
    os.rename(train_label[i], os.path.join(val_path, image))

val_data_path = './data/val/data1500/*.bmp'
val_label_path = './data/val/label1500/*.bmp'
val_data = glob.glob(train_data_path)
val_label = glob.glob(train_label_path)
for i in range(len(val_data)):
    if val_data[i].split('/')[-1].split('\\')[-1] != val_label[i].split('/')[-1].split('\\')[-1]:
        print('error')