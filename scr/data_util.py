import numpy as np
import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from constants import IMG_HEIGHT,IMG_WIDTH, BATCH_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE

def get_class_names(folder_path):
    class_names = np.array(sorted([item.name for item in folder_path.glob('*')]))
    return class_names
    
def get_file_paths(folder_path):
    class_names = get_class_names(folder_path)
    image_count = len(list(folder_path.glob('*/*.jpg')))
    file_paths = tf.data.Dataset.list_files(str(folder_path/'*/*'), shuffle=False)
    return file_paths
    
def get_label(file_path, folder_path):
  class_names=get_class_names(folder_path)
  parts = tf.strings.split(file_path, os.path.sep)
  one_hot = parts[-2] == class_names
  return tf.argmax(one_hot)
  
def process_image(image, 
                  img_height=IMG_HEIGHT, 
                  img_width=IMG_WIDTH,
                  chan=3):
  
  image = tf.io.decode_jpeg(image, channels=chan)
  image = tf.image.resize(image, [img_height, img_width])
  image = (image / 255.0)
  return image
 
def process_path(file_path, folder_path, color: bool=True):
  label = get_label(file_path, folder_path)
  image = tf.io.read_file(file_path)
  if color:
      image = process_image(image)
  else: image = process_image(image, chan=1)
  return image, label
  
def train_val_test_split(dataset, 
                         dataset_size, 
                         folder_path,
                         train_split=0.8, 
                         test_split=0.1,
                         val_split=0.1,
                         seeed=SEED):
    
    
    dataset = get_file_paths(folder_path)
    dataset = dataset.shuffle(buffer_size=dataset_size, 
                              seed=SEED, 
                              reshuffle_each_iteration=False)
    
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    
    train_ds = dataset.take(train_size)    
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size).skip(val_size)

    print("The train set: " + str(train_ds.__len__()))
    print("The validation set: " + str(val_ds.__len__()))
    print("The test set: " + str(test_ds.__len__()))

    return train_ds, val_ds, test_ds


def get_class_proportions(dataset, num):
    zero, one = 0, 0
    for image, label in dataset.take(num):
        if label.numpy() == 0:
            zero += 1
        elif label.numpy() == 1:
            one += 1
    print("zero: " + str(zero))
    print("one: " + str(one))  

def show_examples(dataset, folder_path, number=2):
    dataset = dataset.take(number)
    iterator = iter(dataset)
    for _ in range(number):
        img, lab = iterator.get_next()
        img = img * 255
        try:
            plt.imshow(img.numpy().astype("uint8"))
        except:
            plt.imshow(img.numpy().astype("uint8")[:,:,0], cmap='gray')
        plt.title(get_class_names(folder_path)[lab])
        plt.show()


def prepare_for_training(dataset, batch_size=BATCH_SIZE):
  dataset = dataset.cache()
  dataset = dataset.batch(batch_size)
  ddatasets = dataset.prefetch(buffer_size=AUTOTUNE)
  return dataset
  
def show_batch_examples(dataset, folder_path):
    image_batch, label_batch = next(iter(dataset))
    image_batch = image_batch * 255
    plt.figure(figsize=(12, 12))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        try:
            plt.imshow(image_batch[i].numpy().astype("uint8"))
        except:
            plt.imshow(image_batch[i].numpy().astype("uint8")[:,:,0], cmap='gray')
        label = label_batch[i]
        plt.title(get_class_names(folder_path)[label])
        plt.axis("off")