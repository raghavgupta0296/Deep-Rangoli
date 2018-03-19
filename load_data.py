import cv2
import os
import numpy as np


def load(dataset_path="./deep rangoli dataset/"):
    print("reading image dataset")
    classes = os.listdir(dataset_path)
    classes.remove("deleted pics")

    images = []

    for class_i in classes:
        for im in os.listdir(dataset_path + class_i):
            i = cv2.imread(dataset_path + class_i + "/" + im, 0)
            i = cv2.Canny(i, 60,120)
            i = cv2.resize(i, (200,200))
            i = np.expand_dims(i,-1)
            try:
                i = (i - np.min(i)) / (np.max(i) - np.min(i)) * 2 - 1
                # i = i.tolist()
                # images = images + [i]
                images.append(i)
            except:
                print("Error here")
                print(len(images))
                print(dataset_path + class_i + "/" + im)
                exit()

    print("Read ", len(images), " images")
    return images
