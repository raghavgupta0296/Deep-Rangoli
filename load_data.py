import cv2
import os
import numpy as np


def load(dataset_path="./deep rangoli dataset/"):
    print("reading image dataset")
    classes = os.listdir(dataset_path)
    # classes.remove("deleted pics")

    images = []

    for class_i in classes:
        for im in os.listdir(dataset_path + class_i):
            i = cv2.imread(dataset_path + class_i + "/" + im, 0)
            # i = cv2.GaussianBlur(i,(3,3),0)
            # auto-canny
            # v = np.median(i)
            # sigma = 0.33
            # lower = int(max(0, (1.0 - sigma) * v))
            # upper = int(min(255, (1.0 + sigma) * v))
            # i = cv2.Canny(i, lower, upper)
            i = cv2.resize(i, (189,189))
            try:
                os.mkdir("./cannied/"+class_i)
            except:
                pass
            cv2.imwrite("./cannied/"+class_i+"/"+im, i)
            i = np.expand_dims(i,-1)
            try:
                i = (i / 255.0) * 2 - 1
                images.append(i)
            except:
                print("Error here")
                print(len(images))
                print(dataset_path + class_i + "/" + im)
                exit()

    print("Read ", len(images), " images")
    return np.array(images)

load()
