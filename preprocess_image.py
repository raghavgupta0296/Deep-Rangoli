import numpy as np
import cv2
import os
import imutils
import random


def preprocess_image(ims):
    new_ims = []
    ims = np.array(ims)
    for im in ims:
        im = np.squeeze(im,-1)
        random_angle = int(random.random()*360)
        # rotate by fine angles or by 90x?
        # print(random_angle)
        rows, cols = im.shape[0], im.shape[1]
        # 1st way of rotation
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random_angle, 1)
        dst = cv2.warpAffine(im, M, (cols, rows))
        # 2nd way of rotation
        # dst = imutils.rotate_bound(im,random_angle)

        if random.choice([0,1])==1:  # flip image if 1
            dst = cv2.flip(dst,random.choice([0,1,-1]))

        # for binary
        # dst = cv2.Canny(dst,80,120)

        # print("dst shape",dst.shape)

        # cv2.imshow("", dst)
        # cv2.waitKey(0)
        new_ims.append(np.expand_dims(dst,-1))
    new_ims = np.array(new_ims)
    return new_ims

# if __name__=="__main__":
    # im = cv2.imread("pic_006.jpg",0)
    # im = cv2.imread("pic_098.jpg",0)
    # preprocess_image(im)
