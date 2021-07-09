import cv2
import tensorflow as tf
import dlib


img = cv2.imread('01.jpg')
print(img)
print(img.shape)

cv2.rectangle(img, pt1=(259, 89), pt2=(380, 348),
              color=(255, 0, 0), thickness=2)
cv2.circle(img, center=(320, 220), radius=100, color=(0, 0, 255), thickness=3)
cropped_img = img[89:384, 259:380]  # y axis followed by x axis

img_resized = cv2.resize(img, (512, 256))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', img_gray)
cv2.imshow("resized", img_resized)
cv2.imshow("image", img)
cv2.imshow("crop", cropped_img)
cv2.waitKey(0)
