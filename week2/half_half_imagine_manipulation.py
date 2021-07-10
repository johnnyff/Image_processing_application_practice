import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/eccv16/la_muse.t7')
net2 = cv2.dnn.readNetFromTorch("models/instance_norm/candy.t7")
img = cv2.imread("imgs/03.jpg")

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))


MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

net.setInput(blob)
net2.setInput(blob)
output = net.forward()
output2 = net2.forward()

output = output.squeeze().transpose((1, 2, 0))
output2 = output2.squeeze().transpose((1, 2, 0))

output += MEAN_VALUE
output2 += MEAN_VALUE

output = np.clip(output, 0, 255)
output2 = np.clip(output2, 0, 255)

output = output.astype('uint8')
output2 = output2.astype('uint8')

output = output[:, :250]
output2 = output2[:, 250:]

result = np.concatenate([output, output2], axis=1)
cv2.imshow('output', output)
cv2.imshow('output2', output2)

cv2.imshow("result", result)
cv2.waitKey(0)
