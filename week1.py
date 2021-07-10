import cv2

cap = cv2.VideoCapture("03.mp4")

while True:
    ret, img = cap.read()
    if ret == False:
        break

    cropped_img = img[183:465, 721:878]
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("result", cropped_img)

    if cv2.waitKey(10) == ord('q'):
        break
