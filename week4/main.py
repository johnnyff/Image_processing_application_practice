import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'week4/models/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture(
    'C:/Users/Dong-In Kang/Desktop/2021/Image_processing/week4/videos/01.mp4')
sticker_img = cv2.imread("week4/stickers/glasses.png", cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)
    print("number of faces detected:", len(dets))

    for det in dets:
        shape = predictor(img, det)
        try:
            glasses_x1 = shape.parts()[2].x - 20
            glasses_x2 = shape.parts()[0].x + 20

            h, w, c = sticker_img.shape

            glasses_w = glasses_x2 - glasses_x1
            glasses_h = int(h / w * glasses_w)

            center_y = (shape.parts()[0].y + shape.parts()[2].y) / 2

            glasses_y1 = int(center_y - glasses_h / 2)
            glasses_y2 = glasses_y1 + glasses_h

            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2),
            #               color=(0, 255, 0), thickness=2)

            overlay_img = sticker_img.copy()

            overlay_img = cv2.resize(overlay_img, dsize=(glasses_w, glasses_h))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            img[glasses_y1:glasses_y2, glasses_x1:glasses_x2] = overlay_alpha * overlay_img[:,
                                                                                            :, :3] + background_alpha * img[glasses_y1:glasses_y2, glasses_x1:glasses_x2]
        except:
            pass
    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break
