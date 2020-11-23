import cv2
import numpy as np
import dlib
import make_noice
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midPoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


count = 0
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0))
        landmark = predictor(gray, face)

        left_point = (landmark.part(36).x, landmark.part(36).y)
        right_point = (landmark.part(39).x, landmark.part(39).y)
        center_top = midPoint(landmark.part(37), landmark.part(38))
        center_bottom = midPoint(landmark.part(41), landmark.part(40))

        hor_line = cv2.line(frame, left_point, right_point, (0, 0, 255), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        # check length of horizontal line
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        # find ration of horizontal line length and vertical line length
        ratio = hor_line_length / ver_line_length

        # check eye is open or close
        if ratio > 5.7:
            print('Eye is closed- ', count)
            count += 1
            if count > 5:
                make_noice.sStart()
                count = 0
        elif ratio < 4:
            print('Eye is open')

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
