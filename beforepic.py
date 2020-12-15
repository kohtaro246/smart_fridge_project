from PIL import Image
import cv2
import os
import re
import numpy as np
import datetime


def capture_camera(mirror=True, size=None):
    """Capture video from camera"""
    # print("here")
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        # success?
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(960, 540))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # print(frame.shape)
        if mirror is False:
            frame = frame[:, ::-1]

        # resize frame
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # show frame
        cv2.imshow('take picture', frame)

        k = cv2.waitKey(1)  # wait 1 ms
        if k == 27:  # use ESC to end
            print("Exiting")
            break
        elif k == 99:  # press c to capture
            print("ok?")
            j = cv2.waitKey(0)
            if j == 121:  # press y to end
                print("picture taken")
                break
            elif k == 110:  # press n to continue
                continue
    now = datetime.datetime.now()
    now = now.strftime('%Y_%m_%d_%H_%M_%S')
    cv2.imwrite("./before_pic/" + now + ".jpg", frame)
    cap.release()
    cv2.destroyAllWindows()
    return now


# capture_camera()
