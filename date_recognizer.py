# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import re
import datetime


def date_recognizer(frame):

    # load the example image and convert it to grayscale
    image = frame
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to preprocess the
    # image

    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise

    gray = cv2.medianBlur(gray, 3)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(image, lang='eng',
                                       config='digits')
    os.remove(filename)
    # print(text)
    result = re.findall(r'[0-9]+\.[0-9]+\.[0-9]+', text)
    date = ""
    # print(result)
    if(len(result) == 1):

        tmp = result[0].split('.')
        if((len(tmp[0]) == 4 and tmp[0][0] == 2 and tmp[0][1] == 0) or (len(tmp[0]) == 2)):
            flag = 1
            if(len(tmp[0]) == 2):
                tmp[0] = "20" + tmp[0]
            if(len(tmp[1]) == 1):
                tmp[1] = "0" + tmp[1]
            if(len(tmp[2]) == 1):
                tmp[2] = "0" + tmp[2]
            #date = tmp[0] + "/" + tmp[1] + "/" + tmp[2]
            date = datetime.date(int(tmp[0]), int(tmp[1]), int(tmp[2]))
            print(date)
        else:
            flag = 0
    else:
        flag = 0
        print("Searching")
    return flag, date
    # show the output images
    #cv2.imshow("Image", image)
    #cv2.imshow("Output", gray)
    # cv2.waitKey(0)


def capture_camera(mirror=True, size=None):
    """Capture video from camera"""

    cap = cv2.VideoCapture(1)

    while True:
        flag = 0
        date = ""
        # success?
        ret, frame = cap.read()

        if mirror is False:
            frame = frame[:, ::-1]

        # resize frame
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # show frame
        cv2.imshow('date recognizer', frame)

        flag, date = date_recognizer(frame)
        if(flag == 1):
            key = cv2.waitKey(0)

            if key == 121:
                print("recognized:")
                print(date)
                break
            elif key == 110:
                continue

        k = cv2.waitKey(1)  # wait 1 ms
        if k == 27:  # use ESC to end
            print("Exiting")
            date = datetime.date(1900, 1, 1)
            break

    print(date)

    cap.release()
    cv2.destroyAllWindows()
    return date


# capture_camera()
