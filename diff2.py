# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

x1 = 0
y1 = 0
x2 = 0
y2 = 0
imageB_copy = cv2.imread("./before_pic/2020_12_13_16_19_52.jpg")
draw_flag = 0
flag = 0


def draw_rectangle(event, x, y, flags, param):
    global imageB_copy
    global x1
    global y1
    global x2
    global y2
    global draw_flag
    global flag
    if event == cv2.EVENT_LBUTTONUP:
        if flag == 0:
            x1 = x
            y1 = y
            cv2.line(imageB_copy, (x1, y1), (x1 + 10, y1), (255, 0, 0), 1)
            cv2.line(imageB_copy, (x1, y1), (x1, y1 + 10), (255, 0, 0), 1)
            flag = 1
        elif flag == 1:
            x2 = x
            y2 = y
            cv2.rectangle(imageB_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            draw_flag = 1
            flag = 0


def compare(before_pic, after_pic):
    global imageB_copy
    global x1
    global y1
    global x2
    global y2
    global flag
    x_final = 0
    y_final = 0
    w_final = 0
    h_final = 0
    # load the two input images
    imageA = cv2.imread('./before_pic/' + before_pic + '.jpg')
    imageB = cv2.imread('./after_pic/' + after_pic + '.jpg')
    # print(imageA.shape)
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sum = 0
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)

        if w > 80 and h > 10:
            #print("width:" + str(w))
            #print("height:" + str(h))

            if w + h > sum:
                x_final = x
                y_final = y
                w_final = w
                h_final = h
                sum = w + h
    if x_final != 0:
        cv2.rectangle(imageA, (x_final, y_final), (x_final +
                                                   w_final, y_final + h_final), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x_final, y_final), (x_final +
                                                   w_final, y_final + h_final), (0, 0, 255), 2)
    # show the output images
    #cv2.imshow("A", grayA)
    #cv2.imshow("B", grayB)

    #cv2.imshow("Original", imageA)
    #cv2.imshow("Compare", imageB)
    #cv2.imshow("Diff", diff)
    #cv2.imshow("Thresh", thresh)
    imageB_copy = imageB.copy()

    cv2.namedWindow('Compare')
    cv2.setMouseCallback('Compare', draw_rectangle)

    while(1):
        cv2.imshow("Compare", imageB_copy)

        k = cv2.waitKey(1)
        if k == 121:
            break
        elif k == 110:
            imageB_copy = imageB.copy()
            flag = 0
            continue
        # if (imageB == imageB_orig).all():
        #    print("True")
        # else:
        #    print("False")

    cv2.destroyAllWindows()
    if draw_flag == 1:
        w_final = x2 - x1
        h_final = y2 - y1
        x_final = x1
        y_final = y1
    return x_final, y_final, w_final, h_final


#before_pic = "2020_12_13_16_19_52"
#after_pic = "2020_12_13_16_20_39"
#x, y, w, h = compare(before_pic, after_pic)
#print("x:" + str(x))
#print("y:" + str(y))
#print("w:" + str(w))
#print("h:" + str(h))
