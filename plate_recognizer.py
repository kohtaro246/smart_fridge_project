# import the necessary packages
from PIL import Image
import cv2
import os
import re
import numpy as np
import glob
import pickle
import datetime as dt
from datetime import datetime, date, timedelta


def relu(z):
    a = np.maximum(0, z)
    return a


def softmax(z):
    a = np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True)
    return a


def for_prop_step(x, w, b, activation_type):
    z = np.dot(w, x)+b

    if activation_type == "relu":
        a = relu(z)
        # print(a.shape)
    elif activation_type == "softmax":
        a = softmax(z)

    return z, a


def for_prop(inp, params):
    layer_num = len(params)//2
    tmps = []
    x = inp
    # print(x.shape)
    for i in range(1, layer_num):
        z, a = for_prop_step(x, params['w'+str(i)], params['b'+str(i)], "relu")
        tmps.append([z, a, params['w'+str(i)], params['b'+str(i)], x])
        x = a
        # print(z.shape)

    z, a = for_prop_step(
        x, params['w'+str(layer_num)], params['b'+str(layer_num)], "softmax")
    tmps.append([z, a, params['w'+str(layer_num)],
                 params['b'+str(layer_num)], x])

    return a, tmps


def predict(data, params):
    m = data.shape[1]
    n = len(params) // 2
    prob, tmps = for_prop(data, params)
    #print(prob[:, 0])
    types = prob[:, 0].shape[0]
    # print(types)
    for i in range(0, prob.shape[1]):
        m_index = np.argmax(prob[:, i])
        # print(m_index)
        for j in range(0, types):
            if j == m_index:
                prob[j][i] = 1
                # print("prob")
                # print(prob[j][i])
            else:
                prob[j][i] = 0
    return prob


def recognizer(frame):
    with open('./params.pickle', mode='rb') as f:
        params = pickle.load(f)
    # print(frame.shape)
    frame = frame.reshape([1, 540//2, 960//2, 3])
    flat_frame = frame.reshape(frame.shape[0], -1).T / 255
    pred = predict(flat_frame, params)

    print(pred)
    return pred


def capture_camera(flag, mirror=True, size=None):
    """Capture video from camera"""
    # print("here")
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    exp_date = dt.date(1800, 1, 1)
    while True:
        flag = 0
        date = ""
        # success?
        ret, frame = cap.read()
        # print(frame.shape)
        frame = cv2.resize(frame, (960//2, 540//2))
        print(frame.shape)
        if mirror is False:
            frame = frame[:, ::-1]

        # resize frame
        # if size is not None and len(size) == 2:
            #frame = cv2.resize(frame, size)

            # show frame
        cv2.imshow('plate_recognizer', frame)

        k = cv2.waitKey(1)  # wait 1 ms
        if k == 27:  # use ESC to end
            print("Exiting")
            exp_date = dt.date(1900, 1, 1)
            break
        elif k == 99:  # press c to predict
            print("calculating")
            pred = recognizer(frame)
            j = cv2.waitKey(0)
            if j == 121:  # press y to end
                break
            elif j == 110:  # prenn n to continue
                continue
    #print(pred[:, 0])
    today = datetime.today()
    tomorrow = today + timedelta(days=1)
    three_days = today + timedelta(days=3)
    week = today + timedelta(days=7)
    two_weeks = today + timedelta(days=14)
    month = today + timedelta(days=30)

    if exp_date != dt.date(1900, 1, 1):
        if pred[:, 0][0] == 1:
            print("donburi")
            exp_date = three_days
            exp_date = exp_date.date()
        elif pred[:, 0][1] == 1:
            print("metal_cont")
            exp_date = week
            exp_date = exp_date.date()
        elif pred[:, 0][2] == 1:
            print("smartsnap")
            exp_date = two_weeks
            exp_date = exp_date.date()
        elif pred[:, 0][3] == 1:
            print("snapware_g")
            exp_date = month
            exp_date = exp_date.date()
        elif pred[:, 0][4] == 1:
            print("soup_s")
            exp_date = tomorrow
            exp_date = exp_date.date()

    print(exp_date)
    now = datetime.now()
    now = now.strftime('%Y_%m_%d_%H_%M_%S')
    if flag == 0:
        cv2.imwrite("./picture/" + now + ".jpg", frame)

    cap.release()
    cv2.destroyAllWindows()
    return exp_date, now


# capture_camera()
