#!/usr/bin/env python
# -*- coding: utf8 -*-

# About the program
# This program has the following functions
#
# 1. recognize expiration dates
# 2. recognize ? types of plates
# 3. Store data of the following
#       picture of the product/plate
#       expiration dates
#       (Where it is stored)
# 4. Interface which allows users to do the following
#       select from list of products/plates close to expiration date
#       see picture of the products/plates close to expiration date
#       (see where the product/plate close to expiration is)
# (5.detect where the new product/plate was placed using a camera)


# import necessary libraries
import datetime
import MySQLdb
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import re
import sys
import Tkinter as tk
import tkMessageBox
import tkFont

# import necessary functions defined in other files
import date_recognizer
import plate_recognizer
import take_picture
import store_data
import beforepic
import afterpic
import diff2


# set necessary global variables

# (if Product: detection_flag == 0, if Plate: detection_flag == 1, if not detect: detection_flag == 2)
track_flag = 0
detection_flag = 0
break_flag = 0
exp_date = datetime.date(1800, 1, 1)
picture = ""
before_pic = ""
after_pic = ""

# declare classes and functions necessary for date input
# start
now = datetime.datetime.now()
this_year = now.year
this_month = now.month


class mycalendar(tk.Frame):
    def __init__(self, master=None, cnf={}, **kw):
        "初期化メソッド"

        tk.Frame.__init__(self, master, cnf, **kw)

        # 現在の日付を取得
        now = datetime.datetime.now()
        # 現在の年と月を属性に追加
        self.year = now.year
        self.month = now.month
        self.today = now.day

        # frame_explanation
        frame_explanation = tk.Frame(self)
        frame_explanation.pack(pady=5)
        self.explanation = tk.Label(
            frame_explanation, text="Please Select the expiration date from below", font=("", 14))
        self.explanation.pack(side="left")

        # frame_top部分の作成
        frame_top = tk.Frame(self)
        frame_top.pack(pady=5)
        self.previous_month = tk.Label(frame_top, text="<", font=("", 14))
        self.previous_month.bind("<1>", self.change_month)
        self.previous_month.pack(side="left", padx=10)
        self.current_year = tk.Label(frame_top, text=self.year, font=("", 18))
        self.current_year.pack(side="left")
        self.current_month = tk.Label(
            frame_top, text=self.month, font=("", 18))
        self.current_month.pack(side="left")
        self.next_month = tk.Label(frame_top, text=">", font=("", 14))
        self.next_month.bind("<1>", self.change_month)
        self.next_month.pack(side="left", padx=10)

        # frame_week部分の作成
        frame_week = tk.Frame(self)
        frame_week.pack()
        button_mon = d_button(frame_week, text="Mon")
        button_mon.grid(column=0, row=0)
        button_tue = d_button(frame_week, text="Tue")
        button_tue.grid(column=1, row=0)
        button_wed = d_button(frame_week, text="Wed")
        button_wed.grid(column=2, row=0)
        button_thu = d_button(frame_week, text="Thu")
        button_thu.grid(column=3, row=0)
        button_fri = d_button(frame_week, text="Fri")
        button_fri.grid(column=4, row=0)
        button_sta = d_button(frame_week, text="Sat", fg="blue")
        button_sta.grid(column=5, row=0)
        button_san = d_button(frame_week, text="Sun", fg="red")
        button_san.grid(column=6, row=0)

        # frame_calendar部分の作成
        self.frame_calendar = tk.Frame(self)
        self.frame_calendar.pack()

        # 日付部分を作成するメソッドの呼び出し
        self.create_calendar(self.year, self.month, self.today)

        # create frame_bottom
        frame_bottom = tk.Frame(self)
        frame_bottom.pack(pady=5)
        button_confirm = c_button(frame_bottom, text="Confirm")
        button_confirm.pack(side="left")
        button_confirm.bind("<1>", self.close)

    def create_calendar(self, year, month, today):
        "指定した年(year),月(month)のカレンダーウィジェットを作成する"
        global this_month
        global this_year
        # ボタンがある場合には削除する（初期化）
        try:
            for key, item in self.day.items():
                item.destroy()
        except:
            pass

        # calendarモジュールのインスタンスを作成
        import calendar
        cal = calendar.Calendar()
        # 指定した年月のカレンダーをリストで返す
        days = cal.monthdayscalendar(year, month)

        # 日付ボタンを格納する変数をdict型で作成
        self.day = {}
        # for文を用いて、日付ボタンを生成
        for i in range(0, 42):
            c = i - (7 * int(i/7))
            r = int(i/7)
            try:
                # 日付が0でなかったら、ボタン作成
                if days[r][c] != 0:
                    self.day[i] = d_button(
                        self.frame_calendar, text=days[r][c], bg="gray")
                    self.day[i].grid(column=c, row=r)
                    self.day[i].bind("<1>", self.select_day)
                if days[r][c] != 0 and days[r][c] == today and self.month == this_month and self.year == this_year:
                    self.day[i] = d_button(
                        self.frame_calendar, text=days[r][c], bg="pink")
                    self.day[i].grid(column=c, row=r)
            except:
                """
                月によっては、i=41まで日付がないため、日付がないiのエラー回避が必要
                """
                break

    def select_day(self, event):
        global exp_date
        if event.widget["bg"] == "gray":
            event.widget["bg"] = "blue"
            day = event.widget["text"]
            exp_date = datetime.date(self.year, self.month, day)
            print(exp_date)
        else:
            event.widget["bg"] = "gray"

    def change_month(self, event):
        # 押されたラベルを判定し、月の計算
        if event.widget["text"] == "<":
            self.month -= 1
        else:
            self.month += 1
        # 月が0、13になったときの処理
        if self.month == 0:
            self.year -= 1
            self.month = 12
        elif self.month == 13:
            self.year += 1
            self.month = 1
        # frame_topにある年と月のラベルを変更する
        self.current_year["text"] = self.year
        self.current_month["text"] = self.month
        # 日付部分を作成するメソッドの呼び出し
        self.create_calendar(self.year, self.month, self.today)

    def close(self, event):
        root.destroy()
# デフォルトのボタンクラス


class d_button(tk.Button):
    def __init__(self, master=None, cnf={}, **kw):
        tk.Button.__init__(self, master, cnf, **kw)
        self.configure(font=("", 14), height=2, width=4, relief="flat")


class c_button(tk.Button):
    def __init__(self, master=None, cnf={}, **kw):
        tk.Button.__init__(self, master, cnf, **kw)
        self.configure(font=("", 14), height=2, width=20,
                       relief="flat", bg="gray")

# end


# check if the fridge is open (door_checker.py) (OPTION)
# start up window
# 1. Select track or not track
#   if track:(OPTION)
#       track_flag = 1
#       check if frige door is open
#       take an initial picture
#   if not track:
#       track_flag = 0
#       continue
root = tk.Tk()
root.title(u"Select Option")
root.geometry("600x200")

fontstyle_title = tkFont.Font(family="Lucida Grande", size=20)
#
# チェックボックスのチェック状況を取得する
#


def check(event):
    global Val1
    global track_flag

    text = ""

    if Val1.get() == True:
        track_flag = 1
        root.destroy()
    else:
        track_flag = 0
        root.destroy()


# ラベル
Static1 = tk.Label(text=u'Initial Setup', font=fontstyle_title)
Static1.place(x=220, y=10)


#
# チェックボックスの各項目の初期値
#
Val1 = tk.BooleanVar()


Val1.set(False)


CheckButton1 = tk.Checkbutton(
    text=u"Track? (check if Yes)", variable=Val1)
CheckButton1.place(x=20, y=70)


button1 = tk.Button(root, text=u'Confirm', width=30)
button1.bind("<Button-1>", check)
button1.place(x=170, y=140)

root.mainloop()


counter = 0


def check1(event):
    global Val1
    global Val2
    global Val3
    global detection_flag
    global counter

    text = ""

    if Val1.get() == True and Val2.get() == False and Val3.get() == False:
        detection_flag = 0
        root.destroy()
    elif Val1.get() == False and Val2.get() == True and Val3.get() == False:
        detection_flag = 1
        root.destroy()
    elif Val1.get() == False and Val2.get() == False and Val3.get() == True:
        detection_flag = 2
        root.destroy()
    else:
        text = "Error: You cannot choose more than one.\n"
        tkMessageBox.showinfo('Error', text)


def quit(event):
    global break_flag
    break_flag = 1
    root.destroy()


while True:

    # Select whether to detect? or what to detect. Plate or Product. (if Product: detection_flag == 0, if Plate: detection_flag == 1, if not detect: detection_flag == 2)
    root = tk.Tk()
    root.title(u"Select Option")
    root.geometry("600x300")

    fontstyle_title = tkFont.Font(family="Lucida Grande", size=20)
    fontstyle_subtitle = tkFont.Font(family="Lucida Grande", size=10)
    #
    # チェックボックスのチェック状況を取得する
    #

    # ラベル
    Static1 = tk.Label(text=u'Detection Setup', font=fontstyle_title)
    Static1.place(x=200, y=10)

    if counter > 0:
        Static2 = tk.Label(
            text=u'Item registered. To continue, select detection setup. To end, press quit', font=fontstyle_subtitle)
        Static2.place(x=20, y=60)

    elif counter == 0:
        Static3 = tk.Label(text=u'Select detection setup',
                           font=fontstyle_subtitle)
        Static3.place(x=20, y=60)

    #
    # チェックボックスの各項目の初期値
    #
    Val1 = tk.BooleanVar()
    Val2 = tk.BooleanVar()
    Val3 = tk.BooleanVar()

    Val1.set(False)
    Val2.set(False)
    Val3.set(False)
    CheckButton1 = tk.Checkbutton(
        text=u"Detect expiration date", variable=Val1)
    CheckButton1.place(x=20, y=90)

    CheckButton2 = tk.Checkbutton(
        text=u"Detect plate", variable=Val2)
    CheckButton2.place(x=20, y=120)

    CheckButton3 = tk.Checkbutton(
        text=u"Manual", variable=Val3)
    CheckButton3.place(x=20, y=150)

    button1 = tk.Button(root, text=u'Confirm', width=30)
    button1.bind("<Button-1>", check1)
    button1.place(x=170, y=190)

    if counter > 0:
        button2 = tk.Button(root, text=u'Quit', width=30)
        button2.bind("<Button-1>", quit)
        button2.place(x=170, y=240)

    root.mainloop()

    if break_flag == 1:
        break

    # take initial picture if track_flag == 0
    if track_flag == 1:
        before_pic = beforepic.capture_camera()

    # detection (date_recognizer.py, plate_recognizer.py, input_date.py) and capture
    if detection_flag == 0:
        exp_date = date_recognizer.capture_camera()
    elif detection_flag == 1:
        exp_date, picture = plate_recognizer.capture_camera()
    if detection_flag == 2 or exp_date == datetime.date(1900, 1, 1):
        # ルートフレームの定義
        root = tk.Tk()
        root.title("Calendar App")
        mycal = mycalendar(root)
        mycal.pack()
        root.mainloop()

    # capture picture of the product (return )
    if (detection_flag == 0) or (detection_flag == 2):
        picture = take_picture.capture_camera()

    # take an after picture of the fridge and track where the item was placed if track_flag == 1 (OPTION)
    if track_flag == 1:
        after_pic = afterpic.capture_camera()

        location = diff2.compare(before_pic, after_pic)

    # store_data
    if track_flag == 0:
        location = None
    store_data.store_data(exp_date, picture, track_flag, location)

    # check if fridge door is open? (OPTION)
    # select end? or continue
    counter += 1
