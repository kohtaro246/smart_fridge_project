#!/usr/bin/env python
# -*- coding: utf8 -*-

# 4. Interface which allows users to do the following
#       select from list of products/plates close to expiration date
#       see picture of the products/plates close to expiration date
#       (see where the product/plate close to expiration is)

import Tkinter as tk
import tkMessageBox
import tkFont
import datetime
from PIL import Image, ImageTk

import date_recognizer
import plate_recognizer
import ret_data
import scrolled_listbox as SL


exp_date = datetime.date(1800, 1, 1)

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


counter = 0
break_flag = 0


def check1(event):
    global Val1
    global Val2
    global Val3
    global detection_flag
    global counter

    text = ""

    if Val1.get() == True and Val2.get() == False and Val3.get() == False and Val4.get() == False:
        detection_flag = 0
        root.destroy()
    elif Val1.get() == False and Val2.get() == True and Val3.get() == False and Val4.get() == False:
        detection_flag = 1
        root.destroy()
    elif Val1.get() == False and Val2.get() == False and Val3.get() == True and Val4.get() == False:
        detection_flag = 2
        root.destroy()
    elif Val1.get() == False and Val2.get() == False and Val3.get() == False and Val4.get() == True:
        detection_flag = 3
        root.destroy()
    else:
        text = "Error: You cannot choose more than one.\n"
        tkMessageBox.showinfo('Error', text)


def quit(event):
    global break_flag
    break_flag = 1
    root.destroy()


class Frame(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master.title('item list')
        intro = tk.Label(self, font=('Helvetica', '12'),  justify=tk.LEFT, wraplength='8c',
                         text=u"Select item from the list to show its picture and location")
        intro.pack()
        f = tk.Frame(self, bd=3, relief=tk.RIDGE)
        f.pack(fill=tk.BOTH, expand=1)

        self.listbox = SL.ScrolledListbox(f)
        self.listbox.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.listbox.bind("<Double-Button-1>", self.change_image)
        self.listbox.insert(tk.END, *Images)

        f_button = tk.Frame(f)
        f_button.pack(side=tk.LEFT, padx=5, pady=5)
        img = Image.open(Images[0])
        self.image = ImageTk.PhotoImage(img)
        self.label1 = tk.Label(f, image=self.image, relief=tk.RAISED, bd=3)
        self.label1.pack(side=tk.RIGHT, padx=5)

        if Location[0] != None:
            loc = Image.open(Location[0])
            self.location = ImageTk.PhotoImage(loc)
            self.label2 = tk.Label(f, image=self.location,
                                   relief=tk.RAISED, bd=3)
            self.label2.pack(side=tk.RIGHT, padx=5)
        elif Location[0] == None:
            loc = Image.open("init.jpg")
            self.location = ImageTk.PhotoImage(loc)
            self.label2 = tk.Label(f, image=self.location,
                                   relief=tk.RAISED, bd=3)
            self.label2.pack(side=tk.RIGHT, padx=5)
        

    def change_image(self, event):
        img = Image.open(self.listbox.get(tk.ACTIVE))
        ind = Images.index(self.listbox.get(tk.ACTIVE))
        self.image = ImageTk.PhotoImage(img)
        self.label1.configure(image=self.image)

        if Location[ind] != None:
            loc = Image.open(Location[ind])
            self.location = ImageTk.PhotoImage(loc)
            self.label2.configure(image=self.location)
        elif Location[ind] == None:
            loc = Image.open("init.jpg")
            self.location = ImageTk.PhotoImage(loc)
            self.label2.configure(image=self.location)

while True:
    root = tk.Tk()
    root.title(u"Select Option")
    root.geometry("600x330")

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
    Val4 = tk.BooleanVar()

    Val1.set(False)
    Val2.set(False)
    Val3.set(False)
    Val4.set(False)
    CheckButton1 = tk.Checkbutton(
        text=u"Detect expiration date", variable=Val1)
    CheckButton1.place(x=20, y=90)

    CheckButton2 = tk.Checkbutton(text=u"Detect plate", variable=Val2)
    CheckButton2.place(x=20, y=120)

    CheckButton3 = tk.Checkbutton(text=u"Manual", variable=Val3)
    CheckButton3.place(x=20, y=150)

    CheckButton4 = tk.Checkbutton(text=u"Show all", variable=Val4)
    CheckButton4.place(x=20, y=180)

    button1 = tk.Button(root, text=u'Confirm', width=30)
    button1.bind("<Button-1>", check1)
    button1.place(x=170, y=220)

    if counter > 0:
        button2 = tk.Button(root, text=u'Quit', width=30)
        button2.bind("<Button-1>", quit)
        button2.place(x=170, y=270)

    root.mainloop()

    if break_flag == 1:
        break

    # detection (date_recognizer.py, plate_recognizer.py, input_date.py) and capture
    if detection_flag == 0:
        exp_date = date_recognizer.capture_camera()
    elif detection_flag == 1:
        exp_date, picture = plate_recognizer.capture_camera(1)
    if detection_flag == 2 or exp_date == datetime.date(1900, 1, 1):
        # ルートフレームの定義
        root = tk.Tk()
        root.title("Calendar App")
        mycal = mycalendar(root)
        mycal.pack()
        root.mainloop()
    if detection_flag == 3:
        exp_date = None

    Images, Location = ret_data.ret_data(detection_flag, exp_date)

    f = Frame()
    f.pack()
    f.mainloop()

    counter += 1
