import MySQLdb
import datetime


def del_data(pic):
    conn = MySQLdb.connect(
        user='user',
        passwd='123456789abc',
        host='localhost',
        db='test')
    cur = conn.cursor()

    cur.execute("delete from data where picture=%s", (pic,))

    cur.close()
    conn.commit()
    conn.close()


#pic = "./picture/2020_12_27_22_02_29.jpg"
# del_data(pic)
