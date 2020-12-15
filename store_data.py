import MySQLdb
import datetime


def store_data(exp_date, picture, track_flag, location):

    current = picture
    exp_date = exp_date.strftime('%Y_%m_%d')
    picture = "./picture/" + picture + ".jpg"

    conn = MySQLdb.connect(
        user='user',
        passwd='123456789abc',
        host='localhost',
        db='test')
    cur = conn.cursor()

    if track_flag == 1:
        location = "./location/" + location + ".jpg"
        cur.execute("insert into data values (%s,%s,%s,%s)",
                    (current, exp_date, picture, location))
    elif track_flag == 0:
        cur.execute('insert into data (input_date, exp_date, picture) values (%s,%s,%s)',
                    (current, exp_date, picture))

    cur.execute('select * from data')
    rows = cur.fetchall()

    for row in rows:
        print(row)

    cur.close()
    conn.commit()
    conn.close()


#exp_date = datetime.date(2021, 1, 13)
#picture = "2020_12_10_15_49_32"
#store_data(exp_date, picture, 1, None)
