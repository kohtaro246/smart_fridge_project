import MySQLdb
import datetime


def ret_data(detection_flag, exp_date):
    images = []
    location = []
    if detection_flag != 3:
        exp_date = exp_date.strftime('%Y_%m_%d')

    conn = MySQLdb.connect(
        user='user',
        passwd='123456789abc',
        host='localhost',
        db='test')
    cur = conn.cursor()

    if detection_flag == 3:
        cur.execute("select * from data")
        rows = cur.fetchall()

        for row in rows:
            images.append(row[2])
            location.append(row[3])

    else:
        cur.execute("select * from data where exp_date=%s", (exp_date,))
        rows = cur.fetchall()

        for row in rows:
            images.append(row[2])
            location.append(row[3])

    cur.close()
    conn.commit()
    conn.close()
    return images, location


#exp_date = datetime.date(2020, 12, 19)
#Images, Location = ret_data(1, exp_date)
#print(Location)
