#!/usr/bin/python3

import cgitb
cgitb.enable()

print("Content-Type: text/html")
print()


import pymysql
conn = pymysql.connect(
    db='Project',
    user='root',
    passwd='',
    host='')
c = conn.cursor()
import sys
from time import gmtime, strftime
from datetime import datetime, timezone

utc_dt = datetime.now(timezone.utc) # UTC time
dt = utc_dt.astimezone() # local time
time1=dt.strftime("%Y-%m-%d %H:%M:%S")
print("this is: ")
c.execute("INSERT INTO PredictionData (ID, lat, lng, acc_x, acc_y, acc_z, grav_x, grav_y, grav_z, Steps, Date_Time) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(sys.argv[3],sys.argv[1],sys.argv[2],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],time1))

conn.commit()
number2=0
c.execute("SELECT COUNT(*) FROM `PredictionData` WHERE Date_Time <= NOW() - INTERVAL 2 HOUR")
number=c.fetchall()[0][0]
if(number>=1):
    c.execute("UPDATE `One_value_check` SET thevalue=%s",1)
    conn.commit()
    conn.close()
    while(True):
        time.sleep(60)
        conn = pymysql.connect(
            db='Project',
            user='root',
            passwd='',
            host='localhost')
        c = conn.cursor()
        c.execute("Select * FROM One_value_check")
        number2=c.fetchall()[0][0]
        if(number2==0):
            c.execute("DELETE FROM `PredictionData` WHERE Date_Time <= NOW() - INTERVAL 2 HOUR")
            conn.commit()
            conn.close()
            break
        else:
            conn.close()
            continue



c.execute("DELETE FROM `PredictionData` WHERE Date_Time <= NOW() - INTERVAL 2 HOUR")
conn.commit()
conn.close()
print("Done",number);
