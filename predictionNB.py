#!/usr/bin/python3

############## Prediction Code ###########3
from sklearn.naive_bayes import GaussianNB
import pickle
model_path_filename = 'model_pickle_NB.pkl'
# Open the file to save as pkl file
model_pkl = open(model_path_filename, 'rb')
NB_model=pickle.load(model_pkl)
print("Loaded model specs: ",NB_model)
model_pkl.close()
### Load data of 30 secs
import time as t
import pymysql
import math as m
import numpy as n
import os
import glob
from haversine import haversine 
from time import gmtime, strftime
from datetime import datetime, timezone

tlength=0
time_count=0
RNumFirst=list()
RNumFinal=list()
while(True):
    print("TIME: ",time_count)
    conn = pymysql.connect(
        db='Project',
        user='root',
        passwd='147258369@',
        host='localhost')
    c = conn.cursor()
    # Getting distict IDs
    #c.execute("DELETE FROM `PredictionData` WHERE Date_Time <= NOW() - INTERVAL 2 HOUR")
    #conn.commit()
    c.execute("SELECT DISTINCT ID FROM PredictionData")
    IDs=[(r[0:12]) for r in c.fetchall()]
    Num_IDs=len(IDs)
    pred_float=0.0
    pred='-'
    lat='-'
    lng='-'
    check=0
    first=0
    os.chdir("/var/www/html/CSVfiles")
    print("Number of IDs: ",Num_IDs)
    for k in range(0, Num_IDs):
        do_append=0
        do_predict=1
        #####Checking Row numbers
        check=0
        switch=2
        tlength=c.execute("(SELECT * FROM `PredictionData` where ID = %s order by RNum DESC LIMIT 30) order by RNum ASC",IDs[k])
        data_temp=[(r[0:12]) for r in c.fetchall()]
        ###here comes csv file generation
        #1. check row numbers of read data
        if(len(RNumFirst)==0 or ((len(RNumFirst)-1)<k)):
            first=first+1
            RNumFirst.append(data_temp[0][0])
        
        if(len(RNumFinal)==0 or ((len(RNumFinal)-1)<k)):
            first=first+1
            RNumFinal.append(data_temp[len(data_temp)-1][0])

        if(first>=1):
            print(">>>>Appending in file for the first time\n")
            file_count=0
            for file in glob.glob("*.csv"):
                file_count=file_count+1
            statinfo = os.stat('PData-'+str(file_count)+'.csv')
            F_GB=(statinfo.st_size)/1073741824 ##converting Bytes into GBs
            ##now we compare file size and create new file if size exceeds 5gb
            if(F_GB>=5):
                file_count=file_count+1
            fd=open('PData-'+str(file_count)+'.csv','a+')
            for i1 in range(0,len(data_temp)):
                for i2 in range(0,len(data_temp[i1])-1):
                    fd.write(str(data_temp[i1][i2])+",")
                
                fd.write(str(data_temp[i1][i2+1])+"\n")

            fd.close()


        else:
            #2. if RNumfirst is in bwtween previously recorded RnNumFirst and RNumFinal then don't append but can predict
            if(data_temp[0][0]>RNumFirst[k] and data_temp[0][0]<RNumFinal[k]):
                do_predict=1
                do_append=0
            
            #3. else if RNumFirst and RNumFinal is equal to previously recorded values then don't append and don't predict
            elif(data_temp[0][0]==RNumFirst[k] and data_temp[len(data_temp)-1][0]==RNumFinal[k]):
                do_predict=0
                do_append=0

            #4. else: append and predict
            else:
                do_predict=1
                do_append=1
            RNumFirst[k]=data_temp[0][0]
            RNumFinal[k]=data_temp[len(data_temp)-1][0]

            if(do_append==1):
                print(">>>>Appending in file\n")
                file_count=0
                for file in glob.glob("*.csv"):
                    file_count=file_count+1
                statinfo = os.stat('PData-'+str(file_count)+'.csv')
                F_GB=(statinfo.st_size)/1073741824 ##converting Bytes into GBs
                ##now we compare file size and create new file if size exceeds 5gb
                if(F_GB>=5):
                    file_count=file_count+1
                fd=open('PData-'+str(file_count)+'.csv','a+')
                for i1 in range(0,len(data_temp)):
                    for i2 in range(0,len(data_temp[i1])-1):
                        fd.write(str(data_temp[i1][i2])+",")
                    
                    fd.write(str(data_temp[i1][i2+1])+"\n")

                fd.close()

        if(do_predict==1):
            print(">>>>Towards calculation\n")
            data_in=[(r[2:11]) for r in data_temp]
            lat=data_in[tlength-1][0]
            lng=data_in[tlength-1][1]
            #1. converting string values to float
            for i in range(0, len(data_in)):
                    data_in[i]=list(map(float, data_in[i]))
            data_in=n.matrix(data_in)
            #3. calculating net acceleration and gravity from all rows
            #dataIn= n.matrix(n.zeros(shape=(len(data_in),4)),dtype=float)
            #for i in range(0, len(data_in)):
            #    dataIn[i,1]=m.sqrt(data_in[i][2]**2 + data_in[i][3]**2 + data_in[i][4]**2)
            #    dataIn[i,2]=m.sqrt(data_in[i][5]**2 + data_in[i][6]**2 + data_in[i][7]**2)
            #    dataIn[i,3]=data_in[i][8]
            
            #for i in range(0, len(data_in)-1):
            #    dataIn[i+1,0]=haversine((data_in[i][0],data_in[i][1]),(data_in[i+1][0],data_in[i+1][1]))*100000 #in centimeters
            
            dataFin= n.matrix(n.zeros(shape=(1,8)),dtype=float)
            dataFin[0,0]=haversine((data_in[0,0],data_in[0,1]),(data_in[tlength-1,0],data_in[tlength-1,1])) * 1000 #in meters
            dataFin[0,1]=n.mean(data_in[0:tlength,2])
            dataFin[0,2]=n.mean(data_in[0:tlength,3])
            dataFin[0,3]=n.mean(data_in[0:tlength,4])
            dataFin[0,4]=n.mean(data_in[0:tlength,5])
            dataFin[0,5]=n.mean(data_in[0:tlength,6])
            dataFin[0,6]=n.mean(data_in[0:tlength,7])
            temp= n.array(data_in[0:tlength,8]).transpose()
            dataFin[0,7]= float(n.argmax(n.bincount(list(map(int,temp[0])))))
            check=c.execute("SELECT ID FROM Output WHERE ID = %s", IDs[k][0])
            if(dataFin[0,7]==0.0):
                if(dataFin[0,0]<=185):
                    if(check!=0):
                        c.execute("UPDATE Output SET Updated=0 WHERE ID =%s", IDs[k][0])
                    pred='stationary'
                elif(dataFin[0,0]>185):
                    print(">>> Predicting with speed: ",dataFin[0,0]," and steps: ",dataFin[0,7])
                    pred_float= NB_model.predict(dataFin)
                    print("Prediction: ",pred_float)
                    if(pred_float==0.0):
                        pred='vehicle'
                    else:
                        pred='transport'

            elif(dataFin[0,7]==1.0):
                if(dataFin[0,0]<=185):
                    if(check!=0):
                        c.execute("UPDATE Output SET Updated=0 WHERE ID =%s", IDs[k][0])
                    
                    if(dataFin[0,0]<=10):
                        pred='stationary'
                    else:
                        pred='pedestrian'
                elif(dataFin[0,0]>185):
                    print(">>> Predicting with speed: ",dataFin[0,0]," and steps: ",dataFin[0,7])
                    pred_float= NB_model.predict(dataFin)
                    print("Prediction: ",pred_float)
                    if(pred_float==0.0):
                        pred='vehicle'
                    else:
                        pred='transport'
            #if(dataFin[0,0]<7.0 and dataFin[0,7]==0.0):
            #    pred='stationary'
            #elif(dataFin[0,0]<=185.0 and dataFin[0,7]==1.0):
            #    pred='pedestrian'
            #else:        
            #    pred_float= model.predict(dataFin)
            #    print("Prediction: ",pred_float)
            #    if(pred_float==0.0):
            #        pred='vehicle'
            #    else:
            #        pred='transport'
            conn.commit()
            print("dataFin: ",dataFin)
            print("%s, %s, %s, %s",IDs[k],lat,lng,pred)
            if(check==0):
                c.execute("INSERT INTO Output(ID, lat, lng, Status) VALUES (%s,%s,%s,%s)",(IDs[k][0],lat,lng,pred))
                    
            else:
                c.execute("SELECT Updated FROM Output WHERE ID = %s", IDs[k][0])
                switch=c.fetchall()[0][0]
                if(switch==1):
                    c.execute("UPDATE Output SET lat=%s, lng=%s WHERE ID =%s", (lat, lng, IDs[k][0]))
                elif(switch==0):
                    c.execute("UPDATE Output SET lat=%s, lng=%s, Status=%s WHERE ID =%s", (lat, lng, pred, IDs[k][0]))
            #c.execute("INSERT INTO Output VALUES (%s,%s,%s,%s)",IDs[k],lat,lng,pred)
            if(time_count>=600):
                print("Inserting in timeline\n")
                utc_dt = datetime.now(timezone.utc) # UTC time
                dt = utc_dt.astimezone() # local time
                time1=dt.strftime("%H:%M:%S")
                time2=dt.strftime("%Y-%m-%d")
                c.execute("DELETE FROM `Timeline` WHERE Dates <= NOW() - INTERVAL 3 DAY")
                for itr in range(0,len(IDs)):
                    c.execute("SELECT Status FROM `Output` where ID=%s",IDs[itr][0])
                    pred2=c.fetchall()[0][0]
                    c.execute("INSERT INTO Timeline(ID, Status, Timing, Dates) VALUES (%s,%s,%s,%s)",(IDs[itr][0],pred2,time1,time2))
                
                time_count=0
            
            conn.commit()
        else:
            print(">>>>No calculation\n")
            continue

    conn.close()
    t.sleep(30)
    time_count=time_count+30
    #conn.close()


