#
# Created on Mon Oct 09 2023
#
# Copyright (c) 2023 Malcolm Doering
#

import json
import csv
import datetime
import pandas as pd
from dataclasses import dataclass
import pymysql
import string
import sys
import pickle
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from pydub import AudioSegment, silence
from scipy.io import wavfile
import ast
import shutil
import datetime


import tools


exp_table_name = "experiments_final_copy"
ht_table_name = "ht_fusion_new_final_20230214"

padding = 15

newTableSuffix = "_share"

connection = pymysql.connect(host=tools.host, port=tools.port, user=tools.user, passwd=tools.password, db=tools.database, autocommit=True)


def copy_files(oldParentDir, oldDirs, newParentDir, startTime, stopTime):
    # get the date
    yyyymmdd = datetime.datetime.utcfromtimestamp(startTime).strftime("%Y%m%d")
    hhmmssStart_ = (datetime.datetime.utcfromtimestamp(startTime) + datetime.timedelta(hours=9)).strftime("%H%M%S")
    hhmmssStop_ = (datetime.datetime.utcfromtimestamp(stopTime) + datetime.timedelta(hours=9)).strftime("%H%M%S")
    
    hhmmssStart = int(hhmmssStart_)
    hhmmssStop = int(hhmmssStop_)


    for d in oldDirs:
        oldDir = oldParentDir + d + "/" + yyyymmdd
        newDir = newParentDir + d + "/" + yyyymmdd

        tools.create_directory(newDir)

        # get the list of files in the old dir
        files = os.listdir(oldDir)

        for f in files:
            try:
                fTime = int(f[:6])
            except:
                continue

            # copy to new dir
            if fTime >= (hhmmssStart-60) and fTime <= hhmmssStop:
                shutil.copy2(oldDir+"/"+f, newDir+"/"+f)


#
# get the experiment times
#
with connection:
    with connection.cursor() as cursor:
        sql = "SELECT * from " + exp_table_name
        cursor.execute(sql)
        experimentInfo = cursor.fetchall()

experimentIDToStartStopTime = {}
experimentIDToConditions = {}
experimentIDToParticipantID = {}
durations = []

for e in experimentInfo:
    experimentIDToStartStopTime[e[1]] = (e[4], e[5])
    experimentIDToConditions[e[1]] = e[6]
    experimentIDToParticipantID[e[1]] = int(experimentIDToConditions[e[1]][:2])
    durations.append(e[5] - e[4])


totalDuration = sum(durations)
aveDuration = np.mean(durations)
stdDuration = np.std(durations)

""" 
#
# make a new table with the skeleton data from only the experiment times
#
new_ht_table_name = ht_table_name + newTableSuffix

with connection:
    with connection.cursor() as cursor:
        try:
            createTableSql = "CREATE TABLE {} LIKE {};".format(new_ht_table_name, ht_table_name)
            cursor.execute(createTableSql)
        except:
            pass


for expID in tqdm(experimentIDToStartStopTime):
    # add some padding
    startTime = experimentIDToStartStopTime[expID][0] - padding
    stopTime = experimentIDToStartStopTime[expID][1] + padding
    

    sql = "INSERT IGNORE INTO {} SELECT * FROM {} WHERE time >= {} AND time <= {};".format(new_ht_table_name, ht_table_name, startTime, stopTime)

    with connection:
        with connection.cursor() as cursor:
            try:
                cursor.execute(sql)
            except Exception as e:
                print(e)
 """


#
# load the participant consent information
#
participantInfo, _ = tools.load_csv_data(tools.dataDir+"participant_info.csv", isHeader=True, isJapanese=False)

participantIDsAudioConsent = []
participantIDsVideoConsent = []

for row in participantInfo:
    participantID = int(row["Participant"])

    consent = row["Consent"] # 1 is agree, 2 is agree with conditions, 3 is disagree
    conditions = row["Additional conditions"] # 1 is blur face, 2 is distort audio, 3 is don't use name

    if "1" in consent:
        participantIDsAudioConsent.append(participantID)
        participantIDsVideoConsent.append(participantID)
    elif "2" in consent:
        if "1" not in conditions:
            participantIDsVideoConsent.append(participantID)
        if "2" not in conditions:
            participantIDsAudioConsent.append(participantID)


#
# copy video and audio files from the experiment times to a new folder
#
dataLocation = "Z:/malcolm/2021_malcolm_multipleshopkeepers/"
newDataLocation = "C:/Users/robovie/Desktop/2021_malcolm_multipleshopkeepers_share/"

audioLocations = ["audio_customer_1", "audio_shopkeeper_1", "audio_shopkeeper_2"]
videoLocations = ["video_1", "video_2", "video_3", "video_4", "video_5"]

tools.create_directory(newDataLocation)

for loc in audioLocations + videoLocations:
    newLoc = newDataLocation + loc
    tools.create_directory(newLoc)

for expID in tqdm(experimentIDToStartStopTime):
    startTime = experimentIDToStartStopTime[expID][0]
    stopTime = experimentIDToStartStopTime[expID][1]

    participantID = experimentIDToParticipantID[expID]

    if participantID in participantIDsAudioConsent:
        copy_files(dataLocation, audioLocations, newDataLocation, startTime, stopTime)
    
    if participantID in participantIDsVideoConsent:
        copy_files(dataLocation, videoLocations, newDataLocation, startTime, stopTime)





print("Done.")

