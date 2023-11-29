#
# Created on Fri Oct 06 2023
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


import tools


speechTablePrefix = "speech_"
speechTableSuffix = "_cas_800_16_50_offline_starttime"
newSpeechTableSuffix = "_ge"

participants = ["shopkeeper_1", "shopkeeper_2", "customer_1"]
participants2 = ["S1", "S2", "C1"]
participantsHids = [1, 3, 2]


def upload_speech_to_db(cursor, speechTableName, speechData):
    for sd in speechData:
        hid = sd[6]
        dbTableName = speechTableName  + newSpeechTableSuffix
        
        try:
            createTableSql = "CREATE TABLE {} LIKE {};".format(dbTableName, speechTableName)
            cursor.execute(createTableSql)
        except:
            pass

        if sd[8] != None:
            sd[8] = sd[8].replace('"', "'")


        sql = "REPLACE INTO {} (time, idx, sentence, duration, confidence_level, filename, hid, rich_result, english, correct, experiment, result_start_time, result_end_time, word_transcripts, word_start_times, word_end_times) VALUES ({}, {}, \"{}\", {}, {}, \"{}\", {}, \"{}\", \"{}\", \"{}\", {}, {}, {}, \"{}\", \"{}\",\"{}\")".format(
            dbTableName, 
            sd[0], 
            sd[1], 
            sd[2],
            sd[3],
            sd[4],
            sd[5],
            sd[6],

            sd[7],
            sd[8],
            sd[10],

            sd[11],
            sd[12],
            sd[13],
            sd[14],
            sd[15],
            sd[16])
        
        cursor.execute(sql)




#
# load the corrected English translations
#
englishTranslationData, englishTranslationFieldnames = tools.load_interaction_data(tools.dataDir+"English_translations/20231006_all_participants_speech_with_conditions.csv")
translationCompletedData, translationCompletedFieldnames = tools.load_interaction_data(tools.dataDir+"English_translations/20231006_experiments_final.csv")

translationCompletedExpIDs = []

hidTimeToTranslationData = {}

for row in translationCompletedData:    
    if row["done by Hasegawa-san"] == "1":
        translationCompletedExpIDs.append(int(row["id"]))

for row in englishTranslationData:
    t = float(row["result_start_time"])
    hid = int(participantsHids[participants2.index(row["participant"])])
    expID = int(row["experiment"])

    if expID in translationCompletedExpIDs:
        if hid not in hidTimeToTranslationData:
            hidTimeToTranslationData[hid] = {}
        
        hidTimeToTranslationData[hid][t] = row



#
# load the speech data from the database
#
connection = pymysql.connect(host=tools.host, port=tools.port, user=tools.user, passwd=tools.password, db=tools.database, autocommit=True)

speechDataPerParticipant = {}
allSpeechData = []
timestamps = []
speechTableNamePerParticipant = {}

with connection:
    with connection.cursor() as cursor:
        for p in participants:
            speechTableName = speechTablePrefix + p + speechTableSuffix
            speechTableNamePerParticipant[participantsHids[participants.index(p)]] = speechTableName
            sql = 'select * from {} order by time'.format(speechTableName)
            cursor.execute(sql)
            allSpeechData += cursor.fetchall()

allSpeechData = sorted(allSpeechData)



#
# insert the corrected English translations
#
hidToNewSpeechData = {}

for row in allSpeechData:
    t = row[0]
    hid = row[6]
    japanese = row[2]
    expID = row[11]

    newSpeechData = list(row)

    if newSpeechData[7] == None:
        newSpeechData[7] = ""
    if newSpeechData[8] == None:
        newSpeechData[8] = ""
    if newSpeechData[9] == None:
        newSpeechData[9] = ""
    if newSpeechData[10] == None:
        newSpeechData[10] = ""

    if hid not in hidToNewSpeechData:
            hidToNewSpeechData[hid] = []

    
    if expID in translationCompletedExpIDs:
        transData = hidTimeToTranslationData[hid][t]

        newSpeechData[7] = "good English translation"
        
        originalJapanese = transData["japanese"]
        correctJapanese = transData["correct japanese"]
        
        originalEnglish = transData["automatic_english_translation"]
        correctEnglish = transData["correct_english_translation"]


        if correctJapanese == "":
            newSpeechData[10] = originalJapanese 
        else:
            newSpeechData[10] = correctJapanese 

        if correctEnglish == "":
            newSpeechData[8] = originalEnglish
        else:
            newSpeechData[8] = correctEnglish
        

    hidToNewSpeechData[hid].append(newSpeechData)
    


#
# load the speech data back into the database
#
connection = pymysql.connect(host=tools.host, port=tools.port, user=tools.user, passwd=tools.password, db=tools.database, autocommit=True)
with connection:
    with connection.cursor() as cursor:

        for hid in hidToNewSpeechData:
            upload_speech_to_db(cursor, speechTableNamePerParticipant[hid], hidToNewSpeechData[hid])



print("Done.")



















