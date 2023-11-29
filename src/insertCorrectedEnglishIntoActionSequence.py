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
speechTableSuffix = "_cas_800_16_50_offline_starttime_ge_p_0606_5s"

participants = ["shopkeeper_1", "shopkeeper_2", "customer_1"]
participants2 = ["S1", "S2", "C1"]
participantsHids = [1, 3, 2]


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

hidTToSpeech = {}

for row in allSpeechData:
    t = row[0]
    hid = row[6]

    if hid not in hidTToSpeech:
        hidTToSpeech[hid] = {}
    hidTToSpeech[hid][t] = row



#
# load the new autotranslations
#
japaneseToAutoEnglish = {}

fn = tools.dataDir + "English_translations/20231009-122820_translate_cas_800_16_50_offline_starttime_ge_p_0606_5s_concatwpunc/japaneseToEnglishAutoTranslations.csv"
oldTransData, _ = tools.load_interaction_data(fn)

for row in oldTransData:
    japaneseToAutoEnglish[row["japanese"]] = row["english"]


fn = tools.dataDir + "English_translations/20231007-193609_translate_cas_800_16_50_offline_starttime_ge_p_0606_5s_concat/japaneseToEnglishAutoTranslations.csv"
oldTransData, _ = tools.load_interaction_data(fn)

for row in oldTransData:
    japaneseToAutoEnglish[row["japanese"]] = row["english"]



#
# insert the good english into the action sequence data
#
interactionDataFilename = tools.dataDir + "20230807-141847_processForSpeechClustering/20230623_SSC_3_trueMotionTargets_3_speechMotionCombined.csv"
interactionData, fieldnames = tools.load_interaction_data(interactionDataFilename)


for i in range(len(interactionData)):
    data = interactionData[i]

    if data["participant_speech"] != "":
        hid = int(data["unique_id"])
        t = float(data["speech_time"])

        speechData = hidTToSpeech[hid][t]

        if speechData[7] == "good English translation":
            interactionData[i]["participant_speech_english_autotranslate"] = japaneseToAutoEnglish[speechData[2]]
            interactionData[i]["participant_speech_english_humantranslate"] = speechData[8]
        else:
            interactionData[i]["participant_speech_english_autotranslate"] = speechData[8]
        

fieldnames.insert(fieldnames.index("participant_speech_english_autotranslate"), "participant_speech_english_humantranslate")

tools.save_interaction_data(interactionData, interactionDataFilename[:-4] + "_goodenglish.csv", fieldnames)














print("Done.")



















