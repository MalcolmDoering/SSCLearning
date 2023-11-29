#
# Created on Wed Sep 13 2023
#
# Copyright (c) 2023 Malcolm Doering
#

import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "jiang-yongqiang-gcp-ai-for-hri-a1a37d8b11ab.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "malcolm-doering-gcp-ai-for-hri-cb28d3361550.json"

from google.cloud import translate_v2 as translate
import json
import csv
import datetime
import pandas as pd
from dataclasses import dataclass
import pymysql
import string
import sys
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

import tools



exp_table_name = "experiments_final_copy"

speechTablePrefix = "speech_"
speechTableSuffix = "_cas_800_16_50_offline_starttime_ge_p_0606_5s"

participants = ["shopkeeper_1", "shopkeeper_2", "customer_1"]
participantsHids = [1, 3, 2]


sessionDir = sessionDir = tools.create_session_dir("translate")


#
# get the speech data from the database
#
connection = pymysql.connect(host=tools.host, port=tools.port, user=tools.user, passwd=tools.password, db=tools.database, autocommit=True)

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

japaneseToEnglish = {}


#
# load translations
#
fn = tools.dataDir + "English_translations/20231009-122820_translate_cas_800_16_50_offline_starttime_ge_p_0606_5s_concatwpunc/japaneseToEnglishAutoTranslations.csv"

oldTransData, _ = tools.load_interaction_data(fn)

for row in oldTransData:
    japaneseToEnglish[row["japanese"]] = row["english"]



#
# get the translations
#

# get set of unique utterances
uniqueUtts = []


for i in range(len(allSpeechData)):
    utt = allSpeechData[i][2]
    if utt not in uniqueUtts:
        uniqueUtts.append(utt)

for utt in uniqueUtts:
    if utt not in japaneseToEnglish:
        japaneseToEnglish[utt] = ""

# remove utts that were in the old list, but aren't in the new list
for utt in list(japaneseToEnglish.keys()):
    if utt not in uniqueUtts:
        japaneseToEnglish.pop(utt)


translate_client = translate.Client()

for japanese in tqdm(japaneseToEnglish.keys()):

    if japaneseToEnglish[japanese] != "":
        continue
    
    if isinstance(japanese, bytes):
        japanese = japanese.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(japanese, target_language="en-US")

    japaneseToEnglish[japanese] = result["translatedText"].replace("&#39;", "'")    


# save to csv...

rowsToSave = []
rowsToSave.sort()
for japanese in japaneseToEnglish.keys():
    rowsToSave.append({"japanese": japanese,
                       "english": japaneseToEnglish[japanese]})

tools.save_interaction_data(rowsToSave, sessionDir + "japaneseToEnglishAutoTranslations.csv", ["japanese", "english"])



#
# upload the english translation to the database
#
for data in tqdm(allSpeechData):

    if data[8] == "": # check to make sure there's not already good english here
        speechTableName = speechTableNamePerParticipant[data[6]]
        japanese = data[2]
        english = japaneseToEnglish[japanese]

        if "'" in english and '"' in english:
            print("hello")
            print(e)
            print(data)
            print(english)
            print("")

        elif "'" in english:
            sql = 'UPDATE {} SET english = "{}" WHERE time = {}'.format(speechTableName, english, data[0])
        else:
            sql = "UPDATE {} SET english = '{}' WHERE time = {}".format(speechTableName, english, data[0])
        
        with connection.cursor() as cursor:
            try:
                cursor.execute(sql)
            except Exception as e:
                print(e)
                print(data)
                print(english)
                print("") 

