

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

import tools


speechTablePrefix = "speech_"
speechTableSuffix = "_cas_800_16_50_offline_starttime"
newSpeechTableSuffix = "_p_0606_5s"

participants = ["shopkeeper_1", "shopkeeper_2", "customer_1"]
participantsHids = [1, 3, 2]


def combine_neighboring_utterances(utt1, utt2):
    utt1_time = utt1[0]
    utt1_idx = utt1[1]
    utt1_sentence = utt1[2]
    utt1_duration = utt1[3]
    utt1_confidence_level = utt1[4]
    utt1_filename = utt1[5]
    utt1_hid = utt1[6]
    utt1_rich_result = utt1[7]
    utt1_english = utt1[8]
    utt1_cluster_key = utt1[9]
    utt1_correct = utt1[10]
    utt1_experiment = utt1[11]
    utt1_result_start_time = utt1[12]
    utt1_result_end_time = utt1[13]
    utt1_word_transcripts = utt1[14]
    utt1_word_start_times = utt1[15]
    utt1_word_end_times = utt1[16]

    utt2_time = utt2[0]
    utt2_idx = utt2[1]
    utt2_sentence = utt2[2]
    utt2_duration = utt2[3]
    utt2_confidence_level = utt2[4]
    utt2_filename = utt2[5]
    utt2_hid = utt2[6]
    utt2_rich_result = utt2[7]
    utt2_english = utt2[8]
    utt2_cluster_key = utt2[9]
    utt2_correct = utt2[10]
    utt2_experiment = utt2[11]
    utt2_result_start_time = utt2[12]
    utt2_result_end_time = utt2[13]
    utt2_word_transcripts = utt2[14]
    utt2_word_start_times = utt2[15]
    utt2_word_end_times = utt2[16]

    new_time = utt1_time # start time of first utterance
    new_idx = utt1_idx
    new_sentence = utt1_sentence + utt2_sentence
    new_duration = round((utt2_result_end_time-utt1_result_start_time) * 1000)
    new_confidence_level = (utt1_confidence_level+utt2_confidence_level) / 2.0 
    new_filename = utt1_filename + ";" + utt2_filename
    new_hid = utt1_hid # should be the same for both
    new_rich_result = ""
    new_english = ""
    new_cluster_key = ""
    new_correct = ""
    new_experiment = utt1_experiment
    new_result_start_time = utt1_result_start_time
    new_result_end_time = utt2_result_end_time
    new_word_transcripts = utt1_word_transcripts + ";" + utt2_word_transcripts
    new_word_start_times = utt1_word_start_times + ";" + utt2_word_start_times
    new_word_end_times = utt1_word_end_times + ";" + utt2_word_end_times

    return (new_time, new_idx, new_sentence, new_duration, new_confidence_level, new_filename, new_hid, new_rich_result, new_english, new_cluster_key, new_correct, new_experiment, new_result_start_time, new_result_end_time, new_word_transcripts, new_word_start_times, new_word_end_times)


def check_speech_data(speechData):
    for i in range(len(speechData)):
        if not (isinstance(speechData[i][13], float)):
            print("WARNING: Index {} not float!".format(i))


def upload_speech_to_db(cursor, speechTableNamePerParticipant, speechData):
    for sd in speechData:
        hid = sd[6]
        dbTableName = speechTableNamePerParticipant[hid] + newSpeechTableSuffix
        
        try:
            createTableSql = "CREATE TABLE {} LIKE {};".format(dbTableName, speechTableNamePerParticipant[hid])
            cursor.execute(createTableSql)
        except:
            pass


        sql = "REPLACE INTO {} (time, idx, sentence, duration, confidence_level, filename, hid, experiment, result_start_time, result_end_time, word_transcripts, word_start_times, word_end_times) VALUES ({}, {}, \"{}\", {}, {}, \"{}\", {}, {}, {}, {}, \"{}\", \"{}\",\"{}\")".format(
            dbTableName, 
            sd[0], 
            sd[1], 
            sd[2],
            sd[3],
            sd[4],
            sd[5],
            sd[6],
            sd[11],
            sd[12],
            sd[13],
            sd[14],
            sd[15],
            sd[16])
        
        cursor.execute(sql)






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
            #speechDataPerParticipant[p] = cursor.fetchall()
            allSpeechData += cursor.fetchall()
            
            #for i in range(len(speechDataPerParticipant[p])):
            #    timestamps.append(speechDataPerParticipant[p][i])

        #createTableSql = "CREATE TABLE speech_shopkeeper_1_cas_800_16_50_offline_starttime_p_20230605 LIKE {};".format(speechTableName)
        #cursor.execute(createTableSql)

timestamps = sorted(list(set(timestamps)))
allSpeechData = sorted(allSpeechData)



check_speech_data(allSpeechData)

#
# remove utterances that are subsumed by other utterances
#
speechStatusPerParticipant = {}
for hid in participantsHids:
    speechStatusPerParticipant[hid] = (-1, -1)

indicesToRemove = []
instancesToRemove = []
moreInstancesToRemove = []


for i in range(len(allSpeechData)):
    t = allSpeechData[i][0]
    hid = allSpeechData[i][6]
    dur = allSpeechData[i][3] / 1000.0

    # update speaking status
    speechStatusPerParticipant[hid] = (t, t + dur)
    
    for otherHid in participantsHids:
        if otherHid != hid:
            if t > speechStatusPerParticipant[otherHid][1]:
                speechStatusPerParticipant[otherHid] = (-1, -1)
    

    # check if this speech is within the start and stop of any other concurrent speech
    remove = False

    for otherHid in participantsHids:
        if otherHid != hid:
            if speechStatusPerParticipant[hid][0] >= speechStatusPerParticipant[otherHid][0] and speechStatusPerParticipant[hid][1] <= speechStatusPerParticipant[otherHid][1]:
                remove = True
            
            #if not remove and speechStatusPerParticipant[hid][0] >= speechStatusPerParticipant[otherHid][0] and speechStatusPerParticipant[hid][1] <= speechStatusPerParticipant[otherHid][1]:
            #    moreInstancesToRemove.append(allSpeechData[i])

    if remove:
        indicesToRemove.append(i)
        instancesToRemove.append(allSpeechData[i])

indicesToRemove.sort(reverse=True)

for i in indicesToRemove:
    allSpeechData.pop(i)



#
# combine neighboring utterances from the same participant if they're not interrupted by another participant's utterance
#
for hid in participantsHids:
    speechStatusPerParticipant[hid] = (-1, -1)

indicesToCombine = set()
indexRangesToCombine = []
allSpeechDataAfterCombine = []

durThresh = 5.0 # maximum time between utterances to combine 

for i in range(1, len(allSpeechData)):
    prev_t = allSpeechData[i-1][0]
    prev_hid = allSpeechData[i-1][6]
    prev_dur = allSpeechData[i-1][3] / 1000.0
    
    t = allSpeechData[i][0]
    hid = allSpeechData[i][6]
    dur = allSpeechData[i][3] / 1000.0

    # update speaking status
    speechStatusPerParticipant[prev_hid] = (prev_t, prev_t + prev_dur)
    
    for otherHid in participantsHids:
        if otherHid != prev_hid:
            if prev_t > speechStatusPerParticipant[otherHid][1]:
                speechStatusPerParticipant[otherHid] = (-1, -1)
    

    # check is both utterances are from the same speaker and there is no one else speaking, if yes -> combine
    if hid == prev_hid and (t - (prev_t+prev_dur)) <= durThresh:
        otherSpeaking = False
        for otherHid in participantsHids:
            if otherHid != hid:
                if speechStatusPerParticipant[otherHid][0] == -1:
                    otherSpeaking == True
        
        if not otherSpeaking:
            # for combining
            if len(indexRangesToCombine) > 0 and indexRangesToCombine[-1][1] == i-1:
                indexRangesToCombine[-1][1] = i
            else:
                indexRangesToCombine.append([i-1, i])
            
            indicesToCombine.add(i-1)
            indicesToCombine.add(i)
        

for ir in indexRangesToCombine:
    combinedSpeech = None

    if indexRangesToCombine.index(ir) == 341:
        print("hello")

    for i in range(ir[0], ir[1]):
        
        if combinedSpeech is None:
            combinedSpeech = combine_neighboring_utterances(allSpeechData[i], allSpeechData[i+1])
        else:
            combinedSpeech = combine_neighboring_utterances(combinedSpeech, allSpeechData[i+1])
    
    allSpeechDataAfterCombine.append(combinedSpeech)

for i in range(len(allSpeechData)):
    if i not in indicesToCombine:
        allSpeechDataAfterCombine.append(allSpeechData[i])


allSpeechDataAfterCombine = sorted(allSpeechDataAfterCombine)

print("{} utterances after combine.".format(len(allSpeechDataAfterCombine)))




#    allSpeechDataAfterCombine.append(combine_neighboring_utterances(allSpeechData[i-1], allSpeechData[i]))
#        else:
#            allSpeechDataAfterCombine.append(allSpeechData[i-1])



# upload results to db
connection = pymysql.connect(host=tools.host, port=tools.port, user=tools.user, passwd=tools.password, db=tools.database, autocommit=True)
with connection:
    with connection.cursor() as cursor:
        upload_speech_to_db(cursor, speechTableNamePerParticipant, allSpeechDataAfterCombine)
        pass


print("Done.")











