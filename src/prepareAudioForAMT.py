#
# Created on Tue Sep 12 2023
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


exp_table_name = "experiments_final_copy"

speechTablePrefix = "speech_"
speechTableSuffix = "_cas_800_16_50_offline_starttime" #_p_0606_5s"

participants = ["shopkeeper_1", "shopkeeper_2", "customer_1"]
participantsHids = [1, 3, 2]

audioDir = tools.dataDir + "800_16_50"
newAudioDir = tools.dataDir + "800_16_50_0606_amt"


def remove_json_prefix(filename):
    if filename.endswith(".json"):
        filename = filename[:-5]
    return filename


#
# get the speech data from the database
# these should for the ASR results from after splitting audio, but before combining the results in postprocessing
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

# sort by timestamp
allSpeechData = sorted(allSpeechData)


#
# prepare audio files
#
audioFilesToSpeechResults = {}

#for speechData in tqdm(allSpeechData):
for speechData in allSpeechData:
    audioFiles = speechData[5].split(";")

    # TODO get start and stop times
    wordStartTimes = [ast.literal_eval(wst) for wst in speechData[15].split(";")][0]
    wordEndTimes = [ast.literal_eval(wet) for wet in speechData[16].split(";")][0]

    startTime = wordStartTimes[0]
    endTime = wordEndTimes[-1]

    audioStartTime = int(audioFiles[0].split("_")[1])

    audioPath = audioDir + "/" + audioFiles[0]
    audioPath = remove_json_prefix(audioPath)
    
    if audioPath not in audioFilesToSpeechResults:
        audioFilesToSpeechResults[audioPath] = []
    audioFilesToSpeechResults[audioPath].append([speechData, startTime, endTime, audioStartTime])



endPadding = 0.5
totalDuration = 0.0
numFiles = 0

for audioPath in tqdm(audioFilesToSpeechResults.keys()):
    speechData = audioFilesToSpeechResults[audioPath]

    if len(speechData) > 1:
        # split the audio file so each speech result has it's own file for transcription 
        audioSegment = AudioSegment.from_wav(audioPath)
        sampleRate, data = wavfile.read(audioPath)

        if "906_1671416599270_audio_shopkeeper" in audioPath:
            print("hello")

        for sd in speechData:
            uttStartTime = sd[1]
            uttEndTime = sd[2]
            audioStartTime = sd[3]

            uttFilename = "{}_{}.wav".format(audioPath.split("/")[-1][:-4], int(uttStartTime*1000))
            uttPath = os.path.join(newAudioDir, uttFilename)
            
            # with pydub
            #startIndex = int((uttStartTime * 1000) - audioStartTime)
            #endIndex = int((uttEndTime * 1000) - audioStartTime)
            #uttSegment = audioSegment[startIndex:endIndex]            
            #uttSegment.export(uttPath, format='wav')

            # with wavfile
            startIndex = int((uttStartTime - (audioStartTime / 1000.0)) * sampleRate)
            endIndex = int((uttEndTime - (audioStartTime / 1000.0)) * sampleRate)
            endIndex += int(endPadding * sampleRate)

            wavfile.write(uttPath, sampleRate, data[startIndex:endIndex])

            totalDuration += (startIndex-endIndex) / sampleRate
    
    else:
        sampleRate, data = wavfile.read(audioPath)
        totalDuration += len(data) / sampleRate
        shutil.copy2(audioPath, newAudioDir)

            
    
    """ if len(audioFiles) > 1:
        # combine the files and save it
        fileData = []

        for i in range(len(audioFiles)):
            fn  = audioFiles[i]

            audioPath = audioDir + "/" + fn
            audioPath = remove_json_prefix(audioPath)
            audioSegment = AudioSegment.from_wav(audioPath)
            fileData.append(audioSegment)

            # insert a silence segment?
            if i != len(audioFiles)-1: # if not the last audiofile
                silenceDuration = float(wordStartTimes[i+1][0]) - float(wordEndTimes[i][-1])
                print(silenceDuration)
                if silenceDuration > 0.0:
                    fileData.append(AudioSegment.silent(duration=silenceDuration*1000))

        newFileData = AudioSegment.empty()

        for fd in fileData:
            newFileData += fd

        temp1 = []
        for af in audioFiles:
             temp1 .append(af.split("_")[1]) # get the timestamp
        
        temp2 = audioFiles[0].split("_")
        temp2 = temp2[:1] + temp1 + temp2[2:]

        newFileName = "_".join(temp2)
        newFileName = remove_json_prefix(newFileName)
        
        newFileData.export(os.path.join(newAudioDir, newFileName), format='wav') """



#
# get the experiment condition info 
#
with connection:
    with connection.cursor() as cursor:
        sql = "SELECT * from " + exp_table_name
        cursor.execute(sql)
        experimentInfo = cursor.fetchall()

experimentIdToConditions = {}

for e in experimentInfo:
    experimentIdToConditions[e[1]] = e[6]



#
# load the data that was labeled by H-san
#




#
# prepare audio files
#

# combine and save if necessary

# otherwise, just copy



#
# combine all data into a csv
#

# mark data that already has a label



#
# prepare input csv for AMT
#




