#
# Created on Tue Aug 01 2023
#
# Copyright (c) 2023 Malcolm Doering
#
#
# get some statistics about the dataset
#

import os
import pandas as pd
import copy
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
from scipy.io import wavfile
import ast
import shutil
import datetime

import tools



exp_table_name = "experiments_final_copy"


connection = pymysql.connect(host=tools.host, port=tools.port, user=tools.user, passwd=tools.password, db=tools.database, autocommit=True)


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
medDuration = np.median(durations)

print("totalDuration", totalDuration)
print("aveDuration", aveDuration)
print("stdDuration", stdDuration)
print("medDuration", medDuration)