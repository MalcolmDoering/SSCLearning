'''
Created on Feb 13, 2017

@author: MalcolmD

Modified Nov 2018 by Amal Nanavati
2022.05 by Jiang

This script is used to generate the vectorizer for speech clustering
Some part of the utterance will be ignored, like the stop words
'''

import chardet
from dataclasses import replace
import os
import MeCab
import datetime
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
import pickle
import sys
import intervaltree
sys.path.append('..')
import copy
import openai
import pandas as pd
import pymysql


import tools


with open(tools.dataDir + "openaikey.txt") as f:
    openai.api_key = f.readline()
    

participants = ["shopkeeper_1", "shopkeeper_2", "customer_1"]
participantsHids = [1, 3, 2]

descriptor = "all_participants"

def get_speech_data_from_database(speechTablePrefix, speechTableSuffix):
    #
    # get the speech data from the database
    #
    connection = pymysql.connect(host=tools.host, port=tools.port, user=tools.user, passwd=tools.password, db=tools.database, autocommit=True)

    allSpeechData = []

    with connection:
        with connection.cursor() as cursor:
            for p in participants:
                speechTableName = speechTablePrefix + p + speechTableSuffix
                sql = 'select * from {} order by time'.format(speechTableName)
                cursor.execute(sql)
                allSpeechData += cursor.fetchall()
    
    return allSpeechData




if __name__ == '__main__':

    sessionDir = tools.create_session_dir("utteranceVectorizer")
    
    #
    # load the utterance data from the database
    #
    allSpeechData = []

    speechTablePrefix = "speech_"
    speechTableSuffix = "_cas_800_16_50_offline_starttime_ge_p_0606_5s"

    allSpeechData += get_speech_data_from_database(speechTablePrefix, speechTableSuffix)

    speechTableSuffix = "_cas_800_16_50_offline_starttime_p_0606_5s"
    allSpeechData += get_speech_data_from_database(speechTablePrefix, speechTableSuffix)


    #
    # get more speech data from the interaction data files (maybe some utterances from the database were merged)
    #
    interactionDataFn = tools.dataDir + "20230807-141847_processForSpeechClustering/20230623_SSC_3_trueMotionTargets_3_speechMotionCombined_goodenglish.csv"
    interactionData, interactionDataFieldnames = tools.load_interaction_data(interactionDataFn)


    allUtterances = [x[2] for x in allSpeechData] + [x["participant_speech"] for x in interactionData if x["participant_speech"] != ""]



    print("Getting OpenAI embeddings...")
    uniqueUtterances = list(set(allUtterances))

    EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
    BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

    #embeddingDir = tools.dataDir+"20231220-111317_utteranceVectorizer/" # all_shopkeeper
    embeddingDir = None

    if embeddingDir == None:

        embeddings = []
        for batch_start in range(0, len(uniqueUtterances), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = uniqueUtterances[batch_start:batch_end]

            print(f"Batch {batch_start} to {batch_end-1}")
            
            response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
            
            for i, be in enumerate(response["data"]):
                assert i == be["index"]  # double check embeddings are in same order as input
            
            batch_embeddings = [e["embedding"] for e in response["data"]]
            embeddings.extend(batch_embeddings)
            #break
        
        embeddings = np.asarray(embeddings)

        # save
        np.save(sessionDir+"{}_unique_utterances_openai_embeddings".format(descriptor), embeddings)

        with open(sessionDir+"{}_unique_utterances.txt".format(descriptor), "w") as f:
            f.writelines(line + '\n' for line in uniqueUtterances)

        #df = pd.DataFrame({"utterance": uniqueUtterances, "embedding": embeddings})
        #df.to_csv(sessionDir+"{}_openai_embeddings.csv".format(descriptor), index=False)
    
    else:
        embeddingFile = embeddingDir+"{}_unique_utterances_openai_embeddings.npy".format(descriptor)
        uniqueUtterancesFile = embeddingDir+"{}_unique_utterances.txt".format(descriptor)

        embeddings = np.load(embeddingFile)

        with open(uniqueUtterancesFile, "r") as f:
            uniqueUtterances = f.read().splitlines()
    

        