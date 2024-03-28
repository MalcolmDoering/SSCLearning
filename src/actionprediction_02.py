#
# Created on Thu Jun 29 2023
#
# Copyright (c) 2023 Malcolm Doering
#

import csv
import sys
import os
import time
import numpy as np
import copy
from tabulate import tabulate
import pickle
from multiprocessing import Process
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random

import tools


def split_list(a, n):
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


#################################################################################################################
# file paths
#################################################################################################################

#
# for automatically extracted method
#
evaluationDataDir = tools.dataDir + "20240226-173919_actionPredictionPreprocessing_1leninput/" # data with both shopkeepers actions, 1 len input
#evaluationDataDir = tools.dataDir + "20240304-153902_actionPredictionPreprocessing_3leninput/" # data with both shopkeepers actions, 3 len input



#interactionDataFilename = "20230807-141847_processForSpeechClustering/20230623_SSC_3_trueMotionTargets_3_speechMotionCombined.csv"
keywordsFilename = tools.modelDir + "20230609-141854_unique_utterance_keywords.csv"
uttVectorizerDir = evaluationDataDir
stoppingLocationClusterDir = tools.modelDir + "20230627_stoppingLocationClusters/"


#
# for mementar based method
#
#mementarDataDir = tools.dataDir + "20240318-141639_processmementaroutput_1leninput/" # with no action states removed
mementarDataDir = tools.dataDir + "20240325-130931_processmementaroutput_testxy/" # with no action states removed



#
# for both
#
speechClustersFilename = "20230731-113400_speechClustering/all_shopkeeper- speech_clusters - levenshtein normalized medoid.csv"



mainDir = tools.create_session_dir("actionPrediction_02")



def main(mainDir, condition, gpuCount):
    #################################################################################################################
    # running params
    #################################################################################################################

    DEBUG = False
    RUN_PARALLEL = True
    SPEECH_CLUSTER_LOSS_WEIGHTS = False
    NUM_GPUS = 8
    NUM_FOLDS = 8

    numTrainFolds = 6
    numValFolds = 1
    numTestFolds = 1

    # params that should be the same for all conditions (predictors)
    batchSize = 8
    randomizeTrainingBatches = True
    numEpochs = 50
    numEpochsTillS2Only = None #250
    evalEvery = 1
    minClassCount = 2
    useAttention = False

    learningRate = 1e-5
    embeddingDim = 1000

    numShopkeepers = 2


    sessionDir = mainDir + "/" + condition
    tools.create_directory(sessionDir)
    
    # what to run. only one of these should be true at a time
    bl1_run = False # prediction of whether or not S2 acts
    bl2_run = False # prediction of S2's actions
    bl3_run = False # prediction of S2's actions with split outputs
    bl4_run = False # predict both the shopkeeper's actions
    bl5_run = False # prediction with mementar output
    bl6_run = False # prediction with mementar output, train on both shopkeeper's actions
    bl7_run = False # prediction of whether or not S2 acts with mementar input
    prop_run = False
    
    if condition == "baseline1":
        bl1_run = True
    elif condition == "baseline2":
        bl2_run = True
    elif condition == "baseline3":
        bl3_run = True
    elif condition == "baseline4":
        bl4_run = True
    elif condition == "baseline5":
        bl5_run = True
    elif condition == "baseline6":
        bl6_run = True
    elif condition == "baseline7":
        bl7_run = True
    elif condition == "proposed":
        prop_run = True
    
    

    #################################################################################################################
    # load the data
    #################################################################################################################
    print("Loading data...")

    if bl1_run:
        humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(evaluationDataDir+"humanReadableInputsOutputs.csv")
        bl1_outputActionIDs_shkp1 = np.load(evaluationDataDir+"outputActionIDs_shkp1.npy")
        bl1_outputSpeechClusterIDs_shkp1 = np.load(evaluationDataDir+"outputSpeechClusterIDs_shkp1.npy")
        bl1_outputSpatialInfo_shkp1 = np.load(evaluationDataDir+"outputSpatialInfo_shkp1.npy")
        bl1_outputSpeechClusterIsJunk_shkp1 = np.load(evaluationDataDir+"outputSpeechClusterIsJunk_shkp1.npy")

        bl1_outputActionIDs_shkp2 = np.load(evaluationDataDir+"outputActionIDs_shkp2.npy")
        bl1_outputSpeechClusterIDs_shkp2 = np.load(evaluationDataDir+"outputSpeechClusterIDs_shkp2.npy")
        bl1_outputSpatialInfo_shkp2 = np.load(evaluationDataDir+"outputSpatialInfo_shkp2.npy")
        bl1_outputSpeechClusterIsJunk_shkp2 = np.load(evaluationDataDir+"outputSpeechClusterIsJunk_shkp2.npy")

        bl1_outputDidActionBits = np.load(evaluationDataDir+"outputDidActionBits.npy")
        bl1_inputVectorsCombined = np.load(evaluationDataDir+"inputVectorsCombined.npy")

    elif bl2_run:
        humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(evaluationDataDir+"humanReadableInputsOutputs.csv")
        bl2_outputActionIDs_shkp1 = np.load(evaluationDataDir+"outputActionIDs_shkp1.npy")
        bl2_outputSpeechClusterIDs_shkp1 = np.load(evaluationDataDir+"outputSpeechClusterIDs_shkp1.npy")
        bl2_outputSpatialInfo_shkp1 = np.load(evaluationDataDir+"outputSpatialInfo_shkp1.npy")
        bl2_outputSpeechClusterIsJunk_shkp1 = np.load(evaluationDataDir+"outputSpeechClusterIsJunk_shkp1.npy")

        bl2_outputActionIDs_shkp2 = np.load(evaluationDataDir+"outputActionIDs_shkp2.npy")
        bl2_outputSpeechClusterIDs_shkp2 = np.load(evaluationDataDir+"outputSpeechClusterIDs_shkp2.npy")
        bl2_outputSpatialInfo_shkp2 = np.load(evaluationDataDir+"outputSpatialInfo_shkp2.npy")
        bl2_outputSpeechClusterIsJunk_shkp2 = np.load(evaluationDataDir+"outputSpeechClusterIsJunk_shkp2.npy")

        bl2_outputDidActionBits = np.load(evaluationDataDir+"outputDidActionBits.npy")
        bl2_inputVectorsCombined = np.load(evaluationDataDir+"inputVectorsCombined.npy")
    
    elif bl3_run:
        humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(evaluationDataDir+"humanReadableInputsOutputs.csv")
        bl3_outputActionIDs = np.load(evaluationDataDir+"outputActionIDs.npy")
        bl3_outputSpeechClusterIDs = np.load(evaluationDataDir+"outputSpeechClusterIDs.npy")
        bl3_outputSpatialInfo = np.load(evaluationDataDir+"outputSpatialInfo.npy")
        bl3_toIgnore = np.load(evaluationDataDir+"toIgnore.npy")
        bl3_isHidToImitate = np.load(evaluationDataDir+"isHidToImitate.npy")
        bl3_inputVectorsCombined = np.load(evaluationDataDir+"inputVectorsCombined.npy")
        
        # read the speech clusters
        bl3_speechClusterData, _ = tools.load_csv_data(tools.dataDir+speechClustersFilename, isHeader=True, isJapanese=True)
        bl3_speechClustIDToRepUtt = {}
        bl3_speechClustIDToIsJunk = {}

        for row in bl3_speechClusterData:
            speech = row["Utterance"]
            speechClustID = int(row["Cluster.ID"])

            if int(row["Is.Representative"]) == 1:
                bl3_speechClustIDToRepUtt[speechClustID] = speech
            
            if int(row["Is.Junk"]) == 1:
                bl3_speechClustIDToIsJunk[speechClustID] = 1
            else:
                bl3_speechClustIDToIsJunk[speechClustID] = 0
    
    elif bl4_run:
        humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(evaluationDataDir+"humanReadableInputsOutputs.csv")
        bl4_outputActionIDs_shkp1 = np.load(evaluationDataDir+"outputActionIDs_shkp1.npy")
        bl4_outputSpeechClusterIDs_shkp1 = np.load(evaluationDataDir+"outputSpeechClusterIDs_shkp1.npy")
        bl4_outputSpatialInfo_shkp1 = np.load(evaluationDataDir+"outputSpatialInfo_shkp1.npy")
        bl4_outputSpeechClusterIsJunk_shkp1 = np.load(evaluationDataDir+"outputSpeechClusterIsJunk_shkp1.npy")

        bl4_outputActionIDs_shkp2 = np.load(evaluationDataDir+"outputActionIDs_shkp2.npy")
        bl4_outputSpeechClusterIDs_shkp2 = np.load(evaluationDataDir+"outputSpeechClusterIDs_shkp2.npy")
        bl4_outputSpatialInfo_shkp2 = np.load(evaluationDataDir+"outputSpatialInfo_shkp2.npy")
        bl4_outputSpeechClusterIsJunk_shkp2 = np.load(evaluationDataDir+"outputSpeechClusterIsJunk_shkp2.npy")

        bl4_outputDidActionBits = np.load(evaluationDataDir+"outputDidActionBits.npy")
        bl4_inputVectorsCombined = np.load(evaluationDataDir+"inputVectorsCombined.npy")

        # read the speech clusters
        bl4_speechClusterData, _ = tools.load_csv_data(tools.dataDir+speechClustersFilename, isHeader=True, isJapanese=True)
        bl4_speechClustIDToRepUtt = {}
        bl4_speechClustIDToIsJunk = {}

        for row in bl4_speechClusterData:
            speech = row["Utterance"]
            speechClustID = int(row["Cluster.ID"])

            if int(row["Is.Representative"]) == 1:
                bl4_speechClustIDToRepUtt[speechClustID] = speech
            
            if int(row["Is.Junk"]) == 1:
                bl4_speechClustIDToIsJunk[speechClustID] = 1
            else:
                bl4_speechClustIDToIsJunk[speechClustID] = 0
    
    elif bl5_run:
        humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(mementarDataDir+"humanReadableInputsOutputs.csv")
        bl5_outputActionIDs_shkp1 = np.load(mementarDataDir+"outputActionIDs_shkp1.npy")
        bl5_outputSpeechClusterIDs_shkp1 = np.load(mementarDataDir+"outputSpeechClusterIDs_shkp1.npy")
        bl5_outputSpatialInfo_shkp1 = np.load(mementarDataDir+"outputSpatialInfo_shkp1.npy")
        bl5_outputSpeechClusterIsJunk_shkp1 = np.load(mementarDataDir+"outputSpeechClusterIsJunk_shkp1.npy")

        bl5_outputActionIDs_shkp2 = np.load(mementarDataDir+"outputActionIDs_shkp2.npy")
        bl5_outputSpeechClusterIDs_shkp2 = np.load(mementarDataDir+"outputSpeechClusterIDs_shkp2.npy")
        bl5_outputSpatialInfo_shkp2 = np.load(mementarDataDir+"outputSpatialInfo_shkp2.npy")
        bl5_outputSpeechClusterIsJunk_shkp2 = np.load(mementarDataDir+"outputSpeechClusterIsJunk_shkp2.npy")

        bl5_outputDidActionBits = np.load(mementarDataDir+"outputDidActionBits.npy")
        bl5_inputVectorsCombined = np.load(mementarDataDir+"inputVectors.npy")
    
    elif bl6_run:
        humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(mementarDataDir+"humanReadableInputsOutputs.csv")
        bl6_outputActionIDs_shkp1 = np.load(mementarDataDir+"outputActionIDs_shkp1.npy")
        bl6_outputSpeechClusterIDs_shkp1 = np.load(mementarDataDir+"outputSpeechClusterIDs_shkp1.npy")
        bl6_outputSpatialInfo_shkp1 = np.load(mementarDataDir+"outputSpatialInfo_shkp1.npy")
        bl6_outputSpeechClusterIsJunk_shkp1 = np.load(mementarDataDir+"outputSpeechClusterIsJunk_shkp1.npy")

        bl6_outputActionIDs_shkp2 = np.load(mementarDataDir+"outputActionIDs_shkp2.npy")
        bl6_outputSpeechClusterIDs_shkp2 = np.load(mementarDataDir+"outputSpeechClusterIDs_shkp2.npy")
        bl6_outputSpatialInfo_shkp2 = np.load(mementarDataDir+"outputSpatialInfo_shkp2.npy")
        bl6_outputSpeechClusterIsJunk_shkp2 = np.load(mementarDataDir+"outputSpeechClusterIsJunk_shkp2.npy")

        bl6_outputDidActionBits = np.load(mementarDataDir+"outputDidActionBits.npy")
        bl6_inputVectorsCombined = np.load(mementarDataDir+"inputVectors.npy")
    
    elif bl7_run:
        humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(mementarDataDir+"humanReadableInputsOutputs.csv")
        bl7_outputActionIDs_shkp1 = np.load(mementarDataDir+"outputActionIDs_shkp1.npy")
        bl7_outputSpeechClusterIDs_shkp1 = np.load(mementarDataDir+"outputSpeechClusterIDs_shkp1.npy")
        bl7_outputSpatialInfo_shkp1 = np.load(mementarDataDir+"outputSpatialInfo_shkp1.npy")
        bl7_outputSpeechClusterIsJunk_shkp1 = np.load(mementarDataDir+"outputSpeechClusterIsJunk_shkp1.npy")

        bl7_outputActionIDs_shkp2 = np.load(mementarDataDir+"outputActionIDs_shkp2.npy")
        bl7_outputSpeechClusterIDs_shkp2 = np.load(mementarDataDir+"outputSpeechClusterIDs_shkp2.npy")
        bl7_outputSpatialInfo_shkp2 = np.load(mementarDataDir+"outputSpatialInfo_shkp2.npy")
        bl7_outputSpeechClusterIsJunk_shkp2 = np.load(mementarDataDir+"outputSpeechClusterIsJunk_shkp2.npy")

        bl7_outputDidActionBits = np.load(mementarDataDir+"outputDidActionBits.npy")
        bl7_inputVectorsCombined = np.load(mementarDataDir+"inputVectors.npy")
    
    #################################################################################################################
    # setup terminal output / file
    #################################################################################################################
    if DEBUG:
        sessionTerminalOutputLogFile = None
        sessionTerminalOutputStream = sys.stdout
    else:
        sessionTerminalOutputLogFile = sessionDir + "/terminal_output_log_main.txt"
        sessionTerminalOutputStream = open(sessionTerminalOutputLogFile, "a")
    
        
        
    #################################################################################################################
    # split the data into train, val, and test sets
    #################################################################################################################
    print("splitting data...", flush=True, file=sessionTerminalOutputStream)
    
    # split based on experiment sessions
    expIDToIndices = {}
    for i in range(len(humanReadableInputsOutputs)):
        expID = int(humanReadableInputsOutputs[i]["TRIAL"])
        
        if expID not in expIDToIndices:
            expIDToIndices[expID] = []
        
        expIDToIndices[expID].append(i)

    expIDs = list(expIDToIndices.keys())
    
    random.seed(0)
    random.shuffle(expIDs)
    
    expIDFolds = split_list(expIDs, NUM_FOLDS)

    trainExpIDFolds = []
    valExpIDFolds = []
    testExpIDFolds = []

    for i in range(NUM_FOLDS):
        tempTrain = []
        tempVal = []
        tempTest = []
        
        for j in range(i, i+numTrainFolds):
            tempTrain += expIDFolds[j % NUM_FOLDS]
        for j in range(i+numTrainFolds, i+numTrainFolds+numValFolds):
            tempVal += expIDFolds[j % NUM_FOLDS]
        for j in range(i+numTrainFolds+numValFolds, i+numTrainFolds+numValFolds+numTestFolds):
            tempTest += expIDFolds[j % NUM_FOLDS]
        
        trainExpIDFolds.append(tempTrain)
        valExpIDFolds.append(tempVal)
        testExpIDFolds.append(tempTest)
        
   
    trainIndexFolds = [] # each training set should consist of all the data from 8 databases
    valIndexFolds = [] # each validation set should consist of all the data from 2 databases
    testIndexFolds = [] # each testing set should consist of all the data from 1 database

    for i in range(NUM_FOLDS):
        trainIndexFolds.append([j for expID in trainExpIDFolds[i] for j in expIDToIndices[expID]])
        valIndexFolds.append([j for expID in valExpIDFolds[i] for j in expIDToIndices[expID]])
        testIndexFolds.append([j for expID in testExpIDFolds[i] for j in expIDToIndices[expID]])

    
    
    # for debugging...
    #trainSplits[0] = trainSplits[0][:batchSize]
    #valSplits = valSplits[0][:batchSize]
    #testSplits = testSplits[0][:batchSize]
    
    
    #################################################################################################################
    # parralell process each fold
    #################################################################################################################
    
    def run_fold(randomSeed, foldId, gpu):
        #random.seed(randomSeed)

        #################################################################################################################
        # setup logging directories
        #################################################################################################################
        foldIdentifier = "rs{}_fold{}".format(randomSeed, foldId)
        
        foldDir = tools.create_directory(sessionDir + "/" + foldIdentifier)
        modelDir = tools.create_directory(foldDir+"/model")
        
        
        foldLogFile = sessionDir + "/fold_log_{}.csv".format(foldIdentifier)
        foldTerminalOutputLogFile = foldDir + "/terminal_output_log_{}.txt".format(foldIdentifier)
        
        
        if DEBUG:
            foldTerminalOutputStream = sys.stdout
        else:
            foldTerminalOutputStream = open(foldTerminalOutputLogFile, "a")
        
        
        #
        # training / testing aggregate scores log file
        # and training / testing outputs log file
        #
        if bl1_run:
            interactionsFieldnames = ["SET", "ID"] + humanReadableInputsOutputsFieldnames + ["TARG_SHOPKEEPER_ACTION", "PRED_SHOPKEEPER_ACTION"]
            
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Action Loss Ave ({})".format(foldIdentifier),
                                 "Training Action Loss SD ({})".format(foldIdentifier),

                                 "Training Action Accuracy ({})".format(foldIdentifier),
                                 "Training Action 1 Precision ({})".format(foldIdentifier),
                                 "Training Action 1 Recall ({})".format(foldIdentifier),
                                 "Training Action 1 F-score ({})".format(foldIdentifier),
                                 "Training Action 1 Support ({})".format(foldIdentifier),

                                 "Training Action 0 Precision ({})".format(foldIdentifier),
                                 "Training Action 0 Recall ({})".format(foldIdentifier),
                                 "Training Action 0 F-score ({})".format(foldIdentifier),
                                 "Training Action 0 Support ({})".format(foldIdentifier),


                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "Validation Action Loss SD ({})".format(foldIdentifier),

                                 "Validation Action Accuracy ({})".format(foldIdentifier),
                                 "Validation Action 1 Precision ({})".format(foldIdentifier),
                                 "Validation Action 1 Recall ({})".format(foldIdentifier),
                                 "Validation Action 1 F-score ({})".format(foldIdentifier),
                                 "Validation Action 1 Support ({})".format(foldIdentifier),

                                 "Validation Action 0 Precision ({})".format(foldIdentifier),
                                 "Validation Action 0 Recall ({})".format(foldIdentifier),
                                 "Validation Action 0 F-score ({})".format(foldIdentifier),
                                 "Validation Action 0 Support ({})".format(foldIdentifier),

                                                                  
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "Testing Action Loss SD ({})".format(foldIdentifier),

                                 "Testing Action Accuracy ({})".format(foldIdentifier),
                                 "Testing Action 1 Precision ({})".format(foldIdentifier),
                                 "Testing Action 1 Recall ({})".format(foldIdentifier),
                                 "Testing Action 1 F-score ({})".format(foldIdentifier),
                                 "Testing Action 1 Support ({})".format(foldIdentifier),

                                 "Testing Action 0 Precision ({})".format(foldIdentifier),
                                 "Testing Action 0 Recall ({})".format(foldIdentifier),
                                 "Testing Action 0 F-score ({})".format(foldIdentifier),
                                 "Testing Action 0 Support ({})".format(foldIdentifier) ])
            
            
        elif bl2_run:
            interactionsFieldnames = ["SET", "ID"] + humanReadableInputsOutputsFieldnames + ["LOSS_WEIGHT",
                                                                                             'PRED_SHOPKEEPER_ACTION_CLUSTER',
                                                                                             'PRED_SHOPKEEPER_SPEECH_CLUSTER',  
                                                                                             'PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE', 
                                                                                             'PRED_SHOPKEEPER_SPATIAL_INFO',
                                                                                             'PRED_SHOPKEEPER_SPATIAL_INFO_NAME' 
                                                                                             ]
            
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Action Loss Ave ({})".format(foldIdentifier),
                                 "Training Action Loss SD ({})".format(foldIdentifier),

                                 "Training Action Accuracy ({})".format(foldIdentifier),
                                 "Training Speech Accuracy ({})".format(foldIdentifier),
                                 "Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "Validation Action Loss SD ({})".format(foldIdentifier),

                                 "Validation Action Accuracy ({})".format(foldIdentifier),
                                 "Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "Testing Action Loss SD ({})".format(foldIdentifier),

                                 "Testing Action Accuracy ({})".format(foldIdentifier),
                                 "Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "Testing Spatial Accuracy ({})".format(foldIdentifier) ])
        

        elif bl3_run:
            interactionsFieldnames = ["SET", "ID"] + humanReadableInputsOutputsFieldnames + ["LOSS_WEIGHT",
                                                                                             'PRED_SHOPKEEPER_SPEECH_CLUSTER',
                                                                                             'PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE', 
                                                                                             'PRED_SHOPKEEPER_SPATIAL_INFO',
                                                                                             'PRED_SHOPKEEPER_SPATIAL_INFO_NAME' 
                                                                                             ]
            
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Speech Loss Ave ({})".format(foldIdentifier),
                                 "Training Speech Loss SD ({})".format(foldIdentifier),
                                 "Training Motion Loss Ave ({})".format(foldIdentifier),
                                 "Training Motion Loss SD ({})".format(foldIdentifier),

                                 "Training Action Accuracy ({})".format(foldIdentifier),
                                 "Training Speech Accuracy ({})".format(foldIdentifier),
                                 "Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Speech Loss Ave ({})".format(foldIdentifier),
                                 "Validation Speech Loss SD ({})".format(foldIdentifier),
                                 "Validation Motion Loss Ave ({})".format(foldIdentifier),
                                 "Validation Motion Loss SD ({})".format(foldIdentifier),

                                 "Validation Action Accuracy ({})".format(foldIdentifier),
                                 "Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Speech Loss Ave ({})".format(foldIdentifier),
                                 "Testing Speech Loss SD ({})".format(foldIdentifier),
                                 "Testing Motion Loss Ave ({})".format(foldIdentifier),
                                 "Testing Motion Loss SD ({})".format(foldIdentifier),


                                 "Testing Action Accuracy ({})".format(foldIdentifier),
                                 "Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "Testing Spatial Accuracy ({})".format(foldIdentifier) ])
        

        elif bl4_run:
            interactionsFieldnames = ["SET", "ID"] + humanReadableInputsOutputsFieldnames + ["SHOPKEEPER_1_LOSS_WEIGHT",
                                                                                             'PRED_SHOPKEEPER_1_ACTION_CLUSTER',
                                                                                             'PRED_SHOPKEEPER_1_SPEECH_CLUSTER',  
                                                                                             'PRED_SHOPKEEPER_1_REPRESENTATIVE_UTTERANCE', 
                                                                                             'PRED_SHOPKEEPER_1_SPATIAL_INFO',
                                                                                             'PRED_SHOPKEEPER_1_SPATIAL_INFO_NAME',
                                                                                             "SHOPKEEPER_2_LOSS_WEIGHT",
                                                                                             'PRED_SHOPKEEPER_2_ACTION_CLUSTER',
                                                                                             'PRED_SHOPKEEPER_2_SPEECH_CLUSTER',  
                                                                                             'PRED_SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE', 
                                                                                             'PRED_SHOPKEEPER_2_SPATIAL_INFO',
                                                                                             'PRED_SHOPKEEPER_2_SPATIAL_INFO_NAME' 
                                                                                             ]
            
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Action Loss Ave ({})".format(foldIdentifier),
                                 "Training Action Loss SD ({})".format(foldIdentifier),

                                 "Training Action Accuracy ({})".format(foldIdentifier),
                                 "Training Speech Accuracy ({})".format(foldIdentifier),
                                 "Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "Validation Action Loss SD ({})".format(foldIdentifier),

                                 "Validation Action Accuracy ({})".format(foldIdentifier),
                                 "Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "Testing Action Loss SD ({})".format(foldIdentifier),

                                 "Testing Action Accuracy ({})".format(foldIdentifier),
                                 "Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "Testing Spatial Accuracy ({})".format(foldIdentifier),
                                  
                                 "S1 Training Loss Ave ({})".format(foldIdentifier), 
                                 "S1 Training Loss SD ({})".format(foldIdentifier),
                                 "S1 Training Action Loss Ave ({})".format(foldIdentifier),
                                 "S1 Training Action Loss SD ({})".format(foldIdentifier),

                                 "S1 Training Action Accuracy ({})".format(foldIdentifier),
                                 "S1 Training Speech Accuracy ({})".format(foldIdentifier),
                                 "S1 Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "S1 Validation Loss Ave ({})".format(foldIdentifier), 
                                 "S1 Validation Loss SD ({})".format(foldIdentifier),
                                 "S1 Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "S1 Validation Action Loss SD ({})".format(foldIdentifier),

                                 "S1 Validation Action Accuracy ({})".format(foldIdentifier),
                                 "S1 Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "S1 Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "S1 Testing Loss Ave ({})".format(foldIdentifier), 
                                 "S1 Testing Loss SD ({})".format(foldIdentifier),
                                 "S1 Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "S1 Testing Action Loss SD ({})".format(foldIdentifier),

                                 "S1 Testing Action Accuracy ({})".format(foldIdentifier),
                                 "S1 Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "S1 Testing Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "S2 Training Loss Ave ({})".format(foldIdentifier), 
                                 "S2 Training Loss SD ({})".format(foldIdentifier),
                                 "S2 Training Action Loss Ave ({})".format(foldIdentifier),
                                 "S2 Training Action Loss SD ({})".format(foldIdentifier),

                                 "S2 Training Action Accuracy ({})".format(foldIdentifier),
                                 "S2 Training Speech Accuracy ({})".format(foldIdentifier),
                                 "S2 Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "S2 Validation Loss Ave ({})".format(foldIdentifier), 
                                 "S2 Validation Loss SD ({})".format(foldIdentifier),
                                 "S2 Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "S2 Validation Action Loss SD ({})".format(foldIdentifier),

                                 "S2 Validation Action Accuracy ({})".format(foldIdentifier),
                                 "S2 Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "S2 Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "S2 Testing Loss Ave ({})".format(foldIdentifier), 
                                 "S2 Testing Loss SD ({})".format(foldIdentifier),
                                 "S2 Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "S2 Testing Action Loss SD ({})".format(foldIdentifier),

                                 "S2 Testing Action Accuracy ({})".format(foldIdentifier),
                                 "S2 Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "S2 Testing Spatial Accuracy ({})".format(foldIdentifier)])
                                 
        
        elif bl5_run:
            interactionsFieldnames = ["SET", "ID"] + humanReadableInputsOutputsFieldnames + ["SHOPKEEPER_2_LOSS_WEIGHT",
                                                                                             'PRED_SHOPKEEPER_2_ACTION_CLUSTER',
                                                                                             'PRED_SHOPKEEPER_2_SPEECH_CLUSTER',  
                                                                                             'PRED_SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE', 
                                                                                             'PRED_SHOPKEEPER_2_SPATIAL_INFO',
                                                                                             'PRED_SHOPKEEPER_2_SPATIAL_INFO_NAME' 
                                                                                             ]
            
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Action Loss Ave ({})".format(foldIdentifier),
                                 "Training Action Loss SD ({})".format(foldIdentifier),

                                 "Training Action Accuracy ({})".format(foldIdentifier),
                                 "Training Speech Accuracy ({})".format(foldIdentifier),
                                 "Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "Validation Action Loss SD ({})".format(foldIdentifier),

                                 "Validation Action Accuracy ({})".format(foldIdentifier),
                                 "Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "Testing Action Loss SD ({})".format(foldIdentifier),

                                 "Testing Action Accuracy ({})".format(foldIdentifier),
                                 "Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "Testing Spatial Accuracy ({})".format(foldIdentifier)])
            
        elif bl6_run:
            interactionsFieldnames = ["SET", "ID"] + humanReadableInputsOutputsFieldnames + ["SHOPKEEPER_1_LOSS_WEIGHT",
                                                                                             'PRED_SHOPKEEPER_1_ACTION_CLUSTER',
                                                                                             'PRED_SHOPKEEPER_1_SPEECH_CLUSTER',  
                                                                                             'PRED_SHOPKEEPER_1_REPRESENTATIVE_UTTERANCE', 
                                                                                             'PRED_SHOPKEEPER_1_SPATIAL_INFO',
                                                                                             'PRED_SHOPKEEPER_1_SPATIAL_INFO_NAME',
                                                                                             "SHOPKEEPER_2_LOSS_WEIGHT",
                                                                                             'PRED_SHOPKEEPER_2_ACTION_CLUSTER',
                                                                                             'PRED_SHOPKEEPER_2_SPEECH_CLUSTER',  
                                                                                             'PRED_SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE', 
                                                                                             'PRED_SHOPKEEPER_2_SPATIAL_INFO',
                                                                                             'PRED_SHOPKEEPER_2_SPATIAL_INFO_NAME' 
                                                                                             ]
            
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Action Loss Ave ({})".format(foldIdentifier),
                                 "Training Action Loss SD ({})".format(foldIdentifier),

                                 "Training Action Accuracy ({})".format(foldIdentifier),
                                 "Training Speech Accuracy ({})".format(foldIdentifier),
                                 "Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "Validation Action Loss SD ({})".format(foldIdentifier),

                                 "Validation Action Accuracy ({})".format(foldIdentifier),
                                 "Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "Testing Action Loss SD ({})".format(foldIdentifier),

                                 "Testing Action Accuracy ({})".format(foldIdentifier),
                                 "Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "Testing Spatial Accuracy ({})".format(foldIdentifier),
                                  
                                 "S1 Training Loss Ave ({})".format(foldIdentifier), 
                                 "S1 Training Loss SD ({})".format(foldIdentifier),
                                 "S1 Training Action Loss Ave ({})".format(foldIdentifier),
                                 "S1 Training Action Loss SD ({})".format(foldIdentifier),

                                 "S1 Training Action Accuracy ({})".format(foldIdentifier),
                                 "S1 Training Speech Accuracy ({})".format(foldIdentifier),
                                 "S1 Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "S1 Validation Loss Ave ({})".format(foldIdentifier), 
                                 "S1 Validation Loss SD ({})".format(foldIdentifier),
                                 "S1 Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "S1 Validation Action Loss SD ({})".format(foldIdentifier),

                                 "S1 Validation Action Accuracy ({})".format(foldIdentifier),
                                 "S1 Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "S1 Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "S1 Testing Loss Ave ({})".format(foldIdentifier), 
                                 "S1 Testing Loss SD ({})".format(foldIdentifier),
                                 "S1 Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "S1 Testing Action Loss SD ({})".format(foldIdentifier),

                                 "S1 Testing Action Accuracy ({})".format(foldIdentifier),
                                 "S1 Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "S1 Testing Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "S2 Training Loss Ave ({})".format(foldIdentifier), 
                                 "S2 Training Loss SD ({})".format(foldIdentifier),
                                 "S2 Training Action Loss Ave ({})".format(foldIdentifier),
                                 "S2 Training Action Loss SD ({})".format(foldIdentifier),

                                 "S2 Training Action Accuracy ({})".format(foldIdentifier),
                                 "S2 Training Speech Accuracy ({})".format(foldIdentifier),
                                 "S2 Training Spatial Accuracy ({})".format(foldIdentifier),
                                 

                                 "S2 Validation Loss Ave ({})".format(foldIdentifier), 
                                 "S2 Validation Loss SD ({})".format(foldIdentifier),
                                 "S2 Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "S2 Validation Action Loss SD ({})".format(foldIdentifier),

                                 "S2 Validation Action Accuracy ({})".format(foldIdentifier),
                                 "S2 Validation Speech Accuracy ({})".format(foldIdentifier),
                                 "S2 Validation Spatial Accuracy ({})".format(foldIdentifier),

                                                                  
                                 "S2 Testing Loss Ave ({})".format(foldIdentifier), 
                                 "S2 Testing Loss SD ({})".format(foldIdentifier),
                                 "S2 Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "S2 Testing Action Loss SD ({})".format(foldIdentifier),

                                 "S2 Testing Action Accuracy ({})".format(foldIdentifier),
                                 "S2 Testing Speech Accuracy ({})".format(foldIdentifier),
                                 "S2 Testing Spatial Accuracy ({})".format(foldIdentifier)])
        

        if bl7_run:
            interactionsFieldnames = ["SET", "ID"] + humanReadableInputsOutputsFieldnames + ["TARG_SHOPKEEPER_ACTION", "PRED_SHOPKEEPER_ACTION"]
            
            with open(foldLogFile, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoch",
                                 
                                 "Training Loss Ave ({})".format(foldIdentifier), 
                                 "Training Loss SD ({})".format(foldIdentifier),
                                 "Training Action Loss Ave ({})".format(foldIdentifier),
                                 "Training Action Loss SD ({})".format(foldIdentifier),

                                 "Training Action Accuracy ({})".format(foldIdentifier),
                                 "Training Action 1 Precision ({})".format(foldIdentifier),
                                 "Training Action 1 Recall ({})".format(foldIdentifier),
                                 "Training Action 1 F-score ({})".format(foldIdentifier),
                                 "Training Action 1 Support ({})".format(foldIdentifier),

                                 "Training Action 0 Precision ({})".format(foldIdentifier),
                                 "Training Action 0 Recall ({})".format(foldIdentifier),
                                 "Training Action 0 F-score ({})".format(foldIdentifier),
                                 "Training Action 0 Support ({})".format(foldIdentifier),


                                 "Validation Loss Ave ({})".format(foldIdentifier), 
                                 "Validation Loss SD ({})".format(foldIdentifier),
                                 "Validation Action Loss Ave ({})".format(foldIdentifier),
                                 "Validation Action Loss SD ({})".format(foldIdentifier),

                                 "Validation Action Accuracy ({})".format(foldIdentifier),
                                 "Validation Action 1 Precision ({})".format(foldIdentifier),
                                 "Validation Action 1 Recall ({})".format(foldIdentifier),
                                 "Validation Action 1 F-score ({})".format(foldIdentifier),
                                 "Validation Action 1 Support ({})".format(foldIdentifier),

                                 "Validation Action 0 Precision ({})".format(foldIdentifier),
                                 "Validation Action 0 Recall ({})".format(foldIdentifier),
                                 "Validation Action 0 F-score ({})".format(foldIdentifier),
                                 "Validation Action 0 Support ({})".format(foldIdentifier),

                                                                  
                                 "Testing Loss Ave ({})".format(foldIdentifier), 
                                 "Testing Loss SD ({})".format(foldIdentifier),
                                 "Testing Action Loss Ave ({})".format(foldIdentifier),
                                 "Testing Action Loss SD ({})".format(foldIdentifier),

                                 "Testing Action Accuracy ({})".format(foldIdentifier),
                                 "Testing Action 1 Precision ({})".format(foldIdentifier),
                                 "Testing Action 1 Recall ({})".format(foldIdentifier),
                                 "Testing Action 1 F-score ({})".format(foldIdentifier),
                                 "Testing Action 1 Support ({})".format(foldIdentifier),

                                 "Testing Action 0 Precision ({})".format(foldIdentifier),
                                 "Testing Action 0 Recall ({})".format(foldIdentifier),
                                 "Testing Action 0 F-score ({})".format(foldIdentifier),
                                 "Testing Action 0 Support ({})".format(foldIdentifier) ])
                
        elif prop_run:
            pass
        
        
        #################################################################################################################
        # 
        #################################################################################################################
        trainIndices = trainIndexFolds[foldId]
        valIndices = valIndexFolds[foldId]
        testIndices = testIndexFolds[foldId]
        
        
        #################################################################################################################
        # get set of actions, compute class weights
        #################################################################################################################
        
        if bl1_run:
            bl1_outputClassWeights = np.ones((2))
            bl1_outputTargets = np.zeros(bl1_outputDidActionBits.shape[0])
            bl1_outputTargets[np.where(bl1_outputDidActionBits[:,2] == 1)] = 1


            bl1_isValidCondition = [] # for now, only inlcude standard S2 roles
            for row in humanReadableInputsOutputs:
                if row["SHOPKEEPER2_TYPE"] == "NORMAL":
                    bl1_isValidCondition.append(1)
                else:
                    bl1_isValidCondition.append(0)
            
            
            bl1_outputMasks = np.ones(bl1_outputDidActionBits.shape[0])
            
            # only evaluate using certain roles
            bl1_isValidCondition = np.asarray(bl1_isValidCondition)
            bl1_outputMasks[bl1_isValidCondition == 0] = 0

        
        elif bl2_run:

            # get action and speech cluster info from the interaction data file
            bl2_actionClustCounts = {}
            bl2_speechClustCounts = {}
            bl2_actionClustIDToSpeechClustID = {}
            bl2_actionClustIDToSpatial = {}
            bl2_actionClustIDToSpatialName = {}
            bl2_speechClustIDToRepUtt = {}
            bl2_speechClustIsJunk = {}
            
            bl2_isValidCondition = [] # for now, only inlcude standard S2 roles
            
            for row in humanReadableInputsOutputs:
                if row["y_SHOPKEEPER_2_ACTION_CLUSTER"] == "":
                    actionClustID = ""
                else:
                    actionClustID = int(row["y_SHOPKEEPER_2_ACTION_CLUSTER"])
                spatialInfo = int(row["y_SHOPKEEPER_2_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_2_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER"])
                repUtt = row["y_SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE"]
                isJunk = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER_IS_JUNK"])

                bl2_actionClustIDToSpeechClustID[actionClustID] = speechClustID
                bl2_actionClustIDToSpatial[actionClustID] = spatialInfo
                bl2_actionClustIDToSpatialName[actionClustID] = spatialInfoName
                bl2_speechClustIDToRepUtt[speechClustID] = repUtt
                bl2_speechClustIsJunk[speechClustID] = isJunk

                if actionClustID not in bl2_actionClustCounts:
                    bl2_actionClustCounts[actionClustID] = 0
                bl2_actionClustCounts[actionClustID] += 1

                if speechClustID not in bl2_speechClustCounts:
                    bl2_speechClustCounts[speechClustID] = 0
                bl2_speechClustCounts[speechClustID] += 1
                
                
                # TODO maybe it's better to keep track of cluster counts separately for each shopkeeper?
                # clusters from shopkeeper 1
                if row["y_SHOPKEEPER_1_ACTION_CLUSTER"] != "":
                    actionClustID = int(row["y_SHOPKEEPER_1_ACTION_CLUSTER"])
                else:
                    actionClustID = ""
                
                actionClustID = int(row["y_SHOPKEEPER_1_ACTION_CLUSTER"])
                spatialInfo = int(row["y_SHOPKEEPER_1_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_1_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_1_SPEECH_CLUSTER"])
                repUtt = row["y_SHOPKEEPER_1_REPRESENTATIVE_UTTERANCE"]
                isJunk = int(row["y_SHOPKEEPER_1_SPEECH_CLUSTER_IS_JUNK"])

                bl2_actionClustIDToSpeechClustID[actionClustID] = speechClustID
                bl2_actionClustIDToSpatial[actionClustID] = spatialInfo
                bl2_actionClustIDToSpatialName[actionClustID] = spatialInfoName
                bl2_speechClustIDToRepUtt[speechClustID] = repUtt
                bl2_speechClustIsJunk[speechClustID] = isJunk

                if actionClustID not in bl2_actionClustCounts:
                    bl2_actionClustCounts[actionClustID] = 0
                bl2_actionClustCounts[actionClustID] += 1

                if speechClustID not in bl2_speechClustCounts:
                    bl2_speechClustCounts[speechClustID] = 0
                bl2_speechClustCounts[speechClustID] += 1


                #
                if row["SHOPKEEPER2_TYPE"] == "NORMAL":
                    bl2_isValidCondition.append(1)
                else:
                    bl2_isValidCondition.append(0)
            
            bl2_numActionClusters = len(bl2_actionClustIDToSpeechClustID)

            # give the null action an ID
            nullActionID = bl2_numActionClusters - 1

            bl2_actionClustIDToSpeechClustID[nullActionID] = bl2_actionClustIDToSpeechClustID[-1]
            bl2_actionClustIDToSpatial[nullActionID] = bl2_actionClustIDToSpatial[-1]
            bl2_actionClustIDToSpatialName[nullActionID] = bl2_actionClustIDToSpatialName[-1]

            # replace null actions marked with -1 with the null action ID
            bl2_outputActionIDs_shkp2[np.where(bl2_outputActionIDs_shkp2 == -1)] = nullActionID
            bl2_actionClustCounts[nullActionID] = bl2_actionClustCounts[-1]
            
            bl2_outputClassWeights = np.ones((bl2_numActionClusters))

            # set null action and junk speech clusters to 0 weight 
            bl2_outputMasks = np.copy(bl2_outputSpeechClusterIsJunk_shkp2)
            bl2_outputMasks[bl2_outputMasks == -1] = 1 # -1 marks non S2 actions
            bl2_outputMasks = 1 -bl2_outputMasks
            
            # set actions with less than min count to 0 weight

            # set count of missing classes to 0 (some might be missing becase here we only look at S2 actions)
            for actionID in range(max(bl2_outputActionIDs_shkp2)):
                if actionID not in bl2_actionClustCounts:
                    bl2_actionClustCounts[actionID] = 0
                
            
            bl2_actionClustOverMinCount = np.asarray([0 if bl2_actionClustCounts[x] < minClassCount else 1 for x in bl2_outputActionIDs_shkp2])
            bl2_outputMasks[bl2_actionClustOverMinCount == 0] = 0
            
            # only evaluate using certain roles
            bl2_isValidCondition = np.asarray(bl2_isValidCondition)
            bl2_outputMasks[bl2_isValidCondition == 0] = 0
        

        elif bl3_run:
            # get action and speech cluster info from the interaction data file
            bl3_speechClustCounts = {}
            bl3_motionIDToSpatialName = {}
            bl3_isValidCondition = [] # for now, only inlcude standard S2 roles
            
            for row in humanReadableInputsOutputs:
                spatialInfo = int(row["y_SHOPKEEPER_2_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_2_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER"])

                if speechClustID not in bl3_speechClustCounts:
                    bl3_speechClustCounts[speechClustID] = 0
                
                if spatialInfo not in bl3_motionIDToSpatialName:
                    bl3_motionIDToSpatialName[spatialInfo] = spatialInfoName
                
                if row["SHOPKEEPER2_TYPE"] == "NORMAL":
                    bl3_isValidCondition.append(1)
                    bl3_speechClustCounts[speechClustID] += 1
                else:
                    bl3_isValidCondition.append(0)

            #bl3_numSpeechClusters = len(bl3_speechClustCounts)
            bl3_maxSpeechClusterID = max(list(bl3_speechClustCounts.keys()))

            # assign the null speech cluster ID to something that can be fed to the neural network
            nullSpeechClusterID = bl3_maxSpeechClusterID + 1
            bl3_maxSpeechClusterID = nullSpeechClusterID

            bl3_speechClustCounts[nullSpeechClusterID] = bl3_speechClustCounts[-1]
            bl3_speechClustCounts.pop(-1)

            bl3_speechClustIDToRepUtt[nullSpeechClusterID] = ""
            bl3_speechClustIDToIsJunk[nullSpeechClusterID] = 0
            
            # replace null actions marked with -1 with the null action ID
            bl3_outputSpeechClusterIDs[np.where(bl3_outputSpeechClusterIDs == -1)] = nullSpeechClusterID
            
            # unused
            bl3_outputSpeechClassWeights = np.ones((bl3_maxSpeechClusterID))

            # set null action to 0 weight 
            bl3_outputMasks = np.copy(bl3_toIgnore)
            bl3_outputMasks[bl3_outputMasks == -1] = 1 # -1 marks non S2 actions
            bl3_outputMasks = 1 -bl3_outputMasks

            # set junk speech clusters to 0 weight 
            bl3_speechClustIsJunk = np.asarray([1 if bl3_speechClustIDToIsJunk[x] == 1 else 0 for x in bl3_outputSpeechClusterIDs])
            bl3_outputMasks[bl3_speechClustIsJunk == 1] = 0

            # set actions with less than min count to 0 weight
            bl3_speechClustOverMinCount = np.asarray([0 if bl3_speechClustCounts[x] < minClassCount else 1 for x in bl3_outputSpeechClusterIDs])
            bl3_outputMasks[bl3_speechClustOverMinCount == 0] = 0
            
            # only evaluate using certain roles
            bl3_isValidCondition = np.asarray(bl3_isValidCondition)
            bl3_outputMasks[bl3_isValidCondition == 0] = 0

            # for motion output
            bl3_maxMotionID = max(bl3_outputSpatialInfo)
            bl3_outputSpatialInfo[bl3_outputSpatialInfo == -1] = 0 # change to something we can input to the network
            bl3_motionIDToSpatialName[0] = bl3_motionIDToSpatialName[-1]
            bl3_outputMotionClassWeights = np.ones((bl3_maxMotionID))

            #
            # for speech clusters
            #
            """
            if SPEECH_CLUSTER_LOSS_WEIGHTS:
                # count number of occurrences of each speech cluster in the training dataset
                bl1_speechClustCounts = {}
                
                for i in trainIndices:
                    speechClustId = bl1_outputSpeechClusterIds[i]
                    
                    if speechClustId not in bl1_speechClustCounts:
                        bl1_speechClustCounts[speechClustId] = 0
                    bl1_speechClustCounts[speechClustId] += 1
                
                # remove junk cluster counts
                for speechClustId in bl1_junkSpeechClusterIds:
                    del bl1_speechClustCounts[speechClustId]
                
                numSamples = sum(bl1_speechClustCounts.values())
                
                
                # compute weights
                bl1_speechClustWeights = [None] * bl1_numSpeechClusters
                
                for clustId in bl1_speechClustIdToShkpUtts:
                    # as in scikit learn - The “balanced” heuristic is inspired by Logistic Regression in Rare Events Data, King, Zen, 2001.
                    if clustId in bl1_speechClustCounts:
                        bl1_speechClustWeights[clustId] = numSamples / ((bl1_numSpeechClusters-len(bl1_junkSpeechClusterIds)) * bl1_speechClustCounts[clustId])
                    else:
                        # sometimes a cluster won't appear in the training set, so give it a weight of 1
                        bl1_speechClustWeights[clustId] = 1.0
                        
                if None in bl1_speechClustWeights:
                    print("WARNING: missing training weight for BASELINE 1 shopkeeper speech cluster!", flush=True, file=foldTerminalOutputStream)
            
            else:
                bl1_speechClustWeights = []
                for clustId in range(bl1_numSpeechClusters):
                    bl1_speechClustWeights.append(1.0)
            
            
            for clustId in bl1_junkSpeechClusterIds:
                bl1_speechClustWeights[clustId] = 0.0
            """
        
        elif bl4_run:
            # get action and speech cluster info from the interaction data file
            bl4_actionClustCounts = {}
            bl4_speechClustCounts = {}
            bl4_actionClustIDToSpeechClustID = {}
            bl4_actionClustIDToSpatial = {}
            bl4_actionClustIDToSpatialName = {}
            bl4_speechClustIDToRepUtt = {}
            bl4_speechClustIsJunk = {}
            
            bl4_isValidCondition = [] # for now, only inlcude standard S2 roles
            
            for row in humanReadableInputsOutputs:
                # clusters from shopkeeper 2
                if row["y_SHOPKEEPER_2_ACTION_CLUSTER"] != "":
                    actionClustID = int(row["y_SHOPKEEPER_2_ACTION_CLUSTER"])
                else:
                    actionClustID = ""
                
                spatialInfo = int(row["y_SHOPKEEPER_2_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_2_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER"])
                repUtt = row["y_SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE"]
                isJunk = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER_IS_JUNK"])

                bl4_actionClustIDToSpeechClustID[actionClustID] = speechClustID
                bl4_actionClustIDToSpatial[actionClustID] = spatialInfo
                bl4_actionClustIDToSpatialName[actionClustID] = spatialInfoName
                bl4_speechClustIDToRepUtt[speechClustID] = repUtt
                bl4_speechClustIsJunk[speechClustID] = isJunk

                if actionClustID not in bl4_actionClustCounts:
                    bl4_actionClustCounts[actionClustID] = 0
                bl4_actionClustCounts[actionClustID] += 1

                if speechClustID not in bl4_speechClustCounts:
                    bl4_speechClustCounts[speechClustID] = 0
                bl4_speechClustCounts[speechClustID] += 1


                # TODO maybe it's better to keep track of cluster counts separately for each shopkeeper?
                # clusters from shopkeeper 2
                if row["y_SHOPKEEPER_1_ACTION_CLUSTER"] != "":
                    actionClustID = int(row["y_SHOPKEEPER_1_ACTION_CLUSTER"])
                else:
                    actionClustID = ""
                
                actionClustID = int(row["y_SHOPKEEPER_1_ACTION_CLUSTER"])
                spatialInfo = int(row["y_SHOPKEEPER_1_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_1_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_1_SPEECH_CLUSTER"])
                repUtt = row["y_SHOPKEEPER_1_REPRESENTATIVE_UTTERANCE"]
                isJunk = int(row["y_SHOPKEEPER_1_SPEECH_CLUSTER_IS_JUNK"])

                bl4_actionClustIDToSpeechClustID[actionClustID] = speechClustID
                bl4_actionClustIDToSpatial[actionClustID] = spatialInfo
                bl4_actionClustIDToSpatialName[actionClustID] = spatialInfoName
                bl4_speechClustIDToRepUtt[speechClustID] = repUtt
                bl4_speechClustIsJunk[speechClustID] = isJunk

                if actionClustID not in bl4_actionClustCounts:
                    bl4_actionClustCounts[actionClustID] = 0
                bl4_actionClustCounts[actionClustID] += 1

                if speechClustID not in bl4_speechClustCounts:
                    bl4_speechClustCounts[speechClustID] = 0
                bl4_speechClustCounts[speechClustID] += 1


                #
                if row["SHOPKEEPER2_TYPE"] == "NORMAL":
                    bl4_isValidCondition.append(1)
                else:
                    bl4_isValidCondition.append(0)
            
            bl4_numActionClusters = len(bl4_actionClustIDToSpeechClustID)

            # give the null action an ID
            nullActionID = bl4_numActionClusters - 1

            bl4_actionClustIDToSpeechClustID[nullActionID] = bl4_actionClustIDToSpeechClustID[-1]
            bl4_actionClustIDToSpatial[nullActionID] = bl4_actionClustIDToSpatial[-1]
            bl4_actionClustIDToSpatialName[nullActionID] = bl4_actionClustIDToSpatialName[-1]

            # replace null actions marked with -1 with the null action ID
            bl4_outputActionIDs_shkp1[np.where(bl4_outputActionIDs_shkp1 == -1)] = nullActionID
            bl4_outputActionIDs_shkp2[np.where(bl4_outputActionIDs_shkp2 == -1)] = nullActionID
            
            bl4_actionClustCounts[nullActionID] = bl4_actionClustCounts[-1]
            
            bl4_outputClassWeights = np.ones((bl4_numActionClusters))

            # set null action and junk speech clusters to 0 weight 
            # shkp 1
            bl4_outputMasks_shkp1 = np.copy(bl4_outputSpeechClusterIsJunk_shkp1)
            bl4_outputMasks_shkp1[bl4_outputMasks_shkp1 == -1] = 1 # -1 marks non S1 actions
            bl4_outputMasks_shkp1 = 1 - bl4_outputMasks_shkp1
            # shkp 2
            bl4_outputMasks_shkp2 = np.copy(bl4_outputSpeechClusterIsJunk_shkp2)
            bl4_outputMasks_shkp2[bl4_outputMasks_shkp2 == -1] = 1 # -1 marks non S2 actions
            bl4_outputMasks_shkp2 = 1 - bl4_outputMasks_shkp2
            
            # set actions with less than min count to 0 weight

            # set count of missing classes to 0
            for actionID in range(max(bl4_outputActionIDs_shkp2)):
                if actionID not in bl4_actionClustCounts:
                    bl4_actionClustCounts[actionID] = 0

            # shkp 1
            bl4_actionClustOverMinCount_shkp1 = np.asarray([0 if bl4_actionClustCounts[x] < minClassCount else 1 for x in bl4_outputActionIDs_shkp1])
            bl4_outputMasks_shkp1[bl4_actionClustOverMinCount_shkp1 == 0] = 0
            # shkp 2
            bl4_actionClustOverMinCount_shkp2 = np.asarray([0 if bl4_actionClustCounts[x] < minClassCount else 1 for x in bl4_outputActionIDs_shkp2])
            bl4_outputMasks_shkp2[bl4_actionClustOverMinCount_shkp2 == 0] = 0

            # only evaluate using certain roles
            bl4_isValidCondition = np.asarray(bl4_isValidCondition)
            bl4_outputMasks_shkp1[bl4_isValidCondition == 0] = 0
            bl4_outputMasks_shkp2[bl4_isValidCondition == 0] = 0

            # additional inputs to tell which shopkeeper's action we want to predict
            bl4_additionalInputs_shkp1 = np.zeros((bl4_inputVectorsCombined.shape[0], numShopkeepers))
            bl4_additionalInputs_shkp1[:,0] = 1
            bl4_additionalInputs_shkp2 = np.zeros((bl4_inputVectorsCombined.shape[0], numShopkeepers))
            bl4_additionalInputs_shkp2[:,1] = 1


        elif bl5_run:
            # get action and speech cluster info from the interaction data file
            bl5_actionClustCounts = {}
            bl5_speechClustCounts = {}
            bl5_actionClustIDToSpeechClustID = {}
            bl5_actionClustIDToSpatial = {}
            bl5_actionClustIDToSpatialName = {}
            bl5_speechClustIDToRepUtt = {}
            bl5_speechClustIsJunk = {}
            
            bl5_isValidCondition = [] # for now, only inlcude standard S2 roles
            
            for row in humanReadableInputsOutputs:
                # clusters from shopkeeper 2
                if row["y_SHOPKEEPER_2_ACTION_CLUSTER"] != "":
                    actionClustID = int(row["y_SHOPKEEPER_2_ACTION_CLUSTER"])
                else:
                    actionClustID = ""
                
                spatialInfo = int(row["y_SHOPKEEPER_2_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_2_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER"])
                repUtt = row["y_SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE"]
                isJunk = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER_IS_JUNK"])

                bl5_actionClustIDToSpeechClustID[actionClustID] = speechClustID
                bl5_actionClustIDToSpatial[actionClustID] = spatialInfo
                bl5_actionClustIDToSpatialName[actionClustID] = spatialInfoName
                bl5_speechClustIDToRepUtt[speechClustID] = repUtt
                bl5_speechClustIsJunk[speechClustID] = isJunk

                if actionClustID not in bl5_actionClustCounts:
                    bl5_actionClustCounts[actionClustID] = 0
                bl5_actionClustCounts[actionClustID] += 1

                if speechClustID not in bl5_speechClustCounts:
                    bl5_speechClustCounts[speechClustID] = 0
                bl5_speechClustCounts[speechClustID] += 1


                # TODO maybe it's better to keep track of cluster counts separately for each shopkeeper?
                # clusters from shopkeeper 1
                if row["y_SHOPKEEPER_1_ACTION_CLUSTER"] != "":
                    actionClustID = int(row["y_SHOPKEEPER_1_ACTION_CLUSTER"])
                else:
                    actionClustID = ""
                
                actionClustID = int(row["y_SHOPKEEPER_1_ACTION_CLUSTER"])
                spatialInfo = int(row["y_SHOPKEEPER_1_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_1_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_1_SPEECH_CLUSTER"])
                repUtt = row["y_SHOPKEEPER_1_REPRESENTATIVE_UTTERANCE"]
                isJunk = int(row["y_SHOPKEEPER_1_SPEECH_CLUSTER_IS_JUNK"])

                bl5_actionClustIDToSpeechClustID[actionClustID] = speechClustID
                bl5_actionClustIDToSpatial[actionClustID] = spatialInfo
                bl5_actionClustIDToSpatialName[actionClustID] = spatialInfoName
                bl5_speechClustIDToRepUtt[speechClustID] = repUtt
                bl5_speechClustIsJunk[speechClustID] = isJunk

                if actionClustID not in bl5_actionClustCounts:
                    bl5_actionClustCounts[actionClustID] = 0
                bl5_actionClustCounts[actionClustID] += 1

                if speechClustID not in bl5_speechClustCounts:
                    bl5_speechClustCounts[speechClustID] = 0
                bl5_speechClustCounts[speechClustID] += 1


                #
                if row["SHOPKEEPER2_TYPE"] == "NORMAL":
                    bl5_isValidCondition.append(1)
                else:
                    bl5_isValidCondition.append(0)
            
            bl5_numActionClusters = len(bl5_actionClustIDToSpeechClustID)

            # give the null action an ID
            nullActionID = bl5_numActionClusters - 1

            bl5_actionClustIDToSpeechClustID[nullActionID] = bl5_actionClustIDToSpeechClustID[-1]
            bl5_actionClustIDToSpatial[nullActionID] = bl5_actionClustIDToSpatial[-1]
            bl5_actionClustIDToSpatialName[nullActionID] = bl5_actionClustIDToSpatialName[-1]

            # replace null actions marked with -1 with the null action ID
            bl5_outputActionIDs_shkp1[np.where(bl5_outputActionIDs_shkp1 == -1)] = nullActionID
            bl5_outputActionIDs_shkp2[np.where(bl5_outputActionIDs_shkp2 == -1)] = nullActionID
            
            bl5_actionClustCounts[nullActionID] = bl5_actionClustCounts[-1]
            
            bl5_outputClassWeights = np.ones((bl5_numActionClusters))

            # set null action and junk speech clusters to 0 weight 
            # shkp 1
            bl5_outputMasks_shkp1 = np.copy(bl5_outputSpeechClusterIsJunk_shkp1)
            bl5_outputMasks_shkp1[bl5_outputMasks_shkp1 == -1] = 1 # -1 marks non S1 actions
            bl5_outputMasks_shkp1 = 1 - bl5_outputMasks_shkp1
            # shkp 2
            bl5_outputMasks_shkp2 = np.copy(bl5_outputSpeechClusterIsJunk_shkp2)
            bl5_outputMasks_shkp2[bl5_outputMasks_shkp2 == -1] = 1 # -1 marks non S2 actions
            bl5_outputMasks_shkp2 = 1 - bl5_outputMasks_shkp2
            
            # set actions with less than min count to 0 weight

            # set count of missing classes to 0
            for actionID in range(max(bl5_outputActionIDs_shkp2)):
                if actionID not in bl5_actionClustCounts:
                    bl5_actionClustCounts[actionID] = 0

            # shkp 1
            #bl5_actionClustOverMinCount_shkp1 = np.asarray([0 if bl5_actionClustCounts[x] < minClassCount else 1 for x in bl5_outputActionIDs_shkp1])
            #bl5_outputMasks_shkp1[bl5_actionClustOverMinCount_shkp1 == 0] = 0
            # shkp 2
            bl5_actionClustOverMinCount_shkp2 = np.asarray([0 if bl5_actionClustCounts[x] < minClassCount else 1 for x in bl5_outputActionIDs_shkp2])
            bl5_outputMasks_shkp2[bl5_actionClustOverMinCount_shkp2 == 0] = 0

            # only evaluate using certain roles
            bl5_isValidCondition = np.asarray(bl5_isValidCondition)
            #bl5_outputMasks_shkp1[bl5_isValidCondition == 0] = 0
            bl5_outputMasks_shkp2[bl5_isValidCondition == 0] = 0

        elif bl6_run:
            # get action and speech cluster info from the interaction data file
            bl6_actionClustCounts = {}
            bl6_speechClustCounts = {}
            bl6_actionClustIDToSpeechClustID = {}
            bl6_actionClustIDToSpatial = {}
            bl6_actionClustIDToSpatialName = {}
            bl6_speechClustIDToRepUtt = {}
            bl6_speechClustIsJunk = {}
            
            bl6_isValidCondition = [] # for now, only inlcude standard S2 roles
            
            for row in humanReadableInputsOutputs:
                # clusters from shopkeeper 2
                if row["y_SHOPKEEPER_2_ACTION_CLUSTER"] != "":
                    actionClustID = int(row["y_SHOPKEEPER_2_ACTION_CLUSTER"])
                else:
                    actionClustID = ""
                
                spatialInfo = int(row["y_SHOPKEEPER_2_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_2_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER"])
                repUtt = row["y_SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE"]
                isJunk = int(row["y_SHOPKEEPER_2_SPEECH_CLUSTER_IS_JUNK"])

                bl6_actionClustIDToSpeechClustID[actionClustID] = speechClustID
                bl6_actionClustIDToSpatial[actionClustID] = spatialInfo
                bl6_actionClustIDToSpatialName[actionClustID] = spatialInfoName
                bl6_speechClustIDToRepUtt[speechClustID] = repUtt
                bl6_speechClustIsJunk[speechClustID] = isJunk

                if actionClustID not in bl6_actionClustCounts:
                    bl6_actionClustCounts[actionClustID] = 0
                bl6_actionClustCounts[actionClustID] += 1

                if speechClustID not in bl6_speechClustCounts:
                    bl6_speechClustCounts[speechClustID] = 0
                bl6_speechClustCounts[speechClustID] += 1


                # TODO maybe it's better to keep track of cluster counts separately for each shopkeeper?
                # clusters from shopkeeper 2
                if row["y_SHOPKEEPER_1_ACTION_CLUSTER"] != "":
                    actionClustID = int(row["y_SHOPKEEPER_1_ACTION_CLUSTER"])
                else:
                    actionClustID = ""
                
                actionClustID = int(row["y_SHOPKEEPER_1_ACTION_CLUSTER"])
                spatialInfo = int(row["y_SHOPKEEPER_1_SPATIAL_INFO"])
                spatialInfoName = row["y_SHOPKEEPER_1_SPATIAL_INFO_NAME"]
                speechClustID = int(row["y_SHOPKEEPER_1_SPEECH_CLUSTER"])
                repUtt = row["y_SHOPKEEPER_1_REPRESENTATIVE_UTTERANCE"]
                isJunk = int(row["y_SHOPKEEPER_1_SPEECH_CLUSTER_IS_JUNK"])

                bl6_actionClustIDToSpeechClustID[actionClustID] = speechClustID
                bl6_actionClustIDToSpatial[actionClustID] = spatialInfo
                bl6_actionClustIDToSpatialName[actionClustID] = spatialInfoName
                bl6_speechClustIDToRepUtt[speechClustID] = repUtt
                bl6_speechClustIsJunk[speechClustID] = isJunk

                if actionClustID not in bl6_actionClustCounts:
                    bl6_actionClustCounts[actionClustID] = 0
                bl6_actionClustCounts[actionClustID] += 1

                if speechClustID not in bl6_speechClustCounts:
                    bl6_speechClustCounts[speechClustID] = 0
                bl6_speechClustCounts[speechClustID] += 1


                #
                if row["SHOPKEEPER2_TYPE"] == "NORMAL":
                    bl6_isValidCondition.append(1)
                else:
                    bl6_isValidCondition.append(0)
            
            bl6_numActionClusters = len(bl6_actionClustIDToSpeechClustID)

            # give the null action an ID
            nullActionID = bl6_numActionClusters - 1

            bl6_actionClustIDToSpeechClustID[nullActionID] = bl6_actionClustIDToSpeechClustID[-1]
            bl6_actionClustIDToSpatial[nullActionID] = bl6_actionClustIDToSpatial[-1]
            bl6_actionClustIDToSpatialName[nullActionID] = bl6_actionClustIDToSpatialName[-1]

            # replace null actions marked with -1 with the null action ID
            bl6_outputActionIDs_shkp1[np.where(bl6_outputActionIDs_shkp1 == -1)] = nullActionID
            bl6_outputActionIDs_shkp2[np.where(bl6_outputActionIDs_shkp2 == -1)] = nullActionID
            
            bl6_actionClustCounts[nullActionID] = bl6_actionClustCounts[-1]
            
            bl6_outputClassWeights = np.ones((bl6_numActionClusters))

            # set null action and junk speech clusters to 0 weight 
            # shkp 1
            bl6_outputMasks_shkp1 = np.copy(bl6_outputSpeechClusterIsJunk_shkp1)
            bl6_outputMasks_shkp1[bl6_outputMasks_shkp1 == -1] = 1 # -1 marks non S1 actions
            bl6_outputMasks_shkp1 = 1 - bl6_outputMasks_shkp1
            # shkp 2
            bl6_outputMasks_shkp2 = np.copy(bl6_outputSpeechClusterIsJunk_shkp2)
            bl6_outputMasks_shkp2[bl6_outputMasks_shkp2 == -1] = 1 # -1 marks non S2 actions
            bl6_outputMasks_shkp2 = 1 - bl6_outputMasks_shkp2
            
            # set actions with less than min count to 0 weight

            # set count of missing classes to 0
            for actionID in range(max(bl6_outputActionIDs_shkp2)):
                if actionID not in bl6_actionClustCounts:
                    bl6_actionClustCounts[actionID] = 0

            # shkp 1
            bl6_actionClustOverMinCount_shkp1 = np.asarray([0 if bl6_actionClustCounts[x] < minClassCount else 1 for x in bl6_outputActionIDs_shkp1])
            bl6_outputMasks_shkp1[bl6_actionClustOverMinCount_shkp1 == 0] = 0
            # shkp 2
            bl6_actionClustOverMinCount_shkp2 = np.asarray([0 if bl6_actionClustCounts[x] < minClassCount else 1 for x in bl6_outputActionIDs_shkp2])
            bl6_outputMasks_shkp2[bl6_actionClustOverMinCount_shkp2 == 0] = 0

            # only evaluate using certain roles
            bl6_isValidCondition = np.asarray(bl6_isValidCondition)
            bl6_outputMasks_shkp1[bl6_isValidCondition == 0] = 0
            bl6_outputMasks_shkp2[bl6_isValidCondition == 0] = 0

            # additional inputs to tell which shopkeeper's action we want to predict
            bl6_additionalInputs_shkp1 = np.zeros((bl6_inputVectorsCombined.shape[0], numShopkeepers))
            bl6_additionalInputs_shkp1[:,0] = 1
            bl6_additionalInputs_shkp2 = np.zeros((bl6_inputVectorsCombined.shape[0], numShopkeepers))
            bl6_additionalInputs_shkp2[:,1] = 1


        if bl7_run:
            bl7_outputClassWeights = np.ones((2))
            bl7_outputTargets = np.zeros(bl7_outputDidActionBits.shape[0])
            bl7_outputTargets[np.where(bl7_outputDidActionBits[:,2] == 1)] = 1


            bl7_isValidCondition = [] # for now, only inlcude standard S2 roles
            for row in humanReadableInputsOutputs:
                if row["SHOPKEEPER2_TYPE"] == "NORMAL":
                    bl7_isValidCondition.append(1)
                else:
                    bl7_isValidCondition.append(0)
            
            
            bl7_outputMasks = np.ones(bl7_outputDidActionBits.shape[0])
            
            # only evaluate using certain roles
            bl7_isValidCondition = np.asarray(bl7_isValidCondition)
            bl7_outputMasks[bl7_isValidCondition == 0] = 0


        if prop_run:
            pass


        
        #################################################################################################################
        # prepare the learning model
        #################################################################################################################
        print("setting up the model...", flush=True, file=foldTerminalOutputStream)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        import learning
        
        if bl1_run:
            bl1_inputDim = bl1_inputVectorsCombined.shape[2]
            bl1_inputSeqLen = bl1_inputVectorsCombined.shape[1]
            bl1_numOutputClasses = 2

            learner = learning.SimpleFeedforwardNetwork(bl1_inputDim, 
                                                        bl1_inputSeqLen, 
                                                        bl1_numOutputClasses,
                                                        batchSize, 
                                                        embeddingDim,
                                                        randomSeed,
                                                        bl1_outputClassWeights,
                                                        useAttention=useAttention)
        
        elif bl2_run:
            bl2_inputDim = bl2_inputVectorsCombined.shape[2]
            bl2_inputSeqLen = bl2_inputVectorsCombined.shape[1]
            bl2_numOutputClasses = bl2_numActionClusters

            learner = learning.SimpleFeedforwardNetwork(bl2_inputDim, 
                                                        bl2_inputSeqLen, 
                                                        bl2_numOutputClasses,
                                                        batchSize, 
                                                        embeddingDim,
                                                        randomSeed,
                                                        bl2_outputClassWeights,
                                                        learningRate=learningRate,
                                                        useAttention=useAttention)
            
        elif bl3_run:
            bl3_inputDim = bl3_inputVectorsCombined.shape[2]
            bl3_inputSeqLen = bl3_inputVectorsCombined.shape[1]

            learner = learning.SimpleFeedforwardNetworkSplitOutputs(bl3_inputDim,
                                                                    bl3_inputSeqLen, 
                                                                    bl3_maxSpeechClusterID,
                                                                    bl3_maxMotionID,
                                                                    batchSize, 
                                                                    embeddingDim,
                                                                    randomSeed,
                                                                    bl3_outputSpeechClassWeights,
                                                                    bl3_outputMotionClassWeights,
                                                                    learningRate=learningRate)
        
        elif bl4_run:
            bl4_inputDim = bl4_inputVectorsCombined.shape[2]
            bl4_inputSeqLen = bl4_inputVectorsCombined.shape[1]
            bl4_numOutputClasses = bl4_numActionClusters

            learner = learning.SimpleFeedforwardNetwork(bl4_inputDim, 
                                                        bl4_inputSeqLen, 
                                                        bl4_numOutputClasses,
                                                        batchSize, 
                                                        embeddingDim,
                                                        randomSeed,
                                                        bl4_outputClassWeights,
                                                        learningRate=learningRate,
                                                        useAttention=useAttention,
                                                        numAdditionalInputs=numShopkeepers) # additional inputs to specify which shopkeeper's action we want to predict
        
        elif bl5_run:
            bl5_inputDim = bl5_inputVectorsCombined.shape[2]
            bl5_inputSeqLen = bl5_inputVectorsCombined.shape[1]
            bl5_numOutputClasses = bl5_numActionClusters

            learner = learning.SimpleFeedforwardNetwork(bl5_inputDim, 
                                                        bl5_inputSeqLen, 
                                                        bl5_numOutputClasses,
                                                        batchSize, 
                                                        embeddingDim,
                                                        randomSeed,
                                                        bl5_outputClassWeights,
                                                        learningRate=learningRate,
                                                        useAttention=useAttention)
        
        elif bl6_run:
            bl6_inputDim = bl6_inputVectorsCombined.shape[2]
            bl6_inputSeqLen = bl6_inputVectorsCombined.shape[1]
            bl6_numOutputClasses = bl6_numActionClusters

            learner = learning.SimpleFeedforwardNetwork(bl6_inputDim, 
                                                        bl6_inputSeqLen, 
                                                        bl6_numOutputClasses,
                                                        batchSize, 
                                                        embeddingDim,
                                                        randomSeed,
                                                        bl6_outputClassWeights,
                                                        learningRate=learningRate,
                                                        useAttention=useAttention,
                                                        numAdditionalInputs=numShopkeepers) # additional inputs to specify which shopkeeper's action we want to predict
                                                        
        
        if bl7_run:
            bl7_inputDim = bl7_inputVectorsCombined.shape[2]
            bl7_inputSeqLen = bl7_inputVectorsCombined.shape[1]
            bl7_numOutputClasses = 2

            learner = learning.SimpleFeedforwardNetwork(bl7_inputDim, 
                                                        bl7_inputSeqLen, 
                                                        bl7_numOutputClasses,
                                                        batchSize, 
                                                        embeddingDim,
                                                        randomSeed,
                                                        bl7_outputClassWeights,
                                                        useAttention=useAttention)
            

        elif prop_run:
            pass

        #################################################################################################################
        # train and test...
        #################################################################################################################
        print("training and testing...", flush=True, file=foldTerminalOutputStream)
        
        for e in range(numEpochs+1):
            startTime = time.time()
            
            if bl1_run:
            
                #################################################################################################################
                # BEGIN BASELINE 1 RUN!
                #################################################################################################################
                
                if e != 0:
                    # train
                    if randomizeTrainingBatches:
                        trainIndicesOrder = random.sample(trainIndices, len(trainIndices))
                    else:
                        trainIndicesOrder = trainIndices

                    learner.train(bl1_inputVectorsCombined[trainIndicesOrder], 
                                  bl1_outputTargets[trainIndicesOrder], 
                                  bl1_outputMasks[trainIndicesOrder])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainActionLoss = learner.get_loss(
                        bl1_inputVectorsCombined[trainIndices],
                        bl1_outputTargets[trainIndices],                                       
                        bl1_outputMasks[trainIndices])
                    
                    # validation loss
                    valCost, valActionLoss = learner.get_loss(
                        bl1_inputVectorsCombined[valIndices],
                        bl1_outputTargets[valIndices],                                       
                        bl1_outputMasks[valIndices])
                    
                    # test loss
                    testCost, testActionLoss = learner.get_loss(
                        bl1_inputVectorsCombined[testIndices],
                        bl1_outputTargets[testIndices],                                       
                        bl1_outputMasks[testIndices])
                        
                    
                    # sometimes because of the batch size, some instances don't get a cost computed, so deal with that here
                    trainCost = np.asarray(trainCost).flatten()
                    trainActionLoss = np.asarray(trainActionLoss).flatten()
                    trainEndIndex = trainCost.shape[0]
                    trainCost = trainCost[np.where(bl1_outputMasks[trainIndices][:trainEndIndex])].tolist()
                    trainActionLoss = trainActionLoss[np.where(bl1_outputMasks[trainIndices][:trainEndIndex])].tolist()
                    
                    valCost = np.asarray(valCost).flatten()
                    valActionLoss = np.asarray(valActionLoss).flatten()
                    valEndIndex = valCost.shape[0]
                    valCost = valCost[np.where(bl1_outputMasks[valIndices][:valEndIndex])].tolist()
                    valActionLoss = valActionLoss[np.where(bl1_outputMasks[valIndices][:valEndIndex])].tolist()
                    
                    testCost = np.asarray(testCost).flatten()
                    testActionLoss = np.asarray(testActionLoss).flatten()
                    testEndIndex = testCost.shape[0]
                    testCost = testCost[np.where(bl1_outputMasks[testIndices][:testEndIndex])].tolist()
                    testActionLoss = testActionLoss[np.where(bl1_outputMasks[testIndices][:testEndIndex])].tolist()


                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainActionLossAve = np.mean(trainActionLoss)
                                        
                    trainCostStd = np.std(trainCost)
                    trainActionLossStd = np.std(trainActionLoss)
                    
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valActionLossAve = np.mean(valActionLoss)

                    valCostStd = np.std(valCost)
                    valActionLossStd = np.std(valActionLoss)
                    
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testActionLossAve = np.mean(testActionLoss)

                    testCostStd = np.std(testCost)
                    testActionLossStd = np.std(testActionLoss)
                    
                    
                    # predict
                    predShkpActions = learner.predict(
                        bl1_inputVectorsCombined,
                        bl1_outputTargets,                                       
                        bl1_outputMasks)
                    
                    
                    def bl1_evaluate_predictions(evalSetName, evalIndices, csvLogRows):
                        
                        # for computing accuracies
                        actions_gt = []
                        actions_pred = []
                        
                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpActions):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i
                            
                            #
                            # target info
                            #
                            csvLogRows[i]["TARG_SHOPKEEPER_ACTION"] = bl1_outputTargets[i]
                            
                            #
                            # prediction info
                            #
                            csvLogRows[i]["PRED_SHOPKEEPER_ACTION"] = predShkpActions[i]

                            #
                            # for computing accuracies
                            #
                            actions_gt.append(csvLogRows[i]["TARG_SHOPKEEPER_ACTION"])
                            actions_pred.append(csvLogRows[i]["PRED_SHOPKEEPER_ACTION"])
                        
                        
                        #
                        # compute accuracies
                        #
                        actionCorrAcc = accuracy_score(actions_gt, actions_pred)
                        actionPrec, actionRec, actionFsc, actionSupp = precision_recall_fscore_support(actions_gt, actions_pred)
                        
                        return csvLogRows, actionCorrAcc, actionPrec, actionRec, actionFsc, actionSupp
                    
                    
                    csvLogRows = copy.deepcopy(humanReadableInputsOutputs)
                    
                    csvLogRows, trainActionCorrAcc, trainActionPrec, trainActionRec, trainActionFsc, trainActionSupp = bl1_evaluate_predictions("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valActionCorrAcc, valActionPrec, valActionRec, valActionFsc, valActionSupp = bl1_evaluate_predictions("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testActionCorrAcc, testActionPrec, testActionRec, testActionFsc, testActionSupp = bl1_evaluate_predictions("TEST", testIndices, csvLogRows)
                    
                    
                    #
                    # save the evaluation results
                    #
                    tools.save_interaction_data(csvLogRows, foldDir+"/{:04}_all_outputs.csv".format(e), interactionsFieldnames)
                                        
                    
                    # append to session log   
                    with open(foldLogFile, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainActionLossAve,
                                         trainActionLossStd,
                                         
                                         trainActionCorrAcc,
                                         trainActionPrec[1],
                                         trainActionRec[1],
                                         trainActionFsc[1],
                                         trainActionSupp[1],
                                         trainActionPrec[0],
                                         trainActionRec[0],
                                         trainActionFsc[0],
                                         trainActionSupp[0],

                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valActionLossAve,
                                         valActionLossStd,
                                                                                  
                                         valActionCorrAcc,
                                         valActionPrec[1], 
                                         valActionRec[1], 
                                         valActionFsc[1], 
                                         valActionSupp[1],
                                         valActionPrec[0], 
                                         valActionRec[0], 
                                         valActionFsc[0], 
                                         valActionSupp[0],
                                         
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testActionLossAve,
                                         testActionLossStd,
                                         
                                         testActionCorrAcc,
                                         testActionPrec[1], 
                                         testActionRec[1], 
                                         testActionFsc[1], 
                                         testActionSupp[1],
                                         testActionPrec[0], 
                                         testActionRec[0], 
                                         testActionFsc[0], 
                                         testActionSupp[0]
                                         ])    
                
                
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ActionLossAve", trainActionLossAve, valActionLossAve, testActionLossAve])
                    tableData.append(["ActionLossStd", trainActionLossStd, valActionLossStd, testActionLossStd])
                    
                    tableData.append(["ActionCorrAcc", trainActionCorrAcc, valActionCorrAcc, testActionCorrAcc])
                    tableData.append(["ActionPrec", trainActionPrec, valActionPrec, testActionPrec])
                    tableData.append(["ActionRec", trainActionRec, valActionRec, testActionRec])
                    tableData.append(["ActionFsc", trainActionFsc, valActionFsc, testActionFsc])
                    tableData.append(["ActionSupp", trainActionSupp, valActionSupp, testActionSupp])

                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)

            
            #################################################################################################################
            # END BASELINE 1 RUN!
            #################################################################################################################
            

            elif bl2_run:
            
                #################################################################################################################
                # BEGIN BASELINE 2 RUN!
                #################################################################################################################
                
                if e != 0:
                    # train
                    if randomizeTrainingBatches:
                        trainIndicesOrder = random.sample(trainIndices, len(trainIndices))
                    else:
                        trainIndicesOrder = trainIndices

                    learner.train(bl2_inputVectorsCombined[trainIndicesOrder], 
                                  bl2_outputActionIDs_shkp2[trainIndicesOrder], 
                                  bl2_outputMasks[trainIndicesOrder])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainActionLoss = learner.get_loss(
                        bl2_inputVectorsCombined[trainIndices],
                        bl2_outputActionIDs_shkp2[trainIndices],                                       
                        bl2_outputMasks[trainIndices])
                    
                    # validation loss
                    valCost, valActionLoss = learner.get_loss(
                        bl2_inputVectorsCombined[valIndices],
                        bl2_outputActionIDs_shkp2[valIndices],                                       
                        bl2_outputMasks[valIndices])
                    
                    # test loss
                    testCost, testActionLoss = learner.get_loss(
                        bl2_inputVectorsCombined[testIndices],
                        bl2_outputActionIDs_shkp2[testIndices],                                       
                        bl2_outputMasks[testIndices])
                        
                    
                    # sometimes because of the batch size, some instances don't get a cost computed, so deal with that here
                    trainCost = np.asarray(trainCost).flatten()
                    trainActionLoss = np.asarray(trainActionLoss).flatten()
                    trainEndIndex = trainCost.shape[0]
                    trainCost = trainCost[np.where(bl2_outputMasks[trainIndices][:trainEndIndex])].tolist()
                    trainActionLoss = trainActionLoss[np.where(bl2_outputMasks[trainIndices][:trainEndIndex])].tolist()
                    
                    valCost = np.asarray(valCost).flatten()
                    valActionLoss = np.asarray(valActionLoss).flatten()
                    valEndIndex = valCost.shape[0]
                    valCost = valCost[np.where(bl2_outputMasks[valIndices][:valEndIndex])].tolist()
                    valActionLoss = valActionLoss[np.where(bl2_outputMasks[valIndices][:valEndIndex])].tolist()
                    
                    testCost = np.asarray(testCost).flatten()
                    testActionLoss = np.asarray(testActionLoss).flatten()
                    testEndIndex = testCost.shape[0]
                    testCost = testCost[np.where(bl2_outputMasks[testIndices][:testEndIndex])].tolist()
                    testActionLoss = testActionLoss[np.where(bl2_outputMasks[testIndices][:testEndIndex])].tolist()


                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainActionLossAve = np.mean(trainActionLoss)
                                        
                    trainCostStd = np.std(trainCost)
                    trainActionLossStd = np.std(trainActionLoss)
                    
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valActionLossAve = np.mean(valActionLoss)

                    valCostStd = np.std(valCost)
                    valActionLossStd = np.std(valActionLoss)
                    
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testActionLossAve = np.mean(testActionLoss)

                    testCostStd = np.std(testCost)
                    testActionLossStd = np.std(testActionLoss)
                    
                    
                    # predict
                    predShkpActions = learner.predict(
                        bl2_inputVectorsCombined,
                        bl2_outputActionIDs_shkp2,                                       
                        bl2_outputMasks)
                    
                    
                    def bl2_evaluate_predictions(evalSetName, evalIndices, csvLogRows):
                        # TODO: don't include null actions, etc. in the performance metrics computations

                        # for computing accuracies
                        actions_gt = []
                        actions_pred = []

                        speechClusts_gt = []
                        speechClusts_pred = []

                        spatial_gt = []
                        spatial_pred = []

                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpActions):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i

                            #
                            # get the speech cluster and spatial info predictions
                            #
                            predActionClustID = predShkpActions[i]
                            predSpeechClustID = bl2_actionClustIDToSpeechClustID[predActionClustID]
                            predSpatialInfo = bl2_actionClustIDToSpatial[predActionClustID]
                            predSpatialInfoName = bl2_actionClustIDToSpatialName[predActionClustID]
                            predRepUtt = bl2_speechClustIDToRepUtt[predSpeechClustID]

                            gtActionClusterID = bl2_outputActionIDs_shkp2[i]
                            gtSpeechClustID = bl2_actionClustIDToSpeechClustID[gtActionClusterID]
                            gtSpatialInfo = bl2_actionClustIDToSpatial[gtActionClusterID]


                            #
                            # prediction info
                            #
                            csvLogRows[i]["LOSS_WEIGHT"] = bl2_outputMasks[i]
                            csvLogRows[i]["PRED_SHOPKEEPER_ACTION_CLUSTER"] = predActionClustID
                            csvLogRows[i]["PRED_SHOPKEEPER_SPEECH_CLUSTER"] = predSpeechClustID
                            csvLogRows[i]["PRED_SHOPKEEPER_SPATIAL_INFO"] = predSpatialInfo
                            csvLogRows[i]["PRED_SHOPKEEPER_SPATIAL_INFO_NAME"] = predSpatialInfoName
                            csvLogRows[i]["PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"] = predRepUtt

                            #
                            # for computing accuracies
                            #
                            actions_gt.append(gtActionClusterID)
                            actions_pred.append(predActionClustID)

                            speechClusts_gt.append(gtSpeechClustID)
                            speechClusts_pred.append(predSpeechClustID)

                            spatial_gt.append(gtSpatialInfo)
                            spatial_pred.append(predSpatialInfo)
                        
                        
                        #
                        # compute accuracies
                        # fix the len of the output masks because sometimes test set gets cut off during prediction
                        #
                        actionCorrAcc = accuracy_score(actions_gt, actions_pred, sample_weight=bl2_outputMasks[evalIndices][:len(spatial_gt)])
                        #actionPrec, actionRec, actionFsc, actionSupp = precision_recall_fscore_support(actions_gt, actions_pred, sample_weight=bl2_outputMasks[evalIndices])
                        
                        speechCorrAcc = accuracy_score(speechClusts_gt, speechClusts_pred, sample_weight=bl2_outputMasks[evalIndices][:len(spatial_gt)])
                        #speechPrec, speechRec, speechFsc, speechSupp = precision_recall_fscore_support(speechClusts_gt, speechClusts_pred, sample_weight=bl2_outputMasks[evalIndices])
                        
                        spatialCorrAcc = accuracy_score(spatial_gt, spatial_pred, sample_weight=bl2_outputMasks[evalIndices][:len(spatial_gt)]) 
                        #spatialPrec, spatialRec, spatialFsc, spatialSupp = precision_recall_fscore_support(spatial_gt, spatial_pred, sample_weight=bl2_outputMasks[evalIndices])
                        


                        return csvLogRows, actionCorrAcc, speechCorrAcc, spatialCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(humanReadableInputsOutputs)
                    
                    csvLogRows, trainActionCorrAcc, trainSpeechCorrAcc, trainSpatialCorrAcc = bl2_evaluate_predictions("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valActionCorrAcc, valSpeechCorrAcc, valSpatialCorrAcc = bl2_evaluate_predictions("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testActionCorrAcc, testSpeechCorrAcc, testSpatialCorrAcc = bl2_evaluate_predictions("TEST", testIndices, csvLogRows)
                    
                    
                    #
                    # save the evaluation results
                    #
                    tools.save_interaction_data(csvLogRows, foldDir+"/{:04}_all_outputs.csv".format(e), interactionsFieldnames)


                    # append to session log   
                    with open(foldLogFile, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainActionLossAve,
                                         trainActionLossStd,
                                         
                                         trainActionCorrAcc,
                                         trainSpeechCorrAcc,
                                         trainSpatialCorrAcc,

                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valActionLossAve,
                                         valActionLossStd,
                                                                                  
                                         valActionCorrAcc,
                                         valSpeechCorrAcc, 
                                         valSpatialCorrAcc,
                                        
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testActionLossAve,
                                         testActionLossStd,
                                         
                                         testActionCorrAcc,
                                         testSpeechCorrAcc, 
                                         testSpatialCorrAcc
                                         ])    
                
                
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ActionLossAve", trainActionLossAve, valActionLossAve, testActionLossAve])
                    tableData.append(["ActionLossStd", trainActionLossStd, valActionLossStd, testActionLossStd])
                    
                    tableData.append(["ActionCorrAcc", trainActionCorrAcc, valActionCorrAcc, testActionCorrAcc])
                    tableData.append(["SpeechCorrAcc", trainSpeechCorrAcc, valSpeechCorrAcc, testSpeechCorrAcc])
                    tableData.append(["SpatialCorrAcc", trainSpatialCorrAcc, valSpatialCorrAcc, testSpatialCorrAcc])
                    

                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)

            
            #################################################################################################################
            # END BASELINE 2 RUN!
            #################################################################################################################


            elif bl3_run:
            
                #################################################################################################################
                # BEGIN BASELINE 3 RUN!
                #################################################################################################################
                
                if e != 0:
                    # train
                    if randomizeTrainingBatches:
                        trainIndicesOrder = random.sample(trainIndices, len(trainIndices))
                    else:
                        trainIndicesOrder = trainIndices

                    learner.train(bl3_inputVectorsCombined[trainIndicesOrder], 
                                  bl3_outputSpeechClusterIDs[trainIndicesOrder],
                                  bl3_outputSpatialInfo[trainIndicesOrder],
                                  bl3_outputMasks[trainIndicesOrder])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainSpeechLoss, trainMotionLoss = learner.get_loss(
                        bl3_inputVectorsCombined[trainIndices],
                        bl3_outputSpeechClusterIDs[trainIndices],
                        bl3_outputSpatialInfo[trainIndices],                                 
                        bl3_outputMasks[trainIndices])
                    
                    # validation loss
                    valCost, valSpeechLoss, valMotionLoss = learner.get_loss(
                        bl3_inputVectorsCombined[valIndices],
                        bl3_outputSpeechClusterIDs[valIndices],
                        bl3_outputSpatialInfo[valIndices],                                     
                        bl3_outputMasks[valIndices])
                    
                    # test loss
                    testCost, testSpeechLoss, testMotionLoss = learner.get_loss(
                        bl3_inputVectorsCombined[testIndices],
                        bl3_outputSpeechClusterIDs[testIndices],
                        bl3_outputSpatialInfo[testIndices],                                      
                        bl3_outputMasks[testIndices])
                        
                    
                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainSpeechLossAve = np.mean(trainSpeechLoss)
                    trainMotionLossAve = np.mean(trainMotionLoss)

                    trainCostStd = np.std(trainCost)
                    trainSpeechLossStd = np.std(trainSpeechLoss)
                    trainMotionLossStd = np.std(trainMotionLoss)
                    
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valSpeechLossAve = np.mean(valSpeechLoss)
                    valMotionLossAve = np.mean(valMotionLoss)

                    valCostStd = np.std(valCost)
                    valSpeechLossStd = np.std(valSpeechLoss)
                    valMotionLossStd = np.std(valMotionLoss)
                    
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testSpeechLossAve = np.mean(testSpeechLoss)
                    testMotionLossAve = np.mean(testMotionLoss)

                    testCostStd = np.std(testCost)
                    testSpeechLossStd = np.std(testSpeechLoss)
                    testMotionLossStd = np.std(testMotionLoss)
                    
                    
                    # predict
                    predShkpSpeech, predShkpMotion = learner.predict(
                        bl3_inputVectorsCombined,
                        bl3_outputSpeechClusterIDs,
                        bl3_outputSpatialInfo,                                       
                        bl3_outputMasks)
                    
                    
                    def bl3_evaluate_predictions(evalSetName, evalIndices, csvLogRows):
                        # for computing accuracies
                        actions_gt = []
                        actions_pred = []

                        speechClusts_gt = []
                        speechClusts_pred = []

                        spatial_gt = []
                        spatial_pred = []

                        speechAndMotionCorrect = []

                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpSpeech):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i

                            #
                            # get the speech cluster and spatial info predictions
                            #
                            predSpeechClustID = predShkpSpeech[i]
                            predSpatialInfo = predShkpMotion[i]
                            predSpatialInfoName = bl3_motionIDToSpatialName[predSpatialInfo]
                            predRepUtt = bl3_speechClustIDToRepUtt[predSpeechClustID]

                            gtSpeechClustID = bl3_outputSpeechClusterIDs[i]
                            gtSpatialInfo = bl3_outputSpatialInfo[i]

                            #
                            # prediction info
                            #
                            csvLogRows[i]["LOSS_WEIGHT"] = bl3_outputMasks[i]
                            csvLogRows[i]["PRED_SHOPKEEPER_SPEECH_CLUSTER"] = predSpeechClustID
                            csvLogRows[i]["PRED_SHOPKEEPER_SPATIAL_INFO"] = predSpatialInfo
                            csvLogRows[i]["PRED_SHOPKEEPER_SPATIAL_INFO_NAME"] = predSpatialInfoName
                            csvLogRows[i]["PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"] = predRepUtt

                            #
                            # for computing accuracies
                            #
                            speechClusts_gt.append(gtSpeechClustID)
                            speechClusts_pred.append(predSpeechClustID)

                            spatial_gt.append(gtSpatialInfo)
                            spatial_pred.append(predSpatialInfo)

                            speechAndMotionCorrect.append(predSpeechClustID == gtSpeechClustID and predSpatialInfo == gtSpatialInfo)
                        
                        
                        #
                        # compute accuracies
                        # fix the len of the output masks because sometimes test set gets cut off during prediction
                        #
                        speechAndMotionCorrect = np.asarray(speechAndMotionCorrect)
                        speechAndMotionCorrectMasked = np.copy(speechAndMotionCorrect)
                        speechAndMotionCorrectMasked[np.where(bl3_outputMasks[:len(spatial_gt)] == 0)] = 0
                        numSpeechAndMotionCorrect = sum(speechAndMotionCorrectMasked)
                        numToBeEvaluated = sum(bl3_outputMasks[evalIndices][:len(spatial_gt)])
                        actionCorrAcc = float(numSpeechAndMotionCorrect) / float(numToBeEvaluated)
                        
                        speechCorrAcc = accuracy_score(speechClusts_gt, speechClusts_pred, sample_weight=bl3_outputMasks[evalIndices][:len(spatial_gt)])
                        #speechPrec, speechRec, speechFsc, speechSupp = precision_recall_fscore_support(speechClusts_gt, speechClusts_pred, sample_weight=bl3_outputMasks[evalIndices])
                        
                        spatialCorrAcc = accuracy_score(spatial_gt, spatial_pred, sample_weight=bl3_outputMasks[evalIndices][:len(spatial_gt)]) 
                        #spatialPrec, spatialRec, spatialFsc, spatialSupp = precision_recall_fscore_support(spatial_gt, spatial_pred, sample_weight=bl3_outputMasks[evalIndices])
                        


                        return csvLogRows, actionCorrAcc, speechCorrAcc, spatialCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(humanReadableInputsOutputs)
                    
                    csvLogRows, trainActionCorrAcc, trainSpeechCorrAcc, trainSpatialCorrAcc = bl3_evaluate_predictions("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valActionCorrAcc, valSpeechCorrAcc, valSpatialCorrAcc = bl3_evaluate_predictions("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testActionCorrAcc, testSpeechCorrAcc, testSpatialCorrAcc = bl3_evaluate_predictions("TEST", testIndices, csvLogRows)
                    
                    
                    #
                    # save the evaluation results
                    #
                    tools.save_interaction_data(csvLogRows, foldDir+"/{:04}_all_outputs.csv".format(e), interactionsFieldnames)


                    # append to session log   
                    with open(foldLogFile, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainSpeechLossAve,
                                         trainSpeechLossStd,
                                         trainMotionLossAve,
                                         trainMotionLossStd,
                                         
                                         trainActionCorrAcc,
                                         trainSpeechCorrAcc,
                                         trainSpatialCorrAcc,

                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valSpeechLossAve,
                                         valSpeechLossStd,
                                         valMotionLossAve,
                                         valMotionLossStd,
                                                                                  
                                         valActionCorrAcc,
                                         valSpeechCorrAcc, 
                                         valSpatialCorrAcc,
                                        
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testSpeechLossAve,
                                         testSpeechLossStd,
                                         testMotionLossAve,
                                         testMotionLossStd,
                                         
                                         testActionCorrAcc,
                                         testSpeechCorrAcc, 
                                         testSpatialCorrAcc
                                         ])    
                
                
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["SpeechLossAve", trainSpeechLossAve, valSpeechLossAve, testSpeechLossAve])
                    tableData.append(["SpeechLossStd", trainSpeechLossStd, valSpeechLossStd, testSpeechLossStd])
                    tableData.append(["MotionLossAve", trainMotionLossAve, valMotionLossAve, testMotionLossAve])
                    tableData.append(["MotionLossStd", trainMotionLossStd, valMotionLossStd, testMotionLossStd])
                    
                    tableData.append(["ActionCorrAcc", trainActionCorrAcc, valActionCorrAcc, testActionCorrAcc])
                    tableData.append(["SpeechCorrAcc", trainSpeechCorrAcc, valSpeechCorrAcc, testSpeechCorrAcc])
                    tableData.append(["SpatialCorrAcc", trainSpatialCorrAcc, valSpatialCorrAcc, testSpatialCorrAcc])
                    

                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)

            
            #################################################################################################################
            # END BASELINE 3 RUN!
            #################################################################################################################


            elif bl4_run:
            
                #################################################################################################################
                # BEGIN BASELINE 4 RUN!
                #################################################################################################################
                
                if e != 0:
                    # train
                    
                    if numEpochsTillS2Only == None or e < numEpochsTillS2Only:
                        # combine S1 and S2 data
                        bl4_inputVectorsCombined_temp = np.concatenate([bl4_inputVectorsCombined[trainIndices], bl4_inputVectorsCombined[trainIndices]])
                        bl4_outputActionIDs_temp = np.concatenate([bl4_outputActionIDs_shkp1[trainIndices], bl4_outputActionIDs_shkp2[trainIndices]])
                        bl4_outputMasks_temp = np.concatenate([bl4_outputMasks_shkp1[trainIndices], bl4_outputMasks_shkp2[trainIndices]])
                        bl4_additionalInputs_temp = np.concatenate([bl4_additionalInputs_shkp1[trainIndices], bl4_additionalInputs_shkp2[trainIndices]])

                        trainIndicesOrder = list(range(bl4_inputVectorsCombined_temp.shape[0]))
                        if randomizeTrainingBatches:
                            trainIndicesOrder = random.sample(trainIndicesOrder, len(trainIndicesOrder))

                        learner.train(bl4_inputVectorsCombined_temp[trainIndicesOrder], 
                                    bl4_outputActionIDs_temp[trainIndicesOrder], 
                                    bl4_outputMasks_temp[trainIndicesOrder],
                                    bl4_additionalInputs_temp[trainIndicesOrder])
                    
                    else:
                        # train only on S2
                        if randomizeTrainingBatches:
                            trainIndicesOrder = random.sample(trainIndices, len(trainIndices))
                        else:
                            trainIndicesOrder = trainIndices

                        learner.train(bl4_inputVectorsCombined[trainIndicesOrder],
                                    bl4_outputActionIDs_shkp2[trainIndicesOrder],
                                    bl4_outputMasks_shkp2[trainIndicesOrder],
                                    bl4_additionalInputs_shkp2[trainIndicesOrder])
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # S1
                    # training loss
                    trainCost_shkp1, trainActionLoss_shkp1 = learner.get_loss(
                        bl4_inputVectorsCombined[trainIndices],
                        bl4_outputActionIDs_shkp1[trainIndices],                                       
                        bl4_outputMasks_shkp1[trainIndices],
                        bl4_additionalInputs_shkp1[trainIndices])
                    
                    # validation loss
                    valCost_shkp1, valActionLoss_shkp1 = learner.get_loss(
                        bl4_inputVectorsCombined[valIndices],
                        bl4_outputActionIDs_shkp1[valIndices],                                       
                        bl4_outputMasks_shkp1[valIndices],
                            bl4_additionalInputs_shkp1[valIndices])
                    
                    # test loss
                    testCost_shkp1, testActionLoss_shkp1 = learner.get_loss(
                        bl4_inputVectorsCombined[testIndices],
                        bl4_outputActionIDs_shkp1[testIndices],                                       
                        bl4_outputMasks_shkp1[testIndices],
                        bl4_additionalInputs_shkp1[testIndices])
                    

                    # S2
                    # training loss
                    trainCost_shkp2, trainActionLoss_shkp2 = learner.get_loss(
                        bl4_inputVectorsCombined[trainIndices],
                        bl4_outputActionIDs_shkp2[trainIndices],                                       
                        bl4_outputMasks_shkp2[trainIndices],
                        bl4_additionalInputs_shkp2[trainIndices])
                    
                    # validation loss
                    valCost_shkp2, valActionLoss_shkp2 = learner.get_loss(
                        bl4_inputVectorsCombined[valIndices],
                        bl4_outputActionIDs_shkp2[valIndices],                                       
                        bl4_outputMasks_shkp2[valIndices],
                        bl4_additionalInputs_shkp2[valIndices])
                    
                    # test loss
                    testCost_shkp2, testActionLoss_shkp2 = learner.get_loss(
                        bl4_inputVectorsCombined[testIndices],
                        bl4_outputActionIDs_shkp2[testIndices],                                       
                        bl4_outputMasks_shkp2[testIndices],
                        bl4_additionalInputs_shkp2[testIndices])
                    
                    # sometimes because of the batch size, some instances don't get a cost computed, so deal with that here

                    trainCost_shkp1 = np.asarray(trainCost_shkp1).flatten()
                    trainCost_shkp2 = np.asarray(trainCost_shkp2).flatten()
                    trainActionLoss_shkp1 = np.asarray(trainActionLoss_shkp1).flatten()
                    trainActionLoss_shkp2 = np.asarray(trainActionLoss_shkp2).flatten()
                    trainEndIndex = trainCost_shkp1.shape[0]
                    trainCost_shkp1 = trainCost_shkp1[np.where(bl4_outputMasks_shkp1[trainIndices][:trainEndIndex])].tolist()
                    trainCost_shkp2 = trainCost_shkp2[np.where(bl4_outputMasks_shkp2[trainIndices][:trainEndIndex])].tolist()
                    trainActionLoss_shkp1 = trainActionLoss_shkp1[np.where(bl4_outputMasks_shkp1[trainIndices][:trainEndIndex])].tolist()
                    trainActionLoss_shkp2 = trainActionLoss_shkp2[np.where(bl4_outputMasks_shkp2[trainIndices][:trainEndIndex])].tolist()
                    
                    valCost_shkp1 = np.asarray(valCost_shkp1).flatten()
                    valCost_shkp2 = np.asarray(valCost_shkp2).flatten()
                    valActionLoss_shkp1 = np.asarray(valActionLoss_shkp1).flatten()
                    valActionLoss_shkp2 = np.asarray(valActionLoss_shkp2).flatten()
                    valEndIndex = valCost_shkp1.shape[0]
                    valCost_shkp1 = valCost_shkp1[np.where(bl4_outputMasks_shkp1[valIndices][:valEndIndex])].tolist()
                    valCost_shkp2 = valCost_shkp2[np.where(bl4_outputMasks_shkp2[valIndices][:valEndIndex])].tolist()
                    valActionLoss_shkp1 = valActionLoss_shkp1[np.where(bl4_outputMasks_shkp1[valIndices][:valEndIndex])].tolist()
                    valActionLoss_shkp2 = valActionLoss_shkp2[np.where(bl4_outputMasks_shkp2[valIndices][:valEndIndex])].tolist()
                    
                    testCost_shkp1 = np.asarray(testCost_shkp1).flatten()
                    testCost_shkp2 = np.asarray(testCost_shkp2).flatten()
                    testActionLoss_shkp1 = np.asarray(testActionLoss_shkp1).flatten()
                    testActionLoss_shkp2 = np.asarray(testActionLoss_shkp2).flatten()
                    testEndIndex = testCost_shkp1.shape[0]
                    testCost_shkp1 = testCost_shkp1[np.where(bl4_outputMasks_shkp1[testIndices][:testEndIndex])].tolist()
                    testCost_shkp2 = testCost_shkp2[np.where(bl4_outputMasks_shkp2[testIndices][:testEndIndex])].tolist()
                    testActionLoss_shkp1 = testActionLoss_shkp1[np.where(bl4_outputMasks_shkp1[testIndices][:testEndIndex])].tolist()
                    testActionLoss_shkp2 = testActionLoss_shkp2[np.where(bl4_outputMasks_shkp2[testIndices][:testEndIndex])].tolist()


                    # S1 + S2
                    trainCost = trainCost_shkp1 + trainCost_shkp2
                    trainActionLoss = trainActionLoss_shkp1 + trainActionLoss_shkp2
                    valCost = valCost_shkp1 + valCost_shkp2
                    valActionLoss = valActionLoss_shkp1 + valActionLoss_shkp2
                    testCost = testCost_shkp1 + testCost_shkp2
                    testActionLoss = testActionLoss_shkp1 + testActionLoss_shkp2


                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainActionLossAve = np.mean(trainActionLoss)
                    trainCostStd = np.std(trainCost)
                    trainActionLossStd = np.std(trainActionLoss)

                    trainCostAve_shkp1 = np.mean(trainCost_shkp1)
                    trainActionLossAve_shkp1 = np.mean(trainActionLoss_shkp1)
                    trainCostStd_shkp1 = np.std(trainCost_shkp1)
                    trainActionLossStd_shkp1 = np.std(trainActionLoss_shkp1)

                    trainCostAve_shkp2 = np.mean(trainCost_shkp2)
                    trainActionLossAve_shkp2 = np.mean(trainActionLoss_shkp2)
                    trainCostStd_shkp2 = np.std(trainCost_shkp2)
                    trainActionLossStd_shkp2 = np.std(trainActionLoss_shkp2)
                    
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valActionLossAve = np.mean(valActionLoss)
                    valCostStd = np.std(valCost)
                    valActionLossStd = np.std(valActionLoss)

                    valCostAve_shkp1 = np.mean(valCost_shkp1)
                    valActionLossAve_shkp1 = np.mean(valActionLoss_shkp1)
                    valCostStd_shkp1 = np.std(valCost_shkp1)
                    valActionLossStd_shkp1 = np.std(valActionLoss_shkp1)
                    
                    valCostAve_shkp2 = np.mean(valCost_shkp2)
                    valActionLossAve_shkp2 = np.mean(valActionLoss_shkp2)
                    valCostStd_shkp2 = np.std(valCost_shkp2)
                    valActionLossStd_shkp2 = np.std(valActionLoss_shkp2)


                    # test
                    testCostAve = np.mean(testCost)
                    testActionLossAve = np.mean(testActionLoss)
                    testCostStd = np.std(testCost)
                    testActionLossStd = np.std(testActionLoss)

                    testCostAve_shkp1 = np.mean(testCost_shkp1)
                    testActionLossAve_shkp1 = np.mean(testActionLoss_shkp1)
                    testCostStd_shkp1 = np.std(testCost_shkp1)
                    testActionLossStd_shkp1 = np.std(testActionLoss_shkp1)

                    testCostAve_shkp2 = np.mean(testCost_shkp2)
                    testActionLossAve_shkp2 = np.mean(testActionLoss_shkp2)
                    testCostStd_shkp2 = np.std(testCost_shkp2)
                    testActionLossStd_shkp2 = np.std(testActionLoss_shkp2)
                    
                    
                    # predict

                    # S1
                    predShkpActions_shkp1 = learner.predict(
                        bl4_inputVectorsCombined,
                        bl4_outputActionIDs_shkp1,                                       
                        bl4_outputMasks_shkp1,
                        bl4_additionalInputs_shkp1)

                    # S2
                    predShkpActions_shkp2 = learner.predict(
                        bl4_inputVectorsCombined,
                        bl4_outputActionIDs_shkp2,                                       
                        bl4_outputMasks_shkp2,
                        bl4_additionalInputs_shkp2)
                    
                    
                    def bl4_evaluate_predictions(evalSetName, evalIndices, csvLogRows, shopkeeper):
                        # TODO: don't include null actions, etc. in the performance metrics computations

                        if shopkeeper == "SHOPKEEPER_1":
                            predShkpActions = predShkpActions_shkp1
                            bl4_outputActionIDs = bl4_outputActionIDs_shkp1
                            bl4_outputMasks = bl4_outputMasks_shkp1

                        elif shopkeeper == "SHOPKEEPER_2":
                            predShkpActions = predShkpActions_shkp2
                            bl4_outputActionIDs = bl4_outputActionIDs_shkp2
                            bl4_outputMasks = bl4_outputMasks_shkp2



                        # for computing accuracies
                        actions_gt = []
                        actions_pred = []

                        speechClusts_gt = []
                        speechClusts_pred = []

                        spatial_gt = []
                        spatial_pred = []

                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpActions):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i

                            #
                            # get the speech cluster and spatial info predictions
                            #
                            predActionClustID = predShkpActions[i]
                            predSpeechClustID = bl4_actionClustIDToSpeechClustID[predActionClustID]
                            predSpatialInfo = bl4_actionClustIDToSpatial[predActionClustID]
                            predSpatialInfoName = bl4_actionClustIDToSpatialName[predActionClustID]
                            predRepUtt = bl4_speechClustIDToRepUtt[predSpeechClustID]

                            gtActionClusterID = bl4_outputActionIDs[i]
                            gtSpeechClustID = bl4_actionClustIDToSpeechClustID[gtActionClusterID]
                            gtSpatialInfo = bl4_actionClustIDToSpatial[gtActionClusterID]


                            #
                            # prediction info
                            #
                            csvLogRows[i]["{}_LOSS_WEIGHT".format(shopkeeper)] = bl4_outputMasks[i]
                            csvLogRows[i]["PRED_{}_ACTION_CLUSTER".format(shopkeeper)] = predActionClustID
                            csvLogRows[i]["PRED_{}_SPEECH_CLUSTER".format(shopkeeper)] = predSpeechClustID
                            csvLogRows[i]["PRED_{}_SPATIAL_INFO".format(shopkeeper)] = predSpatialInfo
                            csvLogRows[i]["PRED_{}_SPATIAL_INFO_NAME".format(shopkeeper)] = predSpatialInfoName
                            csvLogRows[i]["PRED_{}_REPRESENTATIVE_UTTERANCE".format(shopkeeper)] = predRepUtt

                            #
                            # for computing accuracies
                            #
                            actions_gt.append(gtActionClusterID)
                            actions_pred.append(predActionClustID)

                            speechClusts_gt.append(gtSpeechClustID)
                            speechClusts_pred.append(predSpeechClustID)

                            spatial_gt.append(gtSpatialInfo)
                            spatial_pred.append(predSpatialInfo)
                        
                        
                        #
                        # compute accuracies
                        # fix the len of the output masks because sometimes test set gets cut off during prediction
                        #
                        actionCorrAcc = accuracy_score(actions_gt, actions_pred, sample_weight=bl4_outputMasks[evalIndices][:len(spatial_gt)])
                        #actionPrec, actionRec, actionFsc, actionSupp = precision_recall_fscore_support(actions_gt, actions_pred, sample_weight=bl4_outputMasks[evalIndices])
                        
                        speechCorrAcc = accuracy_score(speechClusts_gt, speechClusts_pred, sample_weight=bl4_outputMasks[evalIndices][:len(spatial_gt)])
                        #speechPrec, speechRec, speechFsc, speechSupp = precision_recall_fscore_support(speechClusts_gt, speechClusts_pred, sample_weight=bl4_outputMasks[evalIndices])
                        
                        spatialCorrAcc = accuracy_score(spatial_gt, spatial_pred, sample_weight=bl4_outputMasks[evalIndices][:len(spatial_gt)]) 
                        #spatialPrec, spatialRec, spatialFsc, spatialSupp = precision_recall_fscore_support(spatial_gt, spatial_pred, sample_weight=bl4_outputMasks[evalIndices])
                        


                        return csvLogRows, actionCorrAcc, speechCorrAcc, spatialCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(humanReadableInputsOutputs)
                    
                    # S1
                    csvLogRows, trainActionCorrAcc_shkp1, trainSpeechCorrAcc_shkp1, trainSpatialCorrAcc_shkp1 = bl4_evaluate_predictions("TRAIN", trainIndices, csvLogRows, "SHOPKEEPER_1")
                    csvLogRows, valActionCorrAcc_shkp1, valSpeechCorrAcc_shkp1, valSpatialCorrAcc_shkp1 = bl4_evaluate_predictions("VAL", valIndices, csvLogRows, "SHOPKEEPER_1")
                    csvLogRows, testActionCorrAcc_shkp1, testSpeechCorrAcc_shkp1, testSpatialCorrAcc_shkp1 = bl4_evaluate_predictions("TEST", testIndices, csvLogRows, "SHOPKEEPER_1")
                    
                    # S2
                    csvLogRows, trainActionCorrAcc_shkp2, trainSpeechCorrAcc_shkp2, trainSpatialCorrAcc_shkp2 = bl4_evaluate_predictions("TRAIN", trainIndices, csvLogRows, "SHOPKEEPER_2")
                    csvLogRows, valActionCorrAcc_shkp2, valSpeechCorrAcc_shkp2, valSpatialCorrAcc_shkp2 = bl4_evaluate_predictions("VAL", valIndices, csvLogRows, "SHOPKEEPER_2")
                    csvLogRows, testActionCorrAcc_shkp2, testSpeechCorrAcc_shkp2, testSpatialCorrAcc_shkp2 = bl4_evaluate_predictions("TEST", testIndices, csvLogRows, "SHOPKEEPER_2")
                    
                    # both
                    trainActionCorrAcc = (trainActionCorrAcc_shkp1 + trainActionCorrAcc_shkp2) / 2
                    trainSpeechCorrAcc = (trainSpeechCorrAcc_shkp1 + trainSpeechCorrAcc_shkp2) / 2
                    trainSpatialCorrAcc = (trainSpatialCorrAcc_shkp1 + trainSpatialCorrAcc_shkp2) / 2
                                                            
                    valActionCorrAcc = (valActionCorrAcc_shkp1 + valActionCorrAcc_shkp2) / 2
                    valSpeechCorrAcc = (valSpeechCorrAcc_shkp1 + valSpeechCorrAcc_shkp2) / 2
                    valSpatialCorrAcc = (valSpatialCorrAcc_shkp1 + valSpatialCorrAcc_shkp2) / 2

                    testActionCorrAcc = (testActionCorrAcc_shkp1 + testActionCorrAcc_shkp2) / 2
                    testSpeechCorrAcc = (testSpeechCorrAcc_shkp1 + testSpeechCorrAcc_shkp2) / 2
                    testSpatialCorrAcc = (testSpatialCorrAcc_shkp1 + testSpatialCorrAcc_shkp2) / 2

                    
                    #
                    # save the evaluation results
                    #
                    tools.save_interaction_data(csvLogRows, foldDir+"/{:04}_all_outputs.csv".format(e), interactionsFieldnames)


                    # append to session log   
                    with open(foldLogFile, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainActionLossAve,
                                         trainActionLossStd,
                                         trainActionCorrAcc,
                                         trainSpeechCorrAcc,
                                         trainSpatialCorrAcc,

                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valActionLossAve,
                                         valActionLossStd,                 
                                         valActionCorrAcc,
                                         valSpeechCorrAcc, 
                                         valSpatialCorrAcc,
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testActionLossAve,
                                         testActionLossStd,
                                         testActionCorrAcc,
                                         testSpeechCorrAcc, 
                                         testSpatialCorrAcc,

                                         # S1
                                         # training
                                         trainCostAve_shkp1,
                                         trainCostStd_shkp1,
                                         trainActionLossAve_shkp1,
                                         trainActionLossStd_shkp1,
                                         trainActionCorrAcc_shkp1,
                                         trainSpeechCorrAcc_shkp1,
                                         trainSpatialCorrAcc_shkp1,
                                         
                                         # validation
                                         valCostAve_shkp1,
                                         valCostStd_shkp1,
                                         valActionLossAve_shkp1,
                                         valActionLossStd_shkp1,             
                                         valActionCorrAcc_shkp1,
                                         valSpeechCorrAcc_shkp1, 
                                         valSpatialCorrAcc_shkp1,
                                                                                 
                                         # testing
                                         testCostAve_shkp1,
                                         testCostStd_shkp1,
                                         testActionLossAve_shkp1,
                                         testActionLossStd_shkp1,
                                         testActionCorrAcc_shkp1,
                                         testSpeechCorrAcc_shkp1, 
                                         testSpatialCorrAcc_shkp1,

                                         # S2
                                         # training
                                         trainCostAve_shkp2,
                                         trainCostStd_shkp2,
                                         trainActionLossAve_shkp2,
                                         trainActionLossStd_shkp2,
                                         trainActionCorrAcc_shkp2,
                                         trainSpeechCorrAcc_shkp2,
                                         trainSpatialCorrAcc_shkp2,
                                         
                                         # validation
                                         valCostAve_shkp2,
                                         valCostStd_shkp2,
                                         valActionLossAve_shkp2,
                                         valActionLossStd_shkp2,             
                                         valActionCorrAcc_shkp2,
                                         valSpeechCorrAcc_shkp2, 
                                         valSpatialCorrAcc_shkp2,
                                         
                                         # testing
                                         testCostAve_shkp2,
                                         testCostStd_shkp2,
                                         testActionLossAve_shkp2,
                                         testActionLossStd_shkp2,
                                         testActionCorrAcc_shkp2,
                                         testSpeechCorrAcc_shkp2, 
                                         testSpatialCorrAcc_shkp2
                                         ])    
                    

                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ActionLossAve", trainActionLossAve, valActionLossAve, testActionLossAve])
                    tableData.append(["ActionLossStd", trainActionLossStd, valActionLossStd, testActionLossStd])

                    tableData.append(["ActionCorrAcc", trainActionCorrAcc, valActionCorrAcc, testActionCorrAcc])
                    tableData.append(["SpeechCorrAcc", trainSpeechCorrAcc, valSpeechCorrAcc, testSpeechCorrAcc])
                    tableData.append(["SpatialCorrAcc", trainSpatialCorrAcc, valSpatialCorrAcc, testSpatialCorrAcc])


                    tableData.append(["S1 CostAve", trainCostAve_shkp1, valCostAve_shkp1, testCostAve_shkp1])
                    tableData.append(["S1 CostStd", trainCostStd_shkp1, valCostStd_shkp1, testCostStd_shkp1])
                    tableData.append(["S1 ActionLossAve", trainActionLossAve_shkp1, valActionLossAve_shkp1, testActionLossAve_shkp1])
                    tableData.append(["S1 ActionLossStd", trainActionLossStd_shkp1, valActionLossStd_shkp1, testActionLossStd_shkp1])

                    tableData.append(["S1 ActionCorrAcc", trainActionCorrAcc_shkp1, valActionCorrAcc_shkp1, testActionCorrAcc_shkp1])
                    tableData.append(["S1 SpeechCorrAcc", trainSpeechCorrAcc_shkp1, valSpeechCorrAcc_shkp1, testSpeechCorrAcc_shkp1])
                    tableData.append(["S1 SpatialCorrAcc", trainSpatialCorrAcc_shkp1, valSpatialCorrAcc_shkp1, testSpatialCorrAcc_shkp1])


                    tableData.append(["S2 CostAve", trainCostAve_shkp2, valCostAve_shkp2, testCostAve_shkp2])
                    tableData.append(["S2 CostStd", trainCostStd_shkp2, valCostStd_shkp2, testCostStd_shkp2])
                    tableData.append(["S2 ActionLossAve", trainActionLossAve_shkp2, valActionLossAve_shkp2, testActionLossAve_shkp2])
                    tableData.append(["S2 ActionLossStd", trainActionLossStd_shkp2, valActionLossStd_shkp2, testActionLossStd_shkp2])

                    tableData.append(["S2 ActionCorrAcc", trainActionCorrAcc_shkp2, valActionCorrAcc_shkp2, testActionCorrAcc_shkp2])
                    tableData.append(["S2 SpeechCorrAcc", trainSpeechCorrAcc_shkp2, valSpeechCorrAcc_shkp2, testSpeechCorrAcc_shkp2])
                    tableData.append(["S2 SpatialCorrAcc", trainSpatialCorrAcc_shkp2, valSpatialCorrAcc_shkp2, testSpatialCorrAcc_shkp2])
                    
                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)

            
            #################################################################################################################
            # END BASELINE 4 RUN!
            #################################################################################################################
                    
            
            elif bl5_run:
            
                #################################################################################################################
                # BEGIN BASELINE 5 RUN!
                #################################################################################################################
                
                if e != 0:
                    # train
                    if randomizeTrainingBatches:
                        trainIndicesOrder = random.sample(trainIndices, len(trainIndices))
                    else:
                        trainIndicesOrder = trainIndices

                    learner.train(bl5_inputVectorsCombined[trainIndicesOrder], 
                                  bl5_outputActionIDs_shkp2[trainIndicesOrder], 
                                  bl5_outputMasks_shkp2[trainIndicesOrder])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainActionLoss = learner.get_loss(
                        bl5_inputVectorsCombined[trainIndices],
                        bl5_outputActionIDs_shkp2[trainIndices],                                       
                        bl5_outputMasks_shkp2[trainIndices])
                    
                    # validation loss
                    valCost, valActionLoss = learner.get_loss(
                        bl5_inputVectorsCombined[valIndices],
                        bl5_outputActionIDs_shkp2[valIndices],                                       
                        bl5_outputMasks_shkp2[valIndices])
                    
                    # test loss
                    testCost, testActionLoss = learner.get_loss(
                        bl5_inputVectorsCombined[testIndices],
                        bl5_outputActionIDs_shkp2[testIndices],                                       
                        bl5_outputMasks_shkp2[testIndices])
                    
                    # sometimes because of the batch size, some instances don't get a cost computed, so deal with that here
                    trainCost = np.asarray(trainCost).flatten()
                    trainActionLoss = np.asarray(trainActionLoss).flatten()
                    trainEndIndex = trainCost.shape[0]
                    trainCost = trainCost[np.where(bl5_outputMasks_shkp2[trainIndices][:trainEndIndex])].tolist()
                    trainActionLoss = trainActionLoss[np.where(bl5_outputMasks_shkp2[trainIndices][:trainEndIndex])].tolist()
                    
                    valCost = np.asarray(valCost).flatten()
                    valActionLoss = np.asarray(valActionLoss).flatten()
                    valEndIndex = valCost.shape[0]
                    valCost = valCost[np.where(bl5_outputMasks_shkp2[valIndices][:valEndIndex])].tolist()
                    valActionLoss = valActionLoss[np.where(bl5_outputMasks_shkp2[valIndices][:valEndIndex])].tolist()
                    
                    testCost = np.asarray(testCost).flatten()
                    testActionLoss = np.asarray(testActionLoss).flatten()
                    testEndIndex = testCost.shape[0]
                    testCost = testCost[np.where(bl5_outputMasks_shkp2[testIndices][:testEndIndex])].tolist()
                    testActionLoss = testActionLoss[np.where(bl5_outputMasks_shkp2[testIndices][:testEndIndex])].tolist()

                    
                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainActionLossAve = np.mean(trainActionLoss)
                                        
                    trainCostStd = np.std(trainCost)
                    trainActionLossStd = np.std(trainActionLoss)
                    
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valActionLossAve = np.mean(valActionLoss)

                    valCostStd = np.std(valCost)
                    valActionLossStd = np.std(valActionLoss)
                    
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testActionLossAve = np.mean(testActionLoss)

                    testCostStd = np.std(testCost)
                    testActionLossStd = np.std(testActionLoss)
                    
                    
                    # predict
                    predShkpActions = learner.predict(
                        bl5_inputVectorsCombined,
                        bl5_outputActionIDs_shkp2,                                       
                        bl5_outputMasks_shkp2)
                    
                    
                    def bl5_evaluate_predictions(evalSetName, evalIndices, csvLogRows):
                        # TODO: don't include null actions, etc. in the performance metrics computations

                        # for computing accuracies
                        actions_gt = []
                        actions_pred = []

                        speechClusts_gt = []
                        speechClusts_pred = []

                        spatial_gt = []
                        spatial_pred = []

                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpActions):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i

                            #
                            # get the speech cluster and spatial info predictions
                            #
                            predActionClustID = predShkpActions[i]
                            predSpeechClustID = bl5_actionClustIDToSpeechClustID[predActionClustID]
                            predSpatialInfo = bl5_actionClustIDToSpatial[predActionClustID]
                            predSpatialInfoName = bl5_actionClustIDToSpatialName[predActionClustID]
                            predRepUtt = bl5_speechClustIDToRepUtt[predSpeechClustID]

                            gtActionClusterID = bl5_outputActionIDs_shkp2[i]
                            gtSpeechClustID = bl5_actionClustIDToSpeechClustID[gtActionClusterID]
                            gtSpatialInfo = bl5_actionClustIDToSpatial[gtActionClusterID]


                            #
                            # prediction info
                            #
                            csvLogRows[i]["SHOPKEEPER_2_LOSS_WEIGHT"] = bl5_outputMasks_shkp2[i]
                            csvLogRows[i]["PRED_SHOPKEEPER_2_ACTION_CLUSTER"] = predActionClustID
                            csvLogRows[i]["PRED_SHOPKEEPER_2_SPEECH_CLUSTER"] = predSpeechClustID
                            csvLogRows[i]["PRED_SHOPKEEPER_2_SPATIAL_INFO"] = predSpatialInfo
                            csvLogRows[i]["PRED_SHOPKEEPER_2_SPATIAL_INFO_NAME"] = predSpatialInfoName
                            csvLogRows[i]["PRED_SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE"] = predRepUtt

                            #
                            # for computing accuracies
                            #
                            actions_gt.append(gtActionClusterID)
                            actions_pred.append(predActionClustID)

                            speechClusts_gt.append(gtSpeechClustID)
                            speechClusts_pred.append(predSpeechClustID)

                            spatial_gt.append(gtSpatialInfo)
                            spatial_pred.append(predSpatialInfo)
                        
                        
                        #
                        # compute accuracies
                        # fix the len of the output masks because sometimes test set gets cut off during prediction
                        #
                        actionCorrAcc = accuracy_score(actions_gt, actions_pred, sample_weight=bl5_outputMasks_shkp2[evalIndices][:len(spatial_gt)])
                        #actionPrec, actionRec, actionFsc, actionSupp = precision_recall_fscore_support(actions_gt, actions_pred, sample_weight=bl5_outputMasks[evalIndices])
                        
                        speechCorrAcc = accuracy_score(speechClusts_gt, speechClusts_pred, sample_weight=bl5_outputMasks_shkp2[evalIndices][:len(spatial_gt)])
                        #speechPrec, speechRec, speechFsc, speechSupp = precision_recall_fscore_support(speechClusts_gt, speechClusts_pred, sample_weight=bl5_outputMasks[evalIndices])
                        
                        spatialCorrAcc = accuracy_score(spatial_gt, spatial_pred, sample_weight=bl5_outputMasks_shkp2[evalIndices][:len(spatial_gt)]) 
                        #spatialPrec, spatialRec, spatialFsc, spatialSupp = precision_recall_fscore_support(spatial_gt, spatial_pred, sample_weight=bl5_outputMasks[evalIndices])
                        


                        return csvLogRows, actionCorrAcc, speechCorrAcc, spatialCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(humanReadableInputsOutputs)
                    
                    csvLogRows, trainActionCorrAcc, trainSpeechCorrAcc, trainSpatialCorrAcc = bl5_evaluate_predictions("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valActionCorrAcc, valSpeechCorrAcc, valSpatialCorrAcc = bl5_evaluate_predictions("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testActionCorrAcc, testSpeechCorrAcc, testSpatialCorrAcc = bl5_evaluate_predictions("TEST", testIndices, csvLogRows)
                    
                    
                    #
                    # save the evaluation results
                    #
                    tools.save_interaction_data(csvLogRows, foldDir+"/{:04}_all_outputs.csv".format(e), interactionsFieldnames)


                    # append to session log   
                    with open(foldLogFile, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainActionLossAve,
                                         trainActionLossStd,
                                         
                                         trainActionCorrAcc,
                                         trainSpeechCorrAcc,
                                         trainSpatialCorrAcc,

                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valActionLossAve,
                                         valActionLossStd,
                                                                                  
                                         valActionCorrAcc,
                                         valSpeechCorrAcc, 
                                         valSpatialCorrAcc,
                                        
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testActionLossAve,
                                         testActionLossStd,
                                         
                                         testActionCorrAcc,
                                         testSpeechCorrAcc, 
                                         testSpatialCorrAcc
                                         ])    
                
                
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ActionLossAve", trainActionLossAve, valActionLossAve, testActionLossAve])
                    tableData.append(["ActionLossStd", trainActionLossStd, valActionLossStd, testActionLossStd])
                    
                    tableData.append(["ActionCorrAcc", trainActionCorrAcc, valActionCorrAcc, testActionCorrAcc])
                    tableData.append(["SpeechCorrAcc", trainSpeechCorrAcc, valSpeechCorrAcc, testSpeechCorrAcc])
                    tableData.append(["SpatialCorrAcc", trainSpatialCorrAcc, valSpatialCorrAcc, testSpatialCorrAcc])
                    

                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)

            
            #################################################################################################################
            # END BASELINE 5 RUN!
            #################################################################################################################


            elif bl6_run:
            
                #################################################################################################################
                # BEGIN BASELINE 6 RUN!
                #################################################################################################################
                
                if e != 0:
                    # train

                    # combine S1 and S2 data
                    bl6_inputVectorsCombined_temp = np.concatenate([bl6_inputVectorsCombined[trainIndices], bl6_inputVectorsCombined[trainIndices]])
                    bl6_outputActionIDs_temp = np.concatenate([bl6_outputActionIDs_shkp1[trainIndices], bl6_outputActionIDs_shkp2[trainIndices]])
                    bl6_outputMasks_temp = np.concatenate([bl6_outputMasks_shkp1[trainIndices], bl6_outputMasks_shkp2[trainIndices]])
                    bl6_additionalInputs_temp = np.concatenate([bl6_additionalInputs_shkp1[trainIndices], bl6_additionalInputs_shkp2[trainIndices]])

                    trainIndicesOrder = list(range(bl6_inputVectorsCombined_temp.shape[0]))

                    if randomizeTrainingBatches:
                        trainIndicesOrder = random.sample(trainIndicesOrder, len(trainIndicesOrder))
                    

                    learner.train(bl6_inputVectorsCombined_temp[trainIndicesOrder], 
                                  bl6_outputActionIDs_temp[trainIndicesOrder], 
                                  bl6_outputMasks_temp[trainIndicesOrder],
                                  bl6_additionalInputs_temp[trainIndicesOrder])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # S1
                    # training loss
                    trainCost_shkp1, trainActionLoss_shkp1 = learner.get_loss(
                        bl6_inputVectorsCombined[trainIndices],
                        bl6_outputActionIDs_shkp1[trainIndices],                                       
                        bl6_outputMasks_shkp1[trainIndices],
                        bl6_additionalInputs_shkp1[trainIndices])
                    
                    # validation loss
                    valCost_shkp1, valActionLoss_shkp1 = learner.get_loss(
                        bl6_inputVectorsCombined[valIndices],
                        bl6_outputActionIDs_shkp1[valIndices],                                       
                        bl6_outputMasks_shkp1[valIndices],
                            bl6_additionalInputs_shkp1[valIndices])
                    
                    # test loss
                    testCost_shkp1, testActionLoss_shkp1 = learner.get_loss(
                        bl6_inputVectorsCombined[testIndices],
                        bl6_outputActionIDs_shkp1[testIndices],                                       
                        bl6_outputMasks_shkp1[testIndices],
                        bl6_additionalInputs_shkp1[testIndices])
                    

                    # S2
                    # training loss
                    trainCost_shkp2, trainActionLoss_shkp2 = learner.get_loss(
                        bl6_inputVectorsCombined[trainIndices],
                        bl6_outputActionIDs_shkp2[trainIndices],                                       
                        bl6_outputMasks_shkp2[trainIndices],
                        bl6_additionalInputs_shkp2[trainIndices])
                    
                    # validation loss
                    valCost_shkp2, valActionLoss_shkp2 = learner.get_loss(
                        bl6_inputVectorsCombined[valIndices],
                        bl6_outputActionIDs_shkp2[valIndices],                                       
                        bl6_outputMasks_shkp2[valIndices],
                        bl6_additionalInputs_shkp2[valIndices])
                    
                    # test loss
                    testCost_shkp2, testActionLoss_shkp2 = learner.get_loss(
                        bl6_inputVectorsCombined[testIndices],
                        bl6_outputActionIDs_shkp2[testIndices],                                       
                        bl6_outputMasks_shkp2[testIndices],
                        bl6_additionalInputs_shkp2[testIndices])
                    
                    # sometimes because of the batch size, some instances don't get a cost computed, so deal with that here

                    trainCost_shkp1 = np.asarray(trainCost_shkp1).flatten()
                    trainCost_shkp2 = np.asarray(trainCost_shkp2).flatten()
                    trainActionLoss_shkp1 = np.asarray(trainActionLoss_shkp1).flatten()
                    trainActionLoss_shkp2 = np.asarray(trainActionLoss_shkp2).flatten()
                    trainEndIndex = trainCost_shkp1.shape[0]
                    trainCost_shkp1 = trainCost_shkp1[np.where(bl6_outputMasks_shkp1[trainIndices][:trainEndIndex])].tolist()
                    trainCost_shkp2 = trainCost_shkp2[np.where(bl6_outputMasks_shkp2[trainIndices][:trainEndIndex])].tolist()
                    trainActionLoss_shkp1 = trainActionLoss_shkp1[np.where(bl6_outputMasks_shkp1[trainIndices][:trainEndIndex])].tolist()
                    trainActionLoss_shkp2 = trainActionLoss_shkp2[np.where(bl6_outputMasks_shkp2[trainIndices][:trainEndIndex])].tolist()
                    
                    valCost_shkp1 = np.asarray(valCost_shkp1).flatten()
                    valCost_shkp2 = np.asarray(valCost_shkp2).flatten()
                    valActionLoss_shkp1 = np.asarray(valActionLoss_shkp1).flatten()
                    valActionLoss_shkp2 = np.asarray(valActionLoss_shkp2).flatten()
                    valEndIndex = valCost_shkp1.shape[0]
                    valCost_shkp1 = valCost_shkp1[np.where(bl6_outputMasks_shkp1[valIndices][:valEndIndex])].tolist()
                    valCost_shkp2 = valCost_shkp2[np.where(bl6_outputMasks_shkp2[valIndices][:valEndIndex])].tolist()
                    valActionLoss_shkp1 = valActionLoss_shkp1[np.where(bl6_outputMasks_shkp1[valIndices][:valEndIndex])].tolist()
                    valActionLoss_shkp2 = valActionLoss_shkp2[np.where(bl6_outputMasks_shkp2[valIndices][:valEndIndex])].tolist()
                    
                    testCost_shkp1 = np.asarray(testCost_shkp1).flatten()
                    testCost_shkp2 = np.asarray(testCost_shkp2).flatten()
                    testActionLoss_shkp1 = np.asarray(testActionLoss_shkp1).flatten()
                    testActionLoss_shkp2 = np.asarray(testActionLoss_shkp2).flatten()
                    testEndIndex = testCost_shkp1.shape[0]
                    testCost_shkp1 = testCost_shkp1[np.where(bl6_outputMasks_shkp1[testIndices][:testEndIndex])].tolist()
                    testCost_shkp2 = testCost_shkp2[np.where(bl6_outputMasks_shkp2[testIndices][:testEndIndex])].tolist()
                    testActionLoss_shkp1 = testActionLoss_shkp1[np.where(bl6_outputMasks_shkp1[testIndices][:testEndIndex])].tolist()
                    testActionLoss_shkp2 = testActionLoss_shkp2[np.where(bl6_outputMasks_shkp2[testIndices][:testEndIndex])].tolist()


                    # S1 + S2
                    trainCost = trainCost_shkp1 + trainCost_shkp2
                    trainActionLoss = trainActionLoss_shkp1 + trainActionLoss_shkp2
                    valCost = valCost_shkp1 + valCost_shkp2
                    valActionLoss = valActionLoss_shkp1 + valActionLoss_shkp2
                    testCost = testCost_shkp1 + testCost_shkp2
                    testActionLoss = testActionLoss_shkp1 + testActionLoss_shkp2


                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainActionLossAve = np.mean(trainActionLoss)
                    trainCostStd = np.std(trainCost)
                    trainActionLossStd = np.std(trainActionLoss)

                    trainCostAve_shkp1 = np.mean(trainCost_shkp1)
                    trainActionLossAve_shkp1 = np.mean(trainActionLoss_shkp1)
                    trainCostStd_shkp1 = np.std(trainCost_shkp1)
                    trainActionLossStd_shkp1 = np.std(trainActionLoss_shkp1)

                    trainCostAve_shkp2 = np.mean(trainCost_shkp2)
                    trainActionLossAve_shkp2 = np.mean(trainActionLoss_shkp2)
                    trainCostStd_shkp2 = np.std(trainCost_shkp2)
                    trainActionLossStd_shkp2 = np.std(trainActionLoss_shkp2)
                    
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valActionLossAve = np.mean(valActionLoss)
                    valCostStd = np.std(valCost)
                    valActionLossStd = np.std(valActionLoss)

                    valCostAve_shkp1 = np.mean(valCost_shkp1)
                    valActionLossAve_shkp1 = np.mean(valActionLoss_shkp1)
                    valCostStd_shkp1 = np.std(valCost_shkp1)
                    valActionLossStd_shkp1 = np.std(valActionLoss_shkp1)
                    
                    valCostAve_shkp2 = np.mean(valCost_shkp2)
                    valActionLossAve_shkp2 = np.mean(valActionLoss_shkp2)
                    valCostStd_shkp2 = np.std(valCost_shkp2)
                    valActionLossStd_shkp2 = np.std(valActionLoss_shkp2)


                    # test
                    testCostAve = np.mean(testCost)
                    testActionLossAve = np.mean(testActionLoss)
                    testCostStd = np.std(testCost)
                    testActionLossStd = np.std(testActionLoss)

                    testCostAve_shkp1 = np.mean(testCost_shkp1)
                    testActionLossAve_shkp1 = np.mean(testActionLoss_shkp1)
                    testCostStd_shkp1 = np.std(testCost_shkp1)
                    testActionLossStd_shkp1 = np.std(testActionLoss_shkp1)

                    testCostAve_shkp2 = np.mean(testCost_shkp2)
                    testActionLossAve_shkp2 = np.mean(testActionLoss_shkp2)
                    testCostStd_shkp2 = np.std(testCost_shkp2)
                    testActionLossStd_shkp2 = np.std(testActionLoss_shkp2)
                    
                    
                    # predict

                    # S1
                    predShkpActions_shkp1 = learner.predict(
                        bl6_inputVectorsCombined,
                        bl6_outputActionIDs_shkp1,                                       
                        bl6_outputMasks_shkp1,
                        bl6_additionalInputs_shkp1)

                    # S2
                    predShkpActions_shkp2 = learner.predict(
                        bl6_inputVectorsCombined,
                        bl6_outputActionIDs_shkp2,                                       
                        bl6_outputMasks_shkp2,
                        bl6_additionalInputs_shkp2)
                    
                    
                    def bl6_evaluate_predictions(evalSetName, evalIndices, csvLogRows, shopkeeper):
                        # TODO: don't include null actions, etc. in the performance metrics computations

                        if shopkeeper == "SHOPKEEPER_1":
                            predShkpActions = predShkpActions_shkp1
                            bl6_outputActionIDs = bl6_outputActionIDs_shkp1
                            bl6_outputMasks = bl6_outputMasks_shkp1

                        elif shopkeeper == "SHOPKEEPER_2":
                            predShkpActions = predShkpActions_shkp2
                            bl6_outputActionIDs = bl6_outputActionIDs_shkp2
                            bl6_outputMasks = bl6_outputMasks_shkp2



                        # for computing accuracies
                        actions_gt = []
                        actions_pred = []

                        speechClusts_gt = []
                        speechClusts_pred = []

                        spatial_gt = []
                        spatial_pred = []

                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpActions):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i

                            #
                            # get the speech cluster and spatial info predictions
                            #
                            predActionClustID = predShkpActions[i]
                            predSpeechClustID = bl6_actionClustIDToSpeechClustID[predActionClustID]
                            predSpatialInfo = bl6_actionClustIDToSpatial[predActionClustID]
                            predSpatialInfoName = bl6_actionClustIDToSpatialName[predActionClustID]
                            predRepUtt = bl6_speechClustIDToRepUtt[predSpeechClustID]

                            gtActionClusterID = bl6_outputActionIDs[i]
                            gtSpeechClustID = bl6_actionClustIDToSpeechClustID[gtActionClusterID]
                            gtSpatialInfo = bl6_actionClustIDToSpatial[gtActionClusterID]


                            #
                            # prediction info
                            #
                            csvLogRows[i]["{}_LOSS_WEIGHT".format(shopkeeper)] = bl6_outputMasks[i]
                            csvLogRows[i]["PRED_{}_ACTION_CLUSTER".format(shopkeeper)] = predActionClustID
                            csvLogRows[i]["PRED_{}_SPEECH_CLUSTER".format(shopkeeper)] = predSpeechClustID
                            csvLogRows[i]["PRED_{}_SPATIAL_INFO".format(shopkeeper)] = predSpatialInfo
                            csvLogRows[i]["PRED_{}_SPATIAL_INFO_NAME".format(shopkeeper)] = predSpatialInfoName
                            csvLogRows[i]["PRED_{}_REPRESENTATIVE_UTTERANCE".format(shopkeeper)] = predRepUtt

                            #
                            # for computing accuracies
                            #
                            actions_gt.append(gtActionClusterID)
                            actions_pred.append(predActionClustID)

                            speechClusts_gt.append(gtSpeechClustID)
                            speechClusts_pred.append(predSpeechClustID)

                            spatial_gt.append(gtSpatialInfo)
                            spatial_pred.append(predSpatialInfo)
                        
                        
                        #
                        # compute accuracies
                        # fix the len of the output masks because sometimes test set gets cut off during prediction
                        #
                        actionCorrAcc = accuracy_score(actions_gt, actions_pred, sample_weight=bl6_outputMasks[evalIndices][:len(spatial_gt)])
                        #actionPrec, actionRec, actionFsc, actionSupp = precision_recall_fscore_support(actions_gt, actions_pred, sample_weight=bl6_outputMasks[evalIndices])
                        
                        speechCorrAcc = accuracy_score(speechClusts_gt, speechClusts_pred, sample_weight=bl6_outputMasks[evalIndices][:len(spatial_gt)])
                        #speechPrec, speechRec, speechFsc, speechSupp = precision_recall_fscore_support(speechClusts_gt, speechClusts_pred, sample_weight=bl6_outputMasks[evalIndices])
                        
                        spatialCorrAcc = accuracy_score(spatial_gt, spatial_pred, sample_weight=bl6_outputMasks[evalIndices][:len(spatial_gt)]) 
                        #spatialPrec, spatialRec, spatialFsc, spatialSupp = precision_recall_fscore_support(spatial_gt, spatial_pred, sample_weight=bl6_outputMasks[evalIndices])
                        


                        return csvLogRows, actionCorrAcc, speechCorrAcc, spatialCorrAcc
                    
                    
                    csvLogRows = copy.deepcopy(humanReadableInputsOutputs)
                    
                    # S1
                    csvLogRows, trainActionCorrAcc_shkp1, trainSpeechCorrAcc_shkp1, trainSpatialCorrAcc_shkp1 = bl6_evaluate_predictions("TRAIN", trainIndices, csvLogRows, "SHOPKEEPER_1")
                    csvLogRows, valActionCorrAcc_shkp1, valSpeechCorrAcc_shkp1, valSpatialCorrAcc_shkp1 = bl6_evaluate_predictions("VAL", valIndices, csvLogRows, "SHOPKEEPER_1")
                    csvLogRows, testActionCorrAcc_shkp1, testSpeechCorrAcc_shkp1, testSpatialCorrAcc_shkp1 = bl6_evaluate_predictions("TEST", testIndices, csvLogRows, "SHOPKEEPER_1")
                    
                    # S2
                    csvLogRows, trainActionCorrAcc_shkp2, trainSpeechCorrAcc_shkp2, trainSpatialCorrAcc_shkp2 = bl6_evaluate_predictions("TRAIN", trainIndices, csvLogRows, "SHOPKEEPER_2")
                    csvLogRows, valActionCorrAcc_shkp2, valSpeechCorrAcc_shkp2, valSpatialCorrAcc_shkp2 = bl6_evaluate_predictions("VAL", valIndices, csvLogRows, "SHOPKEEPER_2")
                    csvLogRows, testActionCorrAcc_shkp2, testSpeechCorrAcc_shkp2, testSpatialCorrAcc_shkp2 = bl6_evaluate_predictions("TEST", testIndices, csvLogRows, "SHOPKEEPER_2")
                    
                    # both
                    trainActionCorrAcc = (trainActionCorrAcc_shkp1 + trainActionCorrAcc_shkp2) / 2
                    trainSpeechCorrAcc = (trainSpeechCorrAcc_shkp1 + trainSpeechCorrAcc_shkp2) / 2
                    trainSpatialCorrAcc = (trainSpatialCorrAcc_shkp1 + trainSpatialCorrAcc_shkp2) / 2
                                                            
                    valActionCorrAcc = (valActionCorrAcc_shkp1 + valActionCorrAcc_shkp2) / 2
                    valSpeechCorrAcc = (valSpeechCorrAcc_shkp1 + valSpeechCorrAcc_shkp2) / 2
                    valSpatialCorrAcc = (valSpatialCorrAcc_shkp1 + valSpatialCorrAcc_shkp2) / 2

                    testActionCorrAcc = (testActionCorrAcc_shkp1 + testActionCorrAcc_shkp2) / 2
                    testSpeechCorrAcc = (testSpeechCorrAcc_shkp1 + testSpeechCorrAcc_shkp2) / 2
                    testSpatialCorrAcc = (testSpatialCorrAcc_shkp1 + testSpatialCorrAcc_shkp2) / 2

                    
                    #
                    # save the evaluation results
                    #
                    tools.save_interaction_data(csvLogRows, foldDir+"/{:04}_all_outputs.csv".format(e), interactionsFieldnames)


                    # append to session log   
                    with open(foldLogFile, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainActionLossAve,
                                         trainActionLossStd,
                                         trainActionCorrAcc,
                                         trainSpeechCorrAcc,
                                         trainSpatialCorrAcc,

                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valActionLossAve,
                                         valActionLossStd,                 
                                         valActionCorrAcc,
                                         valSpeechCorrAcc, 
                                         valSpatialCorrAcc,
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testActionLossAve,
                                         testActionLossStd,
                                         testActionCorrAcc,
                                         testSpeechCorrAcc, 
                                         testSpatialCorrAcc,

                                         # S1
                                         # training
                                         trainCostAve_shkp1,
                                         trainCostStd_shkp1,
                                         trainActionLossAve_shkp1,
                                         trainActionLossStd_shkp1,
                                         trainActionCorrAcc_shkp1,
                                         trainSpeechCorrAcc_shkp1,
                                         trainSpatialCorrAcc_shkp1,
                                         
                                         # validation
                                         valCostAve_shkp1,
                                         valCostStd_shkp1,
                                         valActionLossAve_shkp1,
                                         valActionLossStd_shkp1,             
                                         valActionCorrAcc_shkp1,
                                         valSpeechCorrAcc_shkp1, 
                                         valSpatialCorrAcc_shkp1,
                                                                                 
                                         # testing
                                         testCostAve_shkp1,
                                         testCostStd_shkp1,
                                         testActionLossAve_shkp1,
                                         testActionLossStd_shkp1,
                                         testActionCorrAcc_shkp1,
                                         testSpeechCorrAcc_shkp1, 
                                         testSpatialCorrAcc_shkp1,

                                         # S2
                                         # training
                                         trainCostAve_shkp2,
                                         trainCostStd_shkp2,
                                         trainActionLossAve_shkp2,
                                         trainActionLossStd_shkp2,
                                         trainActionCorrAcc_shkp2,
                                         trainSpeechCorrAcc_shkp2,
                                         trainSpatialCorrAcc_shkp2,
                                         
                                         # validation
                                         valCostAve_shkp2,
                                         valCostStd_shkp2,
                                         valActionLossAve_shkp2,
                                         valActionLossStd_shkp2,             
                                         valActionCorrAcc_shkp2,
                                         valSpeechCorrAcc_shkp2, 
                                         valSpatialCorrAcc_shkp2,
                                         
                                         # testing
                                         testCostAve_shkp2,
                                         testCostStd_shkp2,
                                         testActionLossAve_shkp2,
                                         testActionLossStd_shkp2,
                                         testActionCorrAcc_shkp2,
                                         testSpeechCorrAcc_shkp2, 
                                         testSpatialCorrAcc_shkp2
                                         ])    
                    

                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ActionLossAve", trainActionLossAve, valActionLossAve, testActionLossAve])
                    tableData.append(["ActionLossStd", trainActionLossStd, valActionLossStd, testActionLossStd])

                    tableData.append(["ActionCorrAcc", trainActionCorrAcc, valActionCorrAcc, testActionCorrAcc])
                    tableData.append(["SpeechCorrAcc", trainSpeechCorrAcc, valSpeechCorrAcc, testSpeechCorrAcc])
                    tableData.append(["SpatialCorrAcc", trainSpatialCorrAcc, valSpatialCorrAcc, testSpatialCorrAcc])


                    tableData.append(["S1 CostAve", trainCostAve_shkp1, valCostAve_shkp1, testCostAve_shkp1])
                    tableData.append(["S1 CostStd", trainCostStd_shkp1, valCostStd_shkp1, testCostStd_shkp1])
                    tableData.append(["S1 ActionLossAve", trainActionLossAve_shkp1, valActionLossAve_shkp1, testActionLossAve_shkp1])
                    tableData.append(["S1 ActionLossStd", trainActionLossStd_shkp1, valActionLossStd_shkp1, testActionLossStd_shkp1])

                    tableData.append(["S1 ActionCorrAcc", trainActionCorrAcc_shkp1, valActionCorrAcc_shkp1, testActionCorrAcc_shkp1])
                    tableData.append(["S1 SpeechCorrAcc", trainSpeechCorrAcc_shkp1, valSpeechCorrAcc_shkp1, testSpeechCorrAcc_shkp1])
                    tableData.append(["S1 SpatialCorrAcc", trainSpatialCorrAcc_shkp1, valSpatialCorrAcc_shkp1, testSpatialCorrAcc_shkp1])


                    tableData.append(["S2 CostAve", trainCostAve_shkp2, valCostAve_shkp2, testCostAve_shkp2])
                    tableData.append(["S2 CostStd", trainCostStd_shkp2, valCostStd_shkp2, testCostStd_shkp2])
                    tableData.append(["S2 ActionLossAve", trainActionLossAve_shkp2, valActionLossAve_shkp2, testActionLossAve_shkp2])
                    tableData.append(["S2 ActionLossStd", trainActionLossStd_shkp2, valActionLossStd_shkp2, testActionLossStd_shkp2])

                    tableData.append(["S2 ActionCorrAcc", trainActionCorrAcc_shkp2, valActionCorrAcc_shkp2, testActionCorrAcc_shkp2])
                    tableData.append(["S2 SpeechCorrAcc", trainSpeechCorrAcc_shkp2, valSpeechCorrAcc_shkp2, testSpeechCorrAcc_shkp2])
                    tableData.append(["S2 SpatialCorrAcc", trainSpatialCorrAcc_shkp2, valSpatialCorrAcc_shkp2, testSpatialCorrAcc_shkp2])
                    
                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)

            
            #################################################################################################################
            # END BASELINE 6 RUN!
            #################################################################################################################
            

            if bl7_run:
            
                #################################################################################################################
                # BEGIN BASELINE 7 RUN!
                #################################################################################################################
                
                if e != 0:
                    # train
                    if randomizeTrainingBatches:
                        trainIndicesOrder = random.sample(trainIndices, len(trainIndices))
                    else:
                        trainIndicesOrder = trainIndices

                    learner.train(bl7_inputVectorsCombined[trainIndicesOrder], 
                                  bl7_outputTargets[trainIndicesOrder], 
                                  bl7_outputMasks[trainIndicesOrder])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainActionLoss = learner.get_loss(
                        bl7_inputVectorsCombined[trainIndices],
                        bl7_outputTargets[trainIndices],                                       
                        bl7_outputMasks[trainIndices])
                    
                    # validation loss
                    valCost, valActionLoss = learner.get_loss(
                        bl7_inputVectorsCombined[valIndices],
                        bl7_outputTargets[valIndices],                                       
                        bl7_outputMasks[valIndices])
                    
                    # test loss
                    testCost, testActionLoss = learner.get_loss(
                        bl7_inputVectorsCombined[testIndices],
                        bl7_outputTargets[testIndices],                                       
                        bl7_outputMasks[testIndices])
                        
                    
                    # sometimes because of the batch size, some instances don't get a cost computed, so deal with that here
                    trainCost = np.asarray(trainCost).flatten()
                    trainActionLoss = np.asarray(trainActionLoss).flatten()
                    trainEndIndex = trainCost.shape[0]
                    trainCost = trainCost[np.where(bl7_outputMasks[trainIndices][:trainEndIndex])].tolist()
                    trainActionLoss = trainActionLoss[np.where(bl7_outputMasks[trainIndices][:trainEndIndex])].tolist()
                    
                    valCost = np.asarray(valCost).flatten()
                    valActionLoss = np.asarray(valActionLoss).flatten()
                    valEndIndex = valCost.shape[0]
                    valCost = valCost[np.where(bl7_outputMasks[valIndices][:valEndIndex])].tolist()
                    valActionLoss = valActionLoss[np.where(bl7_outputMasks[valIndices][:valEndIndex])].tolist()
                    
                    testCost = np.asarray(testCost).flatten()
                    testActionLoss = np.asarray(testActionLoss).flatten()
                    testEndIndex = testCost.shape[0]
                    testCost = testCost[np.where(bl7_outputMasks[testIndices][:testEndIndex])].tolist()
                    testActionLoss = testActionLoss[np.where(bl7_outputMasks[testIndices][:testEndIndex])].tolist()


                    # compute loss averages and s.d. for aggregate log
                    # train
                    trainCostAve = np.mean(trainCost)
                    trainActionLossAve = np.mean(trainActionLoss)
                                        
                    trainCostStd = np.std(trainCost)
                    trainActionLossStd = np.std(trainActionLoss)
                    
                    
                    # validation
                    valCostAve = np.mean(valCost)
                    valActionLossAve = np.mean(valActionLoss)

                    valCostStd = np.std(valCost)
                    valActionLossStd = np.std(valActionLoss)
                    
                    
                    # test
                    testCostAve = np.mean(testCost)
                    testActionLossAve = np.mean(testActionLoss)

                    testCostStd = np.std(testCost)
                    testActionLossStd = np.std(testActionLoss)
                    
                    
                    # predict
                    predShkpActions = learner.predict(
                        bl7_inputVectorsCombined,
                        bl7_outputTargets,                                       
                        bl7_outputMasks)
                    
                    
                    def bl7_evaluate_predictions(evalSetName, evalIndices, csvLogRows):
                        
                        # for computing accuracies
                        actions_gt = []
                        actions_pred = []
                        
                        for i in evalIndices:
                            
                            # check if the index is one of the ones that was cut off because of the batch size
                            if i >= len(predShkpActions):
                                continue
                            
                            csvLogRows[i]["SET"] = evalSetName
                            csvLogRows[i]["ID"] = i
                            
                            #
                            # target info
                            #
                            csvLogRows[i]["TARG_SHOPKEEPER_ACTION"] = bl7_outputTargets[i]
                            
                            #
                            # prediction info
                            #
                            csvLogRows[i]["PRED_SHOPKEEPER_ACTION"] = predShkpActions[i]

                            #
                            # for computing accuracies
                            #
                            actions_gt.append(csvLogRows[i]["TARG_SHOPKEEPER_ACTION"])
                            actions_pred.append(csvLogRows[i]["PRED_SHOPKEEPER_ACTION"])
                        
                        
                        #
                        # compute accuracies
                        #
                        actionCorrAcc = accuracy_score(actions_gt, actions_pred)
                        actionPrec, actionRec, actionFsc, actionSupp = precision_recall_fscore_support(actions_gt, actions_pred)
                        
                        return csvLogRows, actionCorrAcc, actionPrec, actionRec, actionFsc, actionSupp
                    
                    
                    csvLogRows = copy.deepcopy(humanReadableInputsOutputs)
                    
                    csvLogRows, trainActionCorrAcc, trainActionPrec, trainActionRec, trainActionFsc, trainActionSupp = bl7_evaluate_predictions("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valActionCorrAcc, valActionPrec, valActionRec, valActionFsc, valActionSupp = bl7_evaluate_predictions("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testActionCorrAcc, testActionPrec, testActionRec, testActionFsc, testActionSupp = bl7_evaluate_predictions("TEST", testIndices, csvLogRows)
                    
                    
                    #
                    # save the evaluation results
                    #
                    tools.save_interaction_data(csvLogRows, foldDir+"/{:04}_all_outputs.csv".format(e), interactionsFieldnames)
                                        
                    
                    # append to session log   
                    with open(foldLogFile, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([e,
                                         
                                         # training
                                         trainCostAve,
                                         trainCostStd,
                                         trainActionLossAve,
                                         trainActionLossStd,
                                         
                                         trainActionCorrAcc,
                                         trainActionPrec[1],
                                         trainActionRec[1],
                                         trainActionFsc[1],
                                         trainActionSupp[1],
                                         trainActionPrec[0],
                                         trainActionRec[0],
                                         trainActionFsc[0],
                                         trainActionSupp[0],

                                         
                                         # validation
                                         valCostAve,
                                         valCostStd,
                                         valActionLossAve,
                                         valActionLossStd,
                                                                                  
                                         valActionCorrAcc,
                                         valActionPrec[1], 
                                         valActionRec[1], 
                                         valActionFsc[1], 
                                         valActionSupp[1],
                                         valActionPrec[0], 
                                         valActionRec[0], 
                                         valActionFsc[0], 
                                         valActionSupp[0],
                                         
                                         
                                         # testing
                                         testCostAve,
                                         testCostStd,
                                         testActionLossAve,
                                         testActionLossStd,
                                         
                                         testActionCorrAcc,
                                         testActionPrec[1], 
                                         testActionRec[1], 
                                         testActionFsc[1], 
                                         testActionSupp[1],
                                         testActionPrec[0], 
                                         testActionRec[0], 
                                         testActionFsc[0], 
                                         testActionSupp[0]
                                         ])    
                
                
                    # training
                    print("===== {} EPOCH {} LOSSES AND ACCURACIES=====".format(condition.upper(), e), flush=True, file=foldTerminalOutputStream)
                    tableData = []
                    
                    tableData.append(["CostAve", trainCostAve, valCostAve, testCostAve])
                    tableData.append(["CostStd", trainCostStd, valCostStd, testCostStd])
                    tableData.append(["ActionLossAve", trainActionLossAve, valActionLossAve, testActionLossAve])
                    tableData.append(["ActionLossStd", trainActionLossStd, valActionLossStd, testActionLossStd])
                    
                    tableData.append(["ActionCorrAcc", trainActionCorrAcc, valActionCorrAcc, testActionCorrAcc])
                    tableData.append(["ActionPrec", trainActionPrec, valActionPrec, testActionPrec])
                    tableData.append(["ActionRec", trainActionRec, valActionRec, testActionRec])
                    tableData.append(["ActionFsc", trainActionFsc, valActionFsc, testActionFsc])
                    tableData.append(["ActionSupp", trainActionSupp, valActionSupp, testActionSupp])

                    print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"), flush=True, file=foldTerminalOutputStream)
                            
                    print("", flush=True, file=foldTerminalOutputStream)

            
            #################################################################################################################
            # END BASELINE 7 RUN!
            #################################################################################################################
                    
            
            elif prop_run:
                pass
            
            
            print("Epoch {} time: {}s".format(e, round(time.time() - startTime, 2)), flush=True, file=foldTerminalOutputStream)
    

    #
    # start parallel processing
    #
    if not RUN_PARALLEL:
        run_fold(randomSeed=0, foldId=0, gpu=0)
    
    else:
        processes = []
        
        for fold in range(NUM_FOLDS): #numDatabases
            process = Process(target=run_fold, args=[0, fold, gpuCount%NUM_GPUS]) # randomSeed, foldId, gpu
            process.start()
            processes.append(process)
            gpuCount += 1
        
        """
        for gpu in range(8):
            process = Process(target=run_fold, args=[0, gpu, gpu]) # randomSeed, foldId, gpu
            process.start()
            processes.append(process)
        
        for p in processes:
            p.join()
        """
    
    return gpuCount



#
# start here...
#
gpuCount = 0

#gpuCount = main(mainDir, "baseline1", gpuCount)
#gpuCount = main(mainDir, "baseline2", gpuCount)
#gpuCount = main(mainDir, "baseline4", gpuCount)
gpuCount = main(mainDir, "baseline5", gpuCount)
#gpuCount = main(mainDir, "baseline6", gpuCount)
gpuCount = main(mainDir, "baseline7", gpuCount)



print("Done.")

