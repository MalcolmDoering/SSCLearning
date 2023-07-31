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

import tools


def split_list(a, n):
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


#################################################################################################################
# file paths
#################################################################################################################

evaluationDataDir = tools.dataDir + "20230731-125515_actionPredictionPreprocessing/"

interactionDataFilename = "20230710-151337_speechPreprocessing/20230623_SSC_3_trueMotionTargets_3_speechMotionCombined.csv"
speechClustersFilename = "20230731-113400_speechClustering/all_shopkeeper_cos_3gram- speech_clusters.csv"
keywordsFilename = tools.modelDir + "20230609-141854_unique_utterance_keywords.csv"
uttVectorizerDir = evaluationDataDir
stoppingLocationClusterDir = tools.modelDir + "20230627_stoppingLocationClusters/"


mainDir = tools.create_session_dir("actionPrediction_02")



def main(mainDir, condition, gpuCount):
    #################################################################################################################
    # running params
    #################################################################################################################

    DEBUG = True
    RUN_PARALLEL = False
    SPEECH_CLUSTER_LOSS_WEIGHTS = False
    NUM_GPUS = 8
    NUM_FOLDS = 8

    numTrainFolds = 6
    numValFolds = 1
    numTestFolds = 1

    # params that should be the same for all conditions (predictors)
    batchSize = 8
    randomizeTrainingBatches = False
    numEpochs = 100
    evalEvery = 1
    minClassCount = 2



    sessionDir = mainDir + "/" + condition
    tools.create_directory(sessionDir)
    
    # what to run. only one of these should be true at a time
    bl1_run = False # prediction of whether or not S2 acts
    bl2_run = False # prediction of S2's actions
    prop_run = False
    
    if condition == "baseline1":
        bl1_run = True
    elif condition == "baseline2":
        bl2_run = True
    elif condition == "proposed":
        prop_run = True
    
    


    #################################################################################################################
    # load the data
    #################################################################################################################
    print("Loading data...")

    # should be the same for all conditions...
    humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(evaluationDataDir+"humanReadableInputsOutputs.csv")

    if bl1_run:
        bl1_outputActionIDs = np.load(evaluationDataDir+"outputActionIDs.npy")
        bl1_outputSpeechClusterIDs = np.load(evaluationDataDir+"outputSpeechClusterIDs.npy")
        bl1_outputSpatialInfo = np.load(evaluationDataDir+"outputSpatialInfo.npy")
        bl1_toIgnore = np.load(evaluationDataDir+"toIgnore.npy")
        bl1_isHidToImitate = np.load(evaluationDataDir+"isHidToImitate.npy")
        bl1_inputVectorsCombined = np.load(evaluationDataDir+"inputVectorsCombined.npy")

    elif bl2_run:
        bl2_outputActionIDs = np.load(evaluationDataDir+"outputActionIDs.npy")
        bl2_outputSpeechClusterIDs = np.load(evaluationDataDir+"outputSpeechClusterIDs.npy")
        bl2_outputSpatialInfo = np.load(evaluationDataDir+"outputSpatialInfo.npy")
        bl2_toIgnore = np.load(evaluationDataDir+"toIgnore.npy")
        bl2_isHidToImitate = np.load(evaluationDataDir+"isHidToImitate.npy")
        bl2_inputVectorsCombined = np.load(evaluationDataDir+"inputVectorsCombined.npy")
    

    
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
            interactionsFieldnames = ["SET", "ID"] + humanReadableInputsOutputsFieldnames + ["PRED_SHOPKEEPER_ACTION",
                                                                                             'PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE', 
                                                                                             'PRED_SHOPKEEPER_SPEECH_CLUSTER', 
                                                                                             'PRED_SHOPKEEPER_SPATIAL_INFO_NAME', 
                                                                                             'PRED_SHOPKEEPER_ACTION_CLUSTER', 
                                                                                             'PRED_SHOPKEEPER_SPATIAL_INFO']
            
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
            bl1_outputMasks = np.ones(bl1_isHidToImitate.shape)
        
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
            bl2_outputActionIDs[np.where(bl2_outputActionIDs == -1)] = nullActionID
            bl2_actionClustCounts[nullActionID] = bl2_actionClustCounts[-1]
            
            bl2_outputClassWeights = np.ones((bl2_numActionClusters))

            # set null action and junk speech clusters to 0 weight 
            bl2_outputMasks = np.copy(bl2_toIgnore)
            bl2_outputMasks[bl2_outputMasks == -1] = 1 # -1 marks non S2 actions
            bl2_outputMasks = 1 -bl2_outputMasks
            
            # set actions with less than min count to 0 weight
            bl2_actionClustOverMinCount = np.asarray([0 if bl2_actionClustCounts[x] < minClassCount else 1 for x in bl2_outputActionIDs])
            bl2_outputMasks[bl2_actionClustOverMinCount == 0] = 0
            
            # only evaluate using certain roles
            bl2_isValidCondition = np.asarray(bl2_isValidCondition)
            bl2_outputMasks[bl2_isValidCondition == 0] = 0

            
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
            bl1_embeddingSize = 100

            learner = learning.SimpleFeedforwardNetwork(bl1_inputDim, 
                                                        bl1_inputSeqLen, 
                                                        bl1_numOutputClasses,
                                                        batchSize, 
                                                        bl1_embeddingSize,
                                                        randomSeed,
                                                        bl1_outputClassWeights)
        
        elif bl2_run:
            bl2_inputDim = bl2_inputVectorsCombined.shape[2]
            bl2_inputSeqLen = bl2_inputVectorsCombined.shape[1]
            bl2_numOutputClasses = bl2_numActionClusters
            bl2_embeddingSize = 100

            learner = learning.SimpleFeedforwardNetwork(bl2_inputDim, 
                                                        bl2_inputSeqLen, 
                                                        bl2_numOutputClasses,
                                                        batchSize, 
                                                        bl2_embeddingSize,
                                                        randomSeed,
                                                        bl2_outputClassWeights)
        
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

                    learner.train(bl1_inputVectorsCombined[trainIndices], 
                                  bl1_isHidToImitate[trainIndices], 
                                  bl1_outputMasks[trainIndices])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainActionLoss = learner.get_loss(
                        bl1_inputVectorsCombined[trainIndices],
                        bl1_isHidToImitate[trainIndices],                                       
                        bl1_outputMasks[trainIndices])
                    
                    # validation loss
                    valCost, valActionLoss = learner.get_loss(
                        bl1_inputVectorsCombined[valIndices],
                        bl1_isHidToImitate[valIndices],                                       
                        bl1_outputMasks[valIndices])
                    
                    # test loss
                    testCost, testActionLoss = learner.get_loss(
                        bl1_inputVectorsCombined[testIndices],
                        bl1_isHidToImitate[testIndices],                                       
                        bl1_outputMasks[testIndices])
                        
                    
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
                        bl1_isHidToImitate,                                       
                        bl1_outputMasks)
                    
                    
                    def evaluate_predictions_bl1(evalSetName, evalIndices, csvLogRows):
                        
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
                            csvLogRows[i]["TARG_SHOPKEEPER_ACTION"] = bl1_isHidToImitate[i]
                            
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
                    
                    csvLogRows, trainActionCorrAcc, trainActionPrec, trainActionRec, trainActionFsc, trainActionSupp = evaluate_predictions_bl1("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valActionCorrAcc, valActionPrec, valActionRec, valActionFsc, valActionSupp = evaluate_predictions_bl1("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testActionCorrAcc, testActionPrec, testActionRec, testActionFsc, testActionSupp = evaluate_predictions_bl1("TEST", testIndices, csvLogRows)
                    
                    
                    #
                    # save the evaluation results
                    #
                    with open(foldDir+"/{:}_all_outputs.csv".format(e), "w", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, interactionsFieldnames)
                        writer.writeheader()
                        writer.writerows(csvLogRows)
                    
                    
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

                    learner.train(bl2_inputVectorsCombined[trainIndices], 
                                  bl2_outputActionIDs[trainIndices], 
                                  bl2_outputMasks[trainIndices])
                
                
                # evaluate
                if e % evalEvery == 0 or e == numEpochs:
                    
                    # training loss
                    trainCost, trainActionLoss = learner.get_loss(
                        bl2_inputVectorsCombined[trainIndices],
                        bl2_outputActionIDs[trainIndices],                                       
                        bl2_outputMasks[trainIndices])
                    
                    # validation loss
                    valCost, valActionLoss = learner.get_loss(
                        bl2_inputVectorsCombined[valIndices],
                        bl2_outputActionIDs[valIndices],                                       
                        bl2_outputMasks[valIndices])
                    
                    # test loss
                    testCost, testActionLoss = learner.get_loss(
                        bl2_inputVectorsCombined[testIndices],
                        bl2_outputActionIDs[testIndices],                                       
                        bl2_outputMasks[testIndices])
                        
                    
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
                        bl2_outputActionIDs,                                       
                        bl2_outputMasks)
                    
                    
                    def evaluate_predictions_bl2(evalSetName, evalIndices, csvLogRows):
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

                            gtActionClusterID = bl2_outputActionIDs[i]
                            gtSpeechClustID = bl2_actionClustIDToSpeechClustID[gtActionClusterID]
                            gtSpatialInfo = bl2_actionClustIDToSpatial[gtActionClusterID]


                            #
                            # prediction info
                            #
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
                    
                    csvLogRows, trainActionCorrAcc, trainSpeechCorrAcc, trainSpatialCorrAcc = evaluate_predictions_bl2("TRAIN", trainIndices, csvLogRows)
                    
                    csvLogRows, valActionCorrAcc, valSpeechCorrAcc, valSpatialCorrAcc = evaluate_predictions_bl2("VAL", valIndices, csvLogRows)
                    
                    csvLogRows, testActionCorrAcc, testSpeechCorrAcc, testSpatialCorrAcc = evaluate_predictions_bl2("TEST", testIndices, csvLogRows)
                    
                    
                    #
                    # save the evaluation results
                    #
                    tools.save_interaction_data(csvLogRows, foldDir+"/{:}_all_outputs.csv".format(e), interactionsFieldnames)


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
gpuCount = main(mainDir, "baseline2", gpuCount)


print("Done.")

