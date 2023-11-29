#
# Created on Tue Aug 01 2023
#
# Copyright (c) 2023 Malcolm Doering
#
#
# Combine the outputs of the classifier for whether or not to act (baseline 1) and how to act (baseline 2) to get output for evaluation 
#

import os
import pandas as pd
import copy

import tools



#bl2_expLogName = "20230807-191725_actionPrediction_02_/baseline2" # prediction of S2 actions
#bl1_expLogName = "20230807-191725_actionPrediction_02_/baseline1" # binary prediction of whether S2 acts or not
bl1_expLogName = "20231120-171349_actionPrediction_02_/baseline1" # binary prediction of whether S2 acts or not

# with speech and motion class outputs
#bl2_expLogName = "20230808-163410_actionPrediction_02_/baseline3" # prediction of S2 actions
bl2_expLogName = "20231121-134928_actionPrediction_02_/baseline3" # prediction of S2 actions
bl2_expLogName = "20231120-171349_actionPrediction_02_/baseline3" # prediction of S2 actions
bl2_expLogName = "20231122-174002_actionPrediction_02/baseline3" # 800 hidden, 1e-4 learning rate
bl2_expLogName = "20231124-104954_actionPrediction_02/baseline3" # 800 hidden, 1e-3 learning rate
bl2_expLogName = "20231124-144932_actionPrediction_02/baseline3" # 800 hidden, 1e-3 learning rate, attention, 2000 epochs





bl1_expLogDir = tools.logDir+"/"+bl1_expLogName
bl2_expLogDir = tools.logDir+"/"+bl2_expLogName


sessionDir = sessionDir = tools.create_session_dir("analysis_02")


#
# read in the performance metric data
#
def read_performance_metric_data(expLogDir):

    runDirContents = os.listdir(expLogDir)
    runIds = []

    for rdc in runDirContents:
        if "." not in rdc:
            runIds.append(rdc)

    runIds.sort()

    # this will contain the data from all the csv log files
    runIdToData = {}

    for iId in runIds:
        runIdToData[iId] = pd.read_csv("{}/fold_log_{}.csv".format(expLogDir, iId))
    
    return runIdToData


bl1_runIdToData = read_performance_metric_data(bl1_expLogDir)
bl2_runIdToData = read_performance_metric_data(bl2_expLogDir)


#
# find the best performing epochs for each baseline for each fold
#
def find_best_performing_epoch(rundIdToData, metric, greaterIsBetter):
    runIdToBestEpoch = {}

    for runId in rundIdToData:
        bestScore = None
        bestEpoch = None

        for e in rundIdToData[runId]["Epoch"]:
            score = rundIdToData[runId][metric + " ({})".format(runId)][e]

            if greaterIsBetter:
                if bestScore == None or score > bestScore:
                    bestScore = score
                    bestEpoch = e
            else:
                if bestScore == None or score < bestScore:
                    bestScore = score
                    bestEpoch = e
        
        runIdToBestEpoch[runId] = (bestEpoch, bestScore)

    return runIdToBestEpoch


bl1_runIdToBestEpoch = find_best_performing_epoch(bl1_runIdToData, "Validation Action 1 F-score", True)

bl2_runIdToBestEpoch = find_best_performing_epoch(bl2_runIdToData, "Validation Loss Ave", False)



#
# load the prediction data from the best epochs for each baseline
#
def load_prediction_data_for_epoch(runIdToBestEpoch, expLogDir):
    runIdToPredictionData = {}

    for runId in runIdToBestEpoch:
        e = runIdToBestEpoch[runId][0]

        # load the data
        runIdToPredictionData[runId], fieldnames = tools.load_csv_data("{}/{}/{:04}_all_outputs.csv".format(expLogDir, runId, e), isHeader=True, isJapanese=True)
        
    return runIdToPredictionData, fieldnames


bl1_runIdToPredictionData, bl1_fieldnames = load_prediction_data_for_epoch(bl1_runIdToBestEpoch, bl1_expLogDir)
bl2_runIdToPredictionData, bl2_fieldnames = load_prediction_data_for_epoch(bl2_runIdToBestEpoch, bl2_expLogDir)


#
# combine the data
#
runIdToCombinedPredictionData = {}
allPredictionsCombined = []


for runId in bl1_runIdToPredictionData:
    bl1_data = bl1_runIdToPredictionData[runId]
    bl2_data = bl2_runIdToPredictionData[runId]

    runIdToCombinedPredictionData[runId] = []

    for i in range(len(bl1_data)):

        if bl1_data[i]["PRED_SHOPKEEPER_ACTION"] == "":
            continue


        combined = copy.deepcopy(bl2_data[i])

        combined["FOLD"] = runId
        combined["BASELINE1_BEST_EPOCH"] = bl1_runIdToBestEpoch[runId][0]
        combined["BASELINE1_BEST_SCORE"] = bl1_runIdToBestEpoch[runId][1]
        combined["BASELINE2_BEST_EPOCH"] = bl2_runIdToBestEpoch[runId][0]
        combined["BASELINE2_BEST_SCORE"] = bl2_runIdToBestEpoch[runId][1]

        combined["TARG_SHOPKEEPER_ACTS"] = bl1_data[i]["TARG_SHOPKEEPER_ACTION"]
        combined["PRED_SHOPKEEPER_ACTS"] = bl1_data[i]["PRED_SHOPKEEPER_ACTION"]

        if int(combined["PRED_SHOPKEEPER_ACTS"]):
            combined["COMBINED_PRED_SHOPKEEPER_ACTION"] = bl1_data[i]["PRED_SHOPKEEPER_ACTION"]
            combined["COMBINED_PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"] = combined["PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"]
            combined["COMBINED_PRED_SHOPKEEPER_SPEECH_CLUSTER"] = combined["PRED_SHOPKEEPER_SPEECH_CLUSTER"]
            combined["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO_NAME"] = combined["PRED_SHOPKEEPER_SPATIAL_INFO_NAME"]
            #combined["COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER"] = combined["PRED_SHOPKEEPER_ACTION_CLUSTER"]
            combined["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO"] = combined["PRED_SHOPKEEPER_SPATIAL_INFO"]
        else:
            combined["COMBINED_PRED_SHOPKEEPER_ACTION"] = ""
            combined["COMBINED_PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"] = ""
            combined["COMBINED_PRED_SHOPKEEPER_SPEECH_CLUSTER"] = ""
            combined["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO_NAME"] = ""
            #combined["COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER"] = ""
            combined["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO"] = ""
        
        runIdToCombinedPredictionData[runId].append(combined)
        allPredictionsCombined.append(combined)


#
# save the data
#
combinedFieldnames = bl2_fieldnames
combinedFieldnames = ["FOLD", "BASELINE1_BEST_EPOCH", "BASELINE1_BEST_SCORE", "BASELINE2_BEST_EPOCH", "BASELINE2_BEST_SCORE"] + combinedFieldnames
combinedFieldnames.append("TARG_SHOPKEEPER_ACTS")
combinedFieldnames.append("PRED_SHOPKEEPER_ACTS")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_ACTION")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_SPEECH_CLUSTER")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO_NAME")
#combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO")

['PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE', 'PRED_SHOPKEEPER_ACTION_CLUSTER', 'PRED_SHOPKEEPER_SPATIAL_INFO', 'PRED_SHOPKEEPER_SPATIAL_INFO_NAME', 'PRED_SHOPKEEPER_SPEECH_CLUSTER']
tools.save_interaction_data(allPredictionsCombined, sessionDir+"all_predictions_combined.csv", combinedFieldnames)


print("Done.")