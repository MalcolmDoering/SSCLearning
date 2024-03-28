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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate

import tools


# from March 2024
bl1_expLogName = "20240227-171518_actionPrediction_02/baseline1" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, predicting when S2 acts

bl2_expLogName = "20240227-171518_actionPrediction_02/baseline2" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, action prediction
#bl2_expLogName = "20240227-171518_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, both shopkeepers

#bl2_expLogName = "20240307-183915_actionPrediction_02/baseline2" # 1000 hidden, 1e-5 learning rate, with attention, 3-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, action prediction
#bl2_expLogName = "20240307-183915_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, with attention, 3-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, both shopkeepers


# with KM
#bl1_expLogName = "20240227-171518_actionPrediction_02/baseline7" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, predicting when S2 acts

#bl2_expLogName = "20240227-171518_actionPrediction_02/baseline5" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, action prediction
#bl2_expLogName = "20240227-171518_actionPrediction_02/baseline6" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, both shopkeepers

# test xy
bl2_expLogName = "20240325-133211_actionPrediction_02_testxy/baseline5" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, action prediction
bl1_expLogName = "20240325-133211_actionPrediction_02_testxy/baseline7" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, action prediction



bl1_expLogDir = tools.logDir+"/"+bl1_expLogName
bl2_expLogDir = tools.logDir+"/"+bl2_expLogName


sessionDir = sessionDir = tools.create_session_dir("analysis_02")

bl2_useEpoch = 50 #300


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

if bl2_useEpoch is None:
    bl2_runIdToBestEpoch = find_best_performing_epoch(bl2_runIdToData, "Validation Loss Ave", False)
else:
    bl2_runIdToBestEpoch = {}
    for runId in bl2_runIdToData:
        bl2_runIdToBestEpoch[runId] = (bl2_useEpoch, bl2_runIdToData[runId]["Validation Loss Ave" + " ({})".format(runId)][bl2_useEpoch])


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

SHOPKEEPER = "SHOPKEEPER_2"

PRED_SHOPKEEPER = None
if "baseline2" in bl2_expLogName:
    PRED_SHOPKEEPER = "SHOPKEEPER"
#elif "baseline4" in bl2_expLogName:
else:
    PRED_SHOPKEEPER = "SHOPKEEPER_2"

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
            combined["COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER"] = combined["PRED_{}_ACTION_CLUSTER".format(PRED_SHOPKEEPER)]
            combined["COMBINED_PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"] = combined["PRED_{}_REPRESENTATIVE_UTTERANCE".format(PRED_SHOPKEEPER)]
            combined["COMBINED_PRED_SHOPKEEPER_SPEECH_CLUSTER"] = combined["PRED_{}_SPEECH_CLUSTER".format(PRED_SHOPKEEPER)]
            combined["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO_NAME"] = combined["PRED_{}_SPATIAL_INFO_NAME".format(PRED_SHOPKEEPER)]
            #combined["COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER"] = combined["PRED_SHOPKEEPER_ACTION_CLUSTER"]
            combined["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO"] = combined["PRED_{}_SPATIAL_INFO".format(PRED_SHOPKEEPER)]
        else:
            combined["COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER"] = "-1"
            combined["COMBINED_PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"] = ""
            combined["COMBINED_PRED_SHOPKEEPER_SPEECH_CLUSTER"] = "-1"
            combined["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO_NAME"] = "-1" # TODO this may not be correct... training data contains spatial info for null action clusters... use previous S2 location?
            #combined["COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER"] = ""
            combined["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO"] = "-1"
        
        runIdToCombinedPredictionData[runId].append(combined)
        allPredictionsCombined.append(combined)

        # for debug
        #break

def evaluate_predictions(evalSetName):
    #
    # compute accuracy, etc.
    #
    actions_gt = []
    actions_pred = []

    speechClusts_gt = []
    speechClusts_pred = []

    spatial_gt = []
    spatial_pred = []

    output_masks = []

    for data in allPredictionsCombined:
        if data["SET"] != evalSetName:
            continue

        # check if the index is one of the ones that was cut off because of the batch size
        # TODO

        #
        # get the speech cluster and spatial info predictions
        #
        predActionClustID = data["COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER"]
        predSpeechClustID = data["COMBINED_PRED_SHOPKEEPER_SPEECH_CLUSTER"]
        predSpatialInfo = data["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO"]
        predSpatialInfoName = data["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO_NAME"]
        predRepUtt = data["COMBINED_PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"]

        gtActionClusterID = data["y_{}_ACTION_CLUSTER".format(SHOPKEEPER)]
        gtSpeechClustID = data["y_{}_SPEECH_CLUSTER".format(SHOPKEEPER)]
        gtSpatialInfo = data["y_{}_SPATIAL_INFO".format(SHOPKEEPER)]

        if "baseline2" in bl2_expLogName:
            lossWeight = data["LOSS_WEIGHT"]
        #elif "baseline4" in bl2_expLogName:
        else:
            lossWeight = data["{}_LOSS_WEIGHT".format(SHOPKEEPER)]
        
        #
        # for computing accuracies
        #
        actions_gt.append(int(gtActionClusterID))
        actions_pred.append(int(predActionClustID))

        speechClusts_gt.append(int(gtSpeechClustID))
        speechClusts_pred.append(int(predSpeechClustID))

        spatial_gt.append(int(gtSpatialInfo))
        spatial_pred.append(int(predSpatialInfo))

        output_masks.append(int(lossWeight))

    #
    # compute accuracies
    # fix the len of the output masks because sometimes test set gets cut off during prediction
    #
    
    """
    # replace null actions -1 with a bigger number
    max_action_clust = max(actions_gt+actions_pred)
    actions_gt = [max_action_clust if x == -1 else x for x in actions_gt]
    actions_pred = [max_action_clust if x == -1 else x for x in actions_pred]

    max_speech_clust = max(speechClusts_gt+speechClusts_pred)
    speechClusts_gt = [max_speech_clust if x == -1 else x for x in speechClusts_gt]
    speechClusts_pred = [max_speech_clust if x == -1 else x for x in speechClusts_pred]

    max_spatial = max(spatial_gt+spatial_pred)
    spatial_gt = [max_spatial if x == -1 else x for x in spatial_gt]
    spatial_pred = [max_spatial if x == -1 else x for x in spatial_pred]
    """

    actionCorrAccMask = accuracy_score(actions_gt, actions_pred, sample_weight=output_masks)
    actionCorrAcc = accuracy_score(actions_gt, actions_pred)
    #actionPrec, actionRec, actionFsc, actionSupp = precision_recall_fscore_support(actions_gt, actions_pred, sample_weight=bl4_outputMasks[evalIndices])

    speechCorrAccMask = accuracy_score(speechClusts_gt, speechClusts_pred, sample_weight=output_masks)
    speechCorrAcc = accuracy_score(speechClusts_gt, speechClusts_pred)
    #speechPrec, speechRec, speechFsc, speechSupp = precision_recall_fscore_support(speechClusts_gt, speechClusts_pred, sample_weight=bl4_outputMasks[evalIndices])

    spatialCorrAccMask = accuracy_score(spatial_gt, spatial_pred, sample_weight=output_masks) 
    spatialCorrAcc = accuracy_score(spatial_gt, spatial_pred) 
    #spatialPrec, spatialRec, spatialFsc, spatialSupp = precision_recall_fscore_support(spatial_gt, spatial_pred, sample_weight=bl4_outputMasks[evalIndices])

    return actionCorrAcc, actionCorrAccMask, speechCorrAcc, speechCorrAccMask, spatialCorrAcc, spatialCorrAccMask


trainActionCorrAcc, trainActionCorrAccMask, trainSpeechCorrAcc, trainSpeechCorrAccMask, trainSpatialCorrAcc, trainSpatialCorrAccMask = evaluate_predictions("TRAIN")
valActionCorrAcc, valActionCorrAccMask, valSpeechCorrAcc, valSpeechCorrAccMask, valSpatialCorrAcc, valSpatialCorrAccMask = evaluate_predictions("VAL")
testActionCorrAcc, testActionCorrAccMask, testSpeechCorrAcc, testSpeechCorrAccMask, testSpatialCorrAcc, testSpatialCorrAccMask = evaluate_predictions("TEST")

print("===== LOSSES AND ACCURACIES=====")
tableData = []
tableData.append(["ActionCorrAcc", trainActionCorrAcc, valActionCorrAcc, testActionCorrAcc])
tableData.append(["SpeechCorrAcc", trainSpeechCorrAcc, valSpeechCorrAcc, testSpeechCorrAcc])
tableData.append(["SpatialCorrAcc", trainSpatialCorrAcc, valSpatialCorrAcc, testSpatialCorrAcc])

tableData.append(["ActionCorrAccMask", trainActionCorrAccMask, valActionCorrAccMask, testActionCorrAccMask])
tableData.append(["SpeechCorrAccMask", trainSpeechCorrAccMask, valSpeechCorrAccMask, testSpeechCorrAccMask])
tableData.append(["SpatialCorrAccMask", trainSpatialCorrAccMask, valSpatialCorrAccMask, testSpatialCorrAccMask])


print(tabulate(tableData, headers=["METRIC", "TRAINING", "VALIDATION", "TESTING"], floatfmt=".3f", tablefmt="grid"))
                            


#
# save the data
#
combinedFieldnames = bl2_fieldnames
combinedFieldnames = ["FOLD", "BASELINE1_BEST_EPOCH", "BASELINE1_BEST_SCORE", "BASELINE2_BEST_EPOCH", "BASELINE2_BEST_SCORE"] + combinedFieldnames
combinedFieldnames.append("TARG_SHOPKEEPER_ACTS")
combinedFieldnames.append("PRED_SHOPKEEPER_ACTS")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_SPEECH_CLUSTER")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO_NAME")
#combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_ACTION_CLUSTER")
combinedFieldnames.append("COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO")

['PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE', 'PRED_SHOPKEEPER_ACTION_CLUSTER', 'PRED_SHOPKEEPER_SPATIAL_INFO', 'PRED_SHOPKEEPER_SPATIAL_INFO_NAME', 'PRED_SHOPKEEPER_SPEECH_CLUSTER']
tools.save_interaction_data(allPredictionsCombined, sessionDir+"all_predictions_combined.csv", combinedFieldnames)


print("Done.")