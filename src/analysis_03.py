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


englishToJapanese = {"Canon": "キャノン",
                     "CanonLeft": "キャノンの左",
                     "CanonRight": "キャノンの右",
                     "Sony": "ソニー",
                     "SonyLeft": "ソニーの左",
                     "SonyRight": "ソニーの右",
                     "Nikon": "ニコン",
                     "NikonLeft": "ニコンの左",
                     "NikonRight": "ニコンの右",
                     "Printer": "プリンタ",
                     "PrinterLeft": "プリンタの左",
                     "PrinterRight": "プリンタの右",
                     "Counter": "カウンタ",
                     "Door": "ドア",
                     "Entrance": "玄関",
                     "Middle": "真ん中",
                     "Shelf": "棚"
                     }

uniqueIDToIdentifier = {1: "shopkeeper1",
                        2: "customer2",
                        3: "shopkeeper2"
                        }

robotUniqueID = 3


predictionsFilename = "E:/eclipse-log/20240315-134656_analysis_02/all_predictions_combined.csv" #


sessionDir = sessionDir = tools.create_session_dir("analysis_03")


predictionData, _ = tools.load_interaction_data(predictionsFilename)


#
# put the data in the proper format for the coding tool and save it
#
fieldnames = ["Timestamp", "Turn ID", "Is Used For Verification", "Trial ID", "Unique ID", 
              "Utterance", "Robot Utterance Proposed", "Robot Utterance Baseline", 
              "Unique ID 2 X", "Unique ID 2 Y",
              "Unique ID 3 X", "Unique ID 3 Y",
              #"Unique ID 4 X", "Unique ID 4 Y",
              #"Unique ID 5 X", "Unique ID 5 Y",
              #"Unique ID 6 X", "Unique ID 6 Y",
              "Unique ID 1 X", "Unique ID 1 Y",
              #"Rater1 Attention 1", "Rater1 Attention 2", "Rater1 Attention 3", "Rater1 Attention 4", "Rater1 Attention 5", 
              "Rater1 Should Shopkeeper Respond", "Rater1 Proposed", "Rater1 Baseline", 
              #"Rater2 Attention 1", "Rater2 Attention 2", "Rater2 Attention 3", "Rater2 Attention 4", "Rater2 Attention 5", 
              "Rater2 Should Shopkeeper Respond", "Rater2 Proposed", "Rater2 Baseline", 
              #"Rater3 Attention 1", "Rater3 Attention 2", "Rater3 Attention 3", "Rater3 Attention 4", "Rater3 Attention 5", 
              "Rater3 Should Shopkeeper Respond", "Rater3 Proposed", "Rater3 Baseline"]
              
rowsForCoding = []

currTrial = -1
prevTrial = -1
beforeRobotFirstAction = False

for i in range(len(predictionData)):

    row = predictionData[i]

    if row["SET"] != "TEST":
        continue

    
    uniqueID = int(row["4_unique_id"])
    identifier = uniqueIDToIdentifier[uniqueID]
    trialID = int(row["TRIAL"])

    #if trialID == 880:
    #    print("hello")

    if trialID != currTrial:
        prevTrial = currTrial
        currTrial = trialID
        beforeRobotFirstAction = True
    
    if uniqueID == robotUniqueID:
        beforeRobotFirstAction = False
    
    newRow = {}
    newRow["Timestamp"] = row["4_time"]
    newRow["Turn ID"] = row["ID"]
    newRow["Is Used For Verification"] = 0
    newRow["Trial ID"] = trialID
    newRow["Unique ID"] = uniqueID


    

    #
    # the last action before the robot action prediction
    #
    currLocation = row["4_{}_currentLocation_name".format(identifier)]
    motOrigin = row["4_{}_motionOrigin_name".format(identifier)]
    motTarget = row["4_{}_motionTarget_name".format(identifier)]

    speech = row["4_participant_speech"]

    if currLocation == "None" and motTarget == "None" and motOrigin != "None":
        # look ahead to find out where the participant is moving to
        for j in range(i, len(predictionData)):
            if trialID != int(predictionData[j]["TRIAL"]):
                break
            
            targ = row["4_{}_currentLocation_name".format(identifier)]
            if targ != None:
                motTarget = targ
                break
    
    if currLocation != "None":
        motion = englishToJapanese[currLocation]
    elif motOrigin != "None" and motTarget != "None":
        motion = englishToJapanese[motOrigin] + "→" + englishToJapanese[motTarget]
    elif motOrigin != "None" and motTarget == "None":
        motion = englishToJapanese[motOrigin] + "→どこか"
    else:
        motion = ""
        print("WARNING: Invalid motion!", newRow["Trial ID"], newRow["Turn ID"])
    
    motion = "【" + motion + "】"

    newRow["Utterance"] = motion + "　「" + speech + "」"
    
    
    #
    # the proposed robot action prediction
    #

    # the robot's spatial info before its predicted action (to be used for <no action> and getting the motion origin)
    robCurrLocation = row["4_shopkeeper2_currentLocation_name"]
    robMotOrigin = row["4_shopkeeper2_motionOrigin_name"]
    robMotTarget = row["4_shopkeeper2_motionTarget_name"]
    
    if beforeRobotFirstAction and robCurrLocation == "None" and robMotOrigin == "None":
        robCurrLocation = "Entrance"



    if int(row["PRED_SHOPKEEPER_ACTS"]) == 0:
        if robCurrLocation != "None":
            location = robCurrLocation
        elif robMotTarget != "None": # just assume the shopkeeper made it to where they were heading previously
            location = robMotTarget
        elif robMotOrigin != "None": # for some reason we have a movement without the target set... Just use the origin as the location
            location = robMotOrigin
            print("WARNING: Motion origin without motion target...")
        else:
            location = ""
            print("WARNING: Invalid motion for proposed robot <no action> A!", newRow["Trial ID"], newRow["Turn ID"])
        
        proposedRobotAction = "【" + englishToJapanese[location] + "】　「」"

    else:
        newRobMotTarget = row["COMBINED_PRED_SHOPKEEPER_SPATIAL_INFO_NAME"]

        motion = englishToJapanese[newRobMotTarget]

        # if robCurrLocation != "None" and robCurrLocation == newRobMotTarget:
        #     motion = englishToJapanese[newRobMotTarget]
        # elif robCurrLocation != "None" and robCurrLocation != newRobMotTarget:
        #     motion = englishToJapanese[robCurrLocation] + "→" + englishToJapanese[newRobMotTarget]
        # elif robCurrLocation == "None" and robMotTarget != "None":
        #     motion = englishToJapanese[robMotTarget] + "→" + englishToJapanese[newRobMotTarget] # just assume the shopkeeper made it to where they were heading previously
        # else:
        #     motion = ""
        #     print("WARNING: Invalid motion for proposed robot <action> B!")
        
        proposedRobotAction = "【" + motion + "】　「" + row["COMBINED_PRED_SHOPKEEPER_REPRESENTATIVE_UTTERANCE"] +"」"


    newRow["Robot Utterance Proposed"] = proposedRobotAction
    
    newRow["Robot Utterance Baseline"] = ""
    
        
    newRow["Unique ID 1 X"] = row["4_shopkeeper1_x"]
    newRow["Unique ID 1 Y"] = row["4_shopkeeper1_y"]
    newRow["Unique ID 2 X"] = row["4_customer2_x"]
    newRow["Unique ID 2 Y"] = row["4_customer2_y"]
    newRow["Unique ID 3 X"] = row["4_shopkeeper2_x"]
    newRow["Unique ID 3 Y"] = row["4_shopkeeper2_y"]
    
    """
    newRow["Unique ID 4 X"] = ""
    newRow["Unique ID 4 Y"] = ""
    newRow["Unique ID 5 X"] = ""
    newRow["Unique ID 5 Y"] = ""
    newRow["Unique ID 6 X"] = ""
    newRow["Unique ID 6 Y"] = ""
    
    newRow["Rater1 Attention 1"] = row[""]
    newRow["Rater1 Attention 2"] = row[""]
    newRow["Rater1 Attention 3"] = row[""]
    newRow["Rater1 Attention 4"] = row[""]
    newRow["Rater1 Attention 5"] = row[""]
    newRow["Rater1 Should Shopkeeper Respond"] = row[""]
    newRow["Rater1 Proposed"] = row[""]
    newRow["Rater1 Closest_To_Phoebes"] = row[""]
    newRow["Rater2 Attention 1"] = row[""]
    newRow["Rater2 Attention 2"] = row[""]
    newRow["Rater2 Attention 3"] = row[""]
    newRow["Rater2 Attention 4"] = row[""]
    newRow["Rater2 Attention 5"] = row[""]
    newRow["Rater2 Should Shopkeeper Respond"] = row[""]
    newRow["Rater2 Proposed"] = row[""]
    newRow["Rater2 Closest_To_Phoebes"] = row[""]
    newRow["Rater3 Attention 1"] = row[""]
    newRow["Rater3 Attention 2"] = row[""]
    newRow["Rater3 Attention 3"] = row[""]
    newRow["Rater3 Attention 4"] = row[""]
    newRow["Rater3 Attention 5"] = row[""]
    newRow["Rater3 Should Shopkeeper Respond"] = row[""]
    newRow["Rater3 Proposed"] = row[""]
    newRow["Rater3 Closest_To_Phoebes"] = row[""]
    """
    rowsForCoding.append(newRow)


tools.save_interaction_data(rowsForCoding, sessionDir + "predictions.csv", fieldnames)

print("Done.")