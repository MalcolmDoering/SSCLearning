#
# Created on Tue Aug 01 2023
#
# Copyright (c) 2023 Malcolm Doering
#
#
# For predictions using the knowledge management system
# Combine the outputs of the classifier for whether or not to act (baseline 7) and how to act (baseline 5 / 6) to get output for evaluation 
#

import os
import pandas as pd
import copy

import tools

inputStep = "0"
groundTruthParticipant = "shopkeeper2"

englishToJapanese = {"canon_area": "キャノン",
                     "sony_area": "ソニー",
                     "nikon_area": "ニコン",
                     "printer_desk_area": "プリンタ",
                     "service_counter_area": "カウンタ",
                     "entrance_area": "玄関",
                     "shelf_area": "棚",
                     "None":""
                     
                     #"Door": "ドア",
                     #"Middle": "真ん中",
                     }

uniqueIDToIdentifier = {1: "shopkeeper1",
                        2: "customer2",
                        3: "shopkeeper2"
                        }

robotUniqueID = 3


#predictionsFilename = "E:/eclipse-log/20240315-134656_analysis_02/all_predictions_combined.csv"

predictionsFilename = "E:/eclipse-log/20240325-134625_analysis_02_testxy/all_predictions_combined.csv"


sessionDir = sessionDir = tools.create_session_dir("analysis_03")


predictionData, _ = tools.load_interaction_data(predictionsFilename)


#
# put the data in the proper format for the coding tool and save it
#
fieldnames = ["Timestamp", "Turn ID", "Is Used For Verification", "Trial ID", 
              # "Unique ID", "Utterance", # these fields will have to be different for the knowledge management system because 
              
              "Unique ID", # maybe just use this field for something else...
              "S1 Utterance", "S2 Utterance", "C Utterance",
              
              "Robot Utterance Proposed", "Robot Utterance Baseline", 
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

    S1_didAction = int(row["{}_SHOPKEEPER_1_DID_ACTION".format(inputStep)])
    S2_didAction = int(row["{}_SHOPKEEPER_2_DID_ACTION".format(inputStep)])
    C_didAction = int(row["{}_CUSTOMER_DID_ACTION".format(inputStep)])
    
    
    #identifier = uniqueIDToIdentifier[uniqueID]
    trialID = int(row["TRIAL"])

    #if trialID == 880:
    #    print("hello")

    if trialID != currTrial:
        prevTrial = currTrial
        currTrial = trialID
        beforeRobotFirstAction = True
    
    if S2_didAction:
        beforeRobotFirstAction = False
    
    newRow = {}
    newRow["Timestamp"] = row["{}_time".format(inputStep)]
    newRow["Turn ID"] = row["ID"]
    newRow["Is Used For Verification"] = S1_didAction or C_didAction
    newRow["Trial ID"] = trialID
    newRow["Unique ID"] = -1 # TODO


    
    
    #
    # the last action before the robot action prediction
    #
    def get_action(identifier):
        currLocation = row["{}_{}_from_{}_currentLocation_name".format(inputStep, identifier, groundTruthParticipant)]
        motOrigin = row["{}_{}_from_{}_motionOrigin_name".format(inputStep, identifier, groundTruthParticipant)]
        motTarget = row["{}_{}_from_{}_motionTarget_name".format(inputStep, identifier, groundTruthParticipant)]

        speech = row["{}_{}_from_{}_speech".format(inputStep, identifier, groundTruthParticipant)]

        if currLocation == "None" and motTarget == "None" and motOrigin != "None":
            # look ahead to find out where the participant is moving to
            for j in range(i, len(predictionData)):
                if trialID != int(predictionData[j]["TRIAL"]):
                    break
                
                # TODO this code doesn't do anything...
                targ = row["{}_{}_from_{}_currentLocation_name".format(inputStep, identifier, groundTruthParticipant)]
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
        fullAction = motion + "　「" + speech + "」"

        return fullAction
    

    if S1_didAction:
        newRow["S1 Utterance"] = get_action("shopkeeper1") 
    else:
        newRow["S1 Utterance"] = ""
    
    if S2_didAction:
        newRow["S2 Utterance"] = get_action("shopkeeper2") 
    else:
        newRow["S2 Utterance"] = ""
    
    if C_didAction:
        newRow["C Utterance"] = get_action("customer") 
    else:
        newRow["C Utterance"] = ""

    
    #
    # the proposed robot action prediction
    #

    # the robot's spatial info before its predicted action (to be used for <no action> and getting the motion origin)
    robCurrLocation = row["{}_shopkeeper2_from_{}_currentLocation_name".format(inputStep, groundTruthParticipant)]
    robMotOrigin = row["{}_shopkeeper2_from_{}_motionOrigin_name".format(inputStep, groundTruthParticipant)]
    robMotTarget = row["{}_shopkeeper2_from_{}_motionTarget_name".format(inputStep, groundTruthParticipant)]
    
    if beforeRobotFirstAction and robCurrLocation == "None" and robMotOrigin == "None":
        robCurrLocation = "entrance_area"



    if int(row["PRED_SHOPKEEPER_ACTS"]) == 0:
        if robCurrLocation != "None":
            location = robCurrLocation
        elif robMotTarget != "None": # just assume the shopkeeper made it to where they were heading previously
            location = robMotTarget
        elif robMotOrigin != "None": # for some reason we have a movement without the target set... Just use the origin as the location
            location = robMotOrigin
            print("WARNING: Motion origin without motion target...")
        else:
            location = "None"
            print("WARNING: Invalid motion for proposed robot <no action> A!", newRow["Trial ID"], newRow["Turn ID"])
        
        #proposedRobotAction = "【" + englishToJapanese[location] + "】　「」"

        proposedRobotAction = "<NO ACTION>"

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
    
    # TODO
    try:
        newRow["Unique ID 1 X"] = float(row["{}_shopkeeper1_x".format(inputStep)]) * 1000
        newRow["Unique ID 1 Y"] = float(row["{}_shopkeeper1_y".format(inputStep)]) * 1000
    except:
        newRow["Unique ID 1 X"] = 0
        newRow["Unique ID 1 Y"] = 0
    
    try:
        newRow["Unique ID 2 X"] = float(row["{}_customer_x".format(inputStep)]) * 1000
        newRow["Unique ID 2 Y"] = float(row["{}_customer_y".format(inputStep)]) * 1000
    except:
        newRow["Unique ID 2 X"] = 0
        newRow["Unique ID 2 Y"] = 0
    
    try:
        newRow["Unique ID 3 X"] = float(row["{}_shopkeeper2_x".format(inputStep)]) * 1000
        newRow["Unique ID 3 Y"] = float(row["{}_shopkeeper2_y".format(inputStep)]) * 1000
    except:
        newRow["Unique ID 3 X"] = 0
        newRow["Unique ID 3 Y"] = 0


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