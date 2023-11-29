#
# Created on Mon Nov 27 2023
#
# Copyright (c) 2023 Malcolm Doering
#

import ast

import tools


sessionDir = tools.create_session_dir("processmementaroutput")


trial = 169
mementarOutputFile = tools.dataDir + "MementarOutput/{}_merged_state.csv".format(trial)
speechClustersFilename = "20230731-113400_speechClustering/all_shopkeeper- speech_clusters - levenshtein normalized medoid.csv"


participants = ["shopkeeper1", "shopkeeper2", "customer"]
roomAreas = ["sony_area", "canon_area", "nikon_area", "service_counter_area", "printer_desk_area", "shelf_area"]
participantAreas = ["_area".format(p) for p in participants]


#
# load the speech clusters
#
noSpeechClusterID = -1
mainJunkClusterID = 0

speechClusterData, _ = tools.load_csv_data(tools.dataDir+speechClustersFilename, isHeader=True, isJapanese=True)

speechClustIDToUtts = {}
speechClustIDIsJunk = {}
uttToSpeechClustID = {}
speechClustIDToRepUtt = {}

# add no speech
speechClustIDToUtts[noSpeechClusterID] = [""]
speechClustIDIsJunk[noSpeechClusterID] = 0
uttToSpeechClustID[""] = noSpeechClusterID
speechClustIDToRepUtt[noSpeechClusterID] = ""


for row in speechClusterData:
    speech = row["Utterance"]
    speechClustID = int(row["Cluster.ID"])

    if speechClustID not in speechClustIDToUtts:
        speechClustIDToUtts[speechClustID] = []
        speechClustIDIsJunk[speechClustID] = int(row["Is.Junk"])

    
    if speech not in uttToSpeechClustID:
        uttToSpeechClustID[speech] = speechClustID 
    
    speechClustIDToUtts[speechClustID].append(speech)

    if int(row["Is.Representative"]) == 1:
        speechClustIDToRepUtt[speechClustID] = speech

    if speech in uttToSpeechClustID and uttToSpeechClustID[speech] != speechClustID:
        print("WARNING: Same utterance in multiple speech clusters: {} in {} and {}!".format(speech, uttToSpeechClustID[speech], speechClustID))



#
# function for finding "actions"
#
def detectActions(interactionData, participant, index):
    stateKey = "{}_from_{}".format(participant, participant)
    actionKey = "{}_action".format(participant)

    statePrev = interactionData[index-1][stateKey]
    stateCurr = interactionData[index][stateKey]

    stateAdd = stateCurr - statePrev
    stateRem = statePrev - stateCurr
    
    
    if len(stateAdd) > 0 or len(stateRem) > 0:
        print(interactionData[index]["time"])

        if len(stateAdd) > 0:
            print("{} + {}".format(p, stateAdd))

        if len(stateRem) > 0:
            print("{} - {}".format(p, stateRem))

    newActions = []

    for stateChange in stateAdd:
        
        if stateChange[0] == "isInArea":
            if stateChange[1] == "{}_area".format(participant):
                continue

            # has entered a new area
            # go back and fill in the motion target
            for i in reversed(range(index)):
                if actionKey in interactionData[i]:
                    if ("moveTo", None) in interactionData[i][actionKey]:
                        interactionData[i][actionKey].remove(("moveTo", None))
                        interactionData[i][actionKey].append(("moveTo", stateChange[1]))

        if stateChange[0] == "saidSentence":
            # has said something
            newActions.append(("speak", stateChange[1]))
            

        if stateChange[0] == "isPerceiving":
            # is looking at someone
            # should this be treated as an action, or just included as part of the state?
            pass

    for stateChange in stateRem:
        
        if stateChange[0] == "isInArea":
            # has left an area
            # move action, target must be set later, when the participant moves into a new area
            newActions.append(("moveTo", None))
        
        if stateChange[0] == "saidSentence":
            # no meaning? Has said something new?
            pass

        if stateChange[0] == "isPerceiving":
            # is no longer looking at the person
            # should this be treated as an action, or just included as part of the state?
            pass

    return newActions




interactionData, interactionDataFieldnames = tools.load_csv_data(mementarOutputFile, isHeader=True, isJapanese=True)

actionData = []



# convert from string
for i in range(len(interactionData)):

    for key in interactionData[i]:
        interactionData[i][key] = ast.literal_eval(interactionData[i][key])

        if key != "time":
            interactionData[i][key] = set(interactionData[i][key])


# Q - why don't they start "in area" - is there a certain amount of time that must pass before they are recognized as being in an area?
# Q - what is the ground truth state? Is it participantX_from_particiantX for each particiantX?

# go through the data a find wherever there is a change of location or speech - these are the "actions" that we will try to predict
for i in range(1, len(interactionData)):
    for p in participants:
        interactionData[i]["{}_action".format(p)] = detectActions(interactionData, p, i)
        #interactionData[i-1]["action_time"] = interactionData[i]["time"]


interactionDataFieldnames += ["{}_action".format(p) for p in participants]






# TODO get the cluster IDs
# how to deal with movements to positions of other participants?
# make sure it as a previous action that triggers the next action...
















tools.save_interaction_data(interactionData, sessionDir+"{}_interactionData.csv".format(trial), interactionDataFieldnames)

print("Done.")







