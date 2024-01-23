#
# Created on Mon Nov 27 2023
#
# Copyright (c) 2023 Malcolm Doering
#

import ast
import numpy as np
import copy
import os
from tqdm import tqdm

import tools


sessionDir = tools.create_session_dir("processmementaroutput")


groundTruthParticipant = "shopkeeper2"
outputBothShokeeperActions = True
inputLen = 1

mementarDir = tools.dataDir + "MementarOutput_cas_800_16_50_offline_starttime_p_0606_5s/"
#mementarDir = tools.dataDir + "MementarOutputs_cas_800_16_50_offline_starttime_ge_p_0606_5s/"


speechClustersFilename = "20230731-113400_speechClustering/all_shopkeeper- speech_clusters - levenshtein normalized medoid.csv"


participants = ["shopkeeper1", "shopkeeper2", "customer"]
roomAreas = ["None", "sony_area", "canon_area", "nikon_area", "service_counter_area", "printer_desk_area", "shelf_area", "entrance_area"]
participantAreas = [p+"_area" for p in participants]


uniqueIDToIdentifier = {1: "shopkeeper1",
                        2: "customer2",
                        3: "shopkeeper2"
                        }

identifierToDesignator = {"shopkeeper1":"SHOPKEEPER_1",
                          "customer":"CUSTOMER",
                          "shopkeeper2":"SHOPKEEPER_2"
                        }


def get_state_fieldname(p, fromP):
    return "{}_from_{}".format(p, fromP)

def get_action_fieldname(p):
    return "{}_action".format(p)


#
# load the experiment conditions
#
expConditions, expConditionFieldnames = tools.load_csv_data(tools.dataDir+"20240115_experimentConditions.csv", isHeader=True)

expIDToCondition = {}
for row in expConditions:
    expIDToCondition[int(row["TRIAL"])] = row



#
# load the openai utteracnce vectors
#
uttToVec = {}

customerUttVecDir = tools.dataDir+"20240111-141832_utteranceVectorizer/"
shopkeeperUttVecDir = tools.dataDir+"20231220-111317_utteranceVectorizer/"
allParticipantsUttVecDir = tools.dataDir+"20240116-190033_utteranceVectorizer/"

# customerUttVecs = np.load(customerUttVecDir+"customer_unique_utterances_openai_embeddings.npy")
# with open(customerUttVecDir+"customer_unique_utterances.txt", "r") as f:
#     customerUtts = f.read().splitlines()
# for i in range(len(customerUtts)):
#     uttToVec[customerUtts[i]] = customerUttVecs[i]

# shopkeeperUttVecs = np.load(shopkeeperUttVecDir+"all_shopkeeper_unique_utterances_openai_embeddings.npy")
# with open(shopkeeperUttVecDir+"all_shopkeeper_unique_utterances.txt", "r") as f:
#     shopkeeperUtts = f.read().splitlines()
# for i in range(len(shopkeeperUtts)):
#     uttToVec[shopkeeperUtts[i]] = shopkeeperUttVecs[i]

allParticipantsUttVecs = np.load(allParticipantsUttVecDir+"all_participants_unique_utterances_openai_embeddings.npy")
with open(allParticipantsUttVecDir+"all_participants_unique_utterances.txt", "r") as f:
    allParticipantsUtts = f.read().splitlines()
for i in range(len(allParticipantsUtts)):
    uttToVec[allParticipantsUtts[i]] = allParticipantsUttVecs[i]


uttVecDim = allParticipantsUttVecs.shape[1]


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
# load the mementar data and put it into sets
#

# todo now: action segmentation, input vectorization for x timesteps, output classes, output participant, output speech cluster, output spatial, output gaze, output type
# idea: mask on output type + participant (we only want to imitate motions and speech of target shopkeeper)
# todo later: figure out baseline (GT) vs. extra (ToM) inputs, fix the start location information, change robot from S1 to S2


# robot is set to shopkeeper1
# then, does from_shopkeeper1 mean from the view of the participant playing shopkeeper1, or from the view of the sensor network?
# Q - what is the ground truth state? Is it participantX_from_particiantX for each particiantX? A: it should be X_from_shopkeeper1

# output actions: these should be whenever there is change of state to the ground truth (i.e. not just what one of the participants believes)
# inputs: should these be only when GT changes, or when ToM changes? 

# If the target shopkeeper believes that something changes in the mental state of one of the other participants, this may trigger the target shopkeeper to take an action.


# shopkeeper1_from_shopkeeper1 - 
# shopkeeper2_from_shopkeeper1 - 
# customer_from_shopkeeper1 - 

# shopkeeper1_from_shopkeeper2 - 
# shopkeeper2_from_shopkeeper2 - 
# customer_from_shopkeeper2 - 

# shopkeeper1_from_customer - 
# shopkeeper2_from_customer - 
# customer_from_customer - 


mementarOutputFiles = [f for f in os.listdir(mementarDir) if "merged_state" in f]

mementarData = []

for file in mementarOutputFiles:
    trialID = int(file.split("_")[0])

    #if trialID != 177:
    #if trialID != 438:
    #    continue

    mementarDataTemp, mementarDataFieldnames = tools.load_csv_data(mementarDir+file, isHeader=True, isJapanese=True)

    for line in mementarDataTemp:
        line["trial"] = trialID
        mementarData.append(line)
    
    # for debugging
    #break

mementarData.sort(key=lambda x: float(x["time"]))


# convert from strings to sets of fact tuples
print("converting from strings to sets of fact tuples...")
for i in tqdm(range(len(mementarData))):
    for key in mementarData[i]:
        if type(mementarData[i][key]) == str:

            if mementarData[i][key] != "":
                mementarData[i][key] = ast.literal_eval(mementarData[i][key])    
            
            if "_from_" in key:
                mementarData[i][key] = set(mementarData[i][key])


# find states where they are in more that one room area simultaneously and fix it
print("fixing multiple locations problem...")

currLocation = {}
currTrialID = None
prevTrialID = None


for i in tqdm(range(len(mementarData))):

    currTrialID = mementarData[i]["trial"]

    if currTrialID != prevTrialID:
        prevTrialID = currTrialID

        # reset initial locations
        for p in participants:
            currLocation[p] = {}
            for fromP in participants:
                currLocation[p][fromP] = None

    for p in participants:
        for fromP in participants:
            stateKey = get_state_fieldname(p, fromP)
            
            roomLocs = []
            for fact in  mementarData[i][stateKey]:
                if fact[0] == "isInArea" and fact[1] in roomAreas:
                    roomLocs.append(fact)
            
            if len(roomLocs) == 1:
                # this should be the normal case
                currLocation[p][fromP] = roomLocs[0]
            elif len(roomLocs) == 2:
                # assume the newer location is the most up to date and remove the old one
                newLoc = None
                oldLoc = currLocation[p][fromP]
                otherLoc = None

                if oldLoc in roomLocs:
                    # if the older location is in the state, choose the other location as the new location
                    for loc in roomLocs:
                        if loc != oldLoc:
                            newLoc = loc
                else:
                    # otherwise, arbitrarily choose one of the locations (this rarely happens)
                    newLoc = roomLocs[0]
                    otherLoc = roomLocs[1]
                    
                currLocation[p][fromP] = newLoc

                # go through and remove the old location from any subsequent states with more that two locations 
                for j in range(i, len(mementarData)):
                    if mementarData[j]["trial"] != currTrialID:
                        break

                    roomLocs2 = [x for x in mementarData[j][stateKey] if "isInArea" in x]

                    if len(roomLocs2) > 1 and (oldLoc in roomLocs2 or otherLoc in roomLocs2):
                        mementarData[j][stateKey].discard(oldLoc)
                        mementarData[j][stateKey].discard(otherLoc)
                    else:
                        break



            elif len(roomLocs) > 2:
                print("WARNING: {} simultaneous locations ({})!".format(len(roomLocs), stateKey))


# iterate through the data and identify motion between locations
def detect_movement(interactionData, participant, groundTruthParticipant, index):
    trialID = interactionData[index]["trial"]
    
    if interactionData[index-1]["trial"] != trialID:
        return
    
    stateKey = get_state_fieldname(participant, groundTruthParticipant)

    statePrev = interactionData[index-1][stateKey]
    stateCurr = interactionData[index][stateKey]

    stateAdd = stateCurr - statePrev
    stateRem = statePrev - stateCurr
    

    # if len(stateAdd) > 0 or len(stateRem) > 0:
    #     print(interactionData[index]["time"])

    #     if len(stateAdd) > 0:
    #         print("{} + {}".format(p, stateAdd))

    #     if len(stateRem) > 0:
    #         print("{} - {}".format(p, stateRem))

    newLocation = False

    for stateChange in stateAdd:
        if stateChange[0] == "isInArea" and stateChange[1] in roomAreas:
            # has entered a new area
            newLocation = True
            
            #print(stateChange)

            # go back and fill in the motion target
            for i in reversed(range(index)):
                if interactionData[i]["trial"] != trialID:
                    break

                stateToUpdate = None
                stateToRemove = None

                for state in interactionData[i][stateKey]:
                    if state[0] == "moveFromTo" and state[2] == "None":
                        if state[1] != stateChange[1]: # check if motion target and origin are the same. if they are, it's not a move action
                            stateToUpdate = state
                            break
                        else:
                            # if the target and origin are the same, replace with isInArea
                            stateToRemove = state
                            break
                
                if stateToUpdate != None:
                    for j in range(i, index):
                        interactionData[j][stateKey].discard(stateToUpdate)
                        interactionData[j][stateKey].add(("moveFromTo", stateToUpdate[1], stateChange[1]))
                
                if stateToRemove != None:
                    for j in range(i, index):
                        interactionData[j][stateKey].discard(stateToRemove)
                        interactionData[j][stateKey].add(("isInArea", stateChange[1]))
    
    for stateChange in stateRem:
        if stateChange[0] == "isInArea" and stateChange[1] in roomAreas and not newLocation:
            # has left an area
            # move action, target must be set later, when the participant moves into a new area
            interactionData[index][stateKey].add(("moveFromTo", stateChange[1], "None"))


print("detecting movement...")
for p in participants:
    for fromP in participants:
        print(get_state_fieldname(p, fromP))
        for i in tqdm(range(1, len(mementarData))):
            detect_movement(mementarData, p, fromP, i)


# filter out some data to get the "ground truth" data to use as a baseline
mementarDataGT = copy.deepcopy(mementarData)

# remove perceiving states
print("removing perceiving states...")
for i in  tqdm(range(len(mementarDataGT))):
    for p in participants:
        for fromP in participants:
            stateKey = get_state_fieldname(p, fromP)

            factsToRemove = set()

            for fact in mementarDataGT[i][stateKey]:
                if fact[0] == "isPerceiving":
                    factsToRemove.add(fact)
            
            mementarDataGT[i][stateKey] -= factsToRemove

# remove participant areas
print("removing participant areas...")
for i in  tqdm(range(len(mementarDataGT))):
    for p in participants:
        for fromP in participants:
            stateKey = get_state_fieldname(p, fromP)

            factsToRemove = set()

            for fact in mementarDataGT[i][stateKey]:
                if fact[0] == "isInArea" and fact[1] in participantAreas:
                    factsToRemove.add(fact)
            
            mementarDataGT[i][stateKey] -= factsToRemove

# remove speech after it appears the first time
print("removing speech after it appears the first time...")
for i in  tqdm(reversed(range(1, len(mementarDataGT)))):
    for p in participants:
        for fromP in participants: 
            stateKey = get_state_fieldname(p, fromP)
            factsToRemove = []

            for fact in mementarDataGT[i-1][stateKey]:
                if fact[0] == "saidSentence":
                    mementarDataGT[i][stateKey].discard(fact)

# remove non GT states
print("removing non GT states...")
for i in  tqdm(range(len(mementarDataGT))):
    for p in participants:
        for fromP in participants:
            if fromP == groundTruthParticipant:
                continue
            mementarDataGT[i].pop(get_state_fieldname(p, fromP))

# remove states that don't change from the previous state
print("removing states that don't change from the previous state...")
for i in  tqdm(reversed(range(1, len(mementarDataGT)))):
    if mementarDataGT[i]["trial"] != mementarDataGT[i-1]["trial"]:
        continue

    same = True

    for p in participants:
        stateKey = get_state_fieldname(p, groundTruthParticipant)
    
        if mementarDataGT[i][stateKey] != mementarDataGT[i-1][stateKey]:
            same = False

    if same:
        mementarDataGT.pop(i)

# remove arrival states - where the only difference from the previous state is that moveFromTo(X,Y) changes to isInArea(Y)
# note, this must be run after all states that don't change from the previous state are removed
print("removing arrival states...")
for i in tqdm(reversed(range(1, len(mementarDataGT)))):
    if mementarDataGT[i]["trial"] != mementarDataGT[i-1]["trial"]:
        continue

    arrivalState = True

    for p in participants:
        stateKey = get_state_fieldname(p, groundTruthParticipant)
        
        statePrev = mementarDataGT[i-1][stateKey]
        stateCurr = mementarDataGT[i][stateKey]

        stateAdd = stateCurr - statePrev
        stateRem = statePrev - stateCurr

        for fact in stateAdd:
            if fact[0] != "isInArea":
                # some other state has changed, so this isn't just an arrival state
                arrivalState = False
                break
        
        for fact in stateRem:
            if fact[0] != "moveFromTo":
                # some other state has changed, so this isn't just an arrival state
                arrivalState = False
                break

        for factA in stateAdd:
            for factB in stateRem:
                if factA[0] == "isInArea" and factB[0] == "moveFromTo" and factA[1] != factB[2]:
                    # this should never happen (arriving to a different location than the motion target), but check just in case
                    arrivalState = False
                    break
    
    if arrivalState:
        mementarDataGT.pop(i)


# remove states where the only difference with the previous is that the saidSentence is removed
# must be run after removing duplicate states and removing arrival states
print("removing states where the only difference with the previous is that the saidSentence is removed...")
for i in tqdm(reversed(range(1, len(mementarDataGT)))):
    if mementarDataGT[i]["trial"] != mementarDataGT[i-1]["trial"]:
        continue

    speechDisappearsState = True

    for p in participants:
        stateKey = get_state_fieldname(p, groundTruthParticipant)
        
        statePrev = mementarDataGT[i-1][stateKey]
        stateCurr = mementarDataGT[i][stateKey]

        stateAdd = stateCurr - statePrev
        stateRem = statePrev - stateCurr
    
        if len(stateAdd) > 0:
            speechDisappearsState = False
            break

        for fact in stateRem:
            if fact[0] != "saidSentence":
                speechDisappearsState = False
                break
    
    if speechDisappearsState:
        mementarDataGT.pop(i)





print("len original:", len(mementarData))
print("len GT:", len(mementarDataGT))




#
# get output actions
#

# function for finding "actions" - i.e. action segmentation
# when certain state changes happen, treat it as an action
def detect_actions(interactionData, participant, groundTruthParticipant, index):
    if index == 0 or (interactionData[index]["trial"] != interactionData[index-1]["trial"]):
        return []
    
    stateKey = get_state_fieldname(participant, groundTruthParticipant)
    actionKey = get_action_fieldname(participant)

    statePrev = interactionData[index-1][stateKey]
    stateCurr = interactionData[index][stateKey]

    stateAdd = stateCurr - statePrev
    stateRem = statePrev - stateCurr
    
    # if len(stateAdd) > 0 or len(stateRem) > 0:
    #     print(interactionData[index]["time"])

    #     if len(stateAdd) > 0:
    #         print("{} + {}".format(p, stateAdd))

    #     if len(stateRem) > 0:
    #         print("{} - {}".format(p, stateRem))

    newActions = []

    for stateChange in stateRem:
        
        if stateChange[0] == "isInArea":
            # has left an area
            # move action, target must be set later, when the participant moves into a new area
            #newActions.append(("moveTo", "None"))
            pass

        if stateChange[0] == "saidSentence":
            # no meaning? Has said something new?
            pass

        if stateChange[0] == "isPerceiving":
            # is no longer looking at the person
            # should this be treated as an action, or just included as part of the state?
            pass

    moveActionAdded = False

    for stateChange in stateAdd:
        
        if stateChange[0] == "moveFromTo":
            newActions.append(("moveTo", stateChange[2]))
            moveActionAdded = True

        elif stateChange[0] == "isInArea":
            # if moveFromTo is present, then isInArea should not be present
            # but sometimes isInArea will suddenly change from the previous state, so the previous state should trigger a move action
            if stateChange[1] in roomAreas:

                # if there wasn't already a move action triggered, trigger a move action
                prevMovementTarget = None
                try:
                    prevMovement = [x for x in stateRem if "moveFromTo" in x][0]
                    prevMovementTarget = prevMovement[2]

                except:
                    pass

                if prevMovementTarget != stateChange[1] or prevMovementTarget == None:
                        newActions.append(("moveTo", stateChange[1]))
                        moveActionAdded = True

                # has entered a new area
                # go back and fill in the motion target
                #for i in reversed(range(index)):
                #    if actionKey in interactionData[i]:
                #        if ("moveTo", "None") in interactionData[i][actionKey]:
                #            interactionData[i][actionKey].remove(("moveTo", "None"))
                #            interactionData[i][actionKey].append(("moveTo", stateChange[1]))

    for stateChange in stateAdd:
        if stateChange[0] == "saidSentence":
            # has said something
            newActions.append(("speak", stateChange[1]))

            #if not ("moveTo", "None") in newActions:
            if not moveActionAdded:
                for state in stateCurr:
                    if state[0] == "isInArea" and state[1] in roomAreas:
                        newActions.append(state)
        
        if stateChange[0] == "isPerceiving":
            # is looking at someone
            # should this be treated as an action, or just included as part of the state?
            pass

    return newActions


# go through the data a find wherever there is a change of location or speech - these are the "actions" that we will try to predict
print("detecting actions...")
#for i in range(0, len(mementarData)):
#    for p in participants:
#        mementarData[i][get_action_fieldname(p)] = detect_actions(mementarData, p, groundTruthParticipant, i)

for i in tqdm(range(0, len(mementarDataGT))):
    for p in participants:
        mementarDataGT[i][get_action_fieldname(p)] = detect_actions(mementarDataGT, p, groundTruthParticipant, i)

mementarDataFieldnames += [get_action_fieldname(p) for p in participants]
mementarDataGTFieldnames = copy.deepcopy(mementarDataFieldnames)

for p in participants:
    for fromP in participants:
        if fromP != groundTruthParticipant:
            mementarDataGTFieldnames.remove(get_state_fieldname(p, fromP))


#
# vectorize the inputs
#


# if we treat each time the state changes as an input-output instance, then the no action output instances will far outweigh the others...
# todo, first just do for some state changes (motions between room locations and speech for GT state), later try adding everything else...
def vectorize_and_save(mementarData, fileDescriptor):
    
    shopkeeperActionIDToKey = {}
    shopkeeperActionKeyToID = {}

    shkpActSimultaneouslyCount = 0
    isInMultipleAreasCount = 0
    areMultipleUtterancesCount = 0

    missingUttCount = 0
    missingUtteranceFilename = sessionDir+"utterances missing from speech clusters.txt"


    def extract_output_action(participantAction, designator):
        isShopkeeper = "SHOPKEEPER" in designator

        mementarData[i]["{}_SPATIAL_INFO".format(designator)] = -1
        mementarData[i]["{}_SPATIAL_INFO_NAME".format(designator)] = "None"
        mementarData[i]["{}_SPEECH".format(designator)] = ""
        mementarData[i]["{}_DID_ACTION".format(designator)] = 0

        if isShopkeeper:
            mementarData[i]["{}_ACTION_CLUSTER".format(designator)] = -1
            mementarData[i]["{}_SPEECH_CLUSTER".format(designator)] = -1
            mementarData[i]["{}_REPRESENTATIVE_UTTERANCE".format(designator)] = ""
            mementarData[i]["{}_SPEECH_CLUSTER_IS_JUNK".format(designator)] =  0
        
        if len(participantAction) > 0:
            for fact in participantAction:
                # speech
                if fact[0] == "speak":
                    mementarData[i]["{}_DID_ACTION".format(designator)] = 1
                    utterance = fact[1]
                    mementarData[i]["{}_SPEECH".format(designator)] = utterance

                    if isShopkeeper:
                        try:
                            speechClustID = uttToSpeechClustID[utterance]
                        except KeyError as e:
                            #print("WARNING: Missing speech cluster info! Using cluster 0:", e)
                            #missingUttCount += 1
                            with open(missingUtteranceFilename, "a") as f:
                                f.write(utterance + "\n")

                            speechClustID = 0
                        
                        mementarData[i]["{}_SPEECH_CLUSTER".format(designator)] = speechClustID
                        mementarData[i]["{}_REPRESENTATIVE_UTTERANCE".format(designator)] = speechClustIDToRepUtt[speechClustID]
                        mementarData[i]["{}_SPEECH_CLUSTER_IS_JUNK".format(designator)] =  speechClustIDIsJunk[speechClustID]
                
                # location and motion
                if (fact[0] == "isInArea" and fact[1] in roomAreas) or fact[0] == "moveTo":
                    mementarData[i]["{}_DID_ACTION".format(designator)] = 1
                    spatial = fact[1]
                    mementarData[i]["{}_SPATIAL_INFO".format(designator)] = roomAreas.index(spatial)
                    mementarData[i]["{}_SPATIAL_INFO_NAME".format(designator)] = spatial
            
            if isShopkeeper:
                shopkeeperActionKey = "{}:{}".format(mementarData[i]["{}_SPATIAL_INFO_NAME".format(designator)], mementarData[i]["{}_SPEECH_CLUSTER".format(designator)])
                if shopkeeperActionKey not in shopkeeperActionKeyToID:
                    shopkeeperActionKeyToID[shopkeeperActionKey] = len(shopkeeperActionKeyToID)
                    shopkeeperActionIDToKey[shopkeeperActionKeyToID[shopkeeperActionKey]] = shopkeeperActionKey
                
                mementarData[i]["{}_ACTION_CLUSTER".format(designator)] = shopkeeperActionKeyToID[shopkeeperActionKey]

    #
    # add the following data to the mementar output file (to be consistent with previous interaction data files)
    #
    print("preparing inputs and outputs for {}...".format(fileDescriptor))

    for i in tqdm(range(len(mementarData))):
        # mementarData["TRIAL"] = 
        # mementarData["CUSTOMER_ID"] = 
        # mementarData["SHOPKEEPER1_ID"] = 
        # mementarData["SHOPKEEPER2_ID"] = 
        # mementarData["CUSTOMER_TYPE"] = 
        # mementarData["CUSTOMER_BUY"] = 
        # mementarData["SHOPKEEPER2_TYPE"] = 
        # mementarData["NOTES"] = 
        mementarData[i].update(expIDToCondition[mementarData[i]["trial"]])
        
        # Formerly, this was for the person who acted. But now... of who? From who's perspective?
        # mementarData["unique_id"] = 
        # mementarData["participant_speech"] = 
        # mementarData["participant_speech_english_autotranslate"] = 
        
        for p in participants:
            for fromP in participants:
                stateFieldname = get_state_fieldname(p, fromP)
                
                if stateFieldname not in mementarData[i]:
                    continue

                state = mementarData[i][stateFieldname]
                
                mementarData[i][stateFieldname+"_speech"] = ""
                mementarData[i][stateFieldname+"_didAction"] = 0 # TODO what to do with this?
                
                mementarData[i][stateFieldname+"_currentLocation"] = 0
                mementarData[i][stateFieldname+"_motionOrigin"] = 0
                mementarData[i][stateFieldname+"_motionTarget"] = 0
                
                mementarData[i][stateFieldname+"_currentLocation_name"] = "None"
                mementarData[i][stateFieldname+"_motionOrigin_name"] = "None"
                mementarData[i][stateFieldname+"_motionTarget_name"] = "None"

                isInAreaCount = 0

                for fact in state:
                    if fact[0] == "saidSentence":
                        mementarData[i][stateFieldname+"_speech"] = fact[1]
                    if fact[0] == "isInArea" and fact[1] in roomAreas:
                        mementarData[i][stateFieldname+"_currentLocation_name"] = fact[1]
                        isInAreaCount += 1
                    if fact[0] == "moveFromTo":
                        mementarData[i][stateFieldname+"_motionOrigin_name"] = fact[1]
                        mementarData[i][stateFieldname+"_motionTarget_name"] = fact[2]
                
                if isInAreaCount > 1:
                    isInMultipleAreasCount += 1

                mementarData[i][stateFieldname+"_currentLocation"] = roomAreas.index(mementarData[i][stateFieldname+"_currentLocation_name"])
                mementarData[i][stateFieldname+"_motionOrigin"] = roomAreas.index(mementarData[i][stateFieldname+"_motionOrigin_name"])
                mementarData[i][stateFieldname+"_motionTarget"] = roomAreas.index(mementarData[i][stateFieldname+"_motionTarget_name"])
        
        
        shopkeeper1Action = mementarData[i][get_action_fieldname("shopkeeper1")]
        shopkeeper2Action = mementarData[i][get_action_fieldname("shopkeeper2")]
        customerAction = mementarData[i][get_action_fieldname("customer")]

        # what to do if both shopkeepers acted at the same time?
        # create an output action for each participant...

        if len(shopkeeper1Action) > 0 and len(shopkeeper2Action) > 0:
            #print("WARNING: Both shopkeepers acted at the same time!")
            shkpActSimultaneouslyCount += 1

        extract_output_action(shopkeeper1Action, "SHOPKEEPER_1")
        extract_output_action(shopkeeper2Action, "SHOPKEEPER_2")
        extract_output_action(customerAction, "CUSTOMER")
        
        if (bool(mementarData[i]["SHOPKEEPER_1_SPEECH"]) + bool(mementarData[i]["CUSTOMER_SPEECH"]) + bool(mementarData[i]["SHOPKEEPER_2_SPEECH"])) > 1:
            areMultipleUtterancesCount += 1


        #
        # things the old file had, but we won't use
        #
        # mementarData["time"] = # already in mementarOut
        # mementarData["keywords"] = 
        # mementarData["speech_time"] = 
        # mementarData["speech_duration"] = 
        # mementarData["motion_start_time"] = 
        # mementarData["motion_end_time"] = 
        # mementarData["true_motionTarget"] = 
        # mementarData["customer2_x"] = 
        # mementarData["customer2_y"] = 
        # mementarData["shopkeeper1_x"] = 
        # mementarData["shopkeeper1_y"] = 
        # mementarData["shopkeeper2_x"] = 
        # mementarData["shopkeeper2_y"] = 
        # mementarData["shopkeeper1_spatialFormation"] = 
        # mementarData["shopkeeper1_stateTarget"] = 
        # mementarData["shopkeeper2_spatialFormation"] = 
        # mementarData["shopkeeper2_stateTarget"] =  
    
    # save
    fieldnamesNotToDuplicate = ["TRIAL", "CUSTOMER_ID", "SHOPKEEPER1_ID", "SHOPKEEPER2_ID", "CUSTOMER_TYPE", "CUSTOMER_BUY", "SHOPKEEPER2_TYPE", "NOTES"]
    fieldnames = fieldnamesNotToDuplicate + ["SOMEONE_ACTS"] + [fn for fn in mementarData[0].keys() if fn not in fieldnamesNotToDuplicate]

    for i in range(len(mementarData)):
        mementarData[i]["SOMEONE_ACTS"] = 0
        for p in participants:
            if len(mementarData[i][get_action_fieldname(p)]) > 0:
                mementarData[i]["SOMEONE_ACTS"] = 1

    tools.save_interaction_data(mementarData, sessionDir+"{}.csv".format(fileDescriptor), fieldnames)


    #
    # get the inputs with the desired history length
    # and get the outputs
    #
    inputs = []
    outputs = []

    outputActionIDs_shkp1 = []
    outputSpeechClusterIDs_shkp1 = []
    outputSpatialInfo_shkp1 = []
    outputSpeechClusterIsJunk_shkp1 = []

    outputActionIDs_shkp2 = []
    outputSpeechClusterIDs_shkp2 = []
    outputSpatialInfo_shkp2 = []
    outputSpeechClusterIsJunk_shkp2 = []
    
    outputDidActionBits = [] # shkp1, cust, shkp2

    thisExpStartIndex = 0
    thisExp = mementarData[0]["TRIAL"]

    # for keeping tracking of some action count and turn taking metrics
    numActionsPerParticipant = {-1:0}
    numFirstActionsPerParticipant = {-1:0}
    for p in participants:
        numActionsPerParticipant[p] = 0
        numFirstActionsPerParticipant[p] = 0
        
    actionHidSequenceCounts = {"total":0}
    countOverThreshold = {"total":0}
    timeDeltas = {"total":[]}
    threshold = 6.0

    #
    print("prepare inputs with interaction history...")
    for i in tqdm(range(len(mementarData)-1)):
        currAction = mementarData[i]
        nextAction = mementarData[i+1]

        if thisExp != nextAction["TRIAL"]:
            thisExpStartIndex = i + 1
            thisExp = nextAction["TRIAL"]
            continue

        # TODO: somehow treat the "first appearance" actions specially
        inputTemp = []

        for j in range(i-inputLen+1, i+1):
            if j < thisExpStartIndex:
                inputTemp.append(None)
            else:
                inputTemp.append(mementarData[j])

        inputs.append(inputTemp)
        outputs.append(nextAction)

        if nextAction["SHOPKEEPER_1_DID_ACTION"] == 1:
            outputActionIDs_shkp1.append(nextAction["SHOPKEEPER_1_ACTION_CLUSTER"])
            outputSpeechClusterIDs_shkp1.append(nextAction["SHOPKEEPER_1_SPEECH_CLUSTER"])
            outputSpatialInfo_shkp1.append(nextAction["SHOPKEEPER_1_SPATIAL_INFO"])
            outputSpeechClusterIsJunk_shkp1.append(nextAction["SHOPKEEPER_1_SPEECH_CLUSTER_IS_JUNK"])
        else:
            outputActionIDs_shkp1.append(-1)
            outputSpeechClusterIDs_shkp1.append(-1)
            outputSpatialInfo_shkp1.append(-1)
            outputSpeechClusterIsJunk_shkp1.append(-1)
                
        if nextAction["SHOPKEEPER_2_DID_ACTION"] == 1:
            outputActionIDs_shkp2.append(nextAction["SHOPKEEPER_2_ACTION_CLUSTER"])
            outputSpeechClusterIDs_shkp2.append(nextAction["SHOPKEEPER_2_SPEECH_CLUSTER"])
            outputSpatialInfo_shkp2.append(nextAction["SHOPKEEPER_2_SPATIAL_INFO"])
            outputSpeechClusterIsJunk_shkp2.append(nextAction["SHOPKEEPER_2_SPEECH_CLUSTER_IS_JUNK"])
        else:
            outputActionIDs_shkp2.append(-1)
            outputSpeechClusterIDs_shkp2.append(-1)
            outputSpatialInfo_shkp2.append(-1)
            outputSpeechClusterIsJunk_shkp2.append(-1)
        
        outputDidActionBits.append([nextAction["SHOPKEEPER_1_DID_ACTION"],
                                         nextAction["CUSTOMER_DID_ACTION"],
                                         nextAction["SHOPKEEPER_2_DID_ACTION"]])
        
        #
        # compute some statistics
        #
        t = float(currAction["time"])
        nextT = float(nextAction["time"])
        timeDelta = nextT - t

        if (timeDelta) > threshold:
            countOverThreshold["total"] += 1
        timeDeltas["total"].append(timeDelta)
        
        for p in participants:
            if currAction["{}_DID_ACTION".format(identifierToDesignator[p])] == 0:
                continue

            for nextP in participants:
                if nextAction["{}_DID_ACTION".format(identifierToDesignator[nextP])] == 0:
                    continue

                if i == thisExpStartIndex:
                    numFirstActionsPerParticipant[p] += 1

                numActionsPerParticipant[nextP] += 1

                hidSeq = "{}->{}".format(p, nextP)
                if hidSeq not in actionHidSequenceCounts:
                    actionHidSequenceCounts[hidSeq] = 0
                actionHidSequenceCounts[hidSeq] += 1

                actionHidSequenceCounts["total"] += 1

    #
    # print the statistics
    #
    for key, value in numActionsPerParticipant.items():
        print("Unique ID {}: {} actions".format(key, value))

    for key, value in numFirstActionsPerParticipant.items():
        print("Unique ID {}: {} first actions".format(key, value))


    for p in participants:
        for nextP in participants:
            hidSeq = "{}->{}".format(p, nextP)
            try:
                print("Action HID sequence {}: {} instances".format(hidSeq, actionHidSequenceCounts[hidSeq]))
            except:
                pass

    for i in [True, False]:
        for j in [True, False]:
            hidSeq = "{}->{}".format(i, j)
            
            try:
                print("Action HID sequence {}: {} instances, {} timeouts, {} delta mean, {} std.".format(hidSeq, 
                                                                                                    actionHidSequenceCounts[hidSeq], 
                                                                                                    countOverThreshold[hidSeq],
                                                                                                    np.mean(timeDeltas[hidSeq]), 
                                                                                                    np.std(timeDeltas[hidSeq])
                                                                                                    ))
            except:
                pass
            
    hidSeq = "total"
    print("Action HID sequence {}: {} instances, {} timeouts, {} delta mean, {} std.".format(hidSeq, 
                                                                                                    actionHidSequenceCounts[hidSeq], 
                                                                                                    countOverThreshold[hidSeq],
                                                                                                    np.mean(timeDeltas[hidSeq]), 
                                                                                                    np.std(timeDeltas[hidSeq])
                                                                                                    ))
    
    print("Num shopkeepers act simul. instances:", shkpActSimultaneouslyCount)
    print("Num states with multiple isInArea:", isInMultipleAreasCount)
    print("Num instances with multiple utterances:", areMultipleUtterancesCount)
    

    with open(missingUtteranceFilename, "r") as f:
        missingUttCount = len(f.readlines())
    print("Num utts missing from speech clusters:", missingUttCount)

    
    #
    # vectorize the inputs
    #

    # who acted bits + at from to x num participants
    inputVecNonSpeechLen = 3 + (len(roomAreas)) * 3 * 3 # how to do the who acted bits? Whose state changed from previous state?
    inputVecSpeechLen = uttVecDim * 3 # need one for each participant? Is there any case with two utterances at the same time?

    # save speech and non speech separately because speech requires floats, non speech can use ints to save space
    inputVectors = []

    print("vectorizing inputs...")
    for i in range(len(inputs)):
        input = inputs[i]
        inputVec = []

        for inputStep in input:
            if inputStep == None:
                inputVec.append(np.zeros(inputVecNonSpeechLen+inputVecSpeechLen))
            
            else:
                inputVecTemp = []

                # did action bits
                # TODO it would be better to mark whose state changes (to include things like arival to a location), not just who did an action
                # TODO this might also need to be done separately for each fromX
                
                didActionVec = np.asarray(outputDidActionBits[i])
                
                inputVecTemp = np.concatenate([inputVecTemp, didActionVec])

                for p in participants:
                    for fromP in participants:
                        stateKey = get_state_fieldname(p, fromP)

                        if stateKey in inputStep:
                            # spatial info
                            currLoc = np.zeros(len(roomAreas)) 
                            motOri = np.zeros(len(roomAreas))
                            motTar = np.zeros(len(roomAreas))

                            currLoc[int(inputStep["{}_currentLocation".format(stateKey)])] = 1
                            motOri[int(inputStep["{}_motionTarget".format(stateKey)])] = 1
                            motTar[int(inputStep["{}_motionOrigin".format(stateKey)])] = 1

                            inputVecTemp = np.concatenate([inputVecTemp, currLoc, motOri, motTar])
                

                shkp1Speech = inputStep["SHOPKEEPER_1_SPEECH"]
                if shkp1Speech != "":
                    shkp1UttVec = uttToVec[shkp1Speech]
                else:
                    shkp1UttVec = np.zeros(uttVecDim)
                
                custSpeech = inputStep["CUSTOMER_SPEECH"]
                if custSpeech != "":
                    custUttVec = uttToVec[custSpeech]
                else:
                    custUttVec = np.zeros(uttVecDim)
                
                shkp2Speech = inputStep["SHOPKEEPER_2_SPEECH"]
                if shkp2Speech != "":
                    shkp2UttVec = uttToVec[shkp2Speech]
                else:
                    shkp2UttVec = np.zeros(uttVecDim)

                inputVecTemp = np.concatenate([inputVecTemp, shkp1UttVec, custUttVec, shkp2UttVec])

                inputVec.append(inputVecTemp)
        
        inputVec = np.stack(inputVec, axis=0)
        inputVectors.append(inputVec)


    #
    # save the vectors
    #
    outputActionIDs_shkp1 = np.asarray(outputActionIDs_shkp1, dtype=np.int16)
    outputSpeechClusterIDs_shkp1 = np.asarray(outputSpeechClusterIDs_shkp1, dtype=np.int16)
    outputSpatialInfo_shkp1 = np.asarray(outputSpatialInfo_shkp1, dtype=np.int16)
    outputSpeechClusterIsJunk_shkp1 = np.asarray(outputSpeechClusterIsJunk_shkp1, dtype=np.int16)

    outputActionIDs_shkp2 = np.asarray(outputActionIDs_shkp2, dtype=np.int16)
    outputSpeechClusterIDs_shkp2 = np.asarray(outputSpeechClusterIDs_shkp2, dtype=np.int16)
    outputSpatialInfo_shkp2 = np.asarray(outputSpatialInfo_shkp2, dtype=np.int16)
    outputSpeechClusterIsJunk_shkp2 = np.asarray(outputSpeechClusterIsJunk_shkp2, dtype=np.int16)

    outputDidActionBits = np.asarray(outputDidActionBits, dtype=np.int16)

    inputVectors = np.stack(inputVectors, axis=0)    
    inputVectors = inputVectors.astype(np.float32)
    

    np.save(sessionDir+"/outputActionIDs_shkp1", outputActionIDs_shkp1)
    np.save(sessionDir+"/outputSpeechClusterIDs_shkp1", outputSpeechClusterIDs_shkp1)
    np.save(sessionDir+"/outputSpatialInfo_shkp1", outputSpatialInfo_shkp1)
    np.save(sessionDir+"/outputSpeechClusterIsJunk_shkp1", outputSpeechClusterIsJunk_shkp1)

    np.save(sessionDir+"/outputActionIDs_shkp2", outputActionIDs_shkp2)
    np.save(sessionDir+"/outputSpeechClusterIDs_shkp2", outputSpeechClusterIDs_shkp2)
    np.save(sessionDir+"/outputSpatialInfo_shkp2", outputSpatialInfo_shkp2)
    np.save(sessionDir+"/outputSpeechClusterIsJunk_shkp2", outputSpeechClusterIsJunk_shkp2)

    np.save(sessionDir+"/outputDidActionBits", outputDidActionBits)

    np.save(sessionDir+"/inputVectors", inputVectors)
    

vectorize_and_save(mementarDataGT, "mementarDataGT")


print("Done.")







