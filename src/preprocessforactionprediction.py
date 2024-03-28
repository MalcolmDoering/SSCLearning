
import csv
import numpy as np
import pickle
from collections import OrderedDict
import pymysql
import copy

import tools
import utterancevectorizer2


trainUttVectorizer = False
useOpenAIEmbeddings = True
shkpUniqueIDs = [1, 3]

interactionDataFilename = "20230807-141847_processForSpeechClustering/20230623_SSC_3_trueMotionTargets_3_speechMotionCombined.csv"
speechClustersFilename = "20230731-113400_speechClustering/all_shopkeeper- speech_clusters - levenshtein normalized medoid.csv"
keywordsFilename = tools.modelDir + "20230609-141854_unique_utterance_keywords.csv"
uttVectorizerDir = tools.dataDir + "20230807-191124_actionPredictionPreprocessing/"
stoppingLocationClusterDir = tools.modelDir + "20230627_stoppingLocationClusters/"

sessionDir = tools.create_session_dir("actionPredictionPreprocessing")


uniqueIDToIdentifier = {1: "shopkeeper1",
                        2: "customer2",
                        3: "shopkeeper2"
                        }

identifierToDesignator = {"shopkeeper1":"SHOPKEEPER_1",
                          "customer":"CUSTOMER",
                          "shopkeeper2":"SHOPKEEPER_2"}


#
# load the data
#
interactionData, fieldnames = tools.load_interaction_data(tools.dataDir+interactionDataFilename)


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
# load the stopping location clusers
#
print("Loading the stopping location clusters...")

shkpStopLocClustIDToName = {}
shkpStopLocClustData, _ = tools.load_csv_data(stoppingLocationClusterDir+"shopkeeper1StoppingLocationClusters.csv", isHeader=True) # shkp 1 and 2 stopping location clusters are the same
for data in shkpStopLocClustData:
    shkpStopLocClustIDToName[int(data["idx"])] = data["label"]
shkpStopLocClustIDToName[0] = "None"

custStopLocClustIDToName = {}
custStopLocClustData, _ = tools.load_csv_data(stoppingLocationClusterDir+"customerStoppingLocationClusters.csv", isHeader=True) 
for data in custStopLocClustData:
    custStopLocClustIDToName[int(data["idx"])] = data["label"]
custStopLocClustIDToName[0] = "None"


#
# load the openAI embeddings
#
if useOpenAIEmbeddings:
    uttToVec = {}

    customerUttVecDir = tools.dataDir+"20240111-141832_utteranceVectorizer/"
    shopkeeperUttVecDir = tools.dataDir+"20231220-111317_utteranceVectorizer/"
    allParticipantsUttVecDir = tools.dataDir+"20240116-190033_utteranceVectorizer/"


    allParticipantsUttVecs = np.load(allParticipantsUttVecDir+"all_participants_unique_utterances_openai_embeddings.npy")
    with open(allParticipantsUttVecDir+"all_participants_unique_utterances.txt", "r") as f:
        allParticipantsUtts = f.read().splitlines()
    for i in range(len(allParticipantsUtts)):
        uttToVec[allParticipantsUtts[i]] = allParticipantsUttVecs[i]


    uttVecDim = allParticipantsUttVecs.shape[1]


#
# add human readable spatial info to the interaction data
#
for i in range(len(interactionData)):
    interactionData[i]["customer2_currentLocation_name"] = custStopLocClustIDToName[int(interactionData[i]["customer2_currentLocation"])]
    interactionData[i]["customer2_motionOrigin_name"] = custStopLocClustIDToName[int(interactionData[i]["customer2_motionOrigin"])]
    interactionData[i]["customer2_motionTarget_name"] = custStopLocClustIDToName[int(interactionData[i]["customer2_motionTarget"])]

    interactionData[i]["shopkeeper1_currentLocation_name"] = shkpStopLocClustIDToName[int(interactionData[i]["shopkeeper1_currentLocation"])]
    interactionData[i]["shopkeeper1_motionOrigin_name"] = shkpStopLocClustIDToName[int(interactionData[i]["shopkeeper1_motionOrigin"])]
    interactionData[i]["shopkeeper1_motionTarget_name"] = shkpStopLocClustIDToName[int(interactionData[i]["shopkeeper1_motionTarget"])]

    interactionData[i]["shopkeeper2_currentLocation_name"] = shkpStopLocClustIDToName[int(interactionData[i]["shopkeeper2_currentLocation"])]
    interactionData[i]["shopkeeper2_motionOrigin_name"] = shkpStopLocClustIDToName[int(interactionData[i]["shopkeeper2_motionOrigin"])]
    interactionData[i]["shopkeeper2_motionTarget_name"] = shkpStopLocClustIDToName[int(interactionData[i]["shopkeeper2_motionTarget"])]


#
# remove null actions (actions with no speech and no motion - e.g. "first appearance" actions)
#
removedNullActions = []

for i in reversed(range(len(interactionData))):
    identifier = uniqueIDToIdentifier[int(interactionData[i]["unique_id"])]

    if identifier == "customer2":
        continue
    
    if interactionData[i]["participant_speech"] == "" and interactionData[i]["{}_motionTarget_name".format(identifier)] == "None" and interactionData[i]["{}_motionOrigin_name".format(identifier)] == "None":
        removedNullActions.append(interactionData.pop(i))


#
# find the action clusters (unique combinations of shopkeeper locations and speech clusters)
# and add the speech and action cluster data to the action sequence data
#
actionIDToAction = {}
actionKeyToActionID = {}

newJunkUtts = []


for row in interactionData:
    uniqueID = int(row["unique_id"])
    speech = row["participant_speech"]
    location = int(row["{}_currentLocation".format(uniqueIDToIdentifier[uniqueID])])
    motionTarget = int(row["{}_motionTarget".format(uniqueIDToIdentifier[uniqueID])])

    if location != 0:
        spatialInfo = location
    elif motionTarget != 0:
        spatialInfo = motionTarget
    else:
        spatialInfo = 0
        #if uniqueID != 2:
        #    print("WARNING: Invalid spatial infor for row: {}".format(row))


    # get the ID for the shopkeeper actions
    if uniqueID in shkpUniqueIDs:
        if speech in uttToSpeechClustID:
            speechClustID = uttToSpeechClustID[speech]
        else:
            # it is an utterance that was held out because of NaN distances, so just add it to the junk cluster
            speechClustID = mainJunkClusterID
            newJunkUtts.append(speech)
        
        action = (speechClustID, spatialInfo)
        actionKey = "-".join([str(x) for x in action])

        if actionKey not in actionKeyToActionID:
            actionID = len(actionKeyToActionID)
            actionKeyToActionID[actionKey] = actionID
            actionIDToAction[actionID] = action
        
        actionID = actionKeyToActionID[actionKey]
        

    # default values
    # SHOPKEEPER 1 action
    row["SHOPKEEPER_1_SPATIAL_INFO"] = -1
    row["SHOPKEEPER_1_SPATIAL_INFO_NAME"] = "None"
    row["SHOPKEEPER_1_SPEECH"] = ""
    row["SHOPKEEPER_1_DID_ACTION"] = 0
    
    row["SHOPKEEPER_1_ACTION_CLUSTER"] = -1
    row["SHOPKEEPER_1_SPEECH_CLUSTER"] = -1
    row["SHOPKEEPER_1_REPRESENTATIVE_UTTERANCE"] = ""
    row["SHOPKEEPER_1_SPEECH_CLUSTER_IS_JUNK"] = 0
    
    # SHOPKEEPER 2 action
    row["SHOPKEEPER_2_SPATIAL_INFO"] = -1
    row["SHOPKEEPER_2_SPATIAL_INFO_NAME"] = "None"
    row["SHOPKEEPER_2_SPEECH"] = ""
    row["SHOPKEEPER_2_DID_ACTION"] = 0
    
    row["SHOPKEEPER_2_ACTION_CLUSTER"] = -1
    row["SHOPKEEPER_2_SPEECH_CLUSTER"] = -1
    row["SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE"] = ""
    row["SHOPKEEPER_2_SPEECH_CLUSTER_IS_JUNK"] = 0

    # CUSTOMER action
    row["CUSTOMER_SPATIAL_INFO"] = -1
    row["CUSTOMER_SPATIAL_INFO_NAME"] = "None"
    row["CUSTOMER_SPEECH"] = ""
    row["CUSTOMER_DID_ACTION"] = 0


    if uniqueID == 1:
        # SHOPKEEPER 1 action
        row["SHOPKEEPER_1_SPATIAL_INFO"] = spatialInfo
        row["SHOPKEEPER_1_SPATIAL_INFO_NAME"] = shkpStopLocClustIDToName[spatialInfo]
        row["SHOPKEEPER_1_SPEECH"] = speech
        row["SHOPKEEPER_1_DID_ACTION"] = 1

        row["SHOPKEEPER_1_ACTION_CLUSTER"] = actionID            
        row["SHOPKEEPER_1_SPEECH_CLUSTER"] = speechClustID
        row["SHOPKEEPER_1_REPRESENTATIVE_UTTERANCE"] = speechClustIDToRepUtt[speechClustID]
        row["SHOPKEEPER_1_SPEECH_CLUSTER_IS_JUNK"] = speechClustIDIsJunk[speechClustID]
    
    elif uniqueID == 3:
        # SHOPKEEPER 2 action
        row["SHOPKEEPER_2_SPATIAL_INFO"] = spatialInfo
        row["SHOPKEEPER_2_SPATIAL_INFO_NAME"] = shkpStopLocClustIDToName[spatialInfo]
        row["SHOPKEEPER_2_SPEECH"] = speech
        row["SHOPKEEPER_2_DID_ACTION"] = 1

        row["SHOPKEEPER_2_ACTION_CLUSTER"] = actionID
        row["SHOPKEEPER_2_SPEECH_CLUSTER"] = speechClustID
        row["SHOPKEEPER_2_REPRESENTATIVE_UTTERANCE"] = speechClustIDToRepUtt[speechClustID]
        row["SHOPKEEPER_2_SPEECH_CLUSTER_IS_JUNK"] = speechClustIDIsJunk[speechClustID]
    
    else:
        # CUSTOMER action
        row["CUSTOMER_SPATIAL_INFO"] = spatialInfo
        row["CUSTOMER_SPATIAL_INFO_NAME"] = custStopLocClustIDToName[spatialInfo]
        row["CUSTOMER_SPEECH"] = speech
        row["CUSTOMER_DID_ACTION"] = 1


tools.save_interaction_data(interactionData, sessionDir+"20230623_SSC_3_trueMotionTargets_3_speechMotionCombined_humanReadable.csv", interactionData[0].keys() )



#
# get sequences of inputs and outputs
# add NULL shopkeeper 2 actions in between actions of other participants (to teach shopkeeper not to respond to them) (is this good?...)
#

inputLen = 3

inputs = []
outputs = []

outputActionIDs_shkp1 = []
outputActionIDs_shkp2 = []
outputSpeechClusterIDs_shkp1 = []
outputSpeechClusterIDs_shkp2 = []
outputSpatialInfo_shkp1 = []
outputSpatialInfo_shkp2 = []
outputSpeechClusterIsJunk_shkp1 = []
outputSpeechClusterIsJunk_shkp2 = []

outputDidActionBits = []


numActionsPerParticipant = {1:0, 2:0, 3:0}
numFirstActionsPerParticipant = {1:0, 2:0, 3:0}
actionHidSequenceCounts = {"total":0}
countOverThreshold = {"total":0}
timeDeltas = {"total":[]}


threshold = 6.0


thisExpStartIndex = 0
thisExp = interactionData[0]["experiment"]

lastAction = False
veryLastAction = False

for i in range(len(interactionData)):
    currAction = interactionData[i]

    try:
        nextAction = interactionData[i+1]
    except:
        veryLastAction = True

    if veryLastAction or thisExp != nextAction["experiment"]:
        thisExpStartIndex = i + 1
        thisExp = nextAction["experiment"]
        
        # just have a null action for the last input step
        nextAction = {}
        nextAction = copy.deepcopy(currAction)
        nextAction["SHOPKEEPER_1_DID_ACTION"] = 0
        nextAction["CUSTOMER_DID_ACTION"] = 0
        nextAction["SHOPKEEPER_2_DID_ACTION"] = 0

        lastAction = True
    else:
        lastAction = False


    # TODO: somehow treat the "first appearance" actions specially
    inputTemp = []

    for j in range(i-inputLen+1, i+1):
        if j < thisExpStartIndex:
            inputTemp.append(None)
        else:
            inputTemp.append(interactionData[j])

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
    if lastAction:
        continue

    t = float(currAction["time"])
    nextT = float(nextAction["time"])
    timeDelta = nextT - t

    hid = int(currAction["unique_id"])
    nextHid = int(nextAction["unique_id"])

    if i == thisExpStartIndex:
        numFirstActionsPerParticipant[hid] += 1

    numActionsPerParticipant[nextHid] += 1

    hidSeq = "{}->{}".format(hid, nextHid)
    if hidSeq not in actionHidSequenceCounts:
        actionHidSequenceCounts[hidSeq] = 0
    actionHidSequenceCounts[hidSeq] += 1

    #hidSeq = "{}->{}".format(hid in hidToImitate, nextHid in hidToImitate) # TODO this isn't really a meaningful metric anymore... (counts when a shopkeeper action follows another shopkeeper action)
    #if hidSeq not in actionHidSequenceCounts:
    #    actionHidSequenceCounts[hidSeq] = 0
    #actionHidSequenceCounts[hidSeq] += 1

    actionHidSequenceCounts["total"] += 1

    if (timeDelta) > threshold:
        if hidSeq not in countOverThreshold:
            countOverThreshold[hidSeq] = 0
        countOverThreshold[hidSeq] += 1
        countOverThreshold["total"] += 1

    if hidSeq not in timeDeltas:
        timeDeltas[hidSeq] = []
    timeDeltas[hidSeq].append(timeDelta)
    timeDeltas["total"].append(timeDelta)


#
# print the statistics
#
for key, value in numActionsPerParticipant.items():
    print("Unique ID {}: {} actions".format(key, value))

for key, value in numFirstActionsPerParticipant.items():
    print("Unique ID {}: {} first actions".format(key, value))


for i in range(1,4):
    for j in range(1,4):
        hidSeq = "{}->{}".format(i, j)
        print("Action HID sequence {}: {} instances".format(hidSeq, actionHidSequenceCounts[hidSeq]))

# for i in [True, False]:
#     for j in [True, False]:
#         hidSeq = "{}->{}".format(i, j)
        
#         print("Action HID sequence {}: {} instances, {} timeouts, {} delta mean, {} std.".format(hidSeq, 
#                                                                                                  actionHidSequenceCounts[hidSeq], 
#                                                                                                  countOverThreshold[hidSeq],
#                                                                                                  np.mean(timeDeltas[hidSeq]), 
#                                                                                                  np.std(timeDeltas[hidSeq])
#                                                                                                  ))
hidSeq = "total"
print("Action HID sequence {}: {} instances, {} timeouts, {} delta mean, {} std.".format(hidSeq, 
                                                                                                 actionHidSequenceCounts[hidSeq], 
                                                                                                 countOverThreshold[hidSeq],
                                                                                                 np.mean(timeDeltas[hidSeq]), 
                                                                                                 np.std(timeDeltas[hidSeq])
                                                                                                 ))
        

#
# vectorize the inputs, outputs, etc.
#


if trainUttVectorizer and not useOpenAIEmbeddings:
    custUttVectorizer = None
    shkpUttVectorizer = None

    custUttToVector = None
    shkpUttToVector = None
    
    #
    # train the speech vectorizers
    #
    maxNGramLen = 3
    shopkeeperUtterances = []
    customerUtterances = []

    custKeywordToRelevance = {}
    shkpKeywordToRelevance = {}


    # load keywords
    uttToKwData = {}
    keywordData, uttToKws, keywordsList, keywordToRelevance, _ = tools.load_keywords(keywordsFilename)
    for data in keywordData:
        uttToKwData[data["utterance"]] = data


    # compute keyword relevances separately for customer and shopkeeper utterances (Note: should do this for speech clustering too...)
    # get keyword relevances by looking at their relevance per sentence and their occurrences in all the utterances
    # also get the separate lists of customer and shopkeeper utterances

    for data in interactionData:
        hid = data["unique_id"]
        speech = data["participant_speech"]

        if speech != "":
            keywordData = uttToKwData[speech]
            kwTemp = [x for x in keywordData["keywords"].split(";") if x != ""]
            relTemp = [float(x) for x in keywordData["relevances"].split(";") if x != ""]
            
            if hid == "1" or hid == "3":
                shopkeeperUtterances.append(speech)

                # compute kw relevance
                for i in range(len(kwTemp)):
                    if kwTemp[i] not in shkpKeywordToRelevance:
                        shkpKeywordToRelevance[kwTemp[i]] = 0.0
                    shkpKeywordToRelevance[kwTemp[i]] += relTemp[i]
                    
            elif hid == "2":
                customerUtterances.append(speech)

                # compute kw relevance
                for i in range(len(kwTemp)):
                    if kwTemp[i] not in custKeywordToRelevance:
                        custKeywordToRelevance[kwTemp[i]] = 0.0
                    custKeywordToRelevance[kwTemp[i]] += relTemp[i]

            else:
                print("WARNING: Invalid unique ID:", hid)

    #
    # train the customer speech vectorizer
    #
    print("Creating customer utterance vectorizer...")

    custUttVectorizer = utterancevectorizer2.UtteranceVectorizer(
        customerUtterances,
        keywordNGramWeight=1,
        numNGramWeight=1,
        keywords=custKeywordToRelevance,
        minCount=2, # 2 for just SK or cust - min times keyword accours
        maxNGramLen=maxNGramLen,
        svdShare=0.5,
        makeKeywordsOneGrams=False,
        keywordCountThreshold=5,  # 5 for just SK or cust
        runLSA=True,
        useStopwords=False,
        useBackchannels=False
    )

    custUttVectors = custUttVectorizer.get_lsa_vectors(customerUtterances)
    print("Customer LSA uttVectors shape", custUttVectors.shape)

    custUttToVector = {}
    for i in range(len(customerUtterances)):
        custUttToVector[customerUtterances[i]] = custUttVectors[i,:] 


    #
    # train the shopkeeper speech vectorizer
    #
    shkpUttVectorizer = utterancevectorizer2.UtteranceVectorizer(
        shopkeeperUtterances,
        keywordNGramWeight=1,
        numNGramWeight=1,
        keywords=shkpKeywordToRelevance,
        minCount=2, # 2 for just SK or cust - min times keyword accours
        maxNGramLen=maxNGramLen,
        svdShare=0.5,
        makeKeywordsOneGrams=False,
        keywordCountThreshold=5,  # 5 for just SK or cust
        runLSA=True,
        useStopwords=False,
        useBackchannels=False
    )

    shkpUttVectors = shkpUttVectorizer.get_lsa_vectors(shopkeeperUtterances)
    print("Shopkeeper LSA uttVectors shape", shkpUttVectors.shape)

    shkpUttToVector = {}
    for i in range(len(shopkeeperUtterances)):
        shkpUttToVector[shopkeeperUtterances[i]] = shkpUttVectors[i,:] 

elif not trainUttVectorizer and not useOpenAIEmbeddings:
    #
    # load the speech vectorizers
    #
    print("Loading customer utterance vectorizer...")

    with open(uttVectorizerDir+"customerUtteranceVectorizer.pkl", "rb") as f:
        custUttVectorizer = pickle.load(f)
    
    with open(uttVectorizerDir+"custUttToVector.pkl", "rb") as f:
        custUttToVector = pickle.load(f)

    
    print("Loading shopkeeper utterance vectorizer...")

    with open(uttVectorizerDir+"shopkeeperUtteranceVectorizer.pkl", "rb") as f:
        shkpUttVectorizer = pickle.load(f)
    
    with open(uttVectorizerDir+"shkpUttToVector.pkl", "rb") as f:
        shkpUttToVector = pickle.load(f)



if not useOpenAIEmbeddings:
    print("pickling the customer utterance vectorizer...")

    custUttVectorizer.deinitializeMeCab() # Necessary because MeCab cannot get pickled. This will automatically get reinitialized when we lemmatize an utterance
    with open(sessionDir+"customerUtteranceVectorizer.pkl", "wb") as f:
        f.write(pickle.dumps(custUttVectorizer))

    with open(sessionDir+"custUttToVector.pkl", "wb") as f:
        f.write(pickle.dumps(custUttToVector))


    print("pickling the shopkeeper utterance vectorizer...")

    shkpUttVectorizer.deinitializeMeCab() # Necessary because MeCab cannot get pickled. This will automatically get reinitialized when we lemmatize an utterance
    with open(sessionDir+"shopkeeperUtteranceVectorizer.pkl", "wb") as f:
        f.write(pickle.dumps(shkpUttVectorizer))

    with open(sessionDir+"shkpUttToVector.pkl", "wb") as f:
        f.write(pickle.dumps(shkpUttToVector))


# add 0 vec for no speech
if not useOpenAIEmbeddings:
    custUttVecShape = list(custUttToVector.values())[0].shape
    shkpUttVecShape = list(shkpUttToVector.values())[0].shape
    custUttToVector[""] = np.zeros(custUttVecShape)
    shkpUttToVector[""] = np.zeros(shkpUttVecShape)

else:
    uttToVec[""] = np.zeros(uttVecDim)
    

#
# create the input vectors
#
print("Creating the input vectors...")

# who acted bit + customer at from to + shopkeeper at from to (two shopkeepers)
inputVecNonSpeechLen = 3 + (len(custStopLocClustIDToName) + 1) * 3 + (len(shkpStopLocClustIDToName) + 1) * 6
if not useOpenAIEmbeddings:
    inputVecSpeechLen = custUttVecShape[0] + shkpUttVecShape[0]
else:
    inputVecSpeechLen = uttVecDim # because all participant vectorizations are the same dimension and there is only one participant acting at a time, we only need one slot for speech

# save speech and non speech separately because speech requires floats, non speech can use ints to save space
inputVectorsNonSpeech = []
inputVectorsSpeech = []
inputVectorsCombined = []

for input in inputs:
    inputVecNonSpeechTemp = []
    inputVecSpeechTemp = []
    inputVecCombinedTemp = []

    for inputStep in input:
        if inputStep == None:
            inputVecNonSpeechTemp.append(np.zeros(inputVecNonSpeechLen))
            inputVecSpeechTemp.append(np.zeros(inputVecSpeechLen))
            inputVecCombinedTemp.append(np.zeros(inputVecNonSpeechLen+inputVecSpeechLen))
        
        else:
            hid = int(inputStep["unique_id"]) # ranges from 1 to 3
            speech = inputStep["participant_speech"]

            # did action bits
            didActionVec = np.zeros(3) # , dytpe=np.int8)
            didActionVec[hid-1] = 1


            # customer spatial info
            custCurrLoc = np.zeros(len(custStopLocClustIDToName) + 1) # use 0 index to show that it is not set
            custMotOri = np.zeros(len(custStopLocClustIDToName) + 1)
            custMotTar = np.zeros(len(custStopLocClustIDToName) + 1)

            custCurrLoc[int(inputStep["customer2_currentLocation"])] = 1
            custMotOri[int(inputStep["customer2_motionTarget"])] = 1
            custMotTar[int(inputStep["customer2_motionOrigin"])] = 1


            # shopkeeper 1 spatial info
            shkp1CurrLoc = np.zeros(len(shkpStopLocClustIDToName) + 1) # use 0 index to show that it is not set
            shkp1MotOri = np.zeros(len(shkpStopLocClustIDToName) + 1)
            shkp1MotTar = np.zeros(len(shkpStopLocClustIDToName) + 1)

            shkp1CurrLoc[int(inputStep["shopkeeper1_currentLocation"])] = 1
            shkp1MotOri[int(inputStep["shopkeeper1_motionTarget"])] = 1
            shkp1MotTar[int(inputStep["shopkeeper1_motionOrigin"])] = 1


            # shopkeeper 2 spatial info
            shkp2CurrLoc = np.zeros(len(shkpStopLocClustIDToName) + 1) # use 0 index to show that it is not set
            shkp2MotOri = np.zeros(len(shkpStopLocClustIDToName) + 1)
            shkp2MotTar = np.zeros(len(shkpStopLocClustIDToName) + 1)

            shkp2CurrLoc[int(inputStep["shopkeeper2_currentLocation"])] = 1
            shkp2MotOri[int(inputStep["shopkeeper2_motionTarget"])] = 1
            shkp2MotTar[int(inputStep["shopkeeper2_motionOrigin"])] = 1


            if not useOpenAIEmbeddings:
                # customer utterance vector
                if hid == 2:
                    custUttVec = custUttToVector[speech]
                else:
                    custUttVec = custUttToVector[""]


                # shopkeeper utterance vector
                if hid == 1 or hid == 3:
                    shkpUttVec = shkpUttToVector[speech]
                else:
                    shkpUttVec = shkpUttToVector[""]

                inputVecSpeechTemp.append(np.concatenate([custUttVec, shkpUttVec]))
            
            else:
                inputVecSpeechTemp.append(uttToVec[speech])


            inputVecNonSpeechTemp.append(np.concatenate([didActionVec, custCurrLoc, custMotOri, custMotTar, shkp1CurrLoc, shkp1MotOri, shkp1MotTar, shkp2CurrLoc, shkp2MotOri, shkp2MotTar]))
            inputVecCombinedTemp.append(np.concatenate([inputVecNonSpeechTemp[-1], inputVecSpeechTemp[-1]]))

    
    inputVecNonSpeechTemp = np.stack(inputVecNonSpeechTemp, axis=0)
    inputVecSpeechTemp = np.stack(inputVecSpeechTemp, axis=0)
    inputVecCombinedTemp = np.stack(inputVecCombinedTemp, axis=0)
    
    inputVectorsNonSpeech.append(inputVecNonSpeechTemp)
    inputVectorsSpeech.append(inputVecSpeechTemp)
    inputVectorsCombined.append(inputVecCombinedTemp)


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


inputVectorsNonSpeech = np.stack(inputVectorsNonSpeech, axis=0)
inputVectorsSpeech = np.stack(inputVectorsSpeech, axis=0)
inputVectorsCombined = np.stack(inputVectorsCombined, axis=0)

inputVectorsNonSpeech = inputVectorsNonSpeech.astype(np.int16)
inputVectorsSpeech = inputVectorsSpeech.astype(np.float32)
inputVectorsCombined = inputVectorsCombined.astype(np.float32)

#np.save(sessionDir+"/inputVectorsNonSpeech", inputVectorsNonSpeech)
#np.save(sessionDir+"/inputVectorsSpeech", inputVectorsSpeech)
np.save(sessionDir+"/inputVectorsCombined", inputVectorsCombined)


np.save(sessionDir+"/outputActionIDs_shkp1", outputActionIDs_shkp1)
np.save(sessionDir+"/outputSpeechClusterIDs_shkp1", outputSpeechClusterIDs_shkp1)
np.save(sessionDir+"/outputSpatialInfo_shkp1", outputSpatialInfo_shkp1)
np.save(sessionDir+"/outputSpeechClusterIsJunk_shkp1", outputSpeechClusterIsJunk_shkp1)

np.save(sessionDir+"/outputActionIDs_shkp2", outputActionIDs_shkp2)
np.save(sessionDir+"/outputSpeechClusterIDs_shkp2", outputSpeechClusterIDs_shkp2)
np.save(sessionDir+"/outputSpatialInfo_shkp2", outputSpatialInfo_shkp2)
np.save(sessionDir+"/outputSpeechClusterIsJunk_shkp2", outputSpeechClusterIsJunk_shkp2)

np.save(sessionDir+"/outputDidActionBits", outputDidActionBits)




#
# prepare and save the human readable inputs and outputs
#

# generate fieldnames
inputFieldnames = list(inputs[4][0].keys()) # the first one without None
inputFieldnames.pop(inputFieldnames.index("experiment")) # no need to put this info for each input step
inputFieldnamesAllSteps = []
fieldnamesNotToDuplicate = ["TRIAL", "CUSTOMER_ID", "SHOPKEEPER1_ID", "SHOPKEEPER2_ID", "CUSTOMER_TYPE", "CUSTOMER_BUY", "SHOPKEEPER2_TYPE", "NOTES"]

for i in range(inputLen):
    for fieldname in inputFieldnames:
        if fieldname not in fieldnamesNotToDuplicate:
            inputFieldnamesAllSteps.append("{}_{}".format(i, fieldname))

outputFieldnames = list(outputs[0].keys())
outputFieldnames.pop(outputFieldnames.index("experiment"))
outputFieldnamesForHumanReadable = ["y_{}".format(fieldname) for fieldname in outputFieldnames if fieldname not in fieldnamesNotToDuplicate]

inputOutputFieldnames = fieldnamesNotToDuplicate + inputFieldnamesAllSteps + outputFieldnamesForHumanReadable



# generate the rows to be saved to csv
inputsAndOutputsForCsv = []

for i in range(len(inputs)):
    row = OrderedDict()

    for fieldname in fieldnamesNotToDuplicate:
        row[fieldname] = outputs[i][fieldname]
    # input
    input = inputs[i]
    for j in range(len(input)):
        if input[j] == None:
            for fieldname in inputFieldnames:
                if fieldname not in fieldnamesNotToDuplicate:
                    row["{}_{}".format(j, fieldname)] = ""
        else:
            for fieldname in inputFieldnames:
                if fieldname not in fieldnamesNotToDuplicate:
                    row["{}_{}".format(j, fieldname)] = input[j][fieldname]
            
    
    # output
    for fieldname in outputFieldnames:
        if fieldname not in fieldnamesNotToDuplicate:
            try:
                row["y_{}".format(fieldname)] = outputs[i][fieldname]
            except:
                row["y_{}".format(fieldname)] = ""
        

    inputsAndOutputsForCsv.append(row)




#for i in range(len(inputsAndOutputsForCsv)):
#    inputsAndOutputsForCsv[i].update(expIDToCondition[int(interactionData[i]["experiment"])])

#inputOutputFieldnames = [] + inputOutputFieldnames


tools.save_interaction_data(inputsAndOutputsForCsv, sessionDir+"humanReadableInputsOutputs.csv", inputOutputFieldnames)



print("Done.")


