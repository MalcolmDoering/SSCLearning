


import tools

import csv
import pymysql
import copy


def combine_speech_and_motion_actions_shopkeeper2(actionA, actionB):

    speechAction = actionA if actionA["participant_speech"] else actionB
    motionAction = actionA if actionA["motion_start_time"] else actionB

    # use timestamp of the first action
    actionC = copy.deepcopy(actionA)

    # take the speech infor from the speech action
    actionC["participant_speech"] = speechAction["participant_speech"]
    actionC["speech_time"] = speechAction["speech_time"]
    actionC["speech_duration"] = speechAction["speech_duration"]

    # take the motion info from the motion action
    actionC["motion_start_time"] = motionAction["motion_start_time"]
    actionC["motion_end_time"] = motionAction["motion_end_time"]
    actionC["true_motionTarget"] = motionAction["true_motionTarget"]
    actionC["shopkeeper2_currentLocation"] = motionAction["shopkeeper2_currentLocation"]
    actionC["shopkeeper2_motionOrigin"] = motionAction["shopkeeper2_motionOrigin"]
    actionC["shopkeeper2_motionTarget"] = motionAction["shopkeeper2_motionTarget"]

    if int(actionC["shopkeeper2_currentLocation"]) == 0 and int(actionC["shopkeeper2_motionTarget"]) == 0:
        print("WARNING: Invalid motion information!")

    return actionC


interactionDataFilename = "20230623_SSC.csv"
keywordsFilename = "20230609-141854_unique_utterance_keywords.csv"
englishFilename = "20230609-141854_unique_utterances_english_autotranslations.csv"


sessionDir = tools.create_session_dir("processForSpeechClustering")



#
# load the data
#
interactionData, fieldnames = tools.load_interaction_data(tools.dataDir+interactionDataFilename)


#
# set the ground truth motion info for the participant we want to imitate
#
hidToImitate = 3 # junior shopkeeper

trueMotTar = None
motOri = None
motStartTime = None
motEndTime = None

indicesToDelete = []

customerFirstAppearance = False
currExpID = None
prevExpID = None

for i in range(len(interactionData)):

    #if i == 2477:
    #    print("hello")

    data = interactionData[i]
    t = float(data["time"])
    expID = int(data["experiment"])
    uniqueID = int(data["unique_id"])

    if expID != currExpID:
        prevExpID = currExpID
        currExpID = expID
        customerFirstAppearance = False
    
    if uniqueID == 2 and not customerFirstAppearance:
        customerFirstAppearance = True
    

    if float(data["motion_start_time"]) != 0 and int(data["unique_id"]) == hidToImitate:
        # this is a movement action for the participant we want to imitate
        trueMotTar = data["true_motionTarget"]
        motStartTime = float(data["motion_start_time"])
        motEndTime = float(data["motion_end_time"])

        if hidToImitate == 1:
            motOri = data["shopkeeper1_motionOrigin"]
        elif hidToImitate == 2:
            motOri = data["customer_motionOrigin"]
        elif hidToImitate == 3:
            motOri = data["shopkeeper2_motionOrigin"]
    
    elif trueMotTar != None and t > motStartTime and t < motEndTime:
        if hidToImitate == 1:
            data["shopkeeper1_currentLocation"] = "0"
            data["shopkeeper1_motionOrigin"] = motOri
            data["shopkeeper1_motionTarget"] = trueMotTar
        elif hidToImitate == 2:
            data["customer_currentLocation"] = "0"
            data["customer_motionOrigin"] = motOri
            data["customer_motionTarget"] = trueMotTar
        elif hidToImitate == 3:
            data["shopkeeper2_currentLocation"] = "0"
            data["shopkeeper2_motionOrigin"] = motOri
            data["shopkeeper2_motionTarget"] = trueMotTar
        else:
            print("WARNING: Invalid unique_id:", hidToImitate)

    elif trueMotTar != None and t >= motEndTime:
        trueMotTar = None
        motOri = None
        motStartTime = None
        motEndTime = None
    

    # there is no ongoing motion by the participant we want to imitate, so make sure the current location is set
    # this will overwrite situations where a motion was detected but the target ended up being the same as origin
    if hidToImitate == 1 and int(data["shopkeeper1_currentLocation"]) == 0 and int(data["shopkeeper1_motionOrigin"]) != 0:
        data["shopkeeper1_currentLocation"] = data["shopkeeper1_motionOrigin"]
        data["shopkeeper1_motionOrigin"] = "0"
        data["shopkeeper1_motionTarget"] = "0"
    elif hidToImitate == 2 and int(data["customer_currentLocation"]) == 0 and int(data["customer_motionOrigin"]) != 0:
        data["customer_currentLocation"] = data["customer_motionOrigin"]
        data["customer_motionOrigin"] = "0"
        data["customer_motionTarget"] = "0"
    elif hidToImitate == 3 and int(data["shopkeeper2_currentLocation"]) == 0 and int(data["shopkeeper2_motionOrigin"]) != 0:
        data["shopkeeper2_currentLocation"] = data["shopkeeper2_motionOrigin"]
        data["shopkeeper2_motionOrigin"] = "0"
        data["shopkeeper2_motionTarget"] = "0"


    # sometimes overwriting detected motions could result in actions with no speech an no motion, so we can delete these.
    # we can also delete shopkeeper first appearance actions that don't contain speech or motion, because we don't need to imitate these.
    # make sure not to delete the customer first appearance, because this is something the shopkeeper might react to.
    if data["participant_speech"] == "" and not customerFirstAppearance:
        if uniqueID == 1 and int(data["shopkeeper1_motionOrigin"]) == 0:
            indicesToDelete.append(i)
        elif uniqueID == 2 and int(data["customer_motionOrigin"]) == 0:
            indicesToDelete.append(i)
        elif uniqueID == 3 and int(data["shopkeeper2_motionOrigin"]) == 0:
            indicesToDelete.append(i)

indicesToDelete.reverse()
for i in indicesToDelete:
    interactionData.pop(i)


for i in range(len(interactionData)):
    data = interactionData[i]

    if int(data["shopkeeper2_currentLocation"]) == 0 and int(data["shopkeeper2_motionTarget"]) == 0 and int(data["shopkeeper2_didAction"]) == 1:
        print("WARNING: Invalid motion information for index, time:", i, data["time"])


"""
# pre 20230623
hidToImitate = 3 # junior shopkeeper

trueMotTar = None
motStartTime = None

for i in reversed(range(len(interactionData))):
    data = interactionData[i]
    t = float(data["time"])

    if float(data["motion_start_time"]) != 0 and int(data["unique_id"]) == hidToImitate:
        trueMotTar = data["true_motionTarget"]
        motStartTime = float(data["motion_start_time"])

    if trueMotTar != None and motStartTime <= t:
        if hidToImitate == 1:
            interactionData[i]["shopkeeper1_motionTarget"] = trueMotTar
        elif hidToImitate == 2:
            interactionData[i]["customer_motionTarget"] = trueMotTar
        elif hidToImitate == 3:
            interactionData[i]["shopkeeper2_motionTarget"] = trueMotTar
        else:
            print("WARNING: Invalid unique_id:", hidToImitate)

    elif trueMotTar != None and motStartTime > t:
        trueMotTar = None
        motStartTime = None
    

    # remove the true motion marker
    if float(data["motion_start_time"]) != 0:
        interactionData.pop(i)
"""



#
# combine speech and motion actions for the participant we will imitate
#

# combine any two subsequent actions from the participant we want to imitate 
# if there is no action from another participant in-between 
# and then time difference is less than threshold

threshold = 5 # seconds
actionIndicesToCombine = []
actionsToCombine = []
combinedActions = []

i = 0
while i < len(interactionData) - 1:

    data = interactionData[i]
    t = float(data["time"])
    isSpeech = bool(data["participant_speech"])
    isMotion = bool(data["motion_start_time"])
    
    data2 = interactionData[i+1]    
    t2 = float(data2["time"])
    isSpeech2 = bool(data2["participant_speech"])
    isMotion2 = bool(data2["motion_start_time"])
    
    # only combine speech and motion actions, within threshold, from the human we want to imitate
    if int(data["unique_id"]) == hidToImitate and int(data2["unique_id"]) == hidToImitate:
        if t2-t < threshold:
            if (isMotion and isSpeech2): 
                actionIndicesToCombine.append((i, i+1))
                actionsToCombine.append((data, data2))
                combinedActions.append(combine_speech_and_motion_actions_shopkeeper2(data, data2))
                i += 2
                continue

    i += 1


interactionDataBeforeCombine = copy.deepcopy(interactionData)

for i in reversed(range(len(actionIndicesToCombine))):
    indexA = actionIndicesToCombine[i][0]
    indexB = actionIndicesToCombine[i][1]

    interactionData.pop(indexB)
    interactionData[indexA] = combinedActions[i]



    """
        for j in range(i+1, len(interactionData)):
            data2 = interactionData[j]
            
            
            if int(data2["unique_id"]) == hidToImitate and (float(data2["time"]) - t) <= threshold:
                if len(tempIndicesToCombine) == 0:
                    tempIndicesToCombine.append(i)
                    tempActionsToCombine.append(data)
                tempIndicesToCombine.append(j)
                tempActionsToCombine.append(data2)

            else:
                # break if the hid of the next action is not the hid we want to imitate or if the time is over the threshold
                break
    
    if len(tempIndicesToCombine) > 0:
        actionIndicesToCombine.append(tempIndicesToCombine)
        actionsToCombine.append(tempActionsToCombine)
        i = j
    else:
        i += 1
    """

# combine the actions...



"""
# before 20230623
# combines actions that occur after participant speech, within the duration of the speech
# this probably misses cases where the motion starts before the speech...
actionIndicesToCombine = []

i = 0
while i < len(interactionData):
    
    data = interactionData[i]
    tempToCombine = []

    if int(data["unique_id"]) == hidToImitate and data["participant_speech"] != "":
        speechEndTime = float(data["speech_time"]) + (float(data["speech_duration"]) / 1000.0)

        for j in range(i+1, len(interactionData)):
            data2 = interactionData[j]
            
            if float(data2["time"]) > speechEndTime:
                break

            elif int(data2["unique_id"]) == hidToImitate and data2["participant_speech"] != "":
                break

            elif int(data2["unique_id"]) == hidToImitate and data2["participant_speech"] == "":
                if len(tempToCombine) == 0:
                    tempToCombine.append(i)
                tempToCombine.append(j)
    
    if len(tempToCombine) > 0:
        actionIndicesToCombine.append(tempToCombine)
        i = j
    else:
        i += 1
"""


#
# add the keywords
#

# load the keywords
keywordData, uttToKws, keywords, keywordToRelevance, keywordFieldnames = tools.load_keywords(tools.modelDir+keywordsFilename)


speechIndex = fieldnames.index("participant_speech")
fieldnames.insert(speechIndex+1, "keywords")

for i in range(len(interactionData)):
    speech = interactionData[i]["participant_speech"]

    if speech != "":
        try:
            interactionData[i]["keywords"] = uttToKws[speech]
        except:
            print("WARNING: Missing keywords for utterance: '{}'".format(speech))


#
# add the English automatic translations
#
englishTranslationData, _ = tools.load_csv_data(tools.modelDir+englishFilename, isJapanese=True)
jpToEn = dict(englishTranslationData)

enlishIndex = fieldnames.index("participant_speech")
fieldnames.insert(enlishIndex+1, "participant_speech_english_autotranslate")

for i in range(len(interactionData)):
    speech = interactionData[i]["participant_speech"]

    if speech != "":
        try:
            interactionData[i]["participant_speech_english_autotranslate"] = jpToEn[speech]
        except:
            print("WARNING: Missing Enlgish translation for utterance: '{}'".format(speech))


#
# get the experiment condition info
#
expIDToCondition = {}

connection = pymysql.connect(host=tools.host,
                                user=tools.user,
                                password=tools.password,
                                database=tools.database)
with connection:
    with connection.cursor() as cursor:
        sql = "SELECT * from "+tools.exp_table_name
        cursor.execute(sql)
        expData = cursor.fetchall()
        for e in expData:
            expID = int(e[1])
            rawCond = e[6]
            custID = int(rawCond[:2])

            splitCond = rawCond[3:].split(" ")
            splitCond2 = splitCond[0].split("-")

            custType = None

            if splitCond2[0] == "po":
                custType = "PORTRAIT"
            elif splitCond2[0] == "l":
                custType = "LANDSCAPE"
            elif splitCond2[0] == "n":
                custType = "NOVICE"
            elif splitCond2[0] == "w":
                custType = "BROWSING"
            elif splitCond2[0] == "d":
                custType = "DEVELOPMENT"
            elif splitCond2[0] == "pr":
                custType = "PRINTER"
            else:
                print("WARNING: Invalid customer type:", expID, splitCond2[0])
            
            buy = ""

            if len(splitCond2) > 1:
                if splitCond2[1] == "b":
                    buy = "TRUE"
                elif splitCond2[1] == "nb":
                    buy = "FALSE"
                else:
                    print("WARNING: Buy information:", expID, splitCond2[1])
            
            if custType != "DEVELOPMENT" and custType != "PRINTER" and buy == "":
                print("WARNING: Missing buy information:", expID)
            
            s2Type = "NORMAL"

            if len(splitCond) > 2 and "S2+" in splitCond[2]:
                s2Type = splitCond[2][2:]

                if s2Type == "+know":
                    s2Type = "FULL_KNOWLEDGE"
                elif s2Type == "+access":
                    s2Type = "STOCK_ACCESS"
            

            otherNotes = ""

            if len(rawCond[rawCond.index("H"):]) > 1:
                splitCond3 = rawCond[rawCond.index("H")+2:].split(" ")
                
                if "S2" in splitCond3[0]:
                    otherNotes = " ".join(splitCond3[1:])
                else:
                    otherNotes = " ".join(splitCond3)
                
                if "S2+" in otherNotes:
                    print("Hello")

            expIDToCondition[expID] = {"TRIAL": expID,
                                       "CUSTOMER_ID": custID,
                                       "SHOPKEEPER1_ID": tools.custIDToShkpIDs[custID][0],
                                       "SHOPKEEPER2_ID": tools.custIDToShkpIDs[custID][1],
                                       "CUSTOMER_TYPE": custType, 
                                       "CUSTOMER_BUY": buy, 
                                       "SHOPKEEPER2_TYPE": s2Type, 
                                       "NOTES": otherNotes}


#
# add the condition info to the data
#
for i in range(len(interactionData)):
    expID = int(interactionData[i]["experiment"])
    interactionData[i].update(expIDToCondition[expID])

fieldnames = ["TRIAL", "CUSTOMER_ID", "SHOPKEEPER1_ID", "SHOPKEEPER2_ID", "CUSTOMER_TYPE", "CUSTOMER_BUY", "SHOPKEEPER2_TYPE", "NOTES"] + fieldnames





# save
tools.save_interaction_data(interactionData, sessionDir + interactionDataFilename[:-4] + "_{}_trueMotionTargets_{}_speechMotionCombined.csv".format(hidToImitate, hidToImitate), fieldnames)



#
# prepare utterances for speech clustering
#
"""
utterancesPerParticipant = {}

for i in range(len(interactionData)):
    hid = interactionData[i]["unique_id"]
    speech = interactionData[i]["participant_speech"]

    if speech != "":
        if hid not in utterancesPerParticipant:
            utterancesPerParticipant[hid] = []

        utterancesPerParticipant[hid].append(interactionData[i])

for hid in utterancesPerParticipant:
    tools.save_interaction_data(utterancesPerParticipant[hid], sessionDir+interactionDataFilename[:-4] + "_{}_speech.csv".format(hid), fieldnames)

tools.save_interaction_data(utterancesPerParticipant["1"]+utterancesPerParticipant["3"], sessionDir+interactionDataFilename[:-4] + "_{}_trueMotionTargets_{}_speechMotionCombined_all_shopkeeper_speech.csv".format(hidToImitate, hidToImitate), fieldnames)
"""

