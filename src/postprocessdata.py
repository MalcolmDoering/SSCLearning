


import tools

import csv


interactionDataFilename = "20230620_SSC.csv"
keywordsFilename = "20230609-141854_unique_utterance_keywords.csv"
englishFilename = "20230609-141854_unique_utterances_english_autotranslations.csv"


sessionDir = tools.create_session_dir("speechPreprocessing")



#
# load the data
#
interactionData, fieldnames = tools.load_interaction_data(tools.dataDir+interactionDataFilename)


#
# set the ground truth motion info for the participant we want to imitate
#
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


#
# combine speech and motion actions for the participant we will imitate
#
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



#
# add the keywords
#

# load the keywords
keywordData, uttToKws, keywrods, keywordFieldnames = tools.load_keywords(tools.modelDir+keywordsFilename)

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
englishTranslationData, _ = tools.load_japanese_csv_data(tools.modelDir+englishFilename)
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


# save
tools.save_interaction_data(interactionData, sessionDir + interactionDataFilename[:-4] + "_{}_trueMotionTargets_{}_speechMotionCombined.csv".format(hidToImitate, hidToImitate), fieldnames)



#
# prepare utterances for speech clustering
#
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


