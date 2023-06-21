

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "jiang-yongqiang -gcp-ai-for-hri-a1a37d8b11ab.json"
from google.cloud import language
import json
import time
import pymysql
from urllib import request, parse
import csv
import sys
from tqdm import tqdm


import tools


interactionDataFilename = "20230608_SSC.csv"
keywordsFilename = "20230606_unique_utterance_keywords.csv"


class Keyword(object):
    def __init__(self, text, sumRelevance=0.0, count=0, fromWatson=False, fromGoogle=False, fromGoo=False):
        self.text = text
        self.sumRelevance = sumRelevance
        self.count = count
        self.fromWatson = fromWatson
        self.fromGoogle = fromGoogle
        self.fromGoo = fromGoo

    def __repr__(self):
        return "Keyword(%s, %f, %d, %s, %s, %s)" % (self.text, self.sumRelevance, self.count, "True" if self.fromWatson else "False", "True" if self.fromGoogle else "False", "True" if self.fromGoo else "False")

    def __str__(self):
        return repr(self)



if __name__ == "__main__":
    
    sessionDir = tools.create_session_dir("keywords")

    #
    # load data
    #    
    interactionData, fieldnames = tools.load_interaction_data(tools.dataDir+interactionDataFilename)
    
    uniqueUtterances = set()

    for i in range(len(interactionData)):
        speech = interactionData[i]["participant_speech"]

        if speech != "":
            uniqueUtterances.add(speech)
    
    uniqueUtterances = list(uniqueUtterances)
    uniqueUtterances.sort()


    #
    # load old keywords
    #
    oldKeywordData, oldUttToKws, oldKeywordFieldnames = tools.load_keywords(tools.modelDir+keywordsFilename)


    #
    # get the keywords
    #
    keywordData = []

    # Connect to Google Natural Language Understanding API
    client = language.LanguageServiceClient()

    googleMaxRelevance = None
    googleErrorCount = 0
    
    
    for i in tqdm(range(len(uniqueUtterances))):
        utt = uniqueUtterances[i]

        # skip utts we already have keywords for
        if utt in oldUttToKws:
            continue

        print(i, utt)

        document = language.Document(
            content=utt,
            type=language.Document.Type.PLAIN_TEXT,
            language='ja')
        
        # Detects entities in the document. You can also analyze HTML with:
        entities = client.analyze_entities(request={'document': document}).entities
        
        kwListTemp = []
        relListTemp = []

        for entity in entities:
            text = entity.name
            relevance = float(entity.salience)
            
            kwListTemp.append(text)
            relListTemp.append(relevance)
            
            if googleMaxRelevance is None or relevance > googleMaxRelevance:
                googleMaxRelevance = relevance
            

        row = {}
        row["utterance"] = utt
        
        if len(kwListTemp) > 0:
            row["keywords"] = ";".join(kwListTemp)
            row["relevances"] = ";".join([str(r) for r in relListTemp])
        else:
            row["keywords"] = ""
            row["relevances"] = ""
        

        keywordData.append(row)

        #if i > 100:
        #    break
    

#
# save the keywords
#
uniqueUtteranceKeywordsFn = sessionDir + "unique_utterance_keywords.csv"

# prepare rows
keywordData += oldKeywordData
keywordData.sort(key=lambda x: x["utterance"])

with open(uniqueUtteranceKeywordsFn, "w", newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["utterance", "keywords", "relevances"], quoting=csv.QUOTE_ALL)
    writer.writeheader()

    for row in keywordData:
        writer.writerow(row)


#
# save unique utterances
#
with open(sessionDir + "unique_utterances.txt", "w", encoding="utf-8-sig") as f:
    for row in keywordData:
        f.write(row["utterance"] + "\n")



print("Done.")