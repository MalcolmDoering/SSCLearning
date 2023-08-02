'''
Created on Feb 16, 2017

@author: MalcolmD
'''

import numpy as np
from scipy.spatial.distance import cosine, jaccard
import csv
from jellyfish import damerau_levenshtein_distance, levenshtein_distance, jaro_winkler
import editdistance
# import nltk
# from nltk.stem import WordNetLemmatizer

# import tools
from utterancevectorizer2 import UtteranceVectorizer, NGram, Token
import MeCab
import tools
import os
import pickle


class ClusterData(object):
    def __init__(self, clusterID, isRepresentative, utteranceID, utterance,uttLocation = None):
        self.clusterID = clusterID
        self.isRepresentative = isRepresentative
        self.utteranceID = utteranceID
        self.utterance = utterance
        if uttLocation!=None:
            self.uttLocation = uttLocation

class RepresentativeUtteranceFinder(object):
    
    def __init__(self, clustToTopic, clustToUttIds, clustToUtts, clustToRepUtt, utteranceVectorizer, distanceMetric, tfidf):
        
        self.clustToTopic = clustToTopic
        self.clustToUttIds = clustToUttIds
        self.clustToUtts = clustToUtts
        self.clustToRepUtt = clustToRepUtt
        
        self.distanceMetric = distanceMetric
        self.tfidf = tfidf
        
        # self.wnl = WordNetLemmatizer()
        self.initializeMeCab()



        clustToCentroid = {}
        
        for clustId, utts in clustToUtts.items():
            
            uttVecs = []
            
            for utt in utts:
                uttVec = utteranceVectorizer.get_utterance_vector(utt)
                uttVecs.append(uttVec)
            
            clustToCentroid[clustId] = np.average(np.asarray(uttVecs, dtype=np.float64), axis=0)
        
        
        #
        # find the utterance closest to the centroid
        #
        self.clustToClosestToCentroidUtt = {}
        self.clustToClosestToCentroidUtt[0] = "BAD CLUSTER"
        
        count = 0
        
        for clustId, utts in clustToUtts.items():
            
            closestUtt = None
            closestDist = None
            
            for utt in utts:
                uttVec = utteranceVectorizer.get_utterance_vector(utt)
                dist = self.distanceMetric(uttVec,clustToCentroid[clustId])
                
                if closestUtt == None or dist < closestDist:
                    closestUtt = utt
                    closestDist = dist
            
            self.clustToClosestToCentroidUtt[clustId] = closestUtt
            
            count += 1
            print("centroid", count, "of", len(clustToUtts))
        
        
        #
        # find the utterance most similar to all other utterances in the cluster
        #
        self.clustToMinAveDistUtt = {}
        self.clustToMinAveDistUtt[0] = "BAD CLUSTER"
        
        count = 0
        
        for clustId, utts in clustToUtts.items():
            
            mostSimUtt = None
            minAveDistDist = None
            
            if clustId == 0:
                continue
            
            for utt in utts:
                
                # uttLemmas = [self.wnl.lemmatize(t).lower() for t in nltk.word_tokenize(utt)]
                uttLemmas = [t.dictionaryForm for t in utteranceVectorizer.lemmatize_utterance(utt)]
                uttVec = utteranceVectorizer.get_utterance_vector(utt)
                distSum = 0.0
                
                for utt2 in utts:
                    
                    # uttLemmas2 = [self.wnl.lemmatize(t).lower() for t in nltk.word_tokenize(utt2)]
                    uttLemmas2 =  [t.dictionaryForm for t in utteranceVectorizer.lemmatize_utterance(utt2)]
                    if utt2 == 'ハワイ':
                        print(utteranceVectorizer.lemmatize_utterance(utt2))
                    uttVec2 = utteranceVectorizer.get_utterance_vector(utt2)
                    
                    #dist = self.distanceMetric(uttVec,uttVec2)
                    
                    #dist = levenshtein_distance(utt.decode("utf-8"),utt2.decode("utf-8")) / float(max(len(utt), len(utt2)))
                    
                    dist = editdistance.eval(uttLemmas, uttLemmas2) / float(max(len(uttLemmas), len(uttLemmas2)))
                    
                    
                    distSum += dist
                
                aveDist = distSum / float(len(utts))
                
                if mostSimUtt == None or aveDist < minAveDistDist:
                    mostSimUtt = utt
                    minAveDistDist = aveDist
            
            self.clustToMinAveDistUtt[clustId] = mostSimUtt
            
            count += 1
            print("medoid", count, "of", len(clustToUtts))
        
    def initializeMeCab(self):
        self.mecab = MeCab.Tagger('-Ochasen')
        self.isMeCabInitialized = True

    def deinitializeMeCab(self):
        self.mecab = None
        self.isMeCabInitialized = False

    
    def save_speech_clusters(self, filename):
        header = ["topic", "clusterIds", "isOldMedoid", "isNewMedoid", "isClosestToCentroid", "utteranceIds", "utts"]
        rows = []
        
        clustIds = list(set(self.clustToUttIds.keys()))
        clustIds.sort()
        
        # create file contents
        for clustId in clustIds:
            
            for i in range(len(self.clustToUttIds[clustId])):
                row = {}
                
                uttId = self.clustToUttIds[clustId][i]
                utt = self.clustToUtts[clustId][i]
                
                row["clusterIds"] = clustId
                row["utteranceIds"] = uttId
                row["utts"] = '"'+utt+'"'
                
                
                if i == 0 and clustId >= 0: 
                    try:
                        row["topic"] = self.clustToTopic[clustId]
                    except:
                        print ("Warning: No topic for cluster {:}!".format(clustId))
                else:
                    row["topic"] = ""
                
                
                try:
                    if clustId >= 0 and self.clustToRepUtt[clustId] == utt:
                        row["isOldMedoid"] = 1
                    else:
                        row["isOldMedoid"] = 0
                    
                    
                    if clustId >= 0 and self.clustToClosestToCentroidUtt[clustId] == utt:
                        row["isClosestToCentroid"] = 1
                    else:
                        row["isClosestToCentroid"] = 0
                    
                    
                    if clustId >= 0 and self.clustToMinAveDistUtt[clustId] == utt:
                        row["isNewMedoid"] = 1
                    else:
                        row["isNewMedoid"] = 0
                
                except Exception:
                    print("Warning: clust", clustId)
                    # print(str(e))
                    
                
                rows.append(row)
        
        # write to file
        with open(filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
        
            writer.writeheader()
            
            for row in rows:
                writer.writerow(row)
        
    
    
    def save_representative_utterances(self, filename):
        
        header = ["clustId","clustSize","oldMedoid","newMedoid","closestToCentroid"]
        rows = []
        
        for clustId in self.clustToUtts:
            
            row = {"clustId":clustId,
                   "clustSize":len(self.clustToUtts[clustId]),
                   "oldMedoid":self.clustToRepUtt[clustId],
                   "newMedoid":self.clustToMinAveDistUtt[clustId],
                   "closestToCentroid":self.clustToClosestToCentroidUtt[clustId]
                   }
            rows.append(row)
        
        # write to file
        with open(filename, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
        
            writer.writeheader()
            
            for row in rows:
                writer.writerow(row)
            
            


if __name__ == '__main__':
    
    # sessionDir = tools.create_session_dir("representativeUtteranceFinder")

    USELOCATION = True
    testNum = '1'
    weightNum = ''
    dataAnalysisDirDate="20230615"
    savePath = os.path.join("utt_csv/",dataAnalysisDirDate,'test{}'.format(testNum))
    os.makedirs("utt_csv/"+dataAnalysisDirDate,exist_ok=True)
    os.makedirs("utt_csv/"+savePath,exist_ok=True)
    #
    # load the raw sequences
    #
    # dialogMap, cUniqueUtterances, aUniqueUtterances, cAllUtterances, aAllUtterances = tools.load_raw_sequences(tools.dataDir + "new raw sequences - replace 40.csv")

    """
    #
    # load the agent speech clusters
    #
    aClustToTopic, aClustToUttIds, aClustToUtts, aClustToRepUtt = tools.load_r_clusters(tools.modelDir + "action clusterings/new travel agent clusters 4 - with rs - renumbered.csv")
    aUttToClust = tools.get_utt_to_clust_map(aClustToUtts)
    
    
    #
    # train the vectorizer
    #
    agentUtteranceVectorizer = UtteranceVectorizer("agent", aAllUtterances, minCount=0, keywordWeight=2.0)
    
    
    #
    # find the new rep utts
    #
    aRepUttFinder = RepresentativeUtteranceFinder(aClustToTopic, aClustToUttIds, aClustToUtts, aClustToRepUtt, 
                                                 agentUtteranceVectorizer, distanceMetric=cosine, tfidf=False)
    
    aRepUttFinder.save_speech_clusters(sessionDir+"/agent speech clusters.csv")
    
    aRepUttFinder.save_representative_utterances(sessionDir+"/agent representative utterances - old lev cent.csv")
    
    tools.save_representative_utterances(aRepUttFinder.clustToClosestToCentroidUtt, sessionDir+"/agent representative utterances - closest to centroid tf cosine kw2 mc0.csv")
    tools.save_representative_utterances(aRepUttFinder.clustToMinAveDistUtt, sessionDir+"/agent representative utterances - levenshtein normalized medoid.csv")
    
    
        
    """
    #
    # load the customer speech clusters
    #
    # cClustToTopic, cClustToUttIds, cClustToUtts, cClustToRepUtt = tools.load_r_clusters(tools.modelDir + "action clusterings/new customer clusters 4 - with rs 1 - lev rep utt - merged - renumbered.csv")
    # cUttToClust = tools.get_utt_to_clust_map(cClustToUtts)
    
    clustersInputFilename = os.path.join('utt_csv/',dataAnalysisDirDate,'test'+str(testNum),str(weightNum),"shopkeeper_clusters_measurement.csv")
    print("Loading cluster data from files %s..." % clustersInputFilename)
    clusters = {} # utterances ->　list of Cluster objects
    representativeElements = {} # clusterID -> Cluster object
    clusterSizes = {} # clusterID -> size
    numUtterances = 0



    clustToTopic = {}
    clustToUttIds = {}
    clustToUtts = {}
    clustToRepUtt = {}
    
    with open(clustersInputFilename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            clusterID = int(row['originalClusterIds'])
            if clusterID not in clusterSizes: clusterSizes[clusterID] = 0
            clusterSizes[clusterID] += 1
            isRepresentative = True if int(row['isRepresentative']) == 1 else False
            utteranceID = int(row['utteranceIds'])
            utterance = row['utts'].strip()
            topic = ''
            if clusterID not in clustToUttIds:
                
                clustToUttIds[clusterID] = []
                clustToUtts[clusterID] = []
            
            clustToUttIds[clusterID].append(utteranceID)
            clustToUtts[clusterID].append(utterance)
            
            if topic != "":
                clustToTopic[clusterID] = topic
            
            if isRepresentative == 1:
                clustToRepUtt[clusterID] = utterance
            # if USELOCATION:
            #     location = row['locations']
            #     location = location.strip('][').strip()
            #     location = location.split('.')
            #     # location = list(filter(None,location.split('0')))
            #     for i_location in range(len(location)):
            #         if location[i_location].strip() != '0' :
            #             location = i_location
            #             break
            #     utterance = utterance+str(location)
            #     cluster = ClusterData(clusterID, isRepresentative, utteranceID, utterance,uttLocation=location)
            # else:
            #     cluster = ClusterData(clusterID, isRepresentative, utteranceID, utterance)

            # if utterance not in clusters: clusters[utterance] = []
            # clusters[utterance].append(cluster)
            # if isRepresentative:
            #     representativeElements[clusterID] = cluster
            numUtterances += 1
    print("loaded", numUtterances, "utterances and", len(representativeElements), "clusters")

    #
    # train the vectorizer
    #
    # customerUtteranceVectorizer = UtteranceVectorizer("customer", cAllUtterances, minCount=0, keywordWeight=3.0)

    # some of the utteracne has empty lemma in the clustering vectorizer so use the vectorizer for prediction here
    vectorizerFilename = os.path.join('utt_csv/',dataAnalysisDirDate,'test'+str(testNum),"shopkeeper_utterance_vectorizer_for_prediction.pkl")

    customerUtteranceVectorizer = pickle.load(open(vectorizerFilename,'rb'))
    
    #
    # find the new rep utts
    #
    # cRepUttFinder = RepresentativeUtteranceFinder(cClustToTopic, cClustToUttIds, cClustToUtts, cClustToRepUtt, 
    cRepUttFinder = RepresentativeUtteranceFinder(clustToTopic, clustToUttIds, clustToUtts, clustToRepUtt, 
                                                 customerUtteranceVectorizer, distanceMetric=cosine, tfidf=False)
    
    cRepUttFinder.save_speech_clusters(os.path.join('utt_csv/',dataAnalysisDirDate,'test'+str(testNum),str(weightNum))+"/skp speech clusters.csv")
    
    cRepUttFinder.save_representative_utterances(os.path.join('utt_csv/',dataAnalysisDirDate,'test'+str(testNum),str(weightNum))+"/skp representative utterances - old lev cent.csv")
    
    # tools.save_representative_utterances(cRepUttFinder.clustToClosestToCentroidUtt, sessionDir+"/customer representative utterances - closest to centroid tf cosine kw3 mc0.csv")
    # tools.save_representative_utterances(cRepUttFinder.clustToMinAveDistUtt, sessionDir+"/customer representative utterances - levenshtein normalized medoid.csv")
    
    
    
    
