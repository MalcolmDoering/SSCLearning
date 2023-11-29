'''
Created on Feb 16, 2017

@author: MalcolmD
'''

import numpy as np
from scipy.spatial.distance import cosine, jaccard
import csv
from jellyfish import damerau_levenshtein_distance, levenshtein_distance, jaro_winkler
import editdistance
import MeCab

import tools
from utterancevectorizer2 import UtteranceVectorizer, Token



class RepresentativeUtteranceFinder(object):
    
    def __init__(self, clustToUtts):
        
        self.initializeMeCab()
        
        self.uttToLemmas = {}


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
                
                uttLemmas = self.lemmatize_utterance(utt)
                self.uttToLemmas[utt] = uttLemmas
                
                distSum = 0.0
                
                for utt2 in utts:
                    
                    uttLemmas2 = self.lemmatize_utterance(utt2)
                    
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
    
    
    # NOTE (amal): There is an important decision to be made here in terms of
    # how we lemmatize words, because the same lemmas are considered the same
    # words. In English, we use the "dictionary form" of words. That is fine,
    # but doesn't work well for homophones (i.e. right meaning correct and
    # right as in right-handed). However, in English homophonous dictionary
    # forms are not too common, so it is not too bad to ignore that.
    #
    # However, in Japanese there are way more homophonous words, and kanji is
    # used to separate their meaning. However, there are multiple ways to write
    # the same word, in Hiragana and Kanji (and, at times, with multiple
    # different kanjis). Therefore, just using the phonetic spelling as a lemma
    # may result in many words being lumped togther (i.e. いる as "to be", "to
    # need", and "to go in"). However, using the Kanji could result in the same
    # word being split up (i.e. みんな vs. 皆).
    #
    # For now, I will use the Kanji, because I hope Google Speech is consistent
    # in what words it puts into Kanji and what words it keeps in Hiragana.
    # However, if this becomes a problem, there are other approaches (i.e. just
    # phonetic, phonetic + part of speech (useful for しる as "to know" versus
    # soup)). Also, if it turns out that there are multiple words with the same
    # kanji but different pronunciations (i.e. 入る as いる vs. はいる), I could
    # further subdivide along kanji (i.e. kanji + part of speech, or kanji +
    # phonetic).
    #
    # NOTE that the code for this discussion is written in the __hash__ and __eq__
    # methods of the Token class.
    def lemmatize_utterance(self, utt):
        # One row of analyzedTextMatrix corresponds to one token, and the
        # columns correspond to different aspects of that token
        # (see http://www.nltk.org/book-jp/ch12.html#mecab)
        # 0 is the token as written in utt, 1 is the pronunciation of that
        # token in Katakana, 2 is the dictionary form of that token, in EITHER
        # kanji or hiragana/katakana depending on how it was written in utt,
        # 3 is part of speech, etc. Importantly, column 2 does not infer kanji
        # from hiragana/katakana.
        if not self.isMeCabInitialized:
            self.initializeMeCab()
        
        analyzedTextStr = self.mecab.parse(utt)
        analyzedTextMatrix = [line.split("\t") for line in analyzedTextStr.split("\n")]
        
        uttLemmas = []
        wordI = 0

        while len(analyzedTextMatrix[wordI]) > 1:
            #if analyzedTextMatrix[wordI][2] not in self.stopwords: # TODO Malcolm: should this be commented out? Because it will mess up the ngrams....
                # Do we want to create ngrams from the list of lemmas after stopwords have been removed? Or first create the ngrams and then remove ngrams that contain stopwords???

            token = Token(analyzedTextMatrix[wordI][0],
                        analyzedTextMatrix[wordI][1],
                        analyzedTextMatrix[wordI][2],
                        analyzedTextMatrix[wordI][3])
            uttLemmas.append(token)
            wordI += 1
        
        return uttLemmas
    

    def save_speech_clusters(self, speechClusterData, fieldnames, filename):

        for i in range(len(speechClusterData)):
            clustID = int(speechClusterData[i]["Cluster.ID"])
            utt = speechClusterData[i]["Utterance"]
            
            speechClusterData[i]["Old.Is.Representative"] = speechClusterData[i]["Is.Representative"]

            if clustID != 0:
                if self.clustToMinAveDistUtt[clustID] == utt:
                    speechClusterData[i]["Is.Representative"] = 1
                else:
                    speechClusterData[i]["Is.Representative"] = 0
            
            #speechClusterData[i]["Lemmas"] = 

        fieldnames.insert(fieldnames.index("Is.Representative") + 1, "Old.Is.Representative")

        tools.save_interaction_data(speechClusterData, filename, fieldnames)

            
            


if __name__ == '__main__':
    
    sessionDir = tools.create_session_dir("representativeUtteranceFinder")
    
    speechClusterDir = tools.dataDir + "20230731-113400_speechClustering/"
    speechClustersFilename = speechClusterDir + "all_shopkeeper- speech_clusters.csv"

    
    #
    # load the speech clusters
    #
    speechClusterData, speechClusterFieldnames = tools.load_csv_data(speechClustersFilename, isHeader=True, isJapanese=True)

    speechClustIDToUtts = {}


    for row in speechClusterData:
        speech = row["Utterance"]
        speechClustID = int(row["Cluster.ID"])

        if speechClustID not in speechClustIDToUtts:
            speechClustIDToUtts[speechClustID] = []
        
        speechClustIDToUtts[speechClustID].append(speech)


    #
    # find the new rep utts
    #
    repUttFinder = RepresentativeUtteranceFinder(speechClustIDToUtts)
    repUttFinder.save_speech_clusters(speechClusterData, speechClusterFieldnames, speechClusterDir + "all_shopkeeper- speech_clusters - levenshtein normalized medoid.csv")

