'''
Created on Feb 13, 2017

@author: MalcolmD

Modified Nov 2018 by Amal Nanavati
2022.05 by Jiang

This script is used to generate the vectorizer for speech clustering
Some part of the utterance will be ignored, like the stop words
'''
import chardet
from dataclasses import replace
import os
import MeCab
import datetime
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances
import pickle
import sys
sys.path.append('..')
import copy

import tools





class Token(object):
    def __init__(self, tokenAsUsed, pronunciationAsUsed, dictionaryForm, partOfSpeech):
        self.tokenAsUsed = tokenAsUsed
        self.pronunciationAsUsed = pronunciationAsUsed
        self.dictionaryForm = dictionaryForm
        self.partOfSpeech = partOfSpeech

# [chr(u) for u in range(48,58)]+ # English digits
#                      [chr(u) for u in range(65,91)]+ # English capital letters
#                      [chr(u) for u in range(67,123)]+ # English lower letters
#                      # [chr(u) for u in range(65296,65306)]+ # fullwidth Romaji digits
#                      [chr(u) for u in range(65313,65339)]+ # fullwidth Romaji capital letters
#                      [chr(u) for u in range(65345,65371)]+ # fullwidth Romaji lower letters
#                      [chr(u) for u in range(12353,12439)]+ # hiragama characters
#                      [chr(u) for u in range(12449,12539)] # katakana characters

    def isNumber(self):
        """
        To be a number, the token must have at least one numeric digit (0-9 roman
        or japanese, or 10, 100, or 1000 which can be used independentely without
        a digit in front of it) and may have additional numeric symbols
        """
        hasAtLeastOneNumber = False
        for character in self.dictionaryForm:
            unicodeValue = ord(character)
            # Numeric Digits
            if ((48 <= unicodeValue and unicodeValue < 58) or # English digits
                 (65313 <= unicodeValue and unicodeValue < 65306) or # fullwidth Romaji digits
                 unicodeValue == 12295 or # 〇
                 unicodeValue == 19968 or # 一
                 unicodeValue == 20108 or # 二
                 unicodeValue == 19977 or # 三
                 unicodeValue == 22235 or # 四
                 unicodeValue == 20116 or # 五
                 unicodeValue == 20845 or # 六
                 unicodeValue == 19971 or # 七
                 unicodeValue == 20843 or # 八
                 unicodeValue == 20061 or # 九
                 unicodeValue == 21313 or # 十
                 unicodeValue == 30334 or # 百
                 unicodeValue == 21315    # 千
             ):
                hasAtLeastOneNumber = True
            # Numeric Symbols
            elif (unicodeValue == 46 or # .
                unicodeValue == 65294 or # ．
                unicodeValue == 9675  or # ○
                unicodeValue == 19975 or # 万
                unicodeValue == 20740 or # 億
                unicodeValue == 20806    # 兆
            ):
                if (not hasAtLeastOneNumber): return False # A number must start with a digit, not a symbol
                continue
            else:
                return False
        return hasAtLeastOneNumber

    def isNumberSymbol(self):
        for character in self.dictionaryForm:
            unicodeValue = ord(character)
            if (not ((48 <= unicodeValue and unicodeValue < 58) or # English digits
                     (65313 <= unicodeValue and unicodeValue < 65306) or # fullwidth Romaji digits
                     unicodeValue == 46 or # .
                     unicodeValue == 65294 or # ．
                     unicodeValue == 9675  or # ○
                     unicodeValue == 12295 or # 〇
                     unicodeValue == 19968 or # 一
                     unicodeValue == 20108 or # 二
                     unicodeValue == 19977 or # 三
                     unicodeValue == 22235 or # 四
                     unicodeValue == 20116 or # 五
                     unicodeValue == 20845 or # 六
                     unicodeValue == 19971 or # 七
                     unicodeValue == 20843 or # 八
                     unicodeValue == 20061 or # 九
                     unicodeValue == 21313 or # 十
                     unicodeValue == 30334 or # 百
                     unicodeValue == 21315 or # 千
                     unicodeValue == 19975 or # 万
                     unicodeValue == 20740 or # 億
                     unicodeValue == 20806    # 兆
                     )):
                return False
        return True




class NGram(object):
    def __init__(self):
        self.tokens = []

    def addToken(self, token):
        self.tokens.append(token)

    def __hash__(self):
        tokenDictForm = [token.dictionaryForm for token in self.tokens]
        return hash(":".join(tokenDictForm))

    def __eq__(ngram1, ngram2):
        if len(ngram1.tokens) != len(ngram2.tokens):
            return False
        for i in range(len(ngram1.tokens)):
            if ngram1.tokens[i].dictionaryForm != ngram2.tokens[i].dictionaryForm:
                return False
        return True

    def __str__(self):
        tokenDictForm = [token.dictionaryForm for token in self.tokens]
        return ":".join(tokenDictForm)

    def __repr__(self):
        return str(self)

    # NOTE: By default, when sorting lists Python sorts lexicographically
    def keyForSorting(self):
        return [token.pronunciationAsUsed for token in self.tokens]

    def isNumber(self):
        hasAtLeastOneNumber = False
        for token in self.tokens:
            if token.isNumber():
                hasAtLeastOneNumber = True
            elif token.isNumberSymbol():
                if (not hasAtLeastOneNumber): return False # A number must start with a digit, not a symbol
                continue
            else:
                return False
        return hasAtLeastOneNumber




class UtteranceVectorizer(object):

    def __init__(
        self, allUtterances, importantNGramWeight=3.0, keywords=None,
        minCount=2, maxNGramLen=3, svdShare=0.5, makeKeywordsOneGrams=False,
        keywordCountThreshold=None, ngramComponents=None, keywordComponents=None,
        runLSA=False, useStopwords=True, useNoisewords=True):
        """
        Takes in all the utterances, tokenizes and lemmatizes them, converts
        them into ngrams, and generates a dictionary of ngrams where each ngram
        has one-and-only-one index

        NOTE: A max ngram of 3 at a minimum is logical because MeCab splits
        numbers with decimal points into 3 grams so 3grams allows them to stay
        together.
        """

        ########################################################################
        # Configuration Variables
        ########################################################################

        self.maxNGramLen = maxNGramLen # will take ngrams from 1 till self.maxNGramLen
        self.importantNGramWeight = importantNGramWeight
        keywords = dict() if keywords is None else keywords
        keywordCountThreshold = 0 if keywordCountThreshold is None else keywordCountThreshold
        self.initializeMeCab()

        ########################################################################
        # Stopwords
        ########################################################################
        self.stopwords = set()
        self.wordsCannotLemmas = []
        self.noisewords = []

        if useStopwords:
            # self.stopwords = set(
            #                      # [chr(u) for u in range(48,58)]+ # English digits
            #                      [chr(u) for u in range(65,91)]+ # English capital letters
            #                      [chr(u) for u in range(67,123)]+ # English lower letters
            #                      # [chr(u) for u in range(65296,65306)]+ # fullwidth Romaji digits
            #                      [chr(u) for u in range(65313,65339)]+ # fullwidth Romaji capital letters
            #                      [chr(u) for u in range(65345,65371)]+ # fullwidth Romaji lower letters
            #                      [chr(u) for u in range(12353,12439)]+ # hiragama characters
            #                      [chr(u) for u in range(12449,12539)] # katakana characters
            #                      )

            # stopwords from https://github.com/stopwords-iso/stopwords-ja/blob/master/stopwords-ja.txt
            self.stopwords = set(["あ","あっ","あの","あのかた","あの人","い","いう","います","う",
                                "え","お","および","かつて","から","が","き","さ","し","する",
                                "ず","せ","せる","そして","その他","その後","それぞれ","それで","た","ただし",
                                "たち","ため","たり","だ","だっ","だれ","つ","て","で","でき","です","では",
                                "でも","と","という","といった","とき","ところ","として","とともに","とも","と共に","どこ",
                                "どの","なお","なかっ","ながら","に","において","における","について","にて","によって","により",
                                "による","の","ので","は","ば","へ","ほか","ほとんど",
                                "ほど","ます","また","または","まで","も","もの","ものの","や","よう","より","ら","ら","られる","ね",
                                "れ","れる","を","ん","及び","彼","彼女","我々","特に","私","私達","貴方","貴方方","そうですね"
                                ,"はい","はーい","ハワイ","。","、","？","あー","なるほど",'ござる',"いただく","ちょっと","やっぱり",
                                "ぐらい","いただい","えーと","もちろん","こちら","あちら","そちら","それ","あれ","これ","そこ","ここ","あそこ"])
            self.wordsCannotLemmas = ["あそうなんですよね","そうなんですよね","そうなんですね","そうなんですよ","あそうなんですね",
                                    "あそうなんですよ","あそうなんです","そうなんです","あそうですね","そうですよね"]
        
        if useNoisewords:
            # added by Jiang, used to find the utterance that only contained these words.
            self.noisewords = ["あ","あっ","あの","あのかた","あの人","い","いう","います","う",
                                "え","お","および","かつて","から","が","き","さ","し","する",
                                "ず","せ","せる","そして","その他","その後","それぞれ","それで","た","ただし",
                                "たち","ため","たり","だ","だっ","だれ","つ","て","で","でき","です","では",
                                "でも","と","という","といった","とき","ところ","として","とともに","とも","と共に","どこ",
                                "どの","なお","なかっ","ながら","に","において","における","について","にて","によって","により",
                                "による","の","ので","は","ば","へ","ほか","ほとんど",
                                "ほど","ます","また","または","まで","も","もの","ものの","や","よう","より","ら","ら","られる","ね",
                                "れ","れる","を","ん","及び","彼","彼女","我々","特に","私","私達","貴方","貴方方","そうですね"
                                ,"はい","はーい","ハワイ","。","、","？","あー","なるほど",'ござる',"いただく","ちょっと","やっぱり",
                                "ぐらい","いただい","えーと","ございます"]
        
        print("stopwords", self.stopwords)

        ########################################################################
        # Find which n-grams occur and their count
        ########################################################################
        print("finding n-grams")
        # 2D Dict: nGramSize (int) -> (nGramText (str) -> count (int))
        ngrams = {n:{} for n in range(1, self.maxNGramLen+1)}
        lemmasLen = []
        for utt in allUtterances:

            # lemmatize the words
            uttLemmas = self.lemmatize_utterance(utt)
            lemmasLen.append(len(uttLemmas))
            for n in range(1, self.maxNGramLen+1):
                for wordI in range(len(uttLemmas)-(n-1)):
                    ngram = NGram()
                    for i in range(n):
                        ngram.addToken(uttLemmas[wordI + i])
                    if ngram not in ngrams[n]:
                        ngrams[n][ngram] = 0
                    ngrams[n][ngram] += 1
        print(np.mean(lemmasLen),np.std(lemmasLen))
        # for i in ngrams:
        #     for ngram in ngrams[i]:
        #         for token in ngram.tokens:
        #             print(token.dictionaryForm+' ',end='',file=open('test.txt','a+'))
        #         print(ngrams[i][ngram],file=open('test.txt','a+'))
        ########################################################################
        # Process the keywords
        ########################################################################
        print("processing keywords")
        
        processedKeywords = dict() # nGramText (str) -> count (int)
        for kw, count in keywords.items():
            kwLemmas = self.lemmatize_utterance(kw) # Split it
            if (makeKeywordsOneGrams):
                for kwLemma in kwLemmas:
                    ngram = NGram() # By default, all keywords will be 1-grams
                    ngram.addToken(kwLemma)
                    if ngram not in processedKeywords:
                        processedKeywords[ngram] = 0
                    processedKeywords[ngram] += count
            else:
                if (len(kwLemmas) > maxNGramLen):  continue # The Keyword Is Too Long
                ngram = NGram() # By default, all keywords will be n-grams
                for kwLemma in kwLemmas:
                    ngram.addToken(kwLemma)
                if ngram not in processedKeywords:
                    processedKeywords[ngram] = 0
                processedKeywords[ngram] += count
        # i = 0
        # for kw, count in sortedKWs:
        #     print(i, count, kw)
        #     i+=1
        kwsToRemove = set()
        for kw, count in processedKeywords.items():
            if count <= keywordCountThreshold: kwsToRemove.add(kw)
        for kw in kwsToRemove:
            del processedKeywords[kw]

        sortedKWs = sorted(list(processedKeywords.items()), key=lambda x: x[1], reverse=True)
        # print("Keywords: ", sortedKWs)
        # for keyword in sortedKWs:
        #     print(keyword)

        ########################################################################
        # Create the vectorization dict by removing all words that occur
        # infrequently
        ########################################################################
        self.ngramToIndex = {}
        self.indexToNgram = {}
        self.keywordToIndex = {}
        self.indexToKeyword = {}

        numberNGrams = [] # All number nGrams of count >= minCount

        # To make the end matrix a bit more understandable, I will assign IDs in order of
        # the length of the gram, and in alphabetic order by pronunciation
        # after that. For ngrams that occur multiple times, the pronunciation
        # that is used is the pronunciation of the token as written in the
        # first utterance that contained that ngram
        index = 0
        for n in range(1, self.maxNGramLen+1):
            grams = list(ngrams[n].keys())
            grams.sort(key=lambda x: x.keyForSorting())
            for gram in grams:
                if ngrams[n][gram] >= minCount:
                    self.ngramToIndex[gram] = index
                    self.indexToNgram[index] = gram
                    index += 1
                    if (gram.isNumber()):
                        numberNGrams.append(gram)
        self.numWords = index
        print("NumWords: %d" % self.numWords)

        # Assign keywords indices
        keywords = list(processedKeywords.keys())
        # All numbers are by default also keywords
        keywords.extend(numberNGrams)
        # print("Keywords: ", keywords)
        keywords.sort(key=lambda x: x.keyForSorting())
        for kw in keywords:
            if kw in self.ngramToIndex: # Only allow keywords that occur more than minCount times in the training data
                self.keywordToIndex[kw] = index
                self.indexToKeyword[index] = kw
                index += 1
        self.numKeywords = index - self.numWords
        print("NumKeywords: %d" % self.numKeywords)

        ########################################################################
        # Compute the LSA transformation
        ########################################################################
        print("getting bag of words matrix")
        # Get the bag of words matrix
        uttVecs = self.get_utterance_vectors(allUtterances)
        # separate the ngram bag of words and the keywords bag of words, as we
        # will apply LSA separately to them
        ngramUttVecs = uttVecs[:,:self.numWords]
        keywordUttVecs = uttVecs[:,self.numWords:]
        # print("NgramBOW", ngramUttVecs, ngramUttVecs.shape)
        # print("KeywordBOW", keywordUttVecs, keywordUttVecs.shape)

        if runLSA:
            print("fitting TFIDFTransformers")
            # Use it to fit a TfidfTransformer (i.e. calculate an IDF matrix)
            self.ngramTfidfTransformer = TfidfTransformer()
            ngramTfidfUttVecs = self.ngramTfidfTransformer.fit_transform(ngramUttVecs)
            self.keywordTfidfTransformer = TfidfTransformer()
            keywordTfidfUttVecs = self.keywordTfidfTransformer.fit_transform(keywordUttVecs)
            # print("NgramTFIDF", ngramTfidfUttVecs, ngramTfidfUttVecs.shape)
            # print("KeywordTFIDF", keywordTfidfUttVecs, keywordTfidfUttVecs.shape)

            # Get all the singluar values and determine the number of dimensions we
            # want to retain
            if (ngramComponents is None or keywordComponents is None):
                print("getting singularValues ngram")
                s = np.linalg.svd(ngramTfidfUttVecs.toarray(), full_matrices=False, compute_uv=False)
                print(s, np.linalg.norm(s))
                sNormalized = s / np.linalg.norm(s)
                nGramNumDim, retainedShare = 0, 0.0
                desiredShare = svdShare*sum(sNormalized)
                print("NGram SVD components: %d" % len(sNormalized))
                for sv in sNormalized:
                    retainedShare += sv
                    nGramNumDim += 1
                    if retainedShare >= desiredShare: break

                print("getting singularValues keywords")
                s = np.linalg.svd(keywordTfidfUttVecs.toarray(), full_matrices=False, compute_uv=False)
                print(s, np.linalg.norm(s))
                sNormalized = s / np.linalg.norm(s)
                keywordNumDim, retainedShare = 0, 0.0
                desiredShare = svdShare*sum(sNormalized)
                print("Keyword SVD components: %d" % len(sNormalized))
                for sv in sNormalized:
                    retainedShare += sv
                    keywordNumDim += 1
                    if retainedShare >= desiredShare: break
            else:
                nGramNumDim = ngramComponents
                keywordNumDim = keywordComponents

            print("NGram components to retain: %d" % nGramNumDim)
            print("Keyword components to retain: %d" % keywordNumDim)

            # Train a TruncatedSVD model (which is an LSA model)
            self.ngramLsaModel = TruncatedSVD(n_components=nGramNumDim)
            self.ngramLsaModel.fit(ngramTfidfUttVecs)
            self.keywordLsaModel = TruncatedSVD(n_components=keywordNumDim)
            self.keywordLsaModel.fit(keywordTfidfUttVecs)

            self.totalComponents = nGramNumDim + keywordNumDim

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
    def wordBreak(self, s):
        
        n=len(s)
        dp=[False]*(n+1)
        dp[0]=True
        for i in range(n):
            for j in range(i+1,n+1):
                if(dp[i] and (s[i:j] in self.noisewords)):
                    dp[j]=True
        return dp[-1]


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

        onlyContainStopwords = self.wordBreak(utt)

        # remove words that can not be lemmas from the utterance.
        # since the words in the list is in order of their length,
        # so it's ok to use a for loop here?
        uttLen = len(utt)
        temp_utt= utt
        if not onlyContainStopwords:
            for word in self.wordsCannotLemmas:
                temp_utt = ''.join(temp_utt.split(word))
            if not len(temp_utt)< 5:
                utt = temp_utt

        # in case the word is only with stop words after remove
        onlyContainStopwords = self.wordBreak(utt)
        
        analyzedTextStr = self.mecab.parse(utt)


        analyzedTextMatrix = [line.split("\t") for line in analyzedTextStr.split("\n")]
        uttLemmas = []
        wordI = 0

        # modified 230124
        # if utt only contain stopwords, lemmas it anyway
        # otherwise only lemmas the useful parts

        if onlyContainStopwords:
            while len(analyzedTextMatrix[wordI]) > 1:
                token = Token(analyzedTextMatrix[wordI][0],
                                analyzedTextMatrix[wordI][1],
                                analyzedTextMatrix[wordI][2],
                                analyzedTextMatrix[wordI][3])
                uttLemmas.append(token)
                wordI += 1
        else:
            while len(analyzedTextMatrix[wordI]) > 1:
                if analyzedTextMatrix[wordI][2] not in self.stopwords:
                    token = Token(analyzedTextMatrix[wordI][0],
                                analyzedTextMatrix[wordI][1],
                                analyzedTextMatrix[wordI][2],
                                analyzedTextMatrix[wordI][3])
                    uttLemmas.append(token)
                wordI += 1

        # if utt in ['黒のみのご用意でございます。',
        #         'そちら分かりかねも調べさせていただきます。',
        #         '特老の一食のみのご用意となります。',
        #         'はいこちら1000円札8歳でございますのではい動物とか間取りですね、取っていただけると思います。',
        #         'そうですね、黒色のご用意のみとなっております。いいえとんでもございません',
        #         'ぜひぜひお待ちしておりますよろしくお願いします何でもございません',
        #         'そうですねこう準備してる間にもう走りすぎてしまっている可能性がありますよね',
        #         'えーとですね。',
        #         'そうですねちょっとこちらは比較的高価にはなってきますね',
        #         'まあしっかりあの下置きガッチリとした造りになってますね',
        #         'ありがとうございます。何でもございませんまたお待ちしております',
        #         'あちらは55000円でござ。',
        #         'そうですねそうですね昨日はこちらと効果もないのでお探しの物だったらソニーが一番近い。',
        #         'とんでもございません。',
        #         'はいさようでございます',
        #         'そうですね、動物たちびっくりしちゃいますしね。逃げだったら母盗撮映画難しいですし。',
        #         'はいさようでございます',
        #         '是非宜しくお願い致します。',
        #         'はいさようでございます',
        #         '是非宜しくお願い致します。']:
        #     print(utt)
            
        #     for tok in uttLemmas:
        #         print(tok.tokenAsUsed,end=',')
        #     print()
        #     print('----------------')
        return uttLemmas

    def get_utterance_vector(self, utt):
        """
        Vectorizes an utterance such that the entry in a column indicates
        how many times that gram appears.
        """

        uttVec = np.zeros(self.numWords + self.numKeywords)

        uttLemmas = self.lemmatize_utterance(utt)

        for n in range(1, self.maxNGramLen+1):
            for wordI in range(len(uttLemmas)-(n-1)):
                ngram = NGram()
                for i in range(n):
                    ngram.addToken(uttLemmas[wordI + i])
                if ngram in self.ngramToIndex:
                    if ngram.isNumber():
                        weight = 1 # self.importantNGramWeight # removed the extra weight since I instead make all numbers keywords
                        # print("Number: ", ngram)
                    else:
                        weight = 1
                    uttVec[self.ngramToIndex[ngram]] += 1*weight # uttVec[self.ngramToIndex[ngram]] = 1*weight #
                if ngram in self.keywordToIndex:
                    # uttVec[self.keywordToIndex[ngram]] += 1*self.importantNGramWeight # uttVec[self.keywordToIndex[ngram]] = 1*self.importantNGramWeight #
                    # NOTE: There is no point multiplying a keyword entry in the
                    # appended part of the vector by a weight, because we run
                    # LSA separately on that so the result will be the same as
                    # if we did not multiply that matrix by a constant
                    uttVec[self.keywordToIndex[ngram]] += 1 # uttVec[self.keywordToIndex[ngram]] = 1 #
                else:
                    # print("The following ngram does not exist in the dictionary (likely means it occured < minCount times) %s" % ngram)
                    pass
        
        # ngramList = [self.indexToNgram[i] for i in range(len(self.numWords)) if i>0]
        return uttVec

    def get_utterance_vectors(self, uttList):
        uttVecs = np.zeros((len(uttList), self.numWords + self.numKeywords))

        for i in range(len(uttList)):
            utt = uttList[i]
            uttVecs[i] = self.get_utterance_vector(utt)

        return uttVecs

    def get_lsa_vectors(self, uttList):
        uttVecs = np.array(self.get_utterance_vectors(uttList))

        # partition the ngram and keyword components of the bag of words, to
        # do separate LSA
        ngramUttVecs = uttVecs[:,:self.numWords]
        keywordUttVecs = uttVecs[:,self.numWords:]

        # Important question -- should I be doing a fit_transform here or just
        # a transform? If I do fit_transform, I am getting the actual TFIDF
        # matrix for this set of utterances. If I am doing transform, I am
        # multiplying the TF matrix for this set of utterances by the IDF
        # matrix for the training set of utterances. MAYBE that makes sense
        # because the weights of the words should be determined by their
        # frequency in the training set. But on the other hand, does LSA work
        # properly if the input to it is not a proper TFIDF matrix?
        #
        # I think I should do transform. Because if there is only one utterance
        # here, then the TFIDF matrix will just be a TF matrix, and it won't
        # take into account the frequency or lack thereof of the words in the
        # training data. And to answer my above question, LSA is just a general
        # dimensionality reduction technique, so it will work as long as the
        # matrix is a type of count matrix, even if it is not specifically TFIDF.
        ngramTfidfUttVecs = self.ngramTfidfTransformer.transform(ngramUttVecs)
        keywordTfidfUttVecs = self.keywordTfidfTransformer.transform(keywordUttVecs)

        ngramLsaUttVecs = self.ngramLsaModel.transform(ngramTfidfUttVecs)
        keywordLsaUttVecs = self.keywordLsaModel.transform(keywordTfidfUttVecs)
        keywordLsaUttVecs = self.importantNGramWeight*keywordLsaUttVecs

        return np.hstack((ngramLsaUttVecs, keywordLsaUttVecs))




if __name__ == '__main__':

    interactionDataFn = tools.dataDir + "20230620-130816_speechPreprocessing/20230620_SSC_3_trueMotionTargets_3_speechMotionCombined.csv"
    keywordsFn = tools.modelDir + "20230609-141854_unique_utterance_keywords.csv"
    
    participant = "all_shopkeeper"
    maxNGramLen = 3
    
    descriptor = participant + "_cos_{}gram".format(maxNGramLen)


    sessionDir = tools.create_session_dir("utteranceVectorizer")
    
    interactionData, interactionDataFieldnames = tools.load_interaction_data(interactionDataFn)
    keywordData, uttToKws, keywordsList, keywordToRelevance, _ = tools.load_keywords(keywordsFn)

    if participant == "all_shopkeeper":
        utterances = [x["participant_speech"] for x in interactionData if (int(x["unique_id"]) == 1 or int(x["unique_id"]) == 3) and x["participant_speech"] != ""]
        utteranceInteractionData = [x for x in interactionData if (int(x["unique_id"]) == 1 or int(x["unique_id"]) == 3) and x["participant_speech"] != ""]

    outputVectorizationFilename = descriptor+"_vectorizedUtterances.txt"
    outputDistMatrixFilename = descriptor+"_distMatrix.txt"
    outputDictFilename = descriptor+"_dictionary.csv"
    outputUtterancesFilename = descriptor+"_utterances.txt"
    outputUtteranceVectorizerFilename = descriptor+"_utterance_vectorizer.pkl"
    outputUtteranceInteractionDataFilename = descriptor+"_utterance_interaction_data.csv"
    
    print("INPUT Utterances: %s" % interactionDataFn)
    print("OUTPUT Vectorization: %s" % outputVectorizationFilename)
    print("OUTPUT DistMatrix: %s" % outputDistMatrixFilename)
    print("OUTPUT Dicionary: %s" % outputDictFilename)
    print("OUTPUT Utterances: %s" % outputUtterancesFilename)
    print("OUTPUT Utterance Interaction Data: %s" % outputUtteranceInteractionDataFilename)




    ############################################################################
    # Vectorize the utterances
    ############################################################################
    print("Creating utterance vectorizer")
    uttVectorizer = UtteranceVectorizer(
        utterances,
        importantNGramWeight=1.0,
        keywords=keywordToRelevance,
        minCount=2, # 2 for just SK or cust - min times keyword accours
        maxNGramLen=maxNGramLen,
        svdShare=0.5,
        makeKeywordsOneGrams=False,
        keywordCountThreshold=5,  # 5 for just SK or cust
        # ngramComponents=ngramComponents,
        # keywordComponents=keywordComponents,
    )

    print("pickling the utterance vectorizer...")
    uttVectorizer.deinitializeMeCab() # Necessary because MeCab cannot get pickled. This will automatically get reinitialized when we lemmatize an utterance
    with open(sessionDir+outputUtteranceVectorizerFilename, "wb") as f:
        f.write(pickle.dumps(uttVectorizer))
    

    vectors = uttVectorizer.get_utterance_vectors(utterances)
    #vectors = uttVectorizer.get_lsa_vectors(utterances)
    #print("LSA uttVectors shape", vectors.shape)
    print("Regular uttVectors shape", vectors.shape)
    

    utteranceIndicesToRemove = []

    for uttI in range(len(utterances)):
        if not vectors[uttI].any():
            utteranceIndicesToRemove.append(uttI)
    for i in range(len(utteranceIndicesToRemove)):
        uttI = utteranceIndicesToRemove[i] - i # account for the fact that we are deleting as this loop progresses
        utterances.pop(uttI)
        utteranceInteractionData.pop(uttI)

    vectors = np.delete(vectors, utteranceIndicesToRemove, axis=0)

    print("After removing zero vectors shape", vectors.shape)


    print("computing distances...")
    distMatrix = pairwise_distances(vectors, metric="cosine", n_jobs=-1)



    ############################################################################
    # Save Dist Matrix, Utterance Vectors, and Ngram to Index Mapping
    ############################################################################
    print("Saving...")

    # Save Dist Matrix
    #testDistMatrix = distMatrix[:10000,:10000]
    #np.savetxt(os.path.join(sessionDir+descriptor+"_test_distMatrix.txt"), testDistMatrix, fmt="%.4f")
    np.savetxt(os.path.join(sessionDir+outputDistMatrixFilename), distMatrix, fmt="%.4f")
    
    # Save Vectorization
    np.savetxt(os.path.join(sessionDir+outputVectorizationFilename), vectors, fmt="%.4f")
    
    # Save Utterances
    with open(os.path.join(sessionDir+outputUtterancesFilename), "w") as txtfile:
        for utt in utterances:
            txtfile.write(u"%s\n" % utt)
    
    tools.save_interaction_data(utteranceInteractionData, sessionDir+outputUtteranceInteractionDataFilename, interactionDataFieldnames)

    # Save NGram To Index Mapping
    with open(os.path.join(sessionDir+outputDictFilename), "w", newline='', encoding='utf-8-sig', errors='ignore') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'ngram'])

        for i in range(uttVectorizer.numWords):
            writer.writerow([i, '{:}'.format(str(uttVectorizer.indexToNgram[i]))])
        for i in range(uttVectorizer.numWords,uttVectorizer.numWords+uttVectorizer.numKeywords):
            writer.writerow([i, '{:}'.format(str(uttVectorizer.indexToKeyword[i]))])
        
