#
# Created on Thu Jun 29 2023
#
# Copyright (c) 2023 Malcolm Doering
#

import csv
import os
import numpy as np
import pickle
from collections import OrderedDict
from sklearn.metrics import accuracy_score

import tools


os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import learning



evaluationDataDir = tools.dataDir + "20230628-122249_actionPredictionPreprocessing/"

interactionDataFilename = "20230626-162715_speechPreprocessing/20230623_SSC_3_trueMotionTargets_3_speechMotionCombined.csv"
speechClustersFilename = "20230620-132426_speechClustering/all_shopkeeper_cos_3gram- speech_clusters.csv"
keywordsFilename = tools.modelDir + "20230609-141854_unique_utterance_keywords.csv"
uttVectorizerDir = tools.modelDir + "20230627-163306_actionPredictionPreprocessing/"
stoppingLocationClusterDir = tools.modelDir + "20230627_stoppingLocationClusters/"


sessionDir = tools.create_session_dir("actionPrediction_01")



#
# load the data
#
print("Loading data...")

outputActionIDs = np.load(evaluationDataDir+"outputActionIDs.npy")
outputSpeechClusterIDs = np.load(evaluationDataDir+"outputSpeechClusterIDs.npy")
outputSpatialInfo = np.load(evaluationDataDir+"outputSpatialInfo.npy")
toIgnore = np.load(evaluationDataDir+"toIgnore.npy")
isHidToImitate = np.load(evaluationDataDir+"isHidToImitate.npy")
#inputVectorsNonSpeech = np.load(evaluationDataDir+"inputVectorsNonSpeech.npy")
#inputVectorsSpeech = np.load(evaluationDataDir+"inputVectorsSpeech.npy")
inputVectorsCombined = np.load(evaluationDataDir+"inputVectorsCombined.npy")

humanReadableInputsOutputs, humanReadableInputsOutputsFieldnames = tools.load_interaction_data(evaluationDataDir+"humanReadableInputsOutputs.csv")



#
# setup the neural network
#

# parameters
numEpochs = 100
evalEvery = 1

inputDim = inputVectorsCombined.shape[2]
inputSeqLen = inputVectorsCombined.shape[1]
numOutputClasses = 2
batchSize = 8
embeddingSize = 100
seed = 0

outputClassWeights = np.ones((2))
outputMasks = np.ones(isHidToImitate.shape)


# TODO: split train and test set (8 folds)
# TODO: run on multiple GPUs
# TODO: log the results
# TODO: do the same for the action cluster prediction
# TODO: attention network?




simpleFeedforwardNetwork = learning.SimpleFeedforwardNetwork(inputDim, 
                                                                inputSeqLen, 
                                                                numOutputClasses,
                                                                batchSize, 
                                                                embeddingSize,
                                                                seed,
                                                                outputClassWeights)


for e in range(numEpochs+1):

    if e != 0:
        simpleFeedforwardNetwork.train(inputVectorsCombined, isHidToImitate, outputMasks)

    if e % evalEvery == 0 or e == numEpochs:
        lossTotal, lossAction = simpleFeedforwardNetwork.get_loss(inputVectorsCombined, isHidToImitate, outputMasks)
        predAction = simpleFeedforwardNetwork.predict(inputVectorsCombined, isHidToImitate, outputMasks)

        predAcc = accuracy_score(isHidToImitate[:predAction.shape[0],], predAction)

        print(e, predAcc, np.mean(lossTotal), np.std(lossTotal))





print("Done.")


