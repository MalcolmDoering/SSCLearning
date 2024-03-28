'''
Created on July 7, 2023

@author: Malcolm


read the csv log files from the training runs and create graphs for each column
copied from DB learning analysis9
for visualizing output of actionprediction_02
'''


import os
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from matplotlib.pyplot import subplots

import tools


#expLogName = "20230731-173119_actionPrediction_02/baseline1" # binary prediction of whether S2 acts or not
#expLogName = "20230731-173119_actionPrediction_02/baseline2" # prediction of S2 actions

# with action class outputs
#expLogName = "20230807-191725_actionPrediction_02_/baseline1" # binary prediction of whether S2 acts or not
#expLogName = "20230807-191725_actionPrediction_02_/baseline2" # prediction of S2 actions

# with speech and motion class outputs
#expLogName = "20230808-163410_actionPrediction_02_/baseline3" # prediction of S2 actions
expLogName = "20231120-171349_actionPrediction_02/baseline1" # prediction of S2 actions
expLogName = "20231120-171349_actionPrediction_02_/baseline3"
expLogName = "20231122-174002_actionPrediction_02/baseline3" # 800 hidden, 1e-4 learning rate
expLogName = "20231124-104954_actionPrediction_02/baseline3" # 800 hidden, 1e-3 learning rate
expLogName = "20231124-134508_actionPrediction_02/baseline3" # 800 hidden, 1e-3 learning rate, attention
expLogName = "20231124-144932_actionPrediction_02/baseline3" # 800 hidden, 1e-3 learning rate, attention, 2000 epochs
expLogName = "20231127-133651_actionPrediction_02/baseline3" # 800 hidden, 1e-4 learning rate, attention, 2000 epochs

expLogName = "20231129-164249_actionPrediction_02/baseline3" # 800 hidden, 1e-4 learning rate, attention, 2000 epochs, expIDs randomized
expLogName = "20231129-164249_actionPrediction_02/baseline3" # 800 hidden, 1e-4 learning rate, attention, 2000 epochs, expIDs randomized, training randomized

expLogName = "20231130-165457_actionPrediction_02/baseline3" # 800 hidden, 1e-5 learning rate, attention, 2000 epochs, expIDs randomized, training randomized
expLogName = "20231201-124435_actionPrediction_02/baseline3" # 800 hidden, 1e-5 learning rate, attention, 2000 epochs, expIDs randomized, training randomized, 3 input len

expLogName = "20231205-145358_actionPrediction_02/baseline3" # 800 hidden, 1e-5 learning rate, no attention, 500 epochs, expIDs randomized, training randomized, 1 input len
expLogName = "20231206-104339_actionPrediction_02/baseline2" # 800 hidden, 1e-5 learning rate, no attention, 500 epochs, expIDs randomized, training randomized, 1 input len, action clusters
#expLogName = "20231206-153533_actionPrediction_02/baseline2" # 800 hidden, 1e-5 learning rate, no attention, 500 epochs, expIDs randomized, training randomized, 3 input len, action clusters

expLogName = "20231207-105917_actionPrediction_02/baseline2" # 800 hidden, 1e-5 learning rate, no attention, 500 epochs, expIDs randomized, training randomized, 3 input len, action clusters, fixed input embedding
expLogName = "20231207-160643_actionPrediction_02/baseline2" # 800 hidden, 1e-5 learning rate, no attention, 500 epochs, expIDs randomized, training randomized, 3 input len, action clusters, fixed input embedding (accidentally ran it twice)

expLogName = "20231208-121643_actionPrediction_02/baseline2" # 800 hidden, 1e-5 learning rate, with attention, 500 epochs, expIDs randomized, training randomized, 3 input len, action clusters, fixed input embedding
expLogName = "20231208-180817_actionPrediction_02/baseline2" # 1200 hidden, 1e-5 learning rate, with attention, 500 epochs, expIDs randomized, training randomized, 3 input len, action clusters, fixed input embedding
expLogName = "20231214-152749_actionPrediction_02/baseline2" # 1200 hidden, 5e-6 learning rate, with attention, 500 epochs, expIDs randomized, training randomized, 3 input len, action clusters, fixed input embedding

expLogName = "20240124-115548_actionPrediction_02/baseline5" # 1200 hidden, 1e-4 learning rate, no attention, 1-1-1 layers, 1000 epochs, mementar

expLogName = "20240124-194354_actionPrediction_02/baseline5" # 1200 hidden, 1e-4 learning rate, no attention, 1-1-1 layers, 1000 epochs, mementar, loss sum over mask sum
expLogName = "20240125-113525_actionPrediction_02/baseline5" # 1200 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 1000 epochs, mementar, loss sum over mask sum
expLogName = "20240202-191816_actionPrediction_02/baseline5" # 1200 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 1000 epochs, mementar full input, loss sum over mask sum

#expLogName = "20240125-185053_actionPrediction_02/baseline6" # 1200 hidden, 1e-4 learning rate, no attention, 1-1-1 layers, 500 epochs, mementar, loss sum over mask sum, both shopkeepers

#expLogName = "20240206-191349_actionPrediction_02/baseline2" # 1200 hidden, 1e-4 learning rate, no attention, 1-1-1 layers, 500 epochs, original, loss sum over mask sum, S2 only
#expLogName = "20240207-132053_actionPrediction_02/baseline2" # 1200 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 500 epochs, original, loss sum over mask sum, S2 only

expLogName = "20240207-160400_actionPrediction_02/baseline1" # 1200 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 50 epochs, original, loss sum over mask sum, predicting when S2 acts



expLogName = "20240215-124211_actionPrediction_02/baseline5" # 1200 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar full, loss sum over mask sum, action prediction
expLogName = "20240215-124211_actionPrediction_02/baseline7" # 1200 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar full, loss sum over mask sum, predicting when S2 acts


expLogName = "20240226-175614_actionPrediction_02/baseline4" # 1200 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, both shopkeepers


# current push
expLogName = "20240227-171518_actionPrediction_02/baseline1" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, predicting when S2 acts
expLogName = "20240227-171518_actionPrediction_02/baseline2" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, action prediction
expLogName = "20240227-171518_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, both shopkeepers
expLogName = "20240227-171518_actionPrediction_02/baseline5" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, action prediction
#expLogName = "20240227-171518_actionPrediction_02/baseline6" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, both shopkeepers
#expLogName = "20240227-171518_actionPrediction_02/baseline7" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, predicting when S2 acts
"""
expLogName = "20240304-154651_actionPrediction_02/baseline1" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, predicting when S2 acts
expLogName = "20240304-154651_actionPrediction_02/baseline2" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, action prediction
expLogName = "20240304-154651_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, both shopkeepers

expLogName = "20240306-144838_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, both shopkeepers. only S2 after e150
expLogName = "20240306-185213_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 500 epochs, non-mementar, 3 len input, loss sum over mask sum, both shopkeepers. only S2 after e250

expLogName = "20240307-111829_actionPrediction_02/baseline2" # 1000 hidden, 1e-5 learning rate, no attention, 3-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, action prediction
expLogName = "20240307-111829_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, no attention, 3-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, both shopkeepers

expLogName = "20240307-183915_actionPrediction_02/baseline2" # 1000 hidden, 1e-5 learning rate, with attention, 3-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, action prediction
expLogName = "20240307-183915_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, with attention, 3-1-1 layers, 200 epochs, non-mementar, 3 len input, loss sum over mask sum, both shopkeepers

expLogName = "20240314-152228_actionPrediction_02/baseline1" # 1000 hidden, 1e-5 learning rate, no attention, 3-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, predicting when S2 acts
#expLogName = "20240314-152228_actionPrediction_02/baseline2" # 1000 hidden, 1e-5 learning rate, no attention, 3-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, action prediction
#expLogName = "20240314-152228_actionPrediction_02/baseline4" # 1000 hidden, 1e-5 learning rate, no attention, 3-1-1 layers, 200 epochs, non-mementar, loss sum over mask sum, both shopkeepers
"""

# with non action states removed
expLogName = "20240318-175457_actionPrediction_02/baseline5" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, action prediction
#expLogName = "20240318-175457_actionPrediction_02/baseline6" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, both shopkeepers
#expLogName = "20240318-175457_actionPrediction_02/baseline7" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, predicting when S2 acts

expLogName = "20240325-133211_actionPrediction_02_testxy/baseline5" # 1000 hidden, 1e-5 learning rate, no attention, 1-1-1 layers, 200 epochs, mementar, loss sum over mask sum, action prediction





descriptor = "inputlen=1, 1layerencoding "

if "baseline1" in expLogName:
    descriptor += "S2 Turn Taking"
elif "baseline2" in expLogName:
    descriptor += "S2 Action"
elif "baseline4" in expLogName:
    descriptor += "S1+S2 Action"
elif "baseline5" in expLogName:
    descriptor += "S2 Action w/ KnowMan."
elif "baseline6" in expLogName:
    descriptor += "S1+S2 Action w/ KnowMan."
elif "baseline7" in expLogName:
    descriptor += "S2 Turn Taking w/ KnowMan."


maxEpoch = None
#maxEpoch = 200


expLogDir = tools.logDir+"/"+expLogName



sessionDir = tools.create_session_dir("analysis_01_" + expLogName.split("/")[-1])


def plot_2_conditions_3_metrics(runIdToData, runIds, metric1Name, metric2Name, metric3Name):
    
    fig, axes = plt.subplots(3, 2, sharex='col', sharey='row')
    
    
    cmap = plt.get_cmap("tab20")
    colors = list(cmap.colors)
    runIdToColor = {}
    
    ymax = 1.05
    
    i = 0
    for runId in runIdToData:
        runIdToColor[runId] = colors[i % len(colors)]
        i += 1
        
        
        # training
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            metric1Ymax = 100
        else:
            metric1Ymax = ymax
        
        runIdToData[runId].plot(x="Epoch", y="Training {} ({})".format(metric1Name, runId), ax=axes[0,0],
                                color=runIdToColor[runId],
                                legend=None,
                                label=runId) 
                                #ylim=[0, metric1Ymax])
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Training {} ({})".format(metric2Name, runId), ax=axes[1,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Training {} ({})".format(metric3Name, runId), ax=axes[2,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
            
        # testing
        
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            metric1Ymax = 100
        else:
            metric1Ymax = ymax
        
        
        runIdToData[runId].plot(x="Epoch", y="Testing {} ({})".format(metric1Name, runId), ax=axes[0,1],
                                color=runIdToColor[runId],
                                legend=None) 
                                #ylim=[0, metric1Ymax])
        
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Testing {} ({})".format(metric2Name, runId), ax=axes[1,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Testing {} ({})".format(metric3Name, runId), ax=axes[2,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
    
    
    
    plt.legend(runIds,
               loc="upper center",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               title="Run Parameters - rs (random seed)",
               
               # for 120 run gridsearch
               #ncol=12,
               #bbox_to_anchor=(-0.05, -0.2)
               
               # for 8 runs
               ncol=8,
               bbox_to_anchor=(0, -.5)
               )
    
    
        #
    # plot the prob for teacher forcing
    #
    for runId in runIds:
        """
        try:
            axes2_00 = axes[0,0].twinx()  # instantiate a second axes that shares the same x-axis
            axes2_10 = axes[1,0].twinx()
            axes2_20 = axes[2,0].twinx()
            
            axes2_01 = axes[0,1].twinx()
            axes2_11 = axes[1,1].twinx()
            axes2_21 = axes[2,1].twinx()
            
            
            axes2_00.set_ylim(0, 1)
            axes2_10.set_ylim(0, 1)
            axes2_20.set_ylim(0, 1)
            
            axes2_01.set_ylim(0, 1)
            axes2_11.set_ylim(0, 1)
            axes2_21.set_ylim(0, 1)
            
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_00,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_10,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_20,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_01,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_11,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_21,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            
            axes2_11.set_ylabel("Teacher Forcing Decay Schedule", rotation=90, size='medium')
            
        except:
            pass
        """
    
    
        
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    
    # for 120 run gridsearch
    #plt.subplots_adjust(bottom=.3)
    
    # for 8 runs
    plt.subplots_adjust(bottom=.2)
    
    
    cols = ["Training", "Testing"]
    hits = [metric1Name, metric2Name, metric3Name]
    
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axes[:,0], hits):
        ax.set_ylabel(row, rotation=90, size='medium')
    
    
    
    plt.subplots_adjust(wspace=.1, hspace=.05)
    #fig.tight_layout()
    
    
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i,j].xaxis.set_tick_params(which='both', direction="in", length=5)
            axes[i,j].yaxis.set_tick_params(which='both', direction="in", length=5)
            
    
    
    plt.show()



def plot_2_conditions_4_metrics(runIdToData, runIds, metric1Name, metric2Name, metric3Name, metric4Name):
    
    fig, axes = plt.subplots(4, 2, sharex='col', sharey='row')
    
    
    cmap = plt.get_cmap("tab20")
    colors = list(cmap.colors)
    runIdToColor = {}
    
    ymax = 1.05
    
    i = 0
    for runId in runIdToData:
        runIdToColor[runId] = colors[i % len(colors)]
        i += 1
        
        
        # training
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            metric1Ymax = 100
        else:
            metric1Ymax = ymax
        
        runIdToData[runId].plot(x="Epoch", y="Training {} ({})".format(metric1Name, runId), ax=axes[0,0],
                                color=runIdToColor[runId],
                                legend=None,
                                label=runId) 
                                #ylim=[0, metric1Ymax])
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Training {} ({})".format(metric2Name, runId), ax=axes[1,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Training {} ({})".format(metric3Name, runId), ax=axes[2,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        runIdToData[runId].plot(x="Epoch", y="Training {} ({})".format(metric4Name, runId), ax=axes[3,0],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
            
        # testing
        
        # graph Cost Ave
        if metric1Name == "Cost Ave":
            metric1Ymax = 100
        else:
            metric1Ymax = ymax
        
        
        runIdToData[runId].plot(x="Epoch", y="Testing {} ({})".format(metric1Name, runId), ax=axes[0,1],
                                color=runIdToColor[runId],
                                legend=None) 
                                #ylim=[0, metric1Ymax])
        
        
        # graph Substring Correct All
        runIdToData[runId].plot(x="Epoch", y="Testing {} ({})".format(metric2Name, runId), ax=axes[1,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        # graph Substring Correct Ave
        runIdToData[runId].plot(x="Epoch", y="Testing {} ({})".format(metric3Name, runId), ax=axes[2,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
        
        
        runIdToData[runId].plot(x="Epoch", y="Testing {} ({})".format(metric4Name, runId), ax=axes[3,1],
                                color=runIdToColor[runId],
                                legend=None,
                                ylim=[0, ymax])
    
    
    
    plt.legend(runIds,
               loc="upper center",   # Position of legend
               borderaxespad=0.1,    # Small spacing around legend box
               title="Run Parameters - rs (random seed)",
               
               # for 120 run gridsearch
               #ncol=12,
               #bbox_to_anchor=(-0.05, -0.2)
               
               # for 8 runs
               ncol=4,
               bbox_to_anchor=(0, -.5)
               )
    
    
    #
    # plot the prob for teacher forcing
    #
    for runId in runIds:
        """
        try:
            axes2_00 = axes[0,0].twinx()  # instantiate a second axes that shares the same x-axis
            axes2_10 = axes[1,0].twinx()
            axes2_20 = axes[2,0].twinx()
            
            axes2_01 = axes[0,1].twinx()
            axes2_11 = axes[1,1].twinx()
            axes2_21 = axes[2,1].twinx()
            
            
            axes2_00.set_ylim(0, 1)
            axes2_10.set_ylim(0, 1)
            axes2_20.set_ylim(0, 1)
            
            axes2_01.set_ylim(0, 1)
            axes2_11.set_ylim(0, 1)
            axes2_21.set_ylim(0, 1)
            
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_00,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_10,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_20,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_01,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_11,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            runIdToData[runId].plot(x="Epoch", y="Teacher Forcing Probability", ax=axes2_21,
                                    color="black",
                                    linestyle='dashed',
                                    linewidth=1,
                                    legend=None)
            
            
            axes2_11.set_ylabel("Teacher Forcing Decay Schedule", rotation=90, size='medium')
            
        except:
            pass
        """
    
    
        
    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    
    # for 120 run gridsearch
    #plt.subplots_adjust(bottom=.3)
    
    # for 8 runs
    plt.subplots_adjust(bottom=.2)
    
    
    cols = ["Training", "Testing"]
    hits = [metric1Name, metric2Name, metric3Name, metric4Name]
    
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    
    for ax, row in zip(axes[:,0], hits):
        ax.set_ylabel(row, rotation=90, size='medium')
    
    
    
    plt.subplots_adjust(wspace=.1, hspace=.05)
    #fig.tight_layout()
    
    
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i,j].xaxis.set_tick_params(which='both', direction="in", length=5)
            axes[i,j].yaxis.set_tick_params(which='both', direction="in", length=5)
            
    
    
    plt.show()


def save_1_metric_graph(runIdToData, metricName, shopkeeper=None):
    
    datasets = ["Training", "Validation", "Testing"]
    
    cmap = plt.get_cmap("tab20")
    colors = list(cmap.colors)
    runIdToColor = {}
    
    fig, axes = subplots(1, 3, sharex='col', sharey='row')
    
    for i in range(len(datasets)):
        ds = datasets[i]
        
        labels = []
        
        for rId, df in runIdToData.items():
            if maxEpoch != None:
                df = df[:maxEpoch]

            if shopkeeper == None:
                y="{} {} ({})".format(ds, metricName, rId)
            else:
                y="{} {} {} ({})".format(shopkeeper, ds, metricName, rId)

            df.plot(x="Epoch", y=y, kind="line", ax=axes[i], legend=None) #, xlim=(0,1500), ylim=(0,0.1))
            labels.append(rId)

            axes[i].set_title(ds)
            axes[i].set_xlabel(None)

            padding = 0.01

            if "Loss" in metricName:
                #axes[i].set_ylim(-1*padding, 8.0 + padding)
                #axes[i].set_ylim(-1*padding, 1.0 + padding)
                pass
            else:
                axes[i].set_ylim(-1*padding, 1.0 + padding)
    
    axes[i].legend(labels);
    
    if shopkeeper == None:
        fig.suptitle(metricName+" | "+descriptor)
    else:
        fig.suptitle(shopkeeper+" "+metricName+" | "+descriptor)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Epoch")
    
    if shopkeeper == None:        
        figFileName = sessionDir+"/{} Average.png".format(metricName.replace(" Ave", ""))
    else:
        figFileName = sessionDir+"/{} {} Average.png".format(shopkeeper, metricName.replace(" Ave", ""))

    fig.savefig(figFileName, format="png")
    #plt.show()




#
# read the data 
#

runDirContents = os.listdir(expLogDir)
runIds = []

for rdc in runDirContents:
    if "." not in rdc:
        runIds.append(rdc)

runIds.sort()


temp = []
"""
for rdn in runIds:
    
    #if "ct3_" in rdn and rdn.endswith("at2"):
    if rdn.endswith("tf1.0"):
        temp.append(rdn)

runIds = temp
"""

# this will contain the data from all the csv log files
runIdToData = {}

for iId in runIds:
    
    #runIdToData[rdn] = pd.read_csv("{}/{}/session_log_{}.csv".format(expLogDir, rdn, rdn))
    runIdToData[iId] = pd.read_csv("{}/fold_log_{}.csv".format(expLogDir, iId))




#
# graph the data
#


 
#plot_2_conditions_3_metrics(runIdToData, runIds, "Cost Ave", "DB Substring Correct Ave", "DB Substring Correct All")

#plot_2_conditions_3_metrics(runIdToData, runIds, "Cam. Address Correct", "Attr. Address Correct", "Both Addresses Correct")

#plot_2_conditions_4_metrics(runIdToData, runIds, "Loss Ave", "Speech Cluster Correct", "Camera Index Correct", "Attribute Index Exact Match")

#plot_2_conditions_3_metrics(runIdToData, runIds, "Cost Ave", "Action ID Correct", "Attribute Index Correct")


# for baseline 1
if "baseline1" in expLogName or "baseline7" in expLogName:
    save_1_metric_graph(runIdToData, "Loss Ave")
    save_1_metric_graph(runIdToData, "Action Accuracy")

    save_1_metric_graph(runIdToData, "Action 1 Precision")
    save_1_metric_graph(runIdToData, "Action 1 Recall")
    save_1_metric_graph(runIdToData, "Action 1 F-score")

    save_1_metric_graph(runIdToData, "Action 0 Precision")
    save_1_metric_graph(runIdToData, "Action 0 Recall")
    save_1_metric_graph(runIdToData, "Action 0 F-score")


# for baseline 2
elif "baseline2" in expLogDir:
    save_1_metric_graph(runIdToData, "Loss Ave")
    save_1_metric_graph(runIdToData, "Action Accuracy")
    save_1_metric_graph(runIdToData, "Speech Accuracy")
    save_1_metric_graph(runIdToData, "Spatial Accuracy")    


# for baseline 3
elif "baseline3" in expLogDir:
    save_1_metric_graph(runIdToData, "Loss Ave")
    save_1_metric_graph(runIdToData, "Speech Loss Ave")
    save_1_metric_graph(runIdToData, "Motion Loss Ave")
    save_1_metric_graph(runIdToData, "Action Accuracy")
    save_1_metric_graph(runIdToData, "Speech Accuracy")
    save_1_metric_graph(runIdToData, "Spatial Accuracy")

# for baseline 5
elif "baseline5" in expLogDir:
    save_1_metric_graph(runIdToData, "Loss Ave")
    save_1_metric_graph(runIdToData, "Action Accuracy")
    save_1_metric_graph(runIdToData, "Speech Accuracy")
    save_1_metric_graph(runIdToData, "Spatial Accuracy")

# for baseline 6
elif "baseline6" in expLogDir or "baseline4" in expLogDir:
    save_1_metric_graph(runIdToData, "Loss Ave")
    save_1_metric_graph(runIdToData, "Action Accuracy")
    save_1_metric_graph(runIdToData, "Speech Accuracy")
    save_1_metric_graph(runIdToData, "Spatial Accuracy")

    save_1_metric_graph(runIdToData, "Loss Ave", "S1")
    save_1_metric_graph(runIdToData, "Action Accuracy", "S1")
    save_1_metric_graph(runIdToData, "Speech Accuracy", "S1")
    save_1_metric_graph(runIdToData, "Spatial Accuracy", "S1")

    save_1_metric_graph(runIdToData, "Loss Ave", "S2")
    save_1_metric_graph(runIdToData, "Action Accuracy", "S2")
    save_1_metric_graph(runIdToData, "Speech Accuracy", "S2")
    save_1_metric_graph(runIdToData, "Spatial Accuracy", "S2")


with open(sessionDir+"condition.txt", "w") as f:
    f.write(expLogDir)


print("Done.")