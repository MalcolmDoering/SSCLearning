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



expLogDir = tools.logDir+"/"+expLogName



sessionDir = sessionDir = tools.create_session_dir("analysis_01")



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


def save_1_metric_graph(runIdToData, metricName):
    
    datasets = ["Training", "Validation", "Testing"]
    
    cmap = plt.get_cmap("tab20")
    colors = list(cmap.colors)
    runIdToColor = {}
    
    fig, axes = subplots(1, 3, sharex='col', sharey='row')
    
    for i in range(len(datasets)):
        ds = datasets[i]
        
        labels = []
        
        for rId, df in runIdToData.items():
            
            df.plot(x="Epoch", y="{} {} ({})".format(ds, metricName, rId), kind="line", ax=axes[i], legend=None) #, xlim=(0,1500), ylim=(0,0.1))
            labels.append(rId)
        
            axes[i].set_title(ds)
            axes[i].set_xlabel(None)
    
    
    axes[i].legend(labels);
    
    fig.suptitle(metricName)
    
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel("Epoch")
    
    fig.savefig((sessionDir+"/{} Average.png".format(metricName.replace(" Ave", ""))), format="png")
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
if "baseline1" in expLogName:
    save_1_metric_graph(runIdToData, "Loss Ave")
    save_1_metric_graph(runIdToData, "Action Accuracy")

    save_1_metric_graph(runIdToData, "Action 1 Precision")
    save_1_metric_graph(runIdToData, "Action 1 Recall")
    save_1_metric_graph(runIdToData, "Action 1 F-score")

    save_1_metric_graph(runIdToData, "Action 0 Precision")
    save_1_metric_graph(runIdToData, "Action 0 Recall")
    save_1_metric_graph(runIdToData, "Action 0 F-score")


# for baseline 2
elif "baseline2" in expLogDir or "baseline3" in expLogDir:
    save_1_metric_graph(runIdToData, "Loss Ave")
    save_1_metric_graph(runIdToData, "Speech Loss Ave")
    save_1_metric_graph(runIdToData, "Motion Loss Ave")
    save_1_metric_graph(runIdToData, "Action Accuracy")
    save_1_metric_graph(runIdToData, "Speech Accuracy")
    save_1_metric_graph(runIdToData, "Spatial Accuracy")



print("Done.")