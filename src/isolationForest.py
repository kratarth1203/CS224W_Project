import snap
import numpy as np

from generateNetwork import *
from dnoda import *
from sklearn.ensemble import IsolationForest

def isolationForest(epoch):
    graph = getGraphAtEpoch(epoch)
    numFeats = len(nodeFeatureNames)
    allFeats = np.zeros((graph.GetNodes(), 2*numFeats))
    featMap = {}
    i = 0
    for node in graph.Nodes():
        nid = node.GetId()
        nbrfeats = getDnodaScore(nid, epoch)
        nodefeats = getNodeFeatures(nid, graph)
        allFeats[i][:numFeats] = np.array(nodefeats)
        allFeats[i][numFeats:] = nbrfeats
        featMap[nid] = allFeats[i][:]
        i+=1
    isf = IsolationForest()
    isf.fit(allFeats)

    print "Outliers:"
    for nid in featMap:
        if isf.predict(featMap[nid].reshape((1, -1))) < 0:
            print nid

createAllGraphs('../data/mote_locs.txt', '../data/connectivity.txt', '../data/data_medium.txt')

isolationForest(graphAtEpochs.keys()[-1])
