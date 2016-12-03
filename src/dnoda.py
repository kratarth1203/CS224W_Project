#import snap
from FeatGraph import *
from generateNetwork import *
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def getDnodaScore(nid, epoch):
    graph = getGraphAtEpoch(epoch)
    node = graph.GetNI(nid)
    nbrs = [node.GetOutNId(i) for i in xrange(node.GetOutDeg())]
    if epoch > 1:
        graph_tm1 = getGraphAtEpoch(epoch - 1)
    else:
        graph_tm1 = None
    if epoch < max_epoch:
        graph_tp1 = getGraphAtEpoch(epoch + 1)
    else:
        graph_tp1 = None

    nbrFeats = []
    if graph_tm1 is not None:
        nbrFeats.append(getNodeFeatures(nid, graph_tm1))
    if graph_tp1 is not None:
        nbrFeats.append(getNodeFeatures(nid, graph_tp1))
    nbrFeats.extend([getNodeFeatures(nbrid, graph) for nbrid in nbrs])
    nbrFeats = np.array(nbrFeats)
    return np.sum(nbrFeats, axis = 0)/(nbrFeats.shape[0]*1.0)

def getDnodaOutliers(epoch):
    graph = getGraphAtEpoch(epoch)
    dnodaDists = []
    for node in graph.Nodes():
        nid = node.GetId()
        if nid == 1:
            print epoch, getNodeFeatures(nid, graph)
        dnoda_score = np.linalg.norm(np.array(getNodeFeatures(nid, graph)) - getDnodaScore(nid, epoch))
        dnodaDists.append((nid, dnoda_score))
    dnodaDists = sorted(dnodaDists, key=lambda x: x[1], reverse=True)
    print [x for x in dnodaDists[:10]]
    allDists = sorted(dnodaDists, key=lambda x: x[0])
    return [x[1] for x in allDists]


createAllGraphs('../data/mote_locs.txt', '../data/connectivity.txt', '../data/data_medium_epochs.txt')
allDists = getDnodaOutliers(25)

plt.plot(np.arange(1, 55), allDists)
plt.savefig('dnoda_medium_epochs.png')
