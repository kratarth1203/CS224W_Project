#import snap
from FeatGraph import *
from generateNetwork import *
import numpy as np

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
        dnoda_score = np.linalg.norm(np.array(getNodeFeatures(nid, graph)) - getDnodaScore(nid, epoch))
        dnodaDists.append((nid, dnoda_score))
    dnodaDists = sorted(dnodaDists, key=lambda x: x[1], reverse=True)
    print [x for x in dnodaDists[:10]]

createAllGraphs('../data/mote_locs.txt', '../data/connectivity.txt', '../data/data.txt')
getDnodaOutliers(2)

