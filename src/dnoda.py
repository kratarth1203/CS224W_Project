import snap
from generateNetwork import *
import numpy as np

def getDnodaScore(nid, epoch):
    node = graph.GetNI(nid)
    nbrs = [node.GetOutNId(i) for i in xrange(node.GetOutDeg)]
    graph = getGraphAtEpoch(epoch)
    if epoch > 1:
        graph_tm1 = getGraphAtEpoch(epoch - 1)
    else:
        graph_tm1 = None
    if epoch < max_epoch:
        graph_tp1 = getGraphAtEpoch(epoch + 1)
    else:
        graph_tp1 = None

    nodeFeats = getNodeFeatures(nid, graph)
    nbrFeats = []
    if graph_tm1 is not None:
        nbrFeats.append(getNodeFeatures(nid, graph_tm1))
    if graph_tp1 is not None:
        nbrFeats.append(getNodeFeatures(nid, graph_tp1))
    nbrFeats.extend([getNodeFeatures(nbrid, graph) for nbrid in nbrs])
    nbrFeats = np.array(nbrFeats)
    print nbrFeats.shape
    return np.linalg.norm(nodeFeats - (np.sum(nbrFeats, axis = 0))/(nbrFeats.shape[0]*1.0))

def getDnodaOutliers(epoch):
    graph = getGraphAtEpoch(epoch)
    dnodaDists = []
    for node in graph:
        nid = node.GetId()
        dnodaDists.append((nid, getDnodaScore(nid, epoch)))
    dnodaDists = sorted(dnodaDists, key=lambda x: x[1], reverse=True)
    print [x for x in dnodaDists[:10]]

getDnodaOutliers(2)

