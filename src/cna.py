#import snap
from FeatGraph import *
from generateNetwork import *
import numpy as np
from sklearn.cluster import SpectralClustering
from collections import defaultdict

def getCommunities(graph):
    clf = SpectralClustering(affinity='precomputed')
    clf.fit(getProbMatrix()[1:, 1:])
    print clf.labels_
    cmtyToNode = defaultdict(list)
    nodeToCmty = {}
    for node in graph.Nodes():
        nid = node.GetId()
        cmty = clf.labels_[nid-1]
        cmtyToNode[cmty].append(nid)
        nodeToCmty[nid] = cmty
    return cmtyToNode, nodeToCmty

'''
def getCommunities1(graph):
    gUnDir = snap.ConvertGraph(snap.PUNGraph, graph)
    cmtyV = snap.TCnComV()
    modularity = snap.CommunityCNM(gUnDir, cmtyV)
    nodeToCmty = {}
    i = 0
    for cmty in cmtyV:
        for nid in cmty:
            nodeToCmty[nid] = i
        i += 1
    return cmtyToNode, nodeToCmty
'''

def getCnaScore(nid, epoch, cmtyToNode, cmtyId):
    graph = getGraphAtEpoch(epoch)
    node = graph.GetNI(nid)
    nbrs = [nbrid for nbrid in cmtyToNode[cmtyId]]
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

def getCnaOutliers(epoch):
    graph = getGraphAtEpoch(epoch)
    CmtyToNode, nodeToCmty = getCommunities(graph)
    cnaDists = []
    for node in graph.Nodes():
        nid = node.GetId()
        cna_score = np.linalg.norm(np.array(getNodeFeatures(nid, graph)) - getCnaScore(nid, epoch, CmtyToNode, nodeToCmty[nid]))
        cnaDists.append((nid, cna_score))
    cnaDists = sorted(cnaDists, key=lambda x: x[1], reverse=True)
    print [x for x in cnaDists[:10]]


createAllGraphs('../data/mote_locs.txt', '../data/connectivity.txt', '../data/data_medium.txt')
#getCommunities()
getCnaOutliers(2)

