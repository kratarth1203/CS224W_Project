#import snap
import copy
from FeatGraph import *
import numpy as np
import cPickle as pickle
import os

graphAtEpochs = {}
max_epoch = -1
nodeFeatureNames = ['temperature', 'humidity', 'light', 'voltage']
probMatrix = None


def getProbMatrix():
    global probMatrix
    return probMatrix

def getNodeFeatures(nid, graph):
    nodeFeats = []
    for feat in nodeFeatureNames:
        nodeFeats.append(graph.GetFltAttrDatN(nid, feat))
    return nodeFeats

def getBasicGraph(nodeFile, edgeFile):
    #Graph = snap.TNEANet.New()
    Graph = DirectedGraph()
    Graph.AddFltAttrE('probability')
    Graph.AddFltAttrN('xLoc')
    Graph.AddFltAttrN('yLoc')
    for feat in nodeFeatureNames:
        Graph.AddFltAttrN(feat)

    with open(nodeFile, 'rb') as f1:
        for line in f1:
            nodeid, xLoc, yLoc = line.split()
            Graph.AddNode(int(nodeid))
            Graph.AddFltAttrDatN(int(nodeid), float(xLoc), 'xLoc')
            Graph.AddFltAttrDatN(int(nodeid), float(yLoc), 'yLoc')
            for feat in nodeFeatureNames:
                Graph.AddFltAttrDatN(int(nodeid), 0, feat)

    global probMatrix
    probMatrix = np.zeros((Graph.GetNodes() + 1, Graph.GetNodes() + 1))

    with open(edgeFile, 'rb') as f1:
        for line in f1:
            try:
                #print line.split(' ')
                _, src, dst, prob = line.split(' ')
                if src == "0" or dst == "0":
                    continue
                prob = float(prob)
                probMatrix[int(src), int(dst)] = prob
                if prob > 0.75:
                    Eid = Graph.AddEdge(int(src), int(dst))
                    Graph.AddFltAttrDatE(Eid, prob, 'probability')
            except Exception as exc:
                pass
    pickle.dump(probMatrix, open('../data/probMatrix.pkl', 'wb'))
    graphAtEpochs[0] = Graph


def timeFunc(srcId, dstId, Graph):
    srcFeats = np.array(getNodeFeatures(srcId, Graph))
    dstFeats = np.array(getNodeFeatures(dstId, Graph))
    return 1.0 + np.sum(srcFeats - dstFeats)

def getEdges(graph):
    rval = []
    for edge in graph.Edges():
        src = edge.GetSrcNId()
        dst = edge.GetDstNId()
        rval.append((src , dst, getEdgeAttr(src, dst, graph)))
    return rval


def getEdgeAttr(srcId, dstId, Graph):
    return timeFunc(srcId, dstId, Graph)* Graph.GetFltAttrDatE(Graph.GetEI(srcId, dstId), 'probability')

def deepCopyGraph(original):
    #newGraph = type(original).New()
    newGraph = copy.deepcopy(original)
    '''
    newGraph.AddFltAttrE('probability')
    newGraph.AddFltAttrN('xLoc')
    newGraph.AddFltAttrN('yLoc')
    for node in original.Nodes():
        nodeid = node.GetId()
        newGraph.AddNode(nodeid)
        newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), 'xLoc'), 'xLoc')
        newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), 'yLoc'), 'yLoc')
        for feat in nodeFeatureNames:
            newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), feat), feat)

    for edge in original.Edges():
        eid = newGraph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
        newGraph.AddFltAttrDatE(eid, original.GetFltAttrDatE(edge, 'probability'), 'probability')
    '''
    return newGraph

def getNodeIds(Graph):
    nodeIds = [node.GetId() for node in Graph.Nodes()]
    return nodeIds

def addEpochToGraph(epoch, lines):
    lastEpoch = max(graphAtEpochs.keys())
    prevGraph = graphAtEpochs[lastEpoch]
    newGraph = deepCopyGraph(prevGraph)
    for line in lines:
        try:
            date, time, epoch, nodeid, temp, hum, light, volt = line.split()
        except:
            continue
        newGraph.AddFltAttrDatN(int(nodeid), float(temp), 'temperature')
        newGraph.AddFltAttrDatN(int(nodeid), float(hum), 'humidity')
        newGraph.AddFltAttrDatN(int(nodeid), float(light), 'light')
        newGraph.AddFltAttrDatN(int(nodeid), float(volt), 'voltage')
    epoch = int(epoch)
    graphAtEpochs[epoch] = newGraph
    global max_epoch
    if epoch > max_epoch:
        max_epoch = epoch



def getGraphAtEpoch(epoch):
    if epoch < 0:
        return None
    if epoch not in graphAtEpochs:
        return getGraphAtEpoch(epoch - 1)
    return graphAtEpochs[epoch]

def createAllGraphs(nodeFile, edgeFile, dataFile):
    if os.path.isfile(dataFile[:-4] + '.pkl'):
        global graphAtEpochs
        graphAtEpochs = pickle.load(open(dataFile[:-4] + '.pkl', 'rb'))
        if os.path.isfile('../data/probMatrix.pkl'):
            global probMatrix
            probMatrix = pickle.load(open('../data/probMatrix.pkl', 'rb'))
        else:
            getBasicGraph(nodeFile, edgeFile)
        return
    getBasicGraph(nodeFile, edgeFile)
    with open(dataFile, 'rb') as f1:
        oldEpoch = 1
        lines = []
        for line in f1:
            words = line.split()
            epoch = int(words[2])
            if epoch == oldEpoch:
                lines.append(line)
            else:
                #print oldEpoch
                addEpochToGraph(oldEpoch, lines)
                lines = [line]
                oldEpoch = epoch
        if len(lines) > 0:
            print oldEpoch
            addEpochToGraph(oldEpoch, lines)
    pickle.dump(graphAtEpochs, open(dataFile[:-4] + '.pkl', 'wb'))



