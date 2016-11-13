import snap
import numpy as np

graphAtEpochs = {}

def getBasicGraph(nodeFile, edgeFile):
    Graph = snap.TNEANet.New()
    Graph.AddFltAttrE('probability')
    Graph.AddFltAttrN('xLoc')
    Graph.AddFltAttrN('yLoc')
    Graph.AddFltAttrN('temperature')
    Graph.AddFltAttrN('humidity')
    Graph.AddFltAttrN('light')
    Graph.AddFltAttrN('voltage')

    with open(nodeFile, 'rb') as f1:
        for line in f1:
            nodeid, xLoc, yLoc = line.split()
            Graph.AddNode(int(nodeid))
            Graph.AddFltAttrDatN(int(nodeid), float(xLoc), 'xLoc')
            Graph.AddFltAttrDatN(int(nodeid), float(yLoc), 'yLoc')

    with open(edgeFile, 'rb') as f1:
        for line in f1:
            try:
                #print line.split(' ')
                _, src, dst, prob = line.split(' ')
                if src == "0" or dst == "0":
                    continue
                prob = float(prob)
                if prob > 0.75:
                    Eid = Graph.AddEdge(int(src), int(dst))
                    Graph.AddFltAttrDatN(Eid, prob, 'probability')
            except:
                pass
    print Graph.GetEdges()
    graphAtEpochs[0] = Graph

def getEdgeAttr(srcId, dstId, Graph):
    return  Graph.GetFltAttrDatE(Graph.GetEI(srcId, dstId), 'probability')

def deepCopyGraph(original):
    newGraph = type(original).New()
    newGraph.AddFltAttrE('probability')
    newGraph.AddFltAttrN('xLoc')
    newGraph.AddFltAttrN('yLoc')
    for node in original.Nodes():
        nodeid = node.GetId()
        newGraph.AddNode(nodeid)
        newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), 'xLoc'), 'xLoc')
        newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), 'yLoc'), 'yLoc')
        newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), 'temperature'), 'temperature')
        newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), 'humidity'), 'humidity')
        newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), 'light'), 'light')
        newGraph.AddFltAttrDatN(int(nodeid), original.GetFltAttrDatN(node.GetId(), 'voltage'), 'voltage')

    for edge in original.Edges():
        eid = newGraph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
        newGraph.AddFltAttrDatE(eid, original.GetFltAttrDatE(edge, 'probability'), 'probability')
    return newGraph

def getNodeIds(Graph):
    nodeIds = [node.GetId() for node in Graph.Nodes()]
    return nodeIds

def addEpochToGraph(epoch, lines):
    lastEpoch = max(graphAtEpochs.keys())
    prevGraph = graphAtEpochs[lastEpoch]
    newGraph = deepCopyGraph(prevGraph)
    for line in lines:
        date, time, epoch, nodeid, temp, hum, light, volt = line.split()
        newGraph.AddFltAttrDatN(int(nodeid), float(temp), 'temperature')
        newGraph.AddFltAttrDatN(int(nodeid), float(hum), 'humidity')
        newGraph.AddFltAttrDatN(int(nodeid), float(light), 'light')
        newGraph.AddFltAttrDatN(int(nodeid), float(volt), 'voltage')
    graphAtEpochs[epoch] = newGraph


def getGraphAtEpoch(epoch):
    return graphAtEpochs[epoch]

def createAllGraphs(nodeFile, edgeFile, dataFile):
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



