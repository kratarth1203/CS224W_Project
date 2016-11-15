import numpy as np
import sys

class DirectedGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.nNodes = 0
        self.nEdges = 0
        self.nidToNodeMap = {}
        self.eidToEdgeMap = {}
        self.nodeFeatures = {}
        self.edgeFeatures = {}
        self.nodeTupToEdgeMap = {}

    def GetNodes(self):
        return self.nNodes

    def GetEdges(self):
        return self.nEdges

    def Nodes(self):
        return self.nodes

    def Edges(self):
        return self.edges

    def AddNode(self, nid):
        self.nNodes += 1
        newNode = DirectedNode(nid, self)
        self.nidToNodeMap[nid] = newNode
        self.nodes.append(newNode)
        return newNode

    def AddEdge(self, srcId, dstId):
        if (srcId, dstId) in self.nodeTupToEdgeMap:
            return self.nodeTupToEdgeMap[(srcId, dstId)]
        eid = self.nEdges
        srcNode = self.nidToNodeMap[srcId]
        dstNode = self.nidToNodeMap[dstId]
        srcNode.AddOutEdge(dstNode)
        dstNode.AddInEdge(srcNode)
        self.nEdges += 1
        newEdge = DirectedEdge(eid, srcId, dstId, self)
        self.edges.append(newEdge)
        self.eidToEdgeMap[eid] = newEdge
        self.nodeTupToEdgeMap[(srcId, dstId)] = eid
        return eid

    def IsNode(self, nid):
        return nid in self.nidToNodeMap

    def IsEdge(self, srcId, dstId):
        return (srcId, dstId) in self.nodeTupToEdgeMap

    def GetNI(self, nid):
        return self.nidToNodeMap[nid]

    def GetEI(self, srcId, dstId):
        return self.nodeTupToEdgeMap[(srcId, dstId)]

    def AddIntAttrN(self, attr):
        self.nodeFeatures[attr] = 'int'
        for node in self.nodes:
            node.AddAttrDat(-1*sys.maxint - 1, attr)

    def AddFltAttrN(self, attr):
        self.nodeFeatures[attr] = 'float'
        for node in self.nodes:
            node.AddAttrDat(-1.0*np.inf, attr)

    def AddStrAttrN(self, attr):
        self.nodeFeatures[attr] = 'string'
        for node in self.nodes:
            node.AddAttrDat('', attr)

    def AddIntAttrDatN(self, node, val, attr):
        if not attr in self.nodeFeatures:
            raise NameError('No such attribute in graph')
        if not type(val) == int:
            raise TypeError('Value needs to be an integer')
        if type(node) == int:
            node = self.nidToNodeMap[node]
        node.AddAttrDat(val, attr)

    def AddFltAttrDatN(self, node, val, attr):
        if not attr in self.nodeFeatures:
            raise NameError('No such attribute in graph')
        if not type(val) in [int, float]:
            raise TypeError('Value needs to be a float')
        if type(node) == int:
            node = self.nidToNodeMap[node]
        node.AddAttrDat(val, attr)

    def AddStrAttrDatN(self, node, val, attr):
        if not attr in self.nodeFeatures:
            raise NameError('No such attribute in graph')
        if not type(val) == str:
            raise TypeError('Value needs to be a string')
        if type(node) == int:
            node = self.nidToNodeMap[node]
        node.AddAttrDat(val, attr)

    def GetIntAttrDatN(self, node, attr):
        if not attr in self.nodeFeatures:
            raise NameError('No such attribute in graph')
        if type(node) == int:
            node = self.nidToNodeMap[node]
        val = node.GetAttrDat(attr)
        if not type(val) == int:
            raise TypeError('Attribute is not integer type')
        return val

    def GetFltAttrDatN(self, node, attr):
        if not attr in self.nodeFeatures:
            raise NameError('No such attribute in graph')
        if type(node) == int:
            node = self.nidToNodeMap[node]
        if type(node) in [np.int64, np.int32]:
            node = self.nidToNodeMap[int(node)]
        val = node.GetAttrDat(attr)
        if not type(val) in [float, int]:
            raise TypeError('Attribute is not float type')
        return val

    def GetStrAttrDatN(self, node, attr):
        if not attr in self.nodeFeatures:
            raise NameError('No such attribute in graph')
        if type(node) == int:
            node = self.nidToNodeMap[node]
        val = node.GetAttrDat(attr)
        if not type(val) == str:
            raise TypeError('Attribute is not string type')
        return val

    def AddIntAttrE(self, attr):
        self.edgeFeatures[attr] = 'int'
        for edge in self.edges:
            edge.AddAttrDat(-1*sys.maxint - 1, attr)

    def AddFltAttrE(self, attr):
        self.edgeFeatures[attr] = 'float'
        for edge in self.edges:
            edge.AddAttrDat(-1.0*np.inf, attr)

    def AddStrAttrE(self, attr):
        self.edgeFeatures[attr] = 'string'
        for edge in self.edges:
            edge.AddAttrDat('', attr)

    def AddIntAttrDatE(self, edge, val, attr):
        if not attr in self.edgeFeatures:
            raise NameError('No such attribute in graph')
        if not type(val) == int:
            raise TypeError('Value needs to be an integer')
        if type(edge) == int:
            edge = self.eidToEdgeMap[edge]
        edge.AddAttrDat(val, attr)

    def AddFltAttrDatE(self, edge, val, attr):
        if not attr in self.edgeFeatures:
            raise NameError('No such attribute in graph')
        if not type(val) in [int, float]:
            raise TypeError('Value needs to be a float')
        if type(edge) == int:
            edge = self.eidToEdgeMap[edge]
        edge.AddAttrDat(val, attr)

    def AddStrAttrDatE(self, edge, val, attr):
        if not attr in self.edgeFeatures:
            raise NameError('No such attribute in graph')
        if not type(val) == str:
            raise TypeError('Value needs to be a string')
        if type(edge) == int:
            edge = self.eidToEdgeMap[edge]
        edge.AddAttrDat(val, attr)

    def GetIntAttrDatE(self, edge, attr):
        if not attr in self.edgeFeatures:
            raise NameError('No such attribute in graph')
        if type(edge) == int:
            edge = self.eidToEdgeMap[edge]
        val = edge.GetAttrDat(attr)
        if not type(val) == int:
            raise TypeError('Attribute is not integer type')
        return val

    def GetFltAttrDatE(self, edge, attr):
        if not attr in self.edgeFeatures:
            raise NameError('No such attribute in graph')
        if type(edge) == int:
            edge = self.eidToEdgeMap[edge]
        val = edge.GetAttrDat(attr)
        if not type(val) in [float, int]:
            raise TypeError('Attribute is not float type')
        return val

    def GetStrAttrDatE(self, edge, attr):
        if not attr in self.edgeFeatures:
            raise NameError('No such attribute in graph')
        if type(edge) == int:
            edge = self.eidToEdgeMap[edge]
        val = edge.GetAttrDat(attr)
        if not type(val) == str:
            raise TypeError('Attribute is not string type')
        return val

class DirectedNode:
    def __init__(self, nid, graph):
        self.graph = graph
        self.nid = nid
        self.outDeg = 0
        self.inDeg = 0
        self.outEdges = []
        self.inEdges = []
        self.features = {}
        for feat in graph.nodeFeatures:
            if graph.nodeFeatures[feat] == 'int':
                self.features[feat] = -1*sys.maxint - 1
            elif graph.nodeFeatures[feat] == 'float':
                self.features[feat] = -1.0*np.inf
            else:
                self.features[feat] = ''

    def GetId(self):
        return self.nid

    def GetOutDeg(self):
        return self.outDeg

    def GetInDeg(self):
        return self.inDeg

    def GetDeg(self):
        return self.inDeg + self.outDeg

    def GetInNId(self, idx):
        return self.inEdges[idx].GetId()

    def GetOutNId(self, idx):
        return self.outEdges[idx].GetId()

    def GetNbrNId(self, idx):
        if idx < self.inDeg:
            return self.InEdges[idx].GetId()
        return self.outEdges[idx].GetId()

    def IsInNId(self, nid):
        node = self.graph.GetNI(nid)
        return node in self.inEdges

    def IsOutNId(self, nid):
        node = self.graph.GetNI(nid)
        return node in self.outEdges

    def IsNbrNId(self, nid):
        node = self.graph.GetNI(nid)
        return node in self.outEdges or node in self.inEdges

    def AddOutEdge(self, dstNode):
        self.outEdges.append(dstNode)
        self.outDeg += 1

    def AddInEdge(self, srcNode):
        self.inEdges.append(srcNode)
        self.inDeg += 1

    def AddAttrDat(self, val, attr):
        self.features[attr] = val

    def GetAttrDat(self, attr):
        return self.features[attr]

class DirectedEdge:
    def __init__(self, eid, srcId, dstId, graph):
        self.graph = graph
        self.eid = eid
        self.srcId = srcId
        self.dstId = dstId
        self.features = {}
        for feat in graph.edgeFeatures:
            if graph.edgeFeatures[feat] == 'int':
                self.features[feat] = -1*sys.maxint - 1
            elif graph.edgeFeatures[feat] == 'float':
                self.features[feat] = -1.0*np.inf
            else:
                self.features[feat] = ''

    def GetId(self):
        return self.eid

    def GetSrcNId(self):
        return self.srcId

    def GetDstNId(self):
        return self.dstId

    def AddAttrDat(self, val, attr):
        self.features[attr] = val

    def GetAttrDat(self, attr):
        return self.features[attr]

