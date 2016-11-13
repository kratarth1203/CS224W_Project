import snap
import numpy

from generateNetwork import *

'''
Given a nodeId and a graph, return the egonet
@param: graph: The snap graog data structure
        nodeId: The integer id of the node whose
                egonet is desired. 
'''
def getEgonet(nodeId, graph):
  nbr = []
  node = graph.GetNI(nodeId)
  for idx in range(node.GetOutDeg()):
     nbr.append(node.GetOutNId(idx)) 
  return nbr


'''
Given a nodeId, returns its super-egonet
@param: nodeId: The integer id of the node whose
                super-egonet is desired
        graph: snap graph data stucture
'''
def getSuperEgonet(nodeId, graph):
  nbr = []
  nbr.extend(getEgonet(nodeId, graph))
  node = graph.GetNI(nodeId)
  for idx in range(node.GetOutDeg()):
     nbr_id = node.GetOutNId(idx)
     nbr.extend(getEgonet(nbr_id, graph))
  return nbr
  


'''
Find paths from node index n to m
@params: n: integer id of the src node
         m: integer id of the dst node
         graph: snap graph data structure
'''
def adjlist_find_paths( n, m, graph, path=[]):
  n = int(n)
  path = path + [n]
  if n == m:
    return [path]
  paths = []
  nbr = []
  node = graph.GetNI(n)
  for idx in range(node.GetOutDeg()):
    nbr.append(node.GetOutNId(idx))
  for child in nbr:
    if child not in path:
      child_paths = adjlist_find_paths(child, m, graph, path[:])
      for child_path in child_paths:
        paths.append(child_path)
  return paths

'''
Given a nodeId (n) return the sum of weights with all its neighbors
@param: n: nodeId whose sum of weights is needed
        graph: snap graph structure
'''

def getSumWeight(n, graph):
  node = graph.GetNI(n)
  nbrs = []
  for idx in range(node.GetOutDeg()):
     nbrs.append(node.GetOutNId(idx))
  sum_ = 0.0
  for nbr in nbrs:
    sum_ += getEdgeAttr(n, nbr, graph)
  return sum_

'''
Given a core node nodeId_v0, calculates the closeness
between it and the nodeId_v1 in the graph.
@param: nodeId_v0: integer id of the core node
        nodeId_v1: integer id of the node to which closeness is needed
        graph: snap graog data structure
'''
def getCloseness(nodeId_v0, nodeId_v1, graph):
  max_weight = -1.0 * np.inf
  paths = adjlist_find_paths( nodeId_v0, nodeId_v1 , graph)
  flag = False
  for path in paths:
    weight = 1
    flag = True
    for n,m in zip(path[:-1], path[1:]):
      w = getEdgeAttr(n, m , graph)
      sum_ = getSumWeight(n, graph)
      weight *= ((w * 1.0)/sum_)
    if weight > max_weight:
      max_weight = weight
  if flag is False:
    return np.inf
  return max_weight

'''
Given a core nodeId v0, gets the closeness
@param: nodeId: integer id of the node whose k-closeness is needed
        graph: snap graph data structure       
'''
def getkCloseness(nodeId_v0, graph):
  nodes = getNodeIds(graph)
  nodes.remove(nodeId_v0)
  closeness = []
  for node in nodes:
    closeness.append((getCloseness(nodeId_v0, node, graph), node))
  return closeness

'''
Given a integer nodeId_v0 and the hyper parameter k, returns the
k-closeness neighborhood
@param k: hyper parameter
       nodeId_v0: the core node integer id whose k-closeness 
                  neighborhood is needed
       graph: snap graph data structure 
'''
def getkCloseNeighbor(nodeId_v0, k, graph):
  closeness  = getkCloseness(nodeId_v0, graph)
  closeness = sorted(closeness, key=lambda x: x[0])
  closeness = np.array(closeness)
  return closeness[k-1, 0], np.array(closeness[:k,1], dtype=np.int32)


'''
Given an integer code node Id , return the corenet 
@param: nodeId_v0: the integer id of the corenet
        graph: snap graph datra structure
        k: hyperparamter
'''
def getCorenet(nodeId_v0, k, graph):
  closeness = getkCloseness(nodeId_v0,  graph)
  closeness = sorted(closeness, key=lambda x: x[0])
  closeness = np.array(closeness)
  min_closeness = closeness[0,0]
  k_closeness, k_nbr = getkCloseNeighbor(nodeId_v0, k, graph)
  if min_closeness >= k_closeness:
    return getSuperEgonet(nodeId_v0, graph)
  else:
    return k_nbr

'''
Given the graphs at time step T and T-1, calculate the 
outlying score for the node nodeIfd.
@param: nodeId: an integer id of the node whose outlying score
                is required
        graph_tm1: graph and time T-1
        graph_t: graoh at time T
'''
def getOutlyingScore(nodeId, graph_tm1, graph_t, k):
  corenet_tm1 = getCorenet(nodeId, k, graph_tm1)
  corenet_t = getCorenet(nodeId, k , graph_t)

  c_old = list(set(corenet_tm1).intersection(set(corenet_t)))
  c_removed = list(set(corenet_tm1) - set(c_old))
  c_new = list(set(corenet_t) - set(c_old))
  sum_ = 0.0
  for node in c_old:
    sum_ += (getCloseness(nodeId, node, graph_tm1) -\
                      getCloseness( nodeId, node,  graph_t))
  for node in c_removed:
    sum_ += getCloseness(nodeId, node , graph_tm1)

  for n,m in zip(c_new, c_old):
    norm = (getEdgeAttr(n, m, graph_t) * 1.0 ) /\
                     getSumWeight(n, graph_t)
    sum_ += ((1.0 - norm) * getCloseness( nodeId, n, graph))

  return sum_

'''
Get IEOutliers based on the snapshot of the graph at two 
separate time instances.
@param: graph_t: snap object of the graph at time T
        graph_tm1: snap object of the graph at time T-1
'''
def getICLEOD(graph_tm1, graph_t):
  for node in getNodeIds(graph_t):
    print getOutlyingScore(node, graph_tm1, graph_t, 2)
 
createAllGraphs('../data/mote_locs.txt', '../data/connectivity.txt', '../data/data_small.txt')
 
getICLEOD(getGraphAtEpoch(graphAtEpochs.keys()[0]), getGraphAtEpoch(graphAtEpochs.keys()[1])) 
