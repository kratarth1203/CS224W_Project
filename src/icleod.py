#import snap
import numpy

from generateNetwork import *
from collections import *


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def deep_list(x):
  """fully copies trees of tuples or lists to a tree of lists.
     deep_list( (1,2,(3,4)) ) returns [1,2,[3,4]]
     deep_list( (1,2,[3,(4,5)]) ) returns [1,2,[3,[4,5]]]"""
  if not ( type(x) == type( () ) or type(x) == type( [] ) ):
    return x
  return map(deep_list,x)


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
Implementation of Dijktras
@param: graph: snap graph data structure
        initial: source node integer id
'''

from collections import defaultdict
from heapq import *

def dijkstra(src, dst, edges):
  g = defaultdict(list)
  for l,r,c in edges:
    g[l].append((c,r))

  q, seen = [(0,src,())], set()
  while q:
    (cost,v1,path) = heappop(q)
    if v1 not in seen:
      seen.add(v1)
      path = (v1, path)
      if v1 == dst: return (cost, path)

      for c, v2 in g.get(v1, ()):
        if v2 not in seen:
          heappush(q, (cost+c, v2, path))

  return np.inf


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
def getCloseness(nodeId_v0, nodeId_v1, graph, edges = None):
  max_weight = -1.0 * np.inf
  #paths = adjlist_find_paths( nodeId_v0, nodeId_v1 , graph)
  paths = [dijkstra(nodeId_v0, nodeId_v1, edges)]
  if paths[0] is np.inf:
    return 1e6
  flag = False
  for path_ in [paths[0][1]]:
    new_path = deep_list(path_)
    path = []
    def convert( x ):
      if type(x) is not type( [] ):
        path.append(x)
      elif len(x) > 0:
        path.append(x[0])
        convert(x[1:][0])
    convert(new_path)
    path = path[::-1]

    weight = 1.0
    flag = True
    for n,m in zip(path[:-1], path[1:]):
      w = getEdgeAttr(int(n), int(m) , graph)
      sum_ = getSumWeight(n, graph)
      weight *= ((w * 1.0)/sum_)
    if weight > max_weight:
      max_weight = weight
  if flag is False:
    return 1e6
  return max_weight

'''
Given a core nodeId v0, gets the closeness
@param: nodeId: integer id of the node whose k-closeness is needed
        graph: snap graph data structure
'''
def getkCloseness(nodeId_v0, graph, edges):
  nodes = getNodeIds(graph)
  nodes.remove(nodeId_v0)
  closeness = []
  for node in nodes:
    closeness.append((getCloseness(nodeId_v0, node, graph, edges), node))
  return closeness

'''
Given a integer nodeId_v0 and the hyper parameter k, returns the
k-closeness neighborhood
@param k: hyper parameter
       nodeId_v0: the core node integer id whose k-closeness
                  neighborhood is needed
       graph: snap graph data structure
'''
def getkCloseNeighbor(nodeId_v0, k, graph, edges):
  closeness  = getkCloseness(nodeId_v0, graph, edges)
  closeness = sorted(closeness, key=lambda x: x[0])
  closeness = np.array(closeness)
  return closeness[k-1, 0], np.array(closeness[:k,1], dtype=np.int32)


'''
Given an integer code node Id , return the corenet
@param: nodeId_v0: the integer id of the corenet
        graph: snap graph datra structure
        k: hyperparamter
'''
def getCorenet(nodeId_v0, k, graph, edges):
  closeness = getkCloseness(nodeId_v0,  graph, edges)
  closeness = sorted(closeness, key=lambda x: x[0])
  closeness = np.array(closeness)
  min_closeness = closeness[0,0]
  k_closeness, k_nbr = getkCloseNeighbor(nodeId_v0, k, graph, edges)
  if min_closeness >= k_closeness:
    return getSuperEgonet(nodeId_v0, graph)
  else:
    return [int(_) for _ in k_nbr]

'''
Given the graphs at time step T and T-1, calculate the
outlying score for the node nodeIfd.
@param: nodeId: an integer id of the node whose outlying score
                is required
        graph_tm1: graph and time T-1
        graph_t: graoh at time T
'''
def getOutlyingScore(nodeId, graph_tm1, graph_t, k, edges_tm1, edges_t):
  corenet_tm1 = getCorenet(nodeId, k, graph_tm1, edges_tm1)
  corenet_t = getCorenet(nodeId, k , graph_t, edges_t)

  c_old = list(set(corenet_tm1).intersection(set(corenet_t)))
  c_removed = list(set(corenet_tm1) - set(c_old))
  c_new = list(set(corenet_t) - set(c_old))
  sum_ = 0.0
  for node in c_old:
    sum_ += (np.abs(getCloseness(nodeId, node, graph_tm1, edges_tm1) -\
                      getCloseness( nodeId, node,  graph_t, edges_t)))
  for node in c_removed:
    sum_ += getCloseness(nodeId, node , graph_tm1, edges_tm1)

  for n,m in zip(c_new, c_old):
    try:
      norm = (getEdgeAttr(n, m, graph_t) * 1.0 ) /\
                     getSumWeight(n, graph_t)
      sum_ += ((1.0 - norm) * getCloseness( nodeId, n, graph_t, edges_t))
    except:
      pass

  return sum_

'''
Get IEOutliers based on the snapshot of the graph at two
separate time instances.
@param: graph_t: snap object of the graph at time T
        graph_tm1: snap object of the graph at time T-1
'''
def getICLEOD(graph_tm1, graph_t, edges_tm1, edges_t):
  scores = []
  for node in getNodeIds(graph_t):
    scores.append((node, getOutlyingScore(node, graph_tm1, graph_t, 2, edges_tm1, edges_t)))
  dists = sorted(scores, key=lambda x: x[1], reverse=True)
  allDists = sorted(dists, key=lambda x: x[0])
  return [x[1] for x in allDists]

createAllGraphs('../data/mote_locs.txt', '../data/connectivity.txt', '../data/data_medium_epochs.txt')


graph_tm1 = getGraphAtEpoch(24)
graph_t = getGraphAtEpoch(25)
edges_tm1 = getEdges(graph_tm1)
edges_t = getEdges(graph_t)
allDists = getICLEOD(graph_tm1, graph_t, edges_tm1, edges_t)

plt.plot(np.arange(1, 55), allDists)
plt.savefig('icleod_medium_epochs.png')
