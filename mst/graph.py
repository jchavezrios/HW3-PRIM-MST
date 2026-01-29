import numpy as np 
import heapq
from typing import Union
class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

     In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        
        m = self.adj_mat.shape[0]
        self.mst = None # starter number of vertices, like (0,0)
        
        # track what vertices are in mst 
        visited = [False]*m 
        
        # create a priority queue that will store data 
        priorQ = []        
        
        # begin with vertex 0 
        visited[0] = True 
        
        # while loop to go through map / tree 
        for _ in range(m):
            if self.adj_mat[0][_] > 0: # edge exists since weight is greater than 0 
                heapq.heappush(priorQ, (self.adj_mat[0][_], 0 , _ ))
        # goes through until edges added = n - 1 based on mst properties for n # of nodes      
        edge = 0 
        
        while priorQ and edge < m - 1:
            # obtain min weight edge 
            wt , u , v = heapq.heappop( priorQ ) 
            
            # skip if vertex is already in mst to avoid cyclization 
            if visited[v]:
                continue
            
            # if not then add v to mst list 
            visited[v] = True
            edge += 1
            
            # add edge to mst for adj_mat;  consider both directions for undirected graph 
            self.mst[u][v] = wt
            self.mst[v][u] = wt
            
            # add edge(s) from new vertex v to priorQ 
            for x in range(m):
                if self.adj_mat[v][x] > 0 and not visited[x]:
                    heapq.heappush(priorQ, (self.adj_mat[v][x]), v , x )
        
#print('yur')
