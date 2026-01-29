import pytest
import numpy as np
from mst.graph import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    """
    n = mst.shape[0]
    
    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    
    
    # added to check that all edges exist in original graph
    
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0:
                assert adj_mat[i, j] > 0, f'MST contains edge ({i},{j}) not in original graph'
                assert approx_equal(mst[i, j], adj_mat[i, j]), \
                    f'MST edge weight ({i},{j}) does not match original graph'
    
    


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)

    

def test_mst_student():
    """
    
    Test that Graph raises TypeError for invalid input types
    
    """
    
    # test with integer
    with pytest.raises(TypeError, match='Input must be a valid path or an adjacency matrix'):
        Graph(12345)
    
    # test with list
    with pytest.raises(TypeError, match='Input must be a valid path or an adjacency matrix'):
        Graph([1, 2, 3])
    
    # test with dict
    with pytest.raises(TypeError, match='Input must be a valid path or an adjacency matrix'):
        Graph({'a': 1})
        
def test_graph_ok():
    """
    unit test to see if tree has n-1 edges and symmetric 
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    
    
    mst = g.mst
    
    print(g.adj_mat)
    
    # number of vertices
    owl = mst.shape[0]
    
    # does mst, owl, have n-1 edges 
    num_edges = np.count_nonzero(mst) / 2  # Divide by 2 because undirected graph
    assert num_edges == owl - 1, f'MST should have {owl-1} edges, but has {num_edges}'
    
    # determine if MST is symmetric (undirected graph property)
    assert np.allclose(mst, mst.T), 'MST adjacency matrix must be symmetric'