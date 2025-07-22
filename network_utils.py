import numpy as np
import pandas as pd
from collections import deque, defaultdict
import powerlaw
import multiprocessing as mp
import networkx as nx
from scipy.stats import linregress

# General functions used throughout analyses

def threshold_adjacency_list(ppi_mat: np.ndarray, threshold: float) -> list[list[int]]:
    """
    Converts a weighted adjacency matrix into an unweighted adjacency list, including only edges
    with weights >= threshold
    --------------------------
    Args:
        ppi_mat (np.ndarray): Weighted adjacency matrix.
        threshold (float): Minimum weight to include an edge.
    Returns:
        list[list[int]]: Unweighted adjacency list after thresholding.
    """
    result = []
    for i in range(ppi_mat.shape[0]):
        curr = []
        for j in range(ppi_mat.shape[1]):
            if ppi_mat[i, j] >= threshold and i != j: # ensure above threshold and not self loop
                curr.append(j)
        result.append(curr)
    return result



def threshold_weighted_adjacency_list(ppi_mat: np.ndarray, threshold: float) -> list[list[tuple]]:
    """
    Converts a weighted adjacency matrix into an weighted adjacency list, including only edges
    with weights >= threshold
    --------------------------
    Args:
        ppi_mat (np.ndarray): Weighted adjacency matrix.
        threshold (float): Minimum weight to include an edge.
    Returns:
        list[list[tuple]]: Weighted adjacency list after thresholding.
    """
    result = []
    for i in range(ppi_mat.shape[0]):
        curr = []
        for j in range(ppi_mat.shape[1]):
            if ppi_mat[i, j] >= threshold and i != j: # ensure above threshold and not self loop
                curr.append((j, float(ppi_mat[i, j])))
        result.append(curr)
    return result



def connected_components(adjacency_list:list[list[int]]) -> tuple:
    """
    Return the number of multi-node connected components and the number of isolated nodes within
    a graph given its unweighted adjacency list
    --------------------------
    Args:
        adjacency_list (list[list[int]]): Unweighted adjacency list.
    Returns:
        tuple: number of multi-node connected components, number of isolated nodes
    """
    num_components = 0 # multi-node connected component 
    isolated_nodes = 0 
    visited = [False for i in range(len(adjacency_list))] 
    S = deque()
    idx = 0
    size = 0 # size of a connected component
    while idx < len(adjacency_list):
        S.append(idx)
        while len(S) > 0:
            v = S.popleft()
            if visited[v] == False:
                visited[v] = True
                size += 1
                for u in adjacency_list[v]:
                    S.append(u) 
        if size >= 2:
            num_components += 1
        else:
            isolated_nodes += 1
        # reset and ensure next node has not been visited 
        size = 0
        while idx < len(adjacency_list) and visited[idx] == True:
            idx += 1
    return num_components, isolated_nodes



def nodes_and_edges(adjacency_list:list[list[int]]) -> tuple:
    """
    Return the number of connected nodes and the number of edges within
    a graph given its unweighted adjacency list
    --------------------------
    Args:
        adjacency_list (list[list[int]]): Unweighted adjacency list.
    Returns:
        tuple: number of connected nodes, number of edges
    """
    num_nodes = 0
    num_edges = 0
    for i in range(len(adjacency_list)):
        curr = adjacency_list[i]
        if i in curr: # remove self-loops (should not be present in our setup)
            curr.remove(i)
        if len(curr) != 0:
            num_nodes += 1
            num_edges += len(curr)
    return num_nodes, num_edges//2



def construct_network(
    adjacency_list:list[list[tuple]], 
    network_name:str, 
    node_names:list[str]
) -> nx.Graph:
    """
    Construct a NetworkX Graph from a weighted adjacency list
    --------------------------
    Args:
        adjacency_list (list[list[tuple]]): Weighted adjacency list
        network_name (str): Name of the network 
        node_names (list[str]): List of names for each of the nodes within the network
    Returns:
        nx.Graph: NetworkX.Graph
    """
    result = nx.Graph(name=network_name)
    for idx, node_e in enumerate(adjacency_list):
        if len(node_e) != 0: # node in network must have an edge
            result.add_node(idx, name=node_names[idx])
            for e in node_e:
                result.add_edge(idx, e[0], weight=e[1]) # duplicate edges handled by NetworkX
    return result




def scale_free_r2(network:nx.Graph) -> float:
    """
    Compute the coefficient of determination for a linear regression fit on the log-log 
    degree probability distribution of the given network for scale-free analysis 
    --------------------------
    Args:
        network (nx.Graph): NetworkX Graph
    Returns:
       float: Coefficient of determination R2 value for scale-free analysis
    """
    deg_distr = [network.degree(n) for n in network.nodes()]
    n = network.number_of_nodes()
    unique_degs, counts = np.unique(deg_distr, return_counts=True)
    deg_prob = np.array([c / n for c in counts])
    nonzero = (unique_degs > 0) & (deg_prob > 0)
    x = np.log10(unique_degs[nonzero])
    y = np.log10(deg_prob[nonzero])
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2



def jaccard_similarity(
    network1:nx.Graph, 
    network2:nx.Graph, 
    matching_nodes:set[str], 
    node2idx:dict[str, int]
) -> dict[str, float]:
    """
    Compute the Jaccard similarity for a set of matching nodes between two networks. The two 
    networks must have the node_idx for all nodes in the matching_nodes set. 
    --------------------------
    Args:
        network1 (nx.Graph): NetworkX Graph
        network2 (nx.Graph): NetworkX Graph
        matching_nodes (set(str)): Set of matching nodes (names) shared by network1 and network2
        node2idx: A dict that maps the names of the matching nodes to the node_idx in the two networks
    Returns:
       dict: A dict of node names where the value is the Jaccard similarity
    """
    jaccard_similarities = dict()
    for node in matching_nodes:
        neighbors_in_1 = set(network1.neighbors(node2idx[node]))
        neighbors_in_2 = set(network2.neighbors(node2idx[node]))
        
        # calculate the Jaccard similarity using the names of the nodes
        names_1 = {network1.nodes[node]['name'] for node in neighbors_in_1}
        names_2 = {network2.nodes[node]['name'] for node in neighbors_in_2}
        intersection = len(names_1.intersection(names_2))
        union = len(names_1.union(names_2))
        
        if union == 0:
            similarity = 0.0  # Avoid division by zero
        else:
            similarity = intersection / union
        jaccard_similarities[node] = similarity
    return jaccard_similarities