import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
import numpy as np


def get_adj_matrix(x, edge_index):
    adj_matrix = np.zeros(shape=[len(x), len(x)])
    adj_matrix.astype(float)

    for i in range(len(x)):
        for j in range(len(x)):
            for index in range(len(edge_index[0])):
                edge = edge_index[:,index]

                if (edge[0] == i) \
                    and (edge[1] == j):
                    adj_matrix[i][j] = 1
                    break
    
    return adj_matrix


def test_function():
    edge_index = torch.tensor([[3, 6, 3, 0, 3, 2, 2, 6, 0, 6, 0, 5, 5, 6, 5, 4, 5, 1],
                               [6, 3, 0, 3, 2, 3, 6, 2, 6, 0, 5, 0, 6, 5, 4, 5, 1, 5]], dtype=torch.long)
    x = torch.tensor([[0], [1], [2], [3], [4], [5], [6]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    adj_matrix = get_adj_matrix(x, edge_index)

    graph = Graph(data, adj_matrix)

    return graph


def karate_club_loader():
    temp_data = KarateClub().data
    edge_index = temp_data.edge_index

    x_aux = [[x] for x in range(len(temp_data.x))]
    x = torch.tensor(x_aux, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)

    adj_matrix = get_adj_matrix(x, edge_index)

    graph = Graph(data, adj_matrix)
    return graph


class Graph():
    def __init__(self, graph, adj_matrix):
        self.data = graph
        self.x = graph.x
        self.adj_matrix = adj_matrix

        edges = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if (j > i) \
                and (adj_matrix[i][j] == 1):
                    edge = Edge(i, j)
                    edges.append(edge)

        self.edges_list = EdgesList(edges)


class EdgesList():
    def __init__(self, edges):
        adj_edges = []

        for i in range(len(edges)):
            adjs = []
            for j in range(len(edges)):
                if ((edges[i].v_i == edges[j].v_i) \
                    or (edges[i].v_i == edges[j].v_j) \
                    or (edges[i].v_j == edges[j].v_i) \
                    or (edges[i].v_j == edges[j].v_j)) \
                and (i != j):
                    adjs.append(j)
            
            adj_edges.append(adjs)

        self.edges = edges
        self.num_edges = len(edges)
        self.adj_edges = adj_edges


class Edge():
    def __init__(self, v_i, v_j):
        self.v_i = v_i
        self.v_j = v_j