from re import sub
import numpy as np
import torch


def get_communities(graph, ind):
    I = np.copy(ind)
    comms_edges = []
    edge_used = []
    pos = 0

    while len(edge_used) < graph.edges_list.num_edges:
        comm_e = []
        while I[pos] != -1:
            edge_used.append(pos)
            comm_e.append(pos)
            
            edge_id = I[pos]
            I[pos] = -1
            pos = int(edge_id)
        
        comms_edges.append(comm_e)
        for i in range(len(I)):
            if I[i] != -1:
                pos = i
                break
    
    comms_nodes = []
    for i in range(len(comms_edges)):
        comm_e = comms_edges[i]
        comm_n = []
        for j in range(len(comm_e)):
            id_node_source = graph.edges_list.edges[int(comm_e[j])].v_i
            id_node_target = graph.edges_list.edges[int(comm_e[j])].v_j
            comm_n.append(id_node_source)
            comm_n.append(id_node_target)

        comms_nodes.append(np.unique(comm_n))

    return comms_nodes


def sum_as(graph, v, adj_vertices):
    a = 0
    for j in range(len(adj_vertices)):
        for h in range(len(adj_vertices)):
            a += graph.adj_matrix[j][h] \
                * graph.adj_matrix[int(v)][j] \
                * graph.adj_matrix[int(v)][h] \
                * graph.adj_matrix[j][int(v)] \
                * graph.adj_matrix[h][int(v)]
    
    return a


def local_clustering_coefficient(graph, comms):
    lcc_mean = 0
    for comm in comms:
        subset = torch.tensor(comm, dtype=torch.long)
        subgraph = graph.data.subgraph(subset)
        
        lcc_comm = 0
        for v in subgraph.x:
            adj_vertices = []
            for i in range(len(subgraph.edge_index[0])):
                edge = subgraph.edge_index[:,i]
                if (v == edge[0]) \
                and (edge[1] not in adj_vertices):
                    adj_vertices.append(edge[1])
            
            k = len(adj_vertices)

            a = sum_as(graph, v, adj_vertices)

            if k > 1:
                lcc_comm += (2 * a) / (k * (k - 1))
            else:
                lcc_comm += 0
        
        lcc_comm /= len(comm)
        lcc_mean += lcc_comm
    
    lcc_mean /= len(comms)
    
    return lcc_mean


def separability(graph, comms):
    sep = 0
    for comm in comms:
        subset = torch.tensor(comm, dtype=torch.long)
        subgraph = graph.data.subgraph(subset)
        
        internal_edges = 0
        external_edges = 0
        for i in range(len(graph.data.edge_index[0])):
            edge = graph.data.edge_index[:,i]

            if (edge[0] in subgraph.x) \
                and (edge[1] in subgraph.x):
                internal_edges += 1
                
                #if verbose:
                    #print("Edge v in C: {}".format(edge))
            
            if (edge[0] in subgraph.x) \
                and (edge[1] not in subgraph.x):
                external_edges += 1

                #if verbose:
                    #print("Edge v not in C: {}".format(edge))
        
        if external_edges > 0:
            sep += internal_edges / external_edges
        else:
            sep += 0
    
    return sep/len(comms)