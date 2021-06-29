# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:55:23 2018

@author: kw31158

For ingesting text document listing which nodes are connected and the cost
associated with each of the link.

Creates dictionaries (shortest_path,shortest_path_distance,
shortest_path_combined, shortest_path_length, shortest_path_length_distance,
shortest_path_length_combined) which can be queried with [source][target] to
return the desired information.

Also returns if the graph in the .txt file is connected and if it is not, it
returns how many connected components exist and how many nodes are in the
smallest of the connected components.

------------------------
Different shortest paths:
shortest_path gives the shortest path based on the number of nodes.
shortest_path_distance gives the shortest path based on the distance.
shortest_path_combined gives the shortest path based on a weighted combination
of nodes and edges

Their counter-parts with "length" give the length of those paths.

Here's an example on how to determine the shortest path using the dictionaries
produced.
Ex: shortest_path_distance['0']['100'] will return the path from node 0 to 100
that takes the shortest path based on distance.
Ex: shortest_path_length_distance['0']['100'] will return the length of that
path from node 0 to 100.

--------------------
If we need to calculate weight with nodes having different weights, then we need
to convert each of the nodes into a pair of nodes:

G = nx.Graph()
f = open("Links&Distances.txt", 'r')
for line in f:
    test = line.strip("\n").split(' ')
    G.add_edge(test[0]+"b", test[1]+"a", weight=float(test[2]))
for node in list(G.nodes):
    if "a" in node:
        G.add_edge(node, node.strip("a") + "b", weight = 100)
        # or whatever the weight should be (stored in a .txt file?)
"""

import networkx as nx


def shortest_path(route_source_path):
    G = nx.Graph()

    f = open(route_source_path + '\links_distance.txt', 'r')
    for line in f:
        test = line.strip("\n").split(' ')  # remove the new line and spaces
        G.add_edge(test[0], test[1], weight=float(test[2]))  # create the nodes and links
    f.close()

#    connects = {'True':0,'False':0}
#    print("The given graph is connected: {}".format(nx.is_connected(G)))
#    if nx.is_connected(G):
#        connects['True'] += 1
#    else:
#        connects['False'] += 1
    # returns true if the graph is connected (every node can reach every other node)

    # if nx.is_connected(G) is False:
    #     n_connected_components = nx.number_connected_components(G)
    #     print("There are {} connected components.".format(n_connected_components))
    #     # returns the number nodes that are in the smallest of the connected components.
    #     print("The smallest of these components contains {} nodes.".format(len(min(nx.connected_components(G),key=len))))
    #     # returns the number of connected components.
    #     # Ex: 2 indicates that there are two connected sets of nodes.


#    min(nx.connected_components(G),key=len)
#    # returns the nodes in the smallest connected component.

#    # All the different paths and lengths we are interested in:

#    combined_function =  lambda o, d, _dict : G.edges[o,d]["weight"]+100
#    # For the combined path, I am assuming each node adds 100 to the cost of the
#    # path, this can be changed to whatever function we think is reasonable
#
    # shortest_path = dict(nx.all_pairs_dijkstra_path(G, weight=1))
#    shortest_path_distance = dict(nx.all_pairs_dijkstra_path(G,weight = "weight"))
#    shortest_path_combined = dict(nx.all_pairs_dijkstra_path(G,weight = combined_function ))


#    shortest_path_length = dict(nx.all_pairs_dijkstra_path_length(G,weight = 1))
#    shortest_path_length_distance = dict(nx.all_pairs_dijkstra_path_length(G,weight = "weight"))
#    shortest_path_length_combined = dict(nx.all_pairs_dijkstra_path_length(G,weight =  combined_function))


#    '''Note: it is probably faster to use the calculated paths to find the length
#    than to find the length using all_pairs_dijkstra_path_length, since
#    all_pairs_dijsktra_path_length calculates the path again. This is something
#    we could implment later if we are concerned about speed.'''

#    print(connects)
#    shortest_path = None
    return G
