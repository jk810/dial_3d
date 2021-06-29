from shortest_paths_nx import shortest_path
import numpy as np
import matplotlib.cm as cm
from matplotlib import colors
from pathlib import Path
import networkx as nx

import plotly
import plotly.graph_objs as go
from timing import timing


@timing
def plot_2d(data_name, center_node, n_hop, colormap):
    current_file_path = Path(__file__)
    results_path = str(current_file_path.parents[1]) + f'/results/{data_name}/'

    # open xy text file and extract values
    xy = open(results_path + 'xy.txt', 'r')
    pos = {}
    for i, line in enumerate(xy):
        a = line.split()
        pos[str(i)] = (float(a[0]), float(a[1]))
    xy.close()

    n_nodes = len(pos)
    # Create graph object based on links and distances txt files
    G = shortest_path(results_path)

    # get node degrees in sorted list
    node_and_degree = {}
    for node, val in G.degree():
        node_and_degree[node] = val
    # add degrees for nodes with 0 connections
    for i in range(n_nodes):
        if str(i) not in node_and_degree:
            node_and_degree[str(i)] = 0
            G.add_node(str(i), weight=0)

    x_center = pos[str(center_node)][0]
    y_center = pos[str(center_node)][1]

    frontier_list = []
    local_n_list = []
    local_n_list_reduced = []

    # Each frontier is a dictionary of local neighborhood
    for i in range(n_hop + 1):       # includes 0-hop neighborhood
        frontier_list.append(nx.single_source_shortest_path(G, str(center_node), cutoff=i))
        # get frontier keys, which are local neighborhood nodes
        local_n_list.append([str(x) for x in list(frontier_list[i].keys())])
    local_n_size = len(local_n_list[-1])

    # Create 'reduced' list, manually add 0, 1 hop neighbor nodes
    for i in range(0, 2):
        local_n_list_reduced.append(local_n_list[i])
    # Remove lower level (-2 levels) hop nodes so each frontier only includes 'growth'
    for i in range(2, n_hop + 1):
        local_n_list_reduced.append([x for x in local_n_list[i] if
                                    x not in local_n_list[i - 2]])

    # Plotting
    # Iterate over local nbrhood list to extract the xy coordinates of each neighborhood node
    xi = []
    yi = []
    for i in local_n_list[-1]:
        xi.append(pos[i][0])
        yi.append(pos[i][1])
    # mode='markers+text' for labels
    nbr_nodes = go.Scatter(x=xi, y=yi, mode='markers',
                             marker=dict(color='red', size=3),
                             text=local_n_list[-1], hoverinfo='none', name='LN nodes')
    nbr_nodes['showlegend'] = True

    # center node
    center = go.Scatter(x=[x_center], y=[y_center], mode='markers',
                             marker=dict(color='azure', size=6, line=dict(color='black', width=1)),
                             text=str(center_node), hoverinfo='none', name=f'Center node')
    center['showlegend'] = False

    # Plot nodes external to local neighborhood
    xo = []
    yo = []
    all_nodes = [str(x) for x in list(range(n_nodes))]
    for i in all_nodes:
        xo.append(pos[i][0])
        yo.append(pos[i][1])
    ext_nodes = go.Scatter(x=xo, y=yo, mode='markers',
                             marker=dict(color='black', size=2),
                             text=all_nodes, hoverinfo='text', name='External nodes')
    ext_nodes['showlegend'] = True

    # get link colors by hop path length
    link_colors = []
    cmap = cm.get_cmap(colormap, n_hop)
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        link_colors.append(colors.rgb2hex(rgb))
    link_colors.insert(0, '#000000')    # insert color for 0-hop neighborhood

    # Plot local neighborhood edges
    # Iterate over list of edges to get the xy, coordinates of connected nodes
    grow_edges = []
    existing_links = []
    for i, frontier in enumerate(local_n_list_reduced):
        x_grow = []
        y_grow = []
        for j in G.edges(frontier):
            x = [pos[j[0]][0], pos[j[1]][0], None]
            y = [pos[j[0]][1], pos[j[1]][1], None]
            if j[0] in frontier and j[1] in frontier and nx.shortest_path_length(G, str(center_node), j[0]) == i-1:
                x_grow += x
                y_grow += y
                existing_links.append([j[0], j[1]])
        grow_edges.append(go.Scatter(x=x_grow, y=y_grow,
                                       mode='lines', line=dict(color=link_colors[i], width=.7),
                                       name=f'{i}-hop', hoverinfo='none'))

    # Plot edges not in local neighborhood
    x_ext = []
    y_ext = []

    for j in G.edges([str(i) for i in range(n_nodes)]):
        x = [pos[j[0]][0], pos[j[1]][0], None]
        y = [pos[j[0]][1], pos[j[1]][1], None]
        # if [j[0], j[1]] not in existing_links and [j[1], j[0]] not in existing_links:
        x_ext += x
        y_ext += y
    ext_edges = go.Scatter(x=x_ext, y=y_ext, mode='lines',
                             line=dict(color='rgb(0, 0, 0)', width=0.2),
                             hoverinfo='none', name='External links')
    
    # Plot layout
    xaxis = dict(ticks='outside', title='', zeroline=False, showspikes=False, linecolor='grey')
    yaxis = dict(ticks='outside', title='', zeroline=False, showspikes=False, linecolor='grey',
                 scaleanchor='x', scaleratio=1, )
    layout = dict(title=f'{data_name} network<br>'
                    f'Node {center_node}: {local_n_size} nodes in local neighborhood',
                    font=dict(family='Arial'), plot_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis=xaxis, yaxis=yaxis)

    data = grow_edges
    # data = []
    data.append(ext_edges)
    data.append(ext_nodes)
    data.append(nbr_nodes)
    data.append(center)
    fig = dict(data=data, layout=layout)

    plotly.offline.plot(fig, filename=f'{results_path}/{data_name}_network.html')


# -------------------------------------
show_ext = True
ccc = 'jet'     # rainbow, Set1, Set2
n_con = 6
f_n_hop = 4

n_hop = 4
center_node = 7
n_sat = 400

data_name = f'{n_sat}sat_{f_n_hop}hop_{n_con}con'
plot_2d(data_name=data_name, center_node=center_node, n_hop=n_hop, colormap=ccc)
