from shortest_paths_nx import shortest_path

import plotly
import plotly.graph_objs as go

from timing import timing


@timing
def plot_2d(true_xy, true_anchors, abs_mapped_xy, abs_mapped_anchors,
            results_path, anchors):

    anchors_str = [str(x) for x in anchors]

    n_nodes = true_xy.shape[0]
    # get list of nodes
    node_list = []
    for i in range(n_nodes):
        node_list.append(str(i))

    # ################# Plotting
    xa = []
    ya = []
    for i in true_anchors:
        xa.append(i[0])
        ya.append(i[1])
    t_anchors = go.Scatter(x=xa, y=ya, mode='markers',
                           marker=dict(color='dodgerblue', size=8,  # cadet blue
                                       line=dict(color='black', width=0.5)),
                           hoverinfo='text', name='True anchors',
                           text=anchors_str)
    t_anchors['showlegend'] = True

    xr = []
    yr = []
    for i in abs_mapped_anchors:
        xr.append(i[0])
        yr.append(i[1])
    m_anchors = go.Scatter(x=xr, y=yr, mode='markers',
                           marker=dict(color='salmon', size=8,
                                       line=dict(color='black', width=0.5)),
                           hoverinfo='text', name='Mapped anchors',
                           text=anchors_str)
    m_anchors['showlegend'] = True

    xo = []
    yo = []
    for i in true_xy:
        xo.append(i[0])
        yo.append(i[1])
    true_nodes = go.Scatter(x=xo, y=yo, mode='markers',
                            marker=dict(color='white', size=7,
                                        line=dict(color='blue', width=0.5)),
                            text=node_list, hoverinfo='text',
                            name='True nodes')
    true_nodes['showlegend'] = True

    xo = []
    yo = []
    for i in abs_mapped_xy:
        xo.append(i[0])
        yo.append(i[1])
    mapped_nodes = go.Scatter(x=xo, y=yo, mode='markers',
                              marker=dict(color='red', size=7,
                                          line=dict(color='red', width=0.5)),
                              text=node_list, hoverinfo='text',
                              name='Mapped nodes')
    mapped_nodes['showlegend'] = True

    x_stem = []
    y_stem = []
    for j in range(n_nodes):
        x = [true_xy[j][0], abs_mapped_xy[j][0], None]
        y = [true_xy[j][1], abs_mapped_xy[j][1], None]

        x_stem += x
        y_stem += y
    stems = go.Scatter(x=x_stem, y=y_stem, mode='lines',
                       line=dict(color='red', width=0.4),
                       hoverinfo='none', name='Stems')

    # # Create graph object based on links and distances txt files
    # G = shortest_path(results_path)

    # # get node degrees in sorted list
    # node_and_degree = {}
    # for node, val in G.degree():
    #     node_and_degree[node] = val
    # # add degrees for nodes with 0 connections
    # for i in range(n_nodes):
    #     if str(i) not in node_and_degree:
    #         node_and_degree[str(i)] = 0
    #         G.add_node(str(i), weight=0)

    # # Plot edges
    # x_ext = []
    # y_ext = []
    # for j in G.edges([str(i) for i in range(n_nodes)]):
    #     x = [true_xy[int(j[0])][0], true_xy[int(j[1])][0], None]
    #     y = [true_xy[int(j[0])][1], true_xy[int(j[1])][1], None]

    #     x_ext += x
    #     y_ext += y
    # edges = go.Scatter(x=x_ext, y=y_ext, mode='lines',
    #                    line=dict(color='rgb(0, 0, 0)', width=0.4),
    #                    hoverinfo='none', name='Network links')

    # Plot layout
    xaxis = dict(ticks='outside', title='', zeroline=False, showspikes=False,
                 linecolor='grey')
    yaxis = dict(ticks='outside', title='', zeroline=False, showspikes=False,
                 linecolor='grey',
                 scaleanchor='x', scaleratio=1, )
    layout = dict(title=f'network<br>',
                  font=dict(family='Arial'), plot_bgcolor='rgba(0, 0, 0, 0)',
                  xaxis=xaxis, yaxis=yaxis)

    data = [true_nodes, stems, t_anchors, m_anchors]
    fig = dict(data=data, layout=layout)

    plotly.offline.plot(fig, filename=f'{results_path}/{n_nodes}_mapped_network.html')
