import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

from shortest_paths_nx import shortest_path


def network(true_xyz, true_anchors, abs_xyz, abs_anchors,
            anchors, rmse, ln_size, results_path, hop_lim):

    anchors_str = [str(x) for x in anchors]

    n_nodes = true_xyz.shape[0]
    node_list = [str(i) for i in range(n_nodes)]

    # Create graph object based on links and distances txt files
    g = shortest_path(results_path)

    # get node degrees in sorted list
    node_and_degree = {node: val for node, val in g.degree()}
    # add degrees for nodes with 0 connections
    for i in range(n_nodes):
        if str(i) not in node_and_degree:
            node_and_degree[str(i)] = 0
            g.add_node(str(i), weight=0)

    # Plotting
    # Iterate over local nbrhood list to extract the xyz coordinates of each
    # neighborhood node
    xa = []
    ya = []
    za = []
    for i in true_anchors:
        xa.append(i[0])
        ya.append(i[1])
        za.append(i[2])
    t_anchors = go.Scatter3d(x=xa, y=ya, z=za, mode='markers',
                             marker=dict(color='dodgerblue', size=4,
                                         line=dict(color='black', width=0.2)),
                             hoverinfo='text', name='True anchors',
                             text=anchors_str)
    t_anchors['showlegend'] = True

    xr = []
    yr = []
    zr = []
    for i in abs_anchors:
        xr.append(i[0])
        yr.append(i[1])
        zr.append(i[2])
    m_anchors = go.Scatter3d(x=xr, y=yr, z=zr, mode='markers',
                             marker=dict(color='salmon', size=4,
                                         line=dict(color='black', width=0.2)),
                             hoverinfo='text', name='Mapped anchors',
                             text=anchors_str)
    m_anchors['showlegend'] = True

    xo = []
    yo = []
    zo = []
    for i in true_xyz:
        xo.append(i[0])
        yo.append(i[1])
        zo.append(i[2])
    true_nodes = go.Scatter3d(x=xo, y=yo, z=zo, mode='markers',
                              marker=dict(color='white', size=4,
                                          line=dict(color='darkblue',
                                                    width=0.2)),
                              text=node_list, hoverinfo='text',
                              name='True nodes')
    true_nodes['showlegend'] = True

    # xo = []
    # yo = []
    # zo = []
    # for i in abs_xyz:
    #     xo.append(i[0])
    #     yo.append(i[1])
    #     zo.append(i[2])
    # m_nodes = go.Scatter3d(x=xo, y=yo, z=zo, mode='markers',
    #                          marker=dict(color='white', size=4,
    #                                      line=dict(color='red', width=0.2)),
    #                          text=node_list, hoverinfo='text',
    #                          name='Mapped nodes')
    # m_nodes['showlegend'] = True

    x_stem = []
    y_stem = []
    z_stem = []
    for j in range(n_nodes):
        x = [true_xyz[j][0], abs_xyz[j][0], None]
        y = [true_xyz[j][1], abs_xyz[j][1], None]
        z = [true_xyz[j][2], abs_xyz[j][2], None]

        x_stem += x
        y_stem += y
        z_stem += z

    stems = go.Scatter3d(x=x_stem, y=y_stem, z=z_stem, mode='lines',
                         line=dict(color='red', width=0.75),
                         hoverinfo='none', name='Error')

    # # Plot edges
    x_ext = []
    y_ext = []
    z_ext = []
    for j in g.edges([str(i) for i in range(n_nodes)]):
        x = [true_xyz[int(j[0])][0], true_xyz[int(j[1])][0], None]
        y = [true_xyz[int(j[0])][1], true_xyz[int(j[1])][1], None]
        z = [true_xyz[int(j[0])][2], true_xyz[int(j[1])][2], None]

        x_ext += x
        y_ext += y
        z_ext += z
    edges = go.Scatter3d(x=x_ext, y=y_ext, z=z_ext, mode='lines',
                         line=dict(color='rgb(0, 0, 0)', width=0.4),
                         hoverinfo='none', name='Links')

    # Plot earth
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
    xx = 6371 * np.cos(u) * np.sin(v)
    yy = 6371 * np.sin(u) * np.sin(v)
    zz = 6371 * np.cos(v)

    cscale = [[0.0, "rgb(240, 240, 240)"],
              [0.111, "rgb(225, 225, 225)"],
              [0.222, "rgb(210, 210, 210)"],
              [0.333, "rgb(195, 195, 195)"],
              [0.444, "rgb(180, 180, 180)"],
              [0.555, "rgb(165, 165, 165)"],
              [0.666, "rgb(150, 150, 150)"],
              [0.777, "rgb(135, 135, 135)"],
              [0.888, "rgb(120, 120, 120)"],
              [1.0, "rgb(105, 105, 105)"]]

    contours = dict(x=dict(highlight=False), y=dict(highlight=False),
                    z=dict(highlight=False))
    sphere = go.Surface(x=xx, y=yy, z=zz, colorscale=cscale, showscale=False,
                        hoverinfo='none', contours=contours, reversescale=True,
                        name='Surface')
    sphere['showlegend'] = True

    # Plot layout
    noaxis = dict(showbackground=False, showticklabels=False, title='',
                  showspikes=False)
    layout3d = dict(font=dict(family='Arial'),
                    scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis))

    data = [sphere, stems, true_nodes, t_anchors]

    fig = go.Figure(data=data, layout=layout3d)

    t_ = f'True and Mapped Positions<br>RMSE: {rmse:.2f} km<br>Anchors ' \
         f'nodes: [{anchors[0]} - {anchors[1]} - {anchors[2]} - ' \
         f'{anchors[3]}]<br> Average ln size: {ln_size:.2f} nodes'
    fig.update_layout(title_text=t_, title_font_size=20, title_x=0.5)

    plotly.offline.plot(fig, filename=f'{results_path}/{hop_lim}hop_map_p.html')


def rmse_time(al_rmse, avg_sim_time, n_node, hop_lim, n_trial, results_path):
    plt.style.use('bmh')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)
    plt.rcParams['font.sans-serif'] = 'Arial'

    fig, ax = plt.subplots(figsize=(16, 8))
    x = range(1, n_trial+1)
    # all_rmse = [1000*i for i in al_rmse]

    x_ticks = list(range(0, n_trial+1, 10))

    ticks = [1, '', '', '', '', '', '', '', '', '']
    for i in range(1, -(-n_trial // 10)):
        sub_ticks = [i*10, '', '', '', '', '', '', '', '', '']
        ticks += sub_ticks
    ticks = ticks[:n_trial]

    ax.plot(x, al_rmse, color='C1', linewidth=1)
    ax.set_xlabel('Trial #', fontsize=11)
    ax.set_ylabel("Average RMSE [km]", fontsize=11)
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    ax.set_title(f'{n_node} Node Network RMSE', {'fontsize': 13})
    ax.set_xlim(0, n_trial+1)
    ax.set_ylim(min(al_rmse)*.5, max(al_rmse)*1.1)
    ax.set_xticks(x_ticks)
    # ax.xaxis.set_ticklabels([str(i) for i in list(x)])
    mean = r'$\mu_{rmse} = $' + f"{np.mean(al_rmse):.2f} km" + '\n' + \
           r'$\mu_{sim time} = $' + f"{avg_sim_time:.2f} s"
    ax.text(.98, .90, mean, transform=ax.transAxes, va='baseline',
            fontsize=11,
            ha='right', bbox=dict(facecolor='C0', edgecolor='k', alpha=.6))

    fig.savefig(f'{results_path}/{hop_lim}hop_{n_trial}sim_rmse.png', dpi=200)
    plt.show()


def aggregate_rmse_time():
    plt.style.use('bmh')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    plt.rcParams['font.sans-serif'] = 'Arial'

    fig, ax = plt.subplots(figsize=(12, 6))

    x = [100, 200, 300, 400, 500, 1000]
    rmse = [51.49, 31.63, 22.18, 20.86, 16.93, 13.17]
    time = [0.033, 0.065, 0.257, 0.845, 1.301, 5.760]

    # cubic = [i**3/150000000 for i in x]

    ax.plot(x, rmse, color='C0', marker='v', linestyle='dotted', linewidth=1,
            markersize=7)
    ax.set_xlabel('Network size [# nodes]', fontsize=12)
    ax.set_ylabel('RMSE [m]', fontsize=12)
    ax.grid(linestyle=':')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelcolor='C0', labelsize=12, colors='C0')
    ax.yaxis.label.set_color('C0')
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xticks(x)

    ax2 = ax.twinx()
    ax2.set_ylabel('Sim time [s]', fontsize=12)
    ax2.plot(x, time, color='C1', marker='^', linestyle='dotted', linewidth=1,
             markersize=7)
    # ax2.plot(x, cubic)
    ax2.tick_params(axis='y', labelcolor='C1', labelsize=12, colors='C1')
    ax2.yaxis.label.set_color('C1')

    ax.text(.5, .9, '* each point averaged over 1000 trials', transform=ax.transAxes,
            fontsize=12, ha='center',
            bbox=dict(facecolor='C2', edgecolor='k', alpha=0.2))
    ax.set_title('RMSE and Simulation Time vs Network Size', {'fontsize': 14})

    plt.show()
