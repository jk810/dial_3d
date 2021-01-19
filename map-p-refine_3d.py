from pathlib import Path
import random
import copy
import time

import numpy as np
from sklearn import manifold
from sklearn import metrics
from scipy import optimize
from scipy.spatial import distance_matrix
import networkx as nx

from shortest_paths_nx import shortest_path
from timing import timing
import plot_3d


def get_data(dat):
    """
    Reads xyz.txt and distance.txt and creates numpy arrays.
    """
    current_file_path = Path(__file__).resolve()
    code_source_path = str(current_file_path.parents[0])
    r_path = f'{code_source_path}/3d_results_dis/{dat}'

    xyz = []
    xyz_file = open(r_path + '/xyz.txt', 'r')
    for line in xyz_file:
        _a = line.split()
        xyz.append((float(_a[0]), float(_a[1]), float(_a[2])))
    xyz_file.close()

    xyz_np = np.array(xyz, dtype='float')

    with open(r_path + f'/distance.txt') as f:
        d_tab = [line.strip('\n').split(' ') for line in f]
    d_tab = [[float(j_) for j_ in k] for k in d_tab]

    d_tab_np = np.array(d_tab, dtype='float')

    return xyz_np, d_tab_np, r_path


def make_graph(r_path, nodes):
    """
    Call shortest_path, which reads links_distance.txt and creates networkx
    graph object.
    Adds degrees for nodes with 0 connections, returns graph object.
    """
    gg = shortest_path(r_path)

    # add degrees for nodes with 0 connections
    for i in range(nodes):
        if str(i) not in list(gg.nodes()):
            gg.add_node(str(i), weight=0)
    return gg


def mds_map(pairwise_d_array, n_list):
    """
    Perform MDS mapping for local neighborhood using pairwise distance array
    """
    ln_xyz_dict = {}

    # perform multi-dimensional scaling
    mds = manifold.MDS(n_components=3, max_iter=300, eps=1e-6, random_state=1,
                       n_jobs=1, n_init=1, dissimilarity='precomputed')
    ln_xyz_array = mds.fit_transform(pairwise_d_array)

    # create dictionary for xyz estimates
    for i, n_ in enumerate(n_list):
        ln_xyz_dict[n_] = ln_xyz_array[i]
    
    return ln_xyz_dict


def refine_objective(coeff, ln_distance_table, map_xyz_array):
    """
    Objective function for the least squares optimization. Calculates pairwise
    distance of local neighborhood mapping estimate, then calculates distance
    matrix between true ln_distance_table and estimates ln_d_table
    """

    # calculate distance table of mapped xyz_array
    map_distance_table = metrics.pairwise_distances(map_xyz_array)

    err = distance_matrix(ln_distance_table, map_distance_table)

    return np.mean(err)


def refine_map(ln_d_table, initial_mds_map):
    """
    Refine the result of the MDS-MAP with Levenberg-Marquardt optimization.
    Uses the initial MDS-MAP result as the starting estimate.
    Limited to 10 iterations of L-M optimization.
    1 to n-hop neighbors are equally weighted (not tracking n_hop value in 
    ln_d_table)
    """
    keys = list(initial_mds_map.keys())
    xyz_array = np.array(list(initial_mds_map.values()))

    refined_array = optimize.minimize(refine_objective, xyz_array, args=(ln_d_table, xyz_array))

    # not sure how coords is ordered: x1, x2, x3, ... x2, y1, y2, y3, ... yn, 
    # assuming it is x1, y1, z1, ..., xn, yn, zn
    coords = refined_array.x

    grouped_coords = np.split(coords, int(len(coords)/3))

    grouped_coords_dict = {}

    for i, j in enumerate(keys):
        grouped_coords_dict[j] = grouped_coords[i]

    return grouped_coords_dict


def kabsch(temp_map_points, temp_true_points):
    """
    Find the optimal linear transformation between two sets of points using the
    Kabsch algorithm. Returns the rotation matrix and translation vector
    """
    map_points = copy.copy(temp_map_points)
    true_points = copy.copy(temp_true_points)

    map_centroid = np.average(map_points, axis=0)
    true_centroid = np.average(true_points, axis=0)

    map_points -= map_centroid
    true_points -= true_centroid

    h = np.dot(map_points.T, true_points)
    u, s, vt = np.linalg.svd(h)
    v = vt.T

    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    # r = v @ e @ u.T
    r = v @ u.T
    tt = true_centroid - r @ map_centroid

    return r, tt


def assemble_patches(all_ln_maps, anch):
    """
    Assembles the network by merging patches, based on amount of overlap.
    Returns the merged network and the mapped anchors as numpy arrays.
    """
    n_ = len(all_ln_maps)
    core_node = random.choice(anch)

    core_map_size = 0
    core_map = all_ln_maps[core_node]
    del all_ln_maps[core_node]

    # iterate until all nodes are in the core_map_list
    while core_map_size < n_:
        core_map_list = list(core_map.keys())

        # find node/patch with most common nodes. Iterates over every ln
        max_overlap = 0
        max_overlap_node = -5

        for center, ln in all_ln_maps.items():
            overlap_temp = list(set(core_map_list) & set(list(ln.keys())))
            n_overlap = len(overlap_temp)

            # find most overlapping ln that is not entirely contained in the
            # core, and that will actually add new nodes to the core
            if max_overlap < n_overlap < len(list(ln.keys())):
                max_overlap_node = center
                max_overlap = n_overlap

        overlap_nodes = list(set(core_map_list) &
                             set(all_ln_maps[max_overlap_node].keys()))
        max_overlap_ln = all_ln_maps[max_overlap_node]

        core_overlap_xyz = [core_map[node] for node in overlap_nodes]
        prime_overlap_xyz = [max_overlap_ln[node] for node in overlap_nodes]

        core_overlap_np = np.vstack(core_overlap_xyz)
        prime_overlap_np = np.vstack(prime_overlap_xyz)

        # Calculate transformation between core and overlapping ln
        rr, ttt = kabsch(prime_overlap_np, core_overlap_np)

        # Merge core map with most overlapping ln
        for node, i in max_overlap_ln.items():
            new_p = rr @ i + ttt
            p = np.reshape(new_p, (1, 3))

            if node in core_map:
                core_map[node] = (core_map[node] + p) / 2
            else:
                core_map[node] = p

        del all_ln_maps[max_overlap_node]
        core_map_size = len(list(core_map.keys()))

    # Create aggregate numpy array of merged network (in sorted node order)
    # and average all coordinate estimates
    all_xyz = []
    for i in range(n_):
        all_xyz.append(core_map[i])

    assem_xyz = np.vstack(all_xyz)
    assem_anchors = np.array([assem_xyz[i] for i in anch])

    return assem_xyz, assem_anchors


if __name__ == "__main__":
    t0 = time.time()

    n_node = 400
    n_trial = 50
    hop_lim = 3

    data_name = f'{n_node}sat_6con'

    # Get xyz and distance numpy arrays, create networkx graph object
    true_xyz, d_table, results_path = get_data(data_name)
    g = make_graph(results_path, n_node)

    best_rmse = 2000
    all_rmse = []
    all_sim_time = []
    network_dat = []

    for _ in range(n_trial):
        t0 = time.time()

        anchors = random.sample(range(n_node), 4)
        true_anchors = np.array([true_xyz[i] for i in anchors])

        ln_maps = {}
        ln_sizes = []

        for n in range(n_node):
            # Construct local neighborhood based on hop limit
            ln_dict = nx.single_source_shortest_path(g, str(n), cutoff=hop_lim)
            ln_node_list = [int(x) for x in ln_dict.keys()]

            n_ln_nodes = len(ln_node_list)
            ln_sizes.append(n_ln_nodes)

            # Create local neighborhood distance table
            ln_d_table = np.empty((n_ln_nodes, n_ln_nodes))
            for i, a in enumerate(ln_node_list):
                for j, b in enumerate(ln_node_list):
                    ln_d_table[i][j] = d_table[int(a)][int(b)]

            # Add the mapped ln to the aggregate dictionary
            initial_local_map = mds_map(ln_d_table, ln_node_list)
            ln_maps[n] = refine_map(ln_d_table, initial_local_map)
        
        rel_xyz, rel_anchors = assemble_patches(ln_maps, anchors)

        R, t = kabsch(rel_anchors, true_anchors)

        avg_ln_size = sum(ln_sizes)/len(ln_sizes)

        # apply transformation to all points
        abs_ps = []
        for i in rel_xyz:
            abs_p = R @ i + t
            abs_ps.append(np.reshape(abs_p, (1, 3)))

        abs_xyz = np.vstack(abs_ps)
        abs_anchors = np.array([abs_xyz[i] for i in anchors])

        # calculate rmse
        rmse = np.linalg.norm(true_xyz - abs_xyz, axis=1)
        avg_rmse = np.mean(rmse)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            network_dat = [true_xyz, true_anchors, abs_xyz, abs_anchors,
                           anchors, best_rmse, avg_ln_size, results_path]

        all_rmse.append(avg_rmse)
        all_sim_time.append(time.time() - t0)

    print(f'sim time {sum(all_sim_time):.2f} sec')

    avg_sim_time = np.mean(all_sim_time)

    plot_3d.network(network_dat[0], network_dat[1], network_dat[2],
                    network_dat[3], network_dat[4], network_dat[5],
                    network_dat[6], network_dat[7], hop_lim)
    plot_3d.rmse_time(all_rmse, avg_sim_time, n_node, hop_lim, n_trial,
                      results_path)
