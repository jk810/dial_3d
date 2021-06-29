from pathlib import Path
import numpy as np
from sklearn import manifold
import networkx as nx
import copy

from shortest_paths_nx import shortest_path
from timing import timing
from plot_2d import plot_2d


@timing
def get_data(dat):
    """
    Reads xy.txt and creates xy numpy array.
    Reads distance.txt and creates distance table numpy array.
    """
    current_file_path = Path(__file__).resolve()
    code_source_path = str(current_file_path.parents[0])
    results_path = f'{code_source_path}/2d_results/{dat}'

    xy = []
    xy_file = open(results_path + '/xy.txt', 'r')
    for line in xy_file:
        a = line.split()
        xy.append((float(a[0]), float(a[1])))
    xy_file.close()

    xy_np = np.array(xy, dtype='float')

    with open(results_path + f'/distance.txt') as f:
        d_table = [line.strip('\n').split(' ') for line in f]
    d_table = [[float(j) for j in k] for k in d_table]

    d_tab_np = np.array(d_table, dtype='float')

    return xy_np, d_tab_np, results_path


@timing
def make_graph(results_path, nodes):
    """
    Call shortest_path, which reads links_distance.txt and creates networkx
    graph object.
    Adds degrees for nodes with 0 connections, returns graph object.
    """
    G = shortest_path(results_path)

    # get nodes and degrees in dictionary
    node_and_degree = {}
    for node, val in G.degree():
        node_and_degree[node] = val
    # add degrees for nodes with 0 connections
    for i in range(nodes):
        if str(i) not in node_and_degree:
            node_and_degree[str(i)] = 0
            G.add_node(str(i), weight=0)
    return G


def map_patch(ln_, ln_node_list):
    """
    Perform MDS mapping for local neighborhood position array
    """
    ln_xy_estimates = {}

    # perform multi-dimensional scaling
    mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-6, random_state=1,
                       n_jobs=1, n_init=4, dissimilarity='precomputed')
    map_xy_est = mds.fit_transform(ln_)

    # create dictionary for local neighborhood xy estimates
    for i, n in enumerate(ln_node_list):
        ln_xy_estimates[n] = map_xy_est[i]

    return ln_xy_estimates


@timing
def assemble_patches(all_patch_estimates, anchors):
    """
    Assembles the network by merging patches, based on amount of overlap.
    Returns the merged network and the mapped anchors as numpy arrays.
    """
    n_ = len(all_patch_estimates)
    # pick a random node to use as 'core' node and patch
    core_node = 0  # random.randint(0, n_ - 1)

    core_patch = all_patch_estimates[core_node]
    core_patch_count = 0

    # delete core node/patch from dictionary
    del all_patch_estimates[core_node]

    # # add coordinates to compilation of coordinates
    # averaged_estimates = {x: [] for x in list(range(n_))}
    # for k, core_node_coord in core_patch.items():
    #     averaged_estimates[k].append(core_node_coord)

    # iterate to find overlapping patches until all nodes in core_patch_list
    # while len(list(all_patch_estimates.keys())) > 0: # use every single patch
    while core_patch_count < n_:
        core_patch_list = list(core_patch.keys())

        # find node/patch with most common nodes. Iterates over every patch
        overlap_count = 0
        most_overlapping_node = -5
        for key, patch in all_patch_estimates.items():
            overlap_list_temp = list(set(core_patch_list) &
                                     set(list(patch.keys())))
            num_overlapping = len(overlap_list_temp)

            # find most overlapping node that is not entirely contained in the
            # core patch. The second condition checks that the new patch
            # actually adds new nodes to the core
            if (num_overlapping > overlap_count and
                    num_overlapping < len(list(patch.keys()))):
                most_overlapping_node = key
                overlap_count = num_overlapping

        overlap_list = list(set(core_patch_list) & set(all_patch_estimates[most_overlapping_node].keys()))
        most_overlapping_patch = all_patch_estimates[most_overlapping_node]

        # Create numpy arrays for the core and overlap patch with the xy
        # positions of the overlapping nodes
        core_overlap_xys = []
        prime_overlap_xys = []
        for node_key in overlap_list:
            core_overlap_xys.append(core_patch[node_key])
            prime_overlap_xys.append(most_overlapping_patch[node_key])
        core_overlap_np = np.vstack(core_overlap_xys)
        prime_overlap_np = np.vstack(prime_overlap_xys)

        # Calculate least squares linear transformation between the M and M'
        # overlapping nodes
        R, t = kabsch(prime_overlap_np, core_overlap_np)

        # Merge core map with most overlapping patch using lstsq lin transf
        for key, val in most_overlapping_patch.items():
            added_point = np.dot(R, val) + t
            np_added_point = np.reshape(added_point, (1, 2))
            if key not in core_patch:  # add the transformed point to core patch
                core_patch[key] = np_added_point
            # else:
            #     # add transformed node coords to node coord list
            #     averaged_estimates[key].append(np_added_point)
            #     # core_patch[key] = (core_patch[key] + np_added_point ) / 2

        # remove overlapping patch after it has been merged into core patch
        del all_patch_estimates[most_overlapping_node]
        core_patch_count = len(list(core_patch.keys()))

    # Create aggregate numpy array of merged network (in sorted node order)
    # and average all coordinate estimates
    map_list = []
    for i in range(n_):
        # coord_list = averaged_estimates[i]
        # if len(coord_list) > 0:
        #     map_list.append(sum(coord_list) / len(coord_list))
        # else:
        #     map_list.append(core_patch[i])
        map_list.append(core_patch[i])

    mapped_xy = np.vstack(map_list)

    mapped_anchors = np.array([mapped_xy[anchors[0]], mapped_xy[anchors[1]],
                               mapped_xy[anchors[2]], mapped_xy[anchors[3]]])

    return mapped_xy, mapped_anchors


def kabsch(temp_mapping_points, temp_true_points):
    """
    Find the optimal linear transformation between two sets of points using the
    Kabsch algorithm. Returns the rotation matrix and translation vector
    """
    mapping_points = copy.copy(temp_mapping_points)
    true_points = copy.copy(temp_true_points)

    mapped_centroid = np.average(mapping_points, axis=0)
    true_centroid = np.average(true_points, axis=0)

    mapping_points -= mapped_centroid
    true_points -= true_centroid

    H = np.dot(mapping_points.T, true_points)
    u, s, vh = np.linalg.svd(H)

    R = np.dot(vh, u.T)

    detR = np.linalg.det(R)
    if detR < 0:
        vh[:, 1] *= detR
        R = np.dot(vh, u.T)

    t = true_centroid - np.dot(R, mapped_centroid)

    return R, t


if __name__ == "__main__":
    n_ = 400
    n_hop = 4
    max_con = 6
    data_name = f'{n_}sat_4hop_{max_con}con'

    # Get xy and distance table as numpy arrays
    true_xy, d_table, results_path = get_data(data_name)
    # Make graph object
    G = make_graph(results_path, n_)

    # Pick anchor nodes
    # anchors = random.sample(range(n_), 4)
    anchors = [10, 20, 30, 40]
    true_anchors = np.array([true_xy[anchors[0]], true_xy[anchors[1]],
                             true_xy[anchors[2]], true_xy[anchors[3]]])

    all_patch_estimates = {}    # dictionary for each node's local neighborhood
    ln_sizes = []               # tracking size of LNs (not necessary)

    for node in range(n_):
        # Construct local neighborhood based on hop limit
        ln_dict = nx.single_source_shortest_path(G, str(node), cutoff=n_hop)
        ln_node_list = [int(x) for x in ln_dict.keys()]

        num_of_ln_nodes = len(ln_node_list)

        # Create local neighborhood distance table
        ln_d_table = np.empty((num_of_ln_nodes, num_of_ln_nodes))
        for i, a in enumerate(ln_node_list):
            for j, b in enumerate(ln_node_list):
                ln_d_table[i][j] = d_table[int(a)][int(b)]
        # # Create local neighborhood xy list
        # ln_xy = np.array([true_xy[a] for a in ln_node_list])

        # Map the local neighborhood patch and add to the aggregate dict
        all_patch_estimates[node] = map_patch(ln_d_table, ln_node_list)
        # Record LN size
        ln_sizes.append(len(ln_node_list))

    avg_ln_size = sum(ln_sizes) / len(ln_sizes)

    # Assemble patches
    mapped_xy, mapped_anchors = assemble_patches(all_patch_estimates, anchors)

    # Use anchor nodes to get optimal linear transformation
    R, t = kabsch(mapped_anchors, true_anchors)

    # apply transformation to entire mapped network
    map_list = []
    for i in mapped_xy:
        point = np.dot(R, i) + t
        point = np.reshape(point, (1, 2))
        map_list.append(point)
    abs_mapped_xy = np.vstack(map_list)
    abs_mapped_anchors = np.array([abs_mapped_xy[anchors[0]],
                                   abs_mapped_xy[anchors[1]],
                                   abs_mapped_xy[anchors[2]],
                                   abs_mapped_xy[anchors[3]]])

    # plot the network
    plot_2d(true_xy, true_anchors, abs_mapped_xy, abs_mapped_anchors,
            results_path, anchors)
