from pathlib import Path
import random
import copy
import time

import numpy as np
from sklearn import manifold

from timing import timing
import plot_3d


# @timing
def get_data(dat):
    """
    Reads xyz.txt and distance.txt to create numpy arrays.
    """
    current_file_path = Path(__file__).resolve()
    code_source_path = str(current_file_path.parents[0])
    r_path = f'{code_source_path}/3d_results_cen/{dat}'

    xyz = []
    xyz_file = open(r_path + '/xyz.txt', 'r')
    for line in xyz_file:
        a = line.split()
        xyz.append((float(a[0]), float(a[1]), float(a[2])))
    xyz_file.close()

    xyz_np = np.array(xyz, dtype='float')

    with open(r_path + f'/distance.txt') as f:
        d_tab = [line.strip('\n').split(' ') for line in f]
    d_tab = [[float(j) for j in k] for k in d_tab]

    d_tab_np = np.array(d_tab, dtype='float')

    return xyz_np, d_tab_np, r_path


# @timing
def mds_map_table(pairwise_d_array):
    """
    Perform MDS mapping for entire network using pairwise distance array
    """
    mds = manifold.MDS(n_components=3, max_iter=300, eps=1e-6, random_state=1,
                       n_jobs=1, n_init=1, dissimilarity='precomputed')
    xyz_est_array = mds.fit_transform(pairwise_d_array)

    return xyz_est_array


# @timing
def kabsch(temp_mapping_points, temp_true_points):
    # R, res, rank, s = np.linalg.lstsq(mapped_anchors, true_anchors,
    # rcond=None)

    # Q = mapped_anchors[1:] - mapped_anchors[0]
    # Q_prime = true_anchors[1:] - true_anchors[0]
    # # calculate rotation matrix
    # R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
    # np.row_stack((Q_prime, np.cross(*Q_prime))))
    # # calculate translation vector
    # t = true_anchors[0] - np.dot(mapped_anchors[0], R)

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

    h = mapping_points.T @ true_points
    u, s, vt = np.linalg.svd(h)
    v = vt.T

    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    # r = v @ e @ u.T
    r = v @ u.T

    tt = true_centroid - r @ mapped_centroid

    return r, tt


if __name__ == "__main__":
    n_node = 5000
    n_trial = 100

    data_name = f'{n_node}sat_6con'

    # Get xyz and distance table as numpy arrays
    true_xyz, d_table, results_path = get_data(data_name)

    best_rmse = 2000
    all_rmse = []
    all_sim_time = []
    network_dat = []

    for _ in range(n_trial):
        t0 = time.time()
        anchors = random.sample(range(n_node), 4)
        true_anchors = np.array([true_xyz[i] for i in anchors])

        # perform mds on the entire network
        rel_xyz = mds_map_table(d_table)

        rel_anchors = np.array([rel_xyz[i] for i in anchors])

        # Use anchor nodes to absolutely localize the network
        R, t = kabsch(rel_anchors, true_anchors)

        # apply transformation
        map_list = []
        for i in rel_xyz:
            point = R @ i + t
            map_list.append(np.reshape(point, (1, 3)))   # reshape point to 1x3
        mapped_xyz = np.vstack(map_list)
        # mapped_xyz[:, 2] *= -1
        mapped_anchors = np.array([mapped_xyz[i] for i in anchors])

        # calculate rmse
        rmse = np.linalg.norm(true_xyz - mapped_xyz, axis=1)
        avg_rmse = np.mean(rmse)

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            network_dat = [true_xyz, true_anchors, mapped_xyz, mapped_anchors,
                           anchors, best_rmse, 0, results_path, 0]

        all_rmse.append(avg_rmse)
        all_sim_time.append(time.time() - t0)

    print(f'sim time {sum(all_sim_time):.2f} sec')

    avg_sim_time = np.mean(all_sim_time)

    plot_3d.network(network_dat[0], network_dat[1], network_dat[2],
                    network_dat[3], network_dat[4], network_dat[5],
                    network_dat[6], network_dat[7], network_dat[8])
    plot_3d.rmse_time(all_rmse, avg_sim_time, n_node, 0, n_trial, results_path)

    # plot_3d.aggregate_rmse_time()
