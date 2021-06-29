from pathlib import Path
import numpy as np
from sklearn import manifold
import random
import copy

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
    r_path = f'{code_source_path}/2d_results/{dat}'

    xy = []
    xy_file = open(r_path + '/xy.txt', 'r')
    for line in xy_file:
        a = line.split()
        xy.append((float(a[0]), float(a[1])))
    xy_file.close()

    xy_np = np.array(xy, dtype='float')

    with open(r_path + f'/distance.txt') as f:
        d_tab = [line.strip('\n').split(' ') for line in f]
    d_tab = [[float(j) for j in k] for k in d_tab]

    d_tab_np = np.array(d_tab, dtype='float')

    return xy_np, d_tab_np, r_path


@timing
def mds_map_table(pairwise_d_array):
    """
    Perform MDS mapping for entire network pairwise distance array
    """
    mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-6, random_state=1,
                       n_jobs=1, n_init=1, dissimilarity='precomputed')
    xy_est_array = mds.fit_transform(pairwise_d_array)

    return xy_est_array


@timing
def kabsch(temp_mapping_points, temp_true_points):
    # x, res, rank, s = np.linalg.lstsq(mapped_anchors, true_anchors,
    # rcond=None)

    # Q = rel_anchors[1:] - rel_anchors[0]
    # Q_prime = true_anchors[1:] - true_anchors[0]
    # # calculate rotation matrix
    # R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
    # np.row_stack((Q_prime, np.cross(*Q_prime))))
    # # calculate translation vector
    # t = p_prime[0] - np.dot(p[0], R)

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

    h = np.dot(mapping_points.T, true_points)
    u, s, vh = np.linalg.svd(h)

    r = np.dot(vh, u.T)

    det_r = np.linalg.det(r)
    if det_r < 0:
        vh[:, 1] *= det_r
        r = np.dot(vh, u.T)

    tt = true_centroid - np.dot(r, mapped_centroid)

    return r, tt


if __name__ == "__main__":
    n_ = 300
    data_name = f'{n_}sat_4hop_6con'

    # Get xy and distance table as numpy arrays
    true_xy, d_table, results_path = get_data(data_name)

    anchors = [1, 25, 50]
    true_anchors = np.array([true_xy[x] for x in anchors])

    # Map the local neighborhood patch and add to the aggregate dict
    xy_estimate = mds_map_table(d_table)

    mapped_anchors = np.array([xy_estimate[x] for x in anchors])

    # Use anchor nodes to get optimal linear transformation
    R, t = kabsch(mapped_anchors, true_anchors)

    # apply transformation to entire mapped network
    map_list = []
    for i in xy_estimate:
        point = np.dot(R, i) + t           # use transformation matrix to map
        point = np.reshape(point, (1, 2))   # reshape
        map_list.append(point)
    abs_mapped_xy = np.vstack(map_list)
    abs_mapped_anchors = np.array([abs_mapped_xy[x] for x in anchors])

    # plot the network
    plot_2d(true_xy, true_anchors, abs_mapped_xy, abs_mapped_anchors,
            results_path, anchors)
