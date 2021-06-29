from pathlib import Path
import numpy as np
from timing import timing

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# @timing
def get_data(nodes, dat):
    current_file_path = Path(__file__).resolve()
    code_source_path = str(current_file_path.parents[0])
    results_path = f'{code_source_path}/results/{dat}'

    # hops_file = open(results_path + '/hops.txt', 'r')

    xy = []
    xy_file = open(results_path + '/xy.txt', 'r')
    for line in xy_file:
        a = line.split()
        xy.append((float(a[0]), float(a[1])))
    xy_file.close()

    xy_np = np.array(xy, dtype='float')

    link_dict = {key: [] for key in list(range(nodes))}
    links_file = open(results_path + f'/links.txt', 'r')
    for line in links_file:
        b = line.split()
        link_dict[int(b[0])].append(int(b[1]))
    links_file.close()

    with open(results_path + f'/distance.txt') as f:
        d_table = [line.strip('\n').split(' ') for line in f]
    d_table = [[int(float(j)) for j in k] for k in d_table]

    return xy_np, link_dict, d_table


@timing
def localize(nodes, dat):
    true_xy, link_dict, similarities = get_data(nodes, dat)

    # perform multi-dimensional scaling
    mds = manifold.MDS(n_components=2, max_iter=100, eps=1e-9,
                       random_state=None, n_jobs=1, n_init=1,
                       dissimilarity='precomputed')
    rel_map = mds.fit(similarities).embedding_

    # pick anchor nodes
    an = [30, 60, 90]

    true_anchors = np.array([true_xy[an[0]], true_xy[an[1]], true_xy[an[2]]])
    rel_anchors = np.array([rel_map[an[0]], rel_map[an[1]], rel_map[an[2]]])

    # find best linear transformation that maps relative map to absolute anchor nodes
    x, res, rank, s = np.linalg.lstsq(rel_anchors, true_anchors, rcond=None)


    # Q = rel_anchors[1:] - rel_anchors[0]
    # Q_prime = true_anchors[1:] - true_anchors[0]
    # # calculate rotation matrix
    # R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))), np.row_stack((Q_prime, np.cross(*Q_prime))))
    # # calculate translation vector
    # t = p_prime[0] - np.dot(p[0], R)

    map_list = []
    for i in rel_map:
        point = np.dot(x, i)
        point = np.reshape(point, (1, 2))
        map_list.append(point)
    abs_map = np.vstack(map_list)
    abs_anchors = np.array([abs_map[an[0]], abs_map[an[1]], abs_map[an[2]]])

##########
    # W = np.linalg.pinv(rel_anchors.T).dot(true_anchors.T).T
    # abs_map = true_anchors
    # for i in range(rel_map.shape[0]//W.shape[0]):
    #     chunk = np.matmul(W, rel_map[i*3:i*3+W.shape[0]])
    #     abs_map = np.concatenate((abs_map, chunk))
##########

    return true_xy, true_anchors, abs_map, abs_anchors


def plot_map(true_xy, true_anchors, abs_map, abs_anchors):
    # plot data
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)
    fig, ax = plt.subplots()
    ax.set_aspect(1/1)
    ax.add_patch(patches.Rectangle((-50, -50), 100, 100, fill=False, color='grey', linestyle=':'))

    ax.scatter(true_xy[:, 0], true_xy[:, 1], marker='.', color='C0')
    ax.scatter(true_anchors[:, 0], true_anchors[:, 1], marker='x', color='C1')
    ax.scatter(abs_map[:, 0], abs_map[:, 1], marker='.', color='C2')
    ax.scatter(abs_anchors[:, 0], abs_anchors[:, 1], marker='+', color='C3')

    plt.show()


if __name__ == "__main__":
    n_ = 100
    n_hop = 4
    max_con = 6

    data_name = f'{n_}sat_{n_hop}hop_{max_con}con'
    true_xy, true_anchors, abs_map, abs_anchors = localize(n_, data_name)
    plot_map(true_xy, true_anchors, abs_map, abs_anchors)
