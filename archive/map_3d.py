from pathlib import Path
import numpy as np

from sklearn import manifold

from timing import timing
from plot_network import plot_3d


def get_data(nodes, dat):
    current_file_path = Path(__file__).resolve()
    code_source_path = str(current_file_path.parents[0])
    results_path = f'{code_source_path}/3d_results/{dat}'

    # hops_file = open(results_path + '/hops.txt', 'r')

    xyz = []
    xyz_file = open(results_path + '/xyz.txt', 'r')
    for line in xyz_file:
        a = line.split()
        xyz.append((float(a[0]), float(a[1]), float(a[2])))
    xyz_file.close()

    xyz_np = np.array(xyz, dtype='float')

    link_dict = {key: [] for key in list(range(nodes))}
    links_file = open(results_path + f'/links.txt', 'r')
    for line in links_file:
        b = line.split()
        link_dict[int(b[0])].append(int(b[1]))
    links_file.close()

    with open(results_path + f'/distance.txt') as f:
        d_table = [line.strip('\n').split(' ') for line in f]
    d_table = [[int(float(j)) for j in k] for k in d_table]

    return xyz_np, link_dict, d_table, results_path


@timing
def localize(nodes, dat):
    true_xyz, link_dict, similarities, r_path = get_data(nodes, dat)

    # perform multi-dimensional scaling
    mds = manifold.MDS(n_components=3, max_iter=100, eps=1e-9,
                       random_state=None, n_jobs=1, n_init=1,
                       dissimilarity='precomputed')
    rel_xyz = mds.fit(similarities).embedding_

    # pick anchor nodes
    anchors = [1, 25, 50, 75]

    true_anchors = np.array([true_xyz[anchors[0]], true_xyz[anchors[1]], true_xyz[anchors[2]], true_xyz[anchors[3]]])
    rel_anchors = np.array([rel_xyz[anchors[0]], rel_xyz[anchors[1]], rel_xyz[anchors[2]], rel_xyz[anchors[3]]])

    # find best linear transformation that maps relative map to absolute anchor nodes
    x, res, rank, s = np.linalg.lstsq(rel_anchors, true_anchors, rcond=None)


    # Q = rel_anchors[1:] - rel_anchors[0]
    # Q_prime = true_anchors[1:] - true_anchors[0]
    # # calculate rotation matrix
    # R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))), np.row_stack((Q_prime, np.cross(*Q_prime))))
    # # calculate translation vector
    # t = p_prime[0] - np.dot(p[0], R)

    map_list = []
    for i in rel_xyz:
        point = np.dot(x, i)
        point = np.reshape(point, (1, 3))
        map_list.append(point)
    mapped_xyz = np.vstack(map_list)
    mapped_anchors = np.array([mapped_xyz[anchors[0]], mapped_xyz[anchors[1]], mapped_xyz[anchors[2]], mapped_xyz[anchors[3]]])

##########
    # W = np.linalg.pinv(rel_anchors.T).dot(true_anchors.T).T
    # mapped_xyz = true_anchors
    # for i in range(rel_xyz.shape[0]//W.shape[0]):
    #     chunk = np.matmul(W, rel_xyz[i*3:i*3+W.shape[0]])
    #     mapped_xyz = np.concatenate((mapped_xyz, chunk))
##########

    return true_xyz, true_anchors, mapped_xyz, mapped_anchors, r_path, anchors


if __name__ == "__main__":
    n_ = 1000
    n_hop = 4
    max_con = 6

    data_name = f'{n_}sat_{n_hop}hop_{max_con}con'

    true_xyz, true_anchors, mapped_xyz, mapped_anchors, r_path, anchors = localize(n_, data_name)
    plot_3d(true_xyz, true_anchors, mapped_xyz, mapped_anchors, r_path, anchors)
