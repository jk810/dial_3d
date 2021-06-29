from pathlib import Path
import numpy as np
from sklearn import manifold
import random
import scipy.spatial.distance

from timing import timing
from plot_2d import plot_2d


@timing
def map_patch(ln_, ln_node_list):
    '''
    Perform MDS mapping for local neighborhood position array
    '''
    ln_xy_estimates = {}

    # perform multi-dimensional scaling
    mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-6, random_state=1,
                       n_jobs=1, n_init=1, dissimilarity='precomputed')
    map_xy_est = mds.fit_transform(ln_)

    # create dictionary for local neighborhood xy estimates
    for i, n in enumerate(ln_node_list):
        ln_xy_estimates[n] = map_xy_est[i]

    return ln_xy_estimates


@timing
def assemble_patches(all_patch_estimates, anchors):
    '''
    Assembles the network by merging patches, based on amount of overlap.
    Returns the merged network and the mapped anchors as numpy arrays.
    '''
    # pick a random node to use as 'core' node and patch
    core_node = 0  # random.randint(0, n_ - 1)

    core_patch = all_patch_estimates[core_node]

    # # add coordinates to compilation of coordinates
    # averaged_estimates = {x: [] for x in list(range(n_))}
    # for k, core_node_coord in core_patch.items():
    #     averaged_estimates[k].append(core_node_coord)

    core_patch_list = list(core_patch.keys())

    # find node/patch with most common nodes. Iterates over every patch
    most_overlapping_node = 1

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
    x, res, rank, s = np.linalg.lstsq(prime_overlap_np, core_overlap_np,
                                      rcond=None)

    # Merge core map with most overlapping patch using lstsq lin transf
    for key, val in most_overlapping_patch.items():
        added_point = np.dot(x, val)
        np_added_point = np.reshape(added_point, (1, 2))

        if key not in core_patch:  # add the transformed point to core patch
            core_patch[key] = np_added_point
        else:
            core_patch[key] = ( np_added_point + core_patch[key] ) / 2

        # # add transformed node coords to node coord list
        # averaged_estimates[key].append(np_added_point)

    # # Create aggregate numpy array of merged network (in sorted node order)
    # # and average all coordinate estimates
    map_list = []
    # for i in range(n_):
    #     coord_list = averaged_estimates[i]
    #     map_list.append(sum(coord_list) / len(coord_list))
    for i in range(100):
        map_list.append(core_patch[i])
    mapped_xy = np.vstack(map_list)

    mapped_anchors = np.array([mapped_xy[anchors[0]], mapped_xy[anchors[1]],
                               mapped_xy[anchors[2]]])

    # # just taking raw average of all patch estimates (without transformation)
    # averaged_estimates = {x: [] for x in list(range(len(all_patch_estimates)))}
    # for patch_dict in all_patch_estimates.values():
    #     for k, coord in patch_dict.items():
    #         averaged_estimates[k].append(coord)
    # map_list = []
    # for coord_list in averaged_estimates.values():
    #     map_list.append(sum(coord_list) / len(coord_list))
    # mapped_xy = np.vstack(map_list)
    # mapped_anchors = np.array([mapped_xy[anchors[0]], mapped_xy[anchors[1]],
    #                            mapped_xy[anchors[2]], mapped_xy[anchors[3]]])

    return mapped_xy, mapped_anchors


@timing
def localize(mapped_xy, mapped_anchors, true_anchors, anchors):
    '''
    Perform absolute localization by finding best linear transformation between
    the mapped anchor nodes and the true anchor nodes.
    Returns the absolutely mapped network and anchor nodes as numpy arrays
    '''
    x, res, rank, s = np.linalg.lstsq(mapped_anchors, true_anchors, rcond=None)

    # Q = rel_anchors[1:] - rel_anchors[0]
    # Q_prime = true_anchors[1:] - true_anchors[0]
    # # calculate rotation matrix
    # R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))), np.row_stack((Q_prime, np.cross(*Q_prime))))
    # # calculate translation vector
    # t = p_prime[0] - np.dot(p[0], R)

    map_list = []
    for i in mapped_xy:
        point = np.dot(x, i)                # use transformation matrix to map
        point = np.reshape(point, (1, 2))   # reshape
        map_list.append(point)
    abs_mapped_xy = np.vstack(map_list)
    abs_mapped_anchors = np.array([abs_mapped_xy[anchors[0]],
                                   abs_mapped_xy[anchors[1]],
                                   abs_mapped_xy[anchors[2]]])

    # W = np.linalg.pinv(rel_anchors.T).dot(true_anchors.T).T
    # mapped_xy = true_anchors
    # for i in range(rel_xy.shape[0]//W.shape[0]):
    #     chunk = np.matmul(W, rel_xy[i*3:i*3+W.shape[0]])
    #     mapped_xy = np.concatenate((mapped_xy, chunk))

    return abs_mapped_xy, abs_mapped_anchors


def symmetricize(arr1D):
    ID = np.arange(arr1D.size)
    return arr1D[np.abs(ID - ID[:,None])]


if __name__ == "__main__":
    xy = []
    for i in range(300):
        xy.append((random.uniform(0, 100), random.uniform(0, 100)))
    true_xy = np.array(xy, dtype='float')

    anchors = [1, 25, 50]
    true_anchors = np.array([true_xy[anchors[0]], true_xy[anchors[1]],
                             true_xy[anchors[2]]])

    patch_1 = true_xy[0:66]
    ln_list_1 = list(range(66))
    patch_2 = true_xy[33:100]
    ln_list_2 = list(range(33, 100))
    ln_d_tab_1 = scipy.spatial.distance.pdist(patch_1)
    ln_d_tab_1 = symmetricize(ln_d_tab_1)
    ln_d_tab_2 = scipy.spatial.distance.pdist(patch_2)
    ln_d_tab_2 = symmetricize(ln_d_tab_2)

    both_ln_node_list = [ln_list_1, ln_list_2]
    both_ln_d_tab = [ln_d_tab_1, ln_d_tab_2]

    all_patch_estimates = {}    # dictionary for each node's local neighborhood

    for node in range(2):
        # Map the local neighborhood patch and add to the aggregate dict
        all_patch_estimates[node] = map_patch(both_ln_d_tab[node], both_ln_node_list[node])
    
    # Assemble patches
    mapped_xy, mapped_anchors = assemble_patches(all_patch_estimates, anchors)

    # Use anchor nodes to absolutely localize the network
    abs_mapped_xy, abs_mapped_anchors = localize(mapped_xy, mapped_anchors,
                                                 true_anchors, anchors)

    current_file_path = Path(__file__).resolve()
    code_source_path = str(current_file_path.parents[0])
    results_path = f'{code_source_path}'

    # mapped_anchors = 5
    # plot the network
    plot_2d(true_xy, true_anchors, mapped_xy, mapped_anchors,
            results_path, anchors)
    # plot_2d(all_patch_estimates[0], true_anchors, all_patch_estimates[1], mapped_anchors,
    #         results_path, anchors)