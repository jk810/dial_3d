import numpy as np
import pandas as pd
import copy

from timing import timing


@timing
def provision(pos_table, nbr_hop, max_conn, avg_conn):
    '''
    Performs provisioning for all nodes.
    Returns a list of all distance tables and link dictionaries
    '''
    global n_hop
    global max_con
    global avg_con

    # hop limit of local neighborhood
    n_hop = nbr_hop
    # max number of connections a satellite can make
    max_con = max_conn
    # average number of connections a satellite tries to maintain
    avg_con = avg_conn

    pos_xy = pos_table

    keys = [str(i) for i in range(len(pos_xy))]
    links = {key: [] for key in keys}

    # Generate initial distance table and links (with fully connected network)
    d_table = generate_dist_table(pos_xy)
    links = generate_links(links, d_table, pos_xy, initial=True)

    d_table = generate_dist_table(pos_xy)
    links = generate_links(links, d_table, pos_xy, initial=False)

    distable_linkdicts = [d_table, links]

    return distable_linkdicts


def generate_links(link, d_table, pos_xy, initial):
    '''
    generate link dictionaries for all nodes at a single time step.
    returns a table of links
    '''
    global max_dist
    # max distance for a link between satellites
    max_dist = 15
    # max range of RF finder
    rf_range = 3
    # boolean controlling whether or not to use RF reconnection
    rf = False

    # additional link; RF reconnection to guarantee reconnections
    max_con_rf = max_con + 1
    links = copy.deepcopy(link)

    # remove links that are out of range
    for sat in link:
        for con in link[sat]:
            if not check_in_range(pos_xy[int(sat)], pos_xy[int(con)]):
                links[sat].remove(con)

    links2 = copy.deepcopy(links)

    # iterate through all nodes
    for sat in links2:
        if initial:
            neighbors = list(range(len(links)))
        else:
            # Build local neighborhood
            neighbors = []
            frontier = [str(sat)]
            for step in range(n_hop):
                new_frontier = []
                for indiv in frontier:
                    new_frontier = new_frontier + links[indiv]
                frontier = list(set([i for i in new_frontier if i not in neighbors]))
                neighbors = neighbors + frontier
            # create list of nodes that are in range and outside LN
            in_range = []
            for label, value in d_table[sat].iteritems():
                if value < max_dist:
                    in_range.append(str(label))
            in_range.remove(sat)
            in_range_notin_ln = list(set(in_range) - set(neighbors))

        if len(links[sat]) != 0 or initial:
            # search for new links while number of cons is < avg_con
            while len(links[sat]) < avg_con:
                if initial:
                    new = search_new_link(sat, links, d_table, pos_xy, use_uni=False, max_d=max_dist, local=neighbors)
                else:
                    new = search_new_link(sat, links, d_table, pos_xy, use_uni=False, max_d=max_dist, local=in_range_notin_ln)
                if new == False:
                    break
                elif str(new) != sat:
                    links[sat].append(str(new))
                    links[str(new)].append(sat)
                else:
                    break

        elif rf == True:
            for i in range(len(pos_xy)):
                node = pos_xy[i]
                if get_distance(pos_xy[int(sat)], node) <= rf_range and str(i) != sat and len(links[str(i)]) < max_con_rf:
                    links[sat].append(str(i))
                    links[str(i)].append(sat)
                    break
    return links


def search_new_link(sat, links, d_table, pos_xy, use_uni, max_d, local):
    '''
    Searches for a new link in a subset of the network (or all nodes if initial).
    Returns the new connection node number, or False if no connection is found
    '''
    # Using only nodes in local subset
    d_table = d_table.loc[[int(i) for i in local]]
    if int(sat) in d_table.index:
        d_table = d_table.drop(int(sat))

    if len(d_table[sat]) == 0:
        return False
    if max(d_table[sat]) == 0:
        return False

    scoring = d_table[sat]
    # # randomly sort list
    # scoring.iloc[np.random.permutation(len(scoring))]
    # sort by closest
    scoring = scoring.sort_values(ascending=True)

    new_con = False

    for potential_link in scoring.index:
        if (check_in_range(pos_xy[int(sat)], pos_xy[int(potential_link)]) and
                len(links[str(potential_link)]) < max_con and
                str(potential_link) not in links[sat] and
                str(potential_link) != sat):
            new_con = potential_link
            break
    return new_con


def generate_dist_table(pos_xy):
    '''
    Generates distance table at a single time step
    '''
    n_sat = len(pos_xy)
    np_xy = np.array(pos_xy)

    d_table = pd.DataFrame(index=[i for i in range(n_sat)])
    for i, pos in enumerate(np_xy):
        d_table[str(i)] = np.sqrt(np.square(pos - np_xy).sum(axis=1))

    return d_table


def check_in_range(pos1, pos2):
    '''
    Check if 2 nodes are within range of each other
    '''
    dist = get_distance(pos1, pos2)
    if dist > max_dist:
        return False
    else:
        return True


def get_distance(pos1, pos2):
    '''
    Calculate the distance between two satellites
    '''
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return np.sqrt(dx**2 + dy**2)
