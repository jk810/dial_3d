from pathlib import Path
from timing import timing


class node:
    '''
    node class that holds route table, one hop list, and ID
    '''
    def __init__(self, my_id):
        self.my_route_table = {}
        self.one_hop_list = []
        self.my_id = my_id

    # add a one-hop neighbor
    def add_nbr(self, nbr):
        self.one_hop_list.append(nbr)

    # insert an entry into the routing node
    def insert_route(self, dst, next_, hops_away):

        # if a routing entry for this dst exists and is hops_away away
        if dst in self.my_route_table:
            existing_entry = self.my_route_table[dst]
            if existing_entry.hops_away == hops_away:

                # compare the distance to dst between the two entries
                if distance_table[next_][dst] < distance_table[existing_entry.next_hop][dst]:
                    self.my_route_table[dst] = routing_entry(dst, next_, hops_away)

        # the routing entry dst is new, add it to the routing table
        else:
            self.my_route_table[dst] = routing_entry(dst, next_, hops_away)

    # add to the routing table for hop_n, by using neighbors that are 1 hop away
    def build_nbrhood(self, hop_n, nnnode_array):
        new_neighbors = []

        for key, entry in self.my_route_table.items():
            # if the hop_n is exactly 1 hop away from me
            if entry.hops_away == hop_n - 1:

                # for this particular node that is (hop_n - 1) hops away
                # find a 1-hop neighbor that is not yet in my nbrhood (and not myself)
                temp_node = nnnode_array[key]

                for y in temp_node.one_hop_list:
                    if y not in self.my_route_table and y != self.my_id:
                        new_entry = routing_entry(y, entry.next_hop, hop_n)
                        new_neighbors.append(new_entry)

        # add new neighborhood routes to route table
        for z in new_neighbors:
            self.insert_route(z.dst, z.next_hop, z.hops_away)


class routing_entry:
    def __init__(self, dst, next_hop, hops_away):
        self.dst = dst
        self.next_hop = next_hop
        self.hops_away = hops_away


# Go step by step to find the next hop with minimum distance to destination until:
    # 1. destination is found in local neighborhood table
    # 2. there is a loop in the path (not found)
def find_path(src, dst):
    '''
    Find a path for a source-destination pair. Next node in path is found by
    minimum distance to destination until:
        1. Destination is found in current node's routing table
        2. there is no path found (loop, or no availabe next node)
    '''
    global no_path
    global too_long

    max_dist = 999999

    current_node = src
    semi_path = [src]

    if src == dst:
        return True

    # main loop
    while True:
        # if a routing entry exist, no need to step further
        if dst in node_array[current_node].my_route_table:
            exist = node_array[current_node].my_route_table[dst]

            # Extracting complete path from src to dst
            complete_path = semi_path.copy()
            complete_path.append(exist.next_hop)
            while complete_path[-1] != dst:
                intermed_node = node_array[complete_path[-1]].my_route_table[dst]
                complete_path.append(intermed_node.next_hop)
            hops_file.write(f'{src} {dst} {len(complete_path)-1} {complete_path}' + '\n')

            return True

        # dst not found in routing table/LN, need to do min dist calculation
        # among all neighbors
        else:
            min_dist = max_dist
            min_dist_i = -1

            # go through all entries in routing table
            for key, entry in node_array[current_node].my_route_table.items():

                # to determine if current node should be used for comparison
                use_node = True

                if use_border is True and entry.hops_away != nbr_hop:
                    use_node = False

                # find node with min distance to dst, use it as the next hop
                if (use_node and distance_table[key][dst] < min_dist and
                    entry.next_hop not in semi_path):
                    min_dist = distance_table[key][dst]
                    min_dist_i = key
                    next_node = entry.next_hop

            if min_dist_i == -1:
                no_path += 1
                return False

            # # node has already been visited, we are in a loop
            # if next_node in semi_path:
            #     semi_path.append(next_node)
            #     too_long += 1
            #     loops_file.write(f'{src} {dst} {semi_path}' + '\n')
            #     return False

            # path is too long, quit the search
            if len(semi_path) > 30:     # hop limit at x + 4
                semi_path.append(next_node)
                too_long += 1
                return False

            # this node has not been visited before, add node to path
            else:
                semi_path.append(next_node)
                current_node = next_node


@timing
def route(n_node, nbrhood_hop, border, linkdict, distable, results_path):
    '''
    Tries to find routes between source-destination pairs of nodes.
    Reads in distance.txt and links.txt, writes hops.txt and loops.txt
    '''
    global nbr_hop
    global node_array
    global distance_table
    global use_border
    global hops_file
    # global loops_file

    global no_path
    global too_long

    try:
        os.remove(results_path + '/hops.txt')
    except:
        pass
    try:
        os.remove(results_path + '/loops.txt')
    except:
        pass

    hops_file = open(results_path + '/hops.txt', 'w')
    # loops_file = open(results_path + '/loops.txt', 'w')

    # reassign parameters to global variables
    nbr_hop = nbrhood_hop
    use_border = border

    # initialize new node list
    node_array = [node(i) for i in range(n_node)]

    # read in link table, add 1-hop neighbors to nodes
    for key, values in linkdict.items():
        n1 = int(key)
        for con in values:
            n2 = int(con)
            node_array[n1].add_nbr(n2)
            node_array[n2].add_nbr(n1)

    # remove duplicate one hop neighbors and add one hop neighbor into routing table
    for i in range(n_node):
        # Sort one_hop_list and remove duplicates (if the list is not empty)
        if len(node_array[i].one_hop_list) != 0:
            node_array[i].one_hop_list = list(set(node_array[i].one_hop_list))
            # initially add one hop neighbors to the routing tables
            for x in node_array[i].one_hop_list:
                node_array[i].insert_route(x, x, 1)

    distance_table = distable.values.tolist()

    for i in range(n_node):
        # start at 2 because 1 hop neighbors were already inserted
        for k in range(2, nbr_hop + 1):
            node_array[i].build_nbrhood(k, node_array)

    too_long = 0
    no_path = 0
    total = 0

    # go through all possible source destination pairs to see reachability
    for i in range(n_node):
        for j in range(n_node):
            if i != j:
                find_path(i, j)
                total += 1

    bad = too_long + no_path

    hops_file.close()
    # loops_file.close()

    return [n_node, nbr_hop, total, bad, too_long, no_path]
