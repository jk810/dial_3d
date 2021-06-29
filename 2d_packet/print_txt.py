import os


def print_txt(distable_linkdicts, pos_xy, results_path):
    '''
    Record simulation results from each routing time step to .txt files.
    This function will remove existing results in the directory.
    Writes distance.txt, xy.txt, links.txt, and links_distance.txt
    '''

    links = distable_linkdicts[1]
    try:
        os.remove(results_path + f'/links.txt')
    except:
        pass
    try:
        os.remove(results_path + f'/distance.txt')
    except:
        pass
    try:
        os.remove(results_path + f'/links_distance.txt')
    except:
        pass
    try:
        os.remove(results_path + f'/xy.txt')
    except:
        pass

    my_positions = pos_xy
    my_distances = distable_linkdicts[0]

    file = open(results_path + f'/distance.txt', 'w')

    for column in my_distances:
        k = list(my_distances[column])
        for j in range(len(k)):
            if j == 0:
                file.write(str(k[j]))
            elif j == len(k) - 1:
                file.write(' ' + str(k[j]) + '\n')
            else:
                file.write(' ' + str(k[j]))
    file.close()

    # Write file with xy position of every satellite at specified time
    file = open(results_path + f'/xy.txt', 'w')
    for num in range(len(my_positions)):
        pos = my_positions[num]
        x = str(pos[0])
        y = str(pos[1])
        file.write(x + ' ' + y + ' ' + '\n')
    file.close()

    file1 = open(results_path + f'/links.txt', 'w')
    file2 = open(results_path + f'/links_distance.txt', 'w')
    set_check = set()
    for sat in links:
        for conn in links[sat]:
            if (sat, conn) not in set_check and (conn, sat) not in set_check:
                file1.write(sat + ' ' + conn + '\n')
                file2.write(sat + ' ' + conn + ' ' + str(my_distances[sat][int(conn)]) + '\n')
                set_check.add((sat, conn))
                set_check.add((conn, sat))
    file1.close()
    file2.close()
