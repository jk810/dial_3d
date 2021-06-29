import datetime
import os
from pathlib import Path
import random
import pandas as pd

from timing import timing
from print_txt import print_txt
from provision import provision
from route import route


@timing
def run_sim(dat, n_hop, max_con, avg_con, n_sat, perimiter_only):
    current_file_path = Path(__file__).resolve()
    code_source_path = str(current_file_path.parents[0])

    # Create directory for sim results
    results_path = f'{code_source_path}/results/{dat}'
    data_path = f'{results_path}/{dat}_data.csv'

    try:
        os.mkdir(results_path)
    except:
        pass

    print('________ ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          ' ________')
    print(data_name)

    # initialize node positions
    pos_table = []
    for i in range(n_sat):
        pos_table.append([random.uniform(-1, 1)*50, random.uniform(-1, 1)*50])

    # Provision at every time step. Returns results from routing time steps
    distable_linkdicts = provision(pos_table, nbr_hop=n_hop, max_conn=max_con,
                                   avg_conn=avg_con)

    # print links and distance table
    print_txt(distable_linkdicts, pos_table, results_path)

    # Perform routing
    route_data = route(n_node=n_sat, nbrhood_hop=n_hop, border=perimiter_only,
                       linkdict=distable_linkdicts[1],
                       distable=distable_linkdicts[0],
                       results_path=results_path)

    # Write routing results to csv
    np_data = pd.DataFrame(route_data).T    # initialize pd dataframe
    np_data.columns = ['n_node', 'n_hop', 'total', 'bad', 'loop', 'no_path']
    np_data.to_csv(data_path, index=False)

    return


# ---------------------------------------
# hop limit for local neighborhood
n_hop = 4
# max # of connections per satellite
max_con = 6
# average # of connections that are maintained
avg_con = 4
# boolean for using only perimeter nodes for routing
perim = False
# n_ must match name of TLE file
n_ = [200, 400]

for n_node in n_:
    data_name = f'{n_node}sat_{n_hop}hop_{max_con}con'
    run_sim(dat=data_name, n_hop=n_hop, max_con=max_con, avg_con=avg_con,
            n_sat=n_node, perimiter_only=perim)
