import argparse
import logging
import pandas as pd
import networkx as nx
import multiprocessing
import time

from joblib import Parallel, delayed
from random import randint

logger = logging.getLogger(__name__ if __name__ != '__main__' else __package__)


def generate_graph(df):
    """
    Generate two directed graph
    :param df: A df with id1, id2, timestamp columns,
               the tuple id1, id2 is not unique, so for each
               occurrence we update the weight attribute
    :return: The main graph (G1), a the reverse one (G2)
    """

    # Create first directed graph
    G1 = nx.Graph()
    for row in df.itertuples():
        try:
            weight = G1[row.id1][row.id2]['weight']
            G1.add_edge(row.id1, row.id2, weight=weight + 1)
        except KeyError:
            G1.add_edge(row.id1, row.id2, weight=1)

    # Create second directed graph
    G2 = nx.Graph()
    for row in df.itertuples():
        try:
            weight = G2[row.id2][row.id1]['weight']
            G2.add_edge(row.id2, row.id1, weight=weight + 1)
        except KeyError:
            G2.add_edge(row.id2, row.id1, weight=1)

    return G1, G2


def add_plurality_attribute(G1, G2, nodes):
    """
    Add plurality attribute.
    """
    for n in nodes:
        if(G1.degree(n) == 0):
            G1.node[n]['plurality'] = sys.maxsize
        else:
            sum_weight = 0
            for g2_node_source in G2.neighbors(n):
                sum_weight = sum_weight + G1.get_edge_data(g2_node_source, n)['weight']

            G1.node[n]['plurality'] = int(sum_weight/2) + 1

    return G1


def add_min_attribute(G1, G2, nodes):
    """
    Add min attribute.
    """
    for n in nodes:
        if(G1.degree(n) == 0):
            G1.node[n]['plurality'] = sys.maxsize
        else:
            G1.node[n]['plurality'] = 1

    return G1


def add_max_attribute(G1, G2, nodes):
    """
    Add max attribute.
    """
    for n in nodes:
        if(G1.degree(n) == 0):
            G1.node[n]['plurality'] = sys.maxsize
        else:
            sum_weight = 0
            for g2_node_source in G2.neighbors(n):
                sum_weight = sum_weight + G1.get_edge_data(g2_node_source, n)['weight']

            G1.node[n]['plurality'] = sum_weight

    return G1


def add_random_attribute(G1, G2, nodes):
    """
    Add max attribute.
    """
    for n in nodes:
        weights = []

        if(G1.degree(n) == 0):
            G1.node[n]['plurality'] = sys.maxsize
        else:
            sum_weight = 0
            for g2_node_source in G2.neighbors(n):
                weights.append(G1.get_edge_data(g2_node_source, n)['weight'])

            G1.node[n]['plurality'] = randint(a=min(weights),
                                              b=max(weights) + 1)
    return G1


def get_unique_nodes(df):
    """
    Get unique nodes from a source target dataframe
    """
    unique_nodes = set()
    unique_nodes.update(df['id1'].unique())
    unique_nodes.update(df['id2'].unique())
    return unique_nodes


def get_neighborhood(node, g1, g2, total_nodes):
    """
    Calculate nodes according at the neighborhood level.
    Using a node as bootstrap:
    - The first level is only the node,
    - The second level is the first node and their neighbors
    - The third level ....

    :param node: The boostrap node identifier
    :param g1: Main graph
    :param g2: Reverse graph
    :param total_nodes: Number of unique nodes in the main graph
    :return: A list with the initial node, the neighborhood level,
             and their nodes
    """
    old_nodes = set()
    new_nodes = set()

    neighborhood = []
    neighborhood.append({'inital_node': node,
                         's': 0,
                         'nodes': [node]})

    new_nodes.add(node)

    for idx, n in enumerate(range(total_nodes)):

        nodes_to_add = set()
        for new_node in new_nodes:
            nodes_to_add.update(g1.neighbors(new_node))
            nodes_to_add.update(g2.neighbors(new_node))

        old_nodes = old_nodes.union(new_nodes)

        new_nodes.clear()
        new_nodes.update(nodes_to_add)

        nodes_s_level = set(old_nodes).union(new_nodes)

        neighborhood.append({'inital_node': node,
                             's': idx + 1,
                             'nodes': nodes_s_level})

        if len(nodes_s_level) == total_nodes:
            return neighborhood

    return neighborhood


def save_parallel_process(data, algo, process_id):

    df = pd.DataFrame(data)

    file_name = 'parallel_{alg}_{ppid}.csv'.format(alg=algo,
                                                   ppid=process_id)
    with open(file_name, 'a') as file:
        df.to_csv(file,
                  index=False,
                  header=False)


def linear_threshold_rank(node, G1, G2, total_nodes, algo):
    """
    Calculate the lTR for each node
    - https://www.sciencedirect.com/science/article/pii/S0950705117304975
    """
    linear_threshold_rank = []

    PRINT_STEPS = False

    if PRINT_STEPS:
        print('-------------- Node: {} --------------'.format(node))

    neighborhoods = get_neighborhood(node, G1, G2, total_nodes)

    for neighborhood in neighborhoods:

        first_step = True

        nodes_to_add_group = set()

        # Get first neighbors (bootstrap nodes)
        neighbors = set(neighborhood['nodes'])

        group = set()
        group.update(neighbors)

        if PRINT_STEPS:
            print('I: 0 ; Neighbors: {0} ; count: {1}'.format(neighbors, len(neighbors)))

        depth_level = 0

        while(first_step or (len(nodes_to_add_group) >= 1)):

            first_step = False

            neighbors.update(nodes_to_add_group)
            group.update(nodes_to_add_group)

            nodes_to_add_group.clear()

            if PRINT_STEPS:
                print('\t group: {1}'.format(node, group))

            next_nodes = set()
            for next_node in neighbors:
                next_nodes.update(G1.neighbors(next_node))

            # Nodes that can be influenced
            dispersion = set()
            dispersion = next_nodes - group

            if PRINT_STEPS:
                print('\t dispersion {0} '.format(dispersion))

            # For each dispersion node
            for n_sub_level in dispersion:
                plurality = G1.node[n_sub_level]['plurality']

                if PRINT_STEPS:
                    print('\t \t Reach node {0} | plurality {1}'.format(n_sub_level, plurality))

                group_influce = 0
                for node_group in group:
                    if(G1.get_edge_data(node_group, n_sub_level)):
                        group_influce = group_influce + G1.get_edge_data(node_group, n_sub_level)['weight']

                if PRINT_STEPS:
                    print('\t \t group {0} | influce {1}'.format(group, group_influce))

                if(group_influce >= plurality):
                    nodes_to_add_group.add(n_sub_level)

            if PRINT_STEPS:
                print('\t \t \t new group {0} '.format(nodes_to_add_group))
                print('{0} ; {1} ; {2}'.format(node, depth_level, len(group)))
                print()

            linear_threshold_rank.append({'node': node,
                                          'k': depth_level,
                                          'number_influenced': len(group),
                                          's': neighborhood['s']})
            depth_level = depth_level + 1

    current = multiprocessing.current_process()
    process_id = current.name[current.name.index('-') + 1:]

    save_parallel_process(linear_threshold_rank, algo, process_id)

    return None

def main(algorithm_weight, n_jobs):

    logger.info('Starting the previous calculations ...')
    start_total_time = time.time()

    df = pd.read_csv('CA-GrQc.txt', delimiter='\t')

    g1, g2 = generate_graph(df)
    unique_nodes = get_unique_nodes(df)

    if algorithm_weight == 'plurality':
        g1 = add_plurality_attribute(g1, g2, unique_nodes)
    elif algorithm_weight == 'min':
        g1 = add_min_attribute(g1, g2, unique_nodes)
    elif algorithm_weight == 'max':
        g1 = add_max_attribute(g1, g2, unique_nodes)
    elif algorithm_weight == 'random':
        g1 = add_random_attribute(g1, g2, unique_nodes)
    else:
        raise ValueError('Invalid algorithm type')

    largest_cc = max(nx.connected_components(g1), key=len)

    logger.info('Starting LTC calculation ...')
    parallel_results = Parallel(n_jobs=n_jobs,
                                verbose=100)(
                          delayed(linear_threshold_rank)(
                            node=n,
                            G1=g1,
                            G2=g2,
                            total_nodes=len(largest_cc),
                            algo=algorithm_weight) for n in largest_cc)

    logger.info('Finished execution')
    logger.info('Duration: {} seconds'.format(time.time() - start_total_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Attribute selection algorithm')

    parser.add_argument(
        '--algorithm-weight',
        choices=['plurality', 'min', 'max','random'],
        help='Algorithm to define node weights of the nodes',
        required=True)

    parser.add_argument(
        '--n-jobs',
        type=int,
        help='Number of n-jobs (multiprocessing)',
        required=True)

    logging.basicConfig(
        level='DEBUG',
        format='%(asctime)-15s %(name)-20s %(levelname)-8s %(message)s')
    
    args = parser.parse_args()

    main(args.algorithm_weight, args.n_jobs)

