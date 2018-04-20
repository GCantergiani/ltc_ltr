
# coding: utf-8

# In[25]:


import pandas as pd
import networkx as nx

from collections import OrderedDict
from tqdm import tqdm
from IPython.display import Image


# ## Graph

# ![title](./graph_2.jpeg)

# ## Read network file

# In[26]:


df = pd.read_csv('network_2.csv')
df


# ## Generate graph

# In[27]:


def generate_graph(df):
    """
    Generate two directed graph
    :param df: A df with source, target, timestamp columns,
               the tuple source, target is not unique, so for each
               occurrence we update the weight attribute
    :return: The main graph (G1), a the reverse one (G2)
    
    """

    # Create first directed graph
    G1 = nx.DiGraph()
    for row in df.itertuples():
        G1.add_edge(row.source, row.target, weight=row.weight)
            

    # Create second directed graph
    G2 = nx.DiGraph()
    for row in df.itertuples():
        G2.add_edge(row.target, row.source, weight=row.weight)
        
    return G1, G2


# ## Plurality Attr

# In[28]:


def add_plurality_attribute(G1, G2, nodes):
    """
    Add plurality attribute.
    """
    for n in nodes:
        
        if(G1.degree(n) == 0):
            plurality = sys.maxsize
        else:
            sum_weight = 0
            for g2_node_source in G2.neighbors(n):
                sum_weight = sum_weight + G1.get_edge_data(g2_node_source, n)['weight']

            plurality = int(sum_weight/2) + 1
            
        G1.node[n]['plurality'] = plurality
        print('Node {} plurality {}'.format(n, plurality))
    return G1


# In[29]:


g1, g2 = generate_graph(df)


# In[30]:


unique_nodes = set()
unique_nodes.update(df['source'].unique())
unique_nodes.update(df['target'].unique())
len(unique_nodes)


# In[31]:


g1 = add_plurality_attribute(g1, g2, unique_nodes)


# In[33]:


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
    neighborhood.append({'inital_node':node,
                         's': 0, 
                         'nodes':[node]})

    new_nodes.add(node)
    last_nodes = None

    for idx, n in enumerate(range(total_nodes)):

        nodes_to_add = set()
        for new_node in new_nodes:
            nodes_to_add.update(g1.neighbors(new_node))
            nodes_to_add.update(g2.neighbors(new_node))

        old_nodes= old_nodes.union(new_nodes)

        new_nodes.clear()
        new_nodes.update(nodes_to_add)


        nodes_s_level = set(old_nodes).union(new_nodes)

        neighborhood.append({'inital_node':node,
                             's':idx + 1, 
                             'nodes':nodes_s_level})

        if len(nodes_s_level) == total_nodes:
            return neighborhood


# In[40]:


set(get_neighborhood(1, g1, g2,  len(unique_nodes))[1]['nodes'])


# In[41]:


def linear_threshold_rank(node, G1, G2, total_nodes):
    """
    Calculate the lTR for each node 
    - https://www.sciencedirect.com/science/article/pii/S0950705117304975
    """
    linear_threshold_rank = []
    
    PRINT_STEPS = True

    if PRINT_STEPS:
        print('-------------- Node: {} --------------'.format(node))
              
    neighborhoods = get_neighborhood(node, g1, g2, total_nodes)
    
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

        while( first_step or (len(nodes_to_add_group) >= 1) ):

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
                        group_influce = group_influce + G1.get_edge_data(node_group,n_sub_level)['weight']

                if PRINT_STEPS:
                    print('\t \t group {0} | influce {1}'.format(group, group_influce))

                if(group_influce >= plurality):
                    nodes_to_add_group.add(n_sub_level)

            if PRINT_STEPS:
                print('\t \t \t new group {0} '.format(nodes_to_add_group))
                print('{0} ; {1} ; {2}'.format(node, depth_level, len(group)))
                print()
                
                
            linear_threshold_rank.append({'node':node,
                                          'k':depth_level,
                                          'group_influenced':group.copy(),
                                          'number_influenced':len(group),
                                          's':neighborhood['s']})
            
            depth_level = depth_level +1 
        
    return linear_threshold_rank


# In[42]:


get_ipython().run_cell_magic('time', '', 'results = []\nfor node in unique_nodes:\n    results.extend(linear_threshold_rank(node, g1, g2, len(unique_nodes)))')


# In[43]:


df_results = pd.DataFrame(results)
df_results[['node','s','k','number_influenced','group_influenced']].sort_values(by=['node','s','k'])

