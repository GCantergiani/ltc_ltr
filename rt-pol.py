
# coding: utf-8

# In[95]:


import pandas as pd
import networkx as nx


# ## Read network file

# In[96]:


df = pd.read_csv('rt-pol.txt')
df.head()


# ## Generate graph

# In[97]:


def generate_graph(df):

    # Create first directed graph
    G1 = nx.DiGraph()
    for row in df.itertuples():
        try:
            weight = G1[row.source][row.target]['weight']
            G1.add_edge(row.source, row.target, weight=weight+1)
        except KeyError:
            G1.add_edge(row.source, row.target, weight=1)

    # Create second directed graph
    G2 = nx.DiGraph()
    for row in df.itertuples():
        try:
            weight = G2[row.target][row.source]['weight']
            G2.add_edge(row.target, row.source, weight=weight+1)
        except KeyError:
            G2.add_edge(row.target, row.source, weight=1)

    return G1, G2


# ## Plurality Attr

# In[118]:


def add_plurality_attribute(G1, G2, nodes):

    for n in nodes:
        
        if(G1.degree(n) == 0):
            plurality = sys.maxsize
        else:
            sum_weight = 0
            for g2_node_source in G2.neighbors(n):
                sum_weight = sum_weight + G1.get_edge_data(g2_node_source, n)['weight']

            plurality = int(sum_weight/2) + 1
            
        G1.node[n]['plurality'] = plurality

    return G1


# In[119]:


g1, g2 = generate_graph(df)


# In[120]:


unique_nodes = []
unique_nodes.extend(df['source'].unique())
unique_nodes.extend(df['target'].unique())
unique_nodes = list(set(unique_nodes))


# In[121]:


g1 = add_plurality_attribute(g1, g2, unique_nodes)


# In[128]:


def ltm(unique_nodes, G1, G2):
    
    results = []
    
    lineal_threshold_to_pandas = []
    
    PRINT_STEPS = False

    # For each unique nodes
    for idx, n in enumerate(unique_nodes):

        if PRINT_STEPS:
            print('--------- Node: {} --------------'.format(n))

        first_step_per_node = True
        
        nodes_to_add_group = []
        
        # Get first neighbors (bootstrap nodes)
        neighbors = []
        neighbors.extend(G1.neighbors(n))
        neighbors.extend(G2.neighbors(n))
        # Remove duplicates
        neighbors = list(set(neighbors))
        
        group = []
        group.append(n)
        group.extend(neighbors)
        # Remove duplicates
        group = list(set(group))

        if PRINT_STEPS:
            print('I: 0 ; Neighbors: {0} ; count: {1}'.format(neighbors, len(neighbors)))
        
        depth_level = 0

        while( first_step_per_node or (len(nodes_to_add_group) >= 1) ):
            
            first_step_per_node = False
            
            neighbors.extend(nodes_to_add_group)
            group.extend(nodes_to_add_group)
            
            nodes_to_add_group = []
        
            if PRINT_STEPS:
                print('\t group: {1}'.format(n, group))

            dispersion = []

            vei = []
            for v in neighbors:
                vei.extend(G1.neighbors(v))

            # Nodes that can be influenced
            dispersion = list(set(vei) - set(group))

            if PRINT_STEPS:
                print('\t dispersion {0} '.format(dispersion))

            # For each dispersion node
            for n_sub_level in dispersion:
                plurality = G1.node[n_sub_level]['plurality']
                
                if PRINT_STEPS:
                    print('\t \t Reach node {0} | plurality {1}'.format(n_sub_level,plurality))
                
                group_influce = 0
                for node_group in group:
                    if(G1.get_edge_data(node_group,n_sub_level)):
                        group_influce = group_influce + G1.get_edge_data(node_group,n_sub_level)['weight']

                if PRINT_STEPS:
                    print('\t \t group {0} | influce {1}'.format(group, group_influce))

                if(group_influce >= plurality):
                    nodes_to_add_group.append(n_sub_level)

            if PRINT_STEPS:
                print('\t \t \t new group {0} '.format(nodes_to_add_group))
                print('{0} ; {1} ; {2}'.format(n, depth_level, len(group)))
                print()

            results.append({'node':n, 'lvl':depth_level, 'group_influeced':group, 'number_influenced': len(group)})
            depth_level = depth_level +1 
    
    return results


# In[123]:


results = ltm(unique_nodes, g1, g2)


# In[124]:


df_results = pd.DataFrame(results)


# In[125]:


df_results = df_results.sort_values(by=['node','lvl'], ascending=True)
df_results = df_results[['node', 'lvl', 'number_influenced','group_influeced']]
df_results


# In[127]:


df_results.to_csv('rt-pol-LTR.csv', index=False)

