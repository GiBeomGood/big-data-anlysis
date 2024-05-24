import networkx as nx
import random
import numpy as np
import argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from random import choice, sample
from tqdm import tqdm
from utils import eval
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Reading the Network Data
def load_network(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G

def frequentest_label(labels: tuple):
    frequencies = dict((label, labels.count(label)) for label in labels)
    max_freq = max(frequencies.values())
    max_freq_labels = tuple(label for label, count in frequencies.items() if count == max_freq)

    if len(max_freq_labels) == 1:
        return max_freq_labels[0]
    else:
        return choice(max_freq_labels)

def max_freq_check(node, node_label, labels, graph: nx.Graph):
    neighbors = graph.neighbors(node)
    neighbors_labels = [labels[neighbor] for neighbor in neighbors]
    freq_label = frequentest_label(neighbors_labels)
    return freq_label == node_label

def get_relationship(node1, node2, graph: nx.Graph):
    neighbors_1 = set(graph.neighbors(node1))
    neighbors_2 = set(graph.neighbors(node2)) | {node2}

    relationship = len(neighbors_1 & neighbors_2) / len(neighbors_1)
    return relationship

def get_relative_neighbors(node1, epsilon, graph: nx.Graph):
    neighbors_1 = graph.neighbors(node1)
    result = tuple(node for node in neighbors_1 if get_relationship(node1, node, graph) >= epsilon)
    return result

def calculate_coreness(G, labels):
    coreness = {}
    communities = defaultdict(list)
    for node, label in labels.items():
        communities[label].append(node)
    
    for label, nodes in communities.items():
        G_label = G.subgraph(nodes)
        coreness_values = nx.core_number(G_label)
        for node in nodes:
            coreness[node] = coreness_values[node]
    return coreness

def decide_label(neighbors_labels, coreness=None, strategy='prob'):
    if coreness:
        label_weights = defaultdict(float)
        for label in neighbors_labels:
            label_weights[label] += coreness[label] + 1
        
        if strategy == 'prob':
            total_weight = sum(label_weights.values())
            probabilities = {label: weight / total_weight for label, weight in label_weights.items()}
            return random.choices(list(probabilities.keys()), list(probabilities.values()))[0]
        else:  # strategy == 'freq'
            return max(label_weights, key=label_weights.get)
    else:
        if strategy == 'prob':
            return choice(neighbors_labels)
        else:  # strategy == 'freq'
            return frequentest_label(neighbors_labels)

def my_nlta(graph: nx.Graph, epsilon, strategy='prob', use_coreness=False, true_file_path=None, max_epochs=100):
    nodes = tuple(graph.nodes())
    nodes_num = len(nodes)
    stop_iteration = False
    epoch = 0
    assert strategy in ('prob', 'freq')

    indices = sample(range(nodes_num), k=nodes_num)
    nodes = tuple(nodes[index] for index in indices)
    del indices
    labels = dict((node, node) for node in nodes)

    while stop_iteration is False and epoch < max_epochs:
        # phase 1
        pbar = tqdm(total=nodes_num, desc=f'Epoch {epoch:3d}')
        pbar_update = pbar.update
        postfix = dict()
        non_neighbor_count = 0

        # Coreness 계산
        coreness = calculate_coreness(graph, labels) if use_coreness else None

        for node in nodes:
            neighbors = get_relative_neighbors(node, epsilon, graph)
            neighbors_size = len(neighbors)
            if neighbors_size == 0:
                neighbors = tuple(graph.neighbors(node))
                neighbors_size = len(neighbors)
                non_neighbor_count += 1

            neighbors_labels = tuple(labels[neighbor] for neighbor in neighbors)
            new_label = decide_label(neighbors_labels, coreness=coreness if use_coreness else None, strategy=strategy)
            labels[node] = new_label

            pbar_update()
            
        if true_file_path is not None:
            eval_score = eval(labels, true_file_path)
            postfix['eval score'] = f'{eval_score:6.1%}'
            
        postfix['no friend'] = f'{non_neighbor_count:4d}'
        pbar.set_postfix(postfix)
        pbar.close()

        # phase 2
        for node in nodes:
            check = max_freq_check(node, labels[node], labels, graph)
            if check is False:
                stop_iteration = False
                break
            
            stop_iteration = True

        # etc.
        epoch += 1

    labels = dict((key, (round(value) if value is not None else None)) for key, value in labels.items())
    labels = dict(sorted(labels.items(), key=lambda x: x[0]))  # sort by node in ascending order

    return labels

# Convert labels dictionary to list of lists for NMI calculation
def convert_labels_to_communities(labels):
    community_to_nodes = defaultdict(list)
    for node, community in labels.items():
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values())


# Save the result
def save_communities_to_file(communities, file_path):
    community_dict = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            community_dict[node] = community_id
    sorted_community_items = sorted(community_dict.items())
    with open(file_path, 'w') as f:
        for node, community_id in sorted_community_items:
            f.write(f"{node} {community_id}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect communities in a network.')
    parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./data/1-1.dat")
    args = parser.parse_args()

    community_file_path = args.networkFile.replace('.dat', '.cmty')

    G = load_network(args.networkFile)

    # Detect communities using Louvain method
    # my_nlta(G, 0.5, strategy='prob', use_coreness=True, true_file_path="./data/1-1.cmty")
    # my_nlta(G, 0.5, strategy='prob', use_coreness=False, true_file_path="./data/1-1.cmty")
    my_nlta(G, 0.5, strategy='freq', use_coreness=True, true_file_path="./data/1-1.cmty")
    # my_nlta(G, 0.5, strategy='freq', use_coreness=False, true_file_path="./data/1-1.cmty")
