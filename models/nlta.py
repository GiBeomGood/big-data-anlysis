from random import choice, sample

import networkx as nx
from tqdm import tqdm

from utils import eval


def frequentest_label(labels: tuple):
    frequencies = dict((label, labels.count(label)) for label in labels)
    max_freq = max(frequencies.values())
    max_freq_labels = tuple(label for label, count in frequencies.items() if count == max_freq)

    if len(max_freq_labels) == 1:
        return max_freq_labels[0]
    else:
        return choice(max_freq_labels)


def max_freq_check(node, node_label, labels, graph: nx.classes.graph.Graph):
    neighbors = graph.neighbors(node)
    neighbors_labels = [labels[neighbor] for neighbor in neighbors]
    freq_label = frequentest_label(neighbors_labels)
    return freq_label == node_label



def nlta_algorithm(graph: nx.classes.graph.Graph, true_file_path, max_epochs=100):
    nodes = tuple(graph.nodes())
    nodes_num = len(nodes)
    stop_iteration = False
    epoch = 0

    indices = sample(range(nodes_num), k=nodes_num)
    nodes = tuple(nodes[index] for index in indices)
    del indices
    labels = dict((node, node) for node in nodes)

    while stop_iteration is False and epoch < max_epochs:
        # phase 1
        pbar = tqdm(total=nodes_num, desc=f'Epoch {epoch:3d}')
        postfix = dict()
        pbar_update = pbar.update

        for node in nodes:
            neighbors = graph.neighbors(node)
            neighbors_labels = tuple(labels[neighbor] for neighbor in neighbors)
            labels[node] = frequentest_label(neighbors_labels)

            pbar_update()
        
        if true_file_path is not None:
            eval_score = eval(labels, true_file_path)
            postfix['eval score'] = f'{eval_score:6.1%}'
        
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

    labels = dict(sorted(labels.items(), key=lambda x: x[0]))  # sort by node in ascending order
    
    return labels


def get_relationship(node1, node2, graph: nx.classes.graph.Graph):
    neighbors_1 = set(graph.neighbors(node1))
    neighbors_2 = set(graph.neighbors(node2)) | {node2}

    relationship = len(neighbors_1 & neighbors_2) / len(neighbors_1)
    return relationship


def get_relative_neighbors(node1, epsilon, graph: nx.classes.graph.Graph):
    neighbors_1 = graph.neighbors(node1)
    result = tuple(node for node in neighbors_1 if get_relationship(node1, node, graph) >= epsilon)

    return result


def my_nlta(graph: nx.classes.graph.Graph, epsilon, strategy='prob', true_file_path=None, max_epochs=100):
    nodes = tuple(graph.nodes())
    nodes_num = len(nodes)
    stop_iteration = False
    epoch = 0
    assert strategy in ('prob', 'freq')
    decide_label_func: function = choice if strategy == 'prob' else frequentest_label

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

        for node in nodes:
            neighbors = get_relative_neighbors(node, epsilon, graph)
            neighbors_size = len(neighbors)
            if neighbors_size == 0:
                neighbors = tuple(graph.neighbors(node))
                neighbors_size = len(neighbors)
                non_neighbor_count += 1

            neighbors_labels = tuple(labels[neighbor] for neighbor in neighbors)
            labels[node] = decide_label_func(neighbors_labels)

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