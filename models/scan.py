from math import sqrt

import networkx as nx
from networkx.classes.graph import Graph
from tqdm import tqdm

from utils import eval


def structure(node, graph: nx.classes.graph.Graph):
    return tuple(graph.neighbors(node)) + (node, )


def structural_similarity(node1, node2, graph: nx.classes.graph.Graph):
    structure1 = set(graph.neighbors(node1)) | {node1}
    size1 = len(structure1)
    
    structure2 = set(graph.neighbors(node2)) | {node2}
    size2 = len(structure2)

    numer = len(structure1 & structure2)
    del structure1, structure2
    denom = sqrt(size1 * size2)

    return numer / denom


def find_neighbor(node, epsilon, graph: nx.classes.graph.Graph):
    result = [element for element in structure(node, graph) \
              if structural_similarity(node, element, graph) >= epsilon]
    return result

# def core(node, epsilon, mu, graph: nx.classes.graph.Graph):
#     condition = len(find_neighbor(node, epsilon, graph)) >= mu
#     return condition


def core(node, epsilon, mu, graph: nx.classes.graph.Graph):
    neighbor_num = 0
    structure_node = structure(node, graph)
    if len(structure_node) < mu:
        return False
    
    for element in structure_node:
        neighbor_num += (structural_similarity(node, element, graph) >= epsilon)
        
        if neighbor_num >= mu:
            return True
    
    return False


# def dir_reach(node1, node2, epsilon, mu, graph: nx.classes.graph.Graph):
#     condition1 = core(node1, epsilon, mu, graph)
#     condition2 = node2 in find_neighbor(node1, epsilon, graph)

#     return condition1 and condition2


def dir_reach(node1: int, node2: int, epsilon: float, mu: int, graph: nx.classes.graph.Graph):
    structure_node = structure(node1, graph)
    if node2 not in structure_node:
        return False

    condition1 = core(node1, epsilon, mu, graph)
    if condition1 is False:
        return False

    for neighbor in structure_node:
        if neighbor == node2 and structural_similarity(node1, neighbor, graph) >= epsilon:
            return True
    
    return False


def scan_algorithm(graph: Graph, epsilon: float, mu: int):
    nodes = tuple(graph.nodes())
    check_unit = round(len(nodes) * 0.1)
    labels = {node: 'unclassified' for node in set(graph.nodes())}
    new_cluster_id = 0

    pbar = tqdm(total=len(nodes), desc=f'First  Labeling')
    pbar_update = pbar.update
    set_postfix = pbar.set_postfix

    for index, node in enumerate(nodes):
        queue = find_neighbor(node, epsilon, graph)
        if len(queue) >= mu:
            new_cluster_id += 1

            while len(queue) != 0:
                y = queue[0]
                R = (x for x in nodes if dir_reach(y, x, epsilon, mu, graph))  # y is core & x is near y

                for x in R:
                    if labels[x] in ('unclassified', 'non_member'):
                        labels[x] = new_cluster_id

                    if labels[x] == 'unclassified':
                        queue.append(x)
                queue.remove(y)

        else:
            labels[node] = 'non_member'
        
        if index % check_unit == 0:
            set_postfix({'cluster_id': f'{new_cluster_id:2d}'})
        pbar_update()
    pbar.close()

    pbar = tqdm(total=len(nodes), desc=f'Second Labeling')
    pbar_update = pbar.update
    set_postfix = pbar.set_postfix

    for node in nodes:
        if labels[node] == 'non_member':
            structure_labels = structure(node, graph)
            structure_labels = set(labels[element] for element in structure_labels)
            if len(structure_labels) != 1:
                labels[node] = 'hub'
            
            else:
                labels[node] = 'outlier'
        
        pbar_update()
    pbar.close()

    return labels


def my_algorithm(graph: Graph, epsilon, mu, true_file_path=None, epochs=10):
    nodes = graph.nodes()
    node_length = len(nodes)
    labels = {node: None for node in nodes}
    new_label = 1
    core_count = 0

    # phase 1
    pbar = tqdm(total=node_length, desc=f'Core Labeling')
    pbar_update = pbar.update
    set_postfix = pbar.set_postfix

    for index, node in enumerate(nodes):
        neighbors = find_neighbor(node, epsilon, graph)

        if len(neighbors) >= mu:
            core_count += 1
            temp_labels = tuple(labels[neighbor] for neighbor in neighbors)
            neighbor_num = sum(1 for label in temp_labels if label is not None)

            if neighbor_num == 0:
                labels[node] = new_label
                new_label += 1
            else:
                label = sum(label for label in temp_labels if label is not None)
                labels[node] = label / neighbor_num

        if index % round(node_length*0.01) == 0:
            set_postfix({'core prop': f'{core_count/node_length:.1%}', 'label': f'{new_label:3d}'})
        pbar_update()
    pbar.close()

    # phase 2
    for epoch in range(epochs):
        pbar = tqdm(total=node_length, desc=f'Epoch {epoch:7d}')
        pbar_update = pbar.update

        for node in nodes:
            neighbors = find_neighbor(node, epsilon, graph)

            temp_labels = tuple(labels[neighbor] for neighbor in neighbors)
            neighbor_num = sum(1 for label in temp_labels if label is not None)

            if neighbor_num == 0:
                labels[node] = new_label
                new_label += 1
            else:
                label = sum(label for label in temp_labels if label is not None)
                labels[node] = label / neighbor_num
            
            pbar_update()
        eval_score = eval(dict((key, round(value)) for key, value in labels.items()), true_file_path) \
            if true_file_path is not None else -1
        pbar.set_postfix({'eval score': f'{eval_score:.1%}'})
        pbar.close()

    
    for node in nodes:
        labels[node] = round(labels[node])

    labels = dict(sorted(labels.items(), key=lambda x: x[0]))

    return labels


# def my_algorithm(graph, epsilon, mu):
#     nodes = graph.nodes()
#     labels = {node: None for node in nodes}
#     new_label = 1

#     for node in tqdm(nodes):
#         neighbors = find_neighbor(node, epsilon, graph)

#         if len(neighbors) >= mu:
#             temp_labels = tuple(labels[neighbor] for neighbor in neighbors)
#             neighbor_num = sum(1 for label in temp_labels if label is not None)

#             if neighbor_num == 0:
#                 labels[node] = new_label
#                 new_label += 1
#             else:
#                 label = sum(label for label in temp_labels if label is not None)
#                 labels[node] = label / neighbor_num

#     for epoch in range(10):
#         for node in tqdm(nodes):
#             neighbors = find_neighbor(node, epsilon, graph)

#             temp_labels = tuple(labels[neighbor] for neighbor in neighbors)
#             neighbor_num = sum(1 for label in temp_labels if label is not None)

#             if neighbor_num == 0:
#                 labels[node] = new_label
#                 new_label += 1
#             else:
#                 label = sum(label for label in temp_labels if label is not None)
#                 labels[node] = label / neighbor_num
    
#     for node in nodes:
#         labels[node] = round(labels[node])

#     labels = dict(sorted(labels.items(), key=lambda x: x[0]))

#     return labels