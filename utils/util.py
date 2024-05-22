from evaluation import calculate_nmi, load_ground_truth


def community_to_node(labels):
    result = {}
    for node, label in labels.items():
        if label not in result:
            result[label] = [node]
        else:
            result[label].append(node)
    
    return list(result.values())


def eval(labels, true_file_path):
    partition = community_to_node(labels)
    true_partition = load_ground_truth(true_file_path)
    nmi_score = calculate_nmi(true_partition, partition)
    return nmi_score