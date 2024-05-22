from networkx.classes.graph import Graph
from tqdm import tqdm


def modularity(graph: Graph, partition, gamma=1):
	m = graph.number_of_edges()
	find_subgraph = graph.subgraph
	find_degree = graph.degree
	find_neighbors = graph.neighbors

	communities = set(partition.values())
	result = 0

	for community in communities:
		nodes_in_community = [node for node, _community in partition.items() if _community == community]

		subgraph = find_subgraph(nodes_in_community)
		e_c = subgraph.number_of_edges()
		K_c = sum(find_degree(node) for node in nodes_in_community)

		result += (e_c - gamma * K_c**2 / (2*m))
	
	result /= 2*m

	return result


def louvain_step(graph: Graph, partition, step, gamma=1):
	find_neighbors = graph.neighbors
	mod = modularity(graph, partition, gamma=gamma)

	pbar = tqdm(total=len(graph.nodes), desc=f'Step {step:3d}')
	pbar_update = pbar.update
	set_postfix = pbar.set_postfix

	for node in graph.nodes:
		for neighbor in find_neighbors(node):
			new_partition = partition.copy()
			new_partition[neighbor] = new_partition[node]
			new_mod = modularity(graph, new_partition, gamma=gamma)

			if new_mod > mod:
				partition = new_partition
				mod = new_mod

		pbar_update()
		set_postfix({'Modularity': f'{mod:.1%}'})

	pbar.close()

	return partition

def louvain(graph, gamma=1):
	partition = {node: i for i, node in enumerate(graph.nodes())}
	mod = modularity(graph, partition, gamma=gamma)
	step = 1

	while True:
		new_partition = louvain_step(graph, partition, step, gamma=gamma)
		new_mod = modularity(graph, new_partition, gamma=gamma)

		if new_mod > mod:
			partition = new_partition
			mod = new_mod
			step += 1
		
		else:
			break

	return partition, mod