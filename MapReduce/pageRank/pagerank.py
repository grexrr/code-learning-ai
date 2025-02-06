import math

def _map(graph, curr_weight):
    """Map: calculate current PageRank for delivering"""
    mapped_values = {}
    for node, outlinks in graph.items():
        weight = curr_weight[node]
        num_links = len(outlinks)

        if num_links > 0:
            share = weight / num_links
            for target in outlinks:
                mapped_values[target] = mapped_values.get(target, 0) + share
        else:
            for target in graph.keys():
                mapped_values[target] = mapped_values.get(target, 0) + (weight / len(graph))

    return mapped_values


def _reduce(mapped_values, damping_factor=0.85, num_nodes=4):
    """Reduce: calculate new PageRank"""
    new_weight = {}

    for node, rank_sum in mapped_values.items(): 
        new_rank = (1 - damping_factor) / num_nodes + damping_factor * rank_sum
        new_weight[node] = new_rank
    return new_weight

def _map_reduce(graph, curr_weight):
    mapped_values = _map(graph, curr_weight)
    print("Current Mapped Values")
    for node, rank in mapped_values.items():
        print(f"{node}: {rank:.4f}")

    new_weights = _reduce(mapped_values, num_nodes=len(graph))
    
    print()
    print("Current Weight Values")
    for node, rank in new_weights.items():
        print(f"{node}: {rank:.4f}")

    return new_weights

def iterative_mapreduce_pagerank(graph, max_iteration=40):
    num_nodes = len(graph)
    curr_weight = {node: 1 / num_nodes for node in graph}
    
    converge = False
    iteration = 0
    while not converge and iteration < max_iteration:
        old_weight = curr_weight.copy()
        print("iteration", iteration)
        curr_weight = _map_reduce(graph, curr_weight)
        
        mse = sum((curr_weight[node] - old_weight[node]) ** 2 for node in curr_weight)
        if math.sqrt(mse) <= 0.00001:
            converge = True

        iteration += 1
        print("---------------------------------------------------------------------")
    
    return curr_weight, iteration
    
def main():
    graph = {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'E', 'F'],
        'C': ['A', 'D', 'G'],
        'D': ['C', 'H'],
        'E': ['B', 'F', 'I'],
        'F': ['E', 'G'],
        'G': ['C', 'H', 'J'],
        'H': ['D', 'I'],
        'I': ['E', 'H', 'J'],
        'J': []  # `J` 是孤立节点，没有出链
    }
    final_pagerank, iteration = iterative_mapreduce_pagerank(graph)
    print(f"{iteration} of iteration")
    print("final result:")
    for node, rank in final_pagerank.items():
        print(f"{node}: {rank:.4f}")

if __name__ == "__main__":
    main()
