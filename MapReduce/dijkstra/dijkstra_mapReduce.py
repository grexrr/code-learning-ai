from collections import defaultdict

def _map(curr_vertex, edges, curr_distances):
    emit = []
    if curr_distances[curr_vertex] == float('inf'):
        return [(curr_vertex, (curr_vertex, float('inf')))]

    emit.append((curr_vertex, (curr_vertex, curr_distances[curr_vertex])))

    for neighbor, weight in edges.items():
        new_distance = curr_distances[curr_vertex] + weight
        emit.append((neighbor, (curr_vertex, new_distance)))

    return emit

def _reduce(source_weight_pairs):
    """Reducer function that finds minimum distances to each vertex"""
    return min(source_weight_pairs, key=lambda x: x[1])  

def _map_reduce(graph, current_distances):
    mapped_data = []
    for v, e in graph.items():
        mapped_data.extend(_map(v, e, current_distances))
    
    group_data = defaultdict(list)
    for v, dist in mapped_data:
        group_data[v].append(dist)

    new_distances = {}
    for curr_node, source_weight_pairs in group_data.items():
        source, min_dist = _reduce(source_weight_pairs)
        new_distances[curr_node] = min_dist 
    return new_distances

def iterative_mapreduce_dijkstra(graph, source, max_iterations=None):
    """Implementation of Dijkstra's algorithm using iterative MapReduce"""
    if max_iterations is None:
        max_iterations = len(graph) - 1 

    distances = {v: float('inf') for v in graph}
    distances[source] = 0

    iteration = 0
    is_converge = False
    while (not is_converge) and (iteration < max_iterations):
        old_distances = distances.copy()
        distances = _map_reduce(graph, distances)

        is_converge = all(old_distances[v] == distances[v] for v in distances)
        iteration += 1
        
        print(f"\nIteration {iteration}:")
        for vertex, dist in sorted(distances.items()):
            print(f"{vertex}: {dist}")
    
    return distances, iteration

def main():
    graph = {
        'A': {'B': 4, 'C': 2, 'F': 3},
        'B': {'A': 4, 'C': 1, 'D': 5},
        'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
        'D': {'B': 5, 'C': 8, 'E': 2},
        'E': {'C': 10, 'D': 2},
        'F': {'A': 3, 'D': 5}
    }

    source = 'A'
    shortest_paths, num_iterations = iterative_mapreduce_dijkstra(graph, source)

    print(f"\nFinal shortest paths from vertex {source} after {num_iterations} iterations:")
    for vertex, distances in sorted(shortest_paths.items()):
        print(f"{vertex}: {distances}")

if __name__ == "__main__":
    main()
