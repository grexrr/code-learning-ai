from collections import defaultdict
def _map(vertex, edges, curr_distances):
    emit = []
    emit.append((vertex, (vertex, curr_distances[vertex])))

    for neighbor, weight in edges.items():
        if curr_distances[vertex] != float('inf'):
            new_distances = curr_distances[vertex] + weight
            emit.append((neighbor, (vertex, new_distances)))
    return emit


def _reduce(values):
    """Reducer function that finds minimum distances to each vertex"""
    min_distance = float('inf')

    for src, dist in values:
        if dist < min_distance:
            min_distance = dist

    return min_distance

def _map_reduce(graph, current_distances):
    mapped_data = []
    for v, e in graph.items():
        mapped_data.extend(_map(v, e, current_distances))
    
    group_data = defaultdict(list)
    for v, dist in mapped_data:
        group_data[v].append(dist)

    # example
    # grouped_data = {
    #     'A': [(A, 0)],
    #     'B': [(A, 4), (C, 3)],
    #     'C': [(A, 2)],
    #     'D': [(B, 9), (C, 10)],
    #     'E': [(C, 12)]
    # }

    new_distances = {}
    for key, values in group_data.items():
        distance = _reduce(values)
        new_distances[key] = distance

    return new_distances


def iterative_mapreduce_dijkstra(graph, source, max_iterations=None):
    """Implementation of Dijkstra's algorithm using iterative MapReduce"""
    if max_iterations is None:
        max_iterations = len(graph) - 1  # Maximum path length in the graph

    # Initialize distancess
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
    # Example graph represented as adjacency list with weights
    graph = {
        'A': {'B': 4, 'C': 2},
        'B': {'A': 4, 'C': 1, 'D': 5},
        'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
        'D': {'B': 5, 'C': 8, 'E': 2},
        'E': {'C': 10, 'D': 2}
    }

    source = 'A'
    shortest_paths, num_iterations = iterative_mapreduce_dijkstra(graph, source)

    print(f"\nFinal shortest paths from vertex {source} after {num_iterations} iterations:")
    for vertex, distances in sorted(shortest_paths.items()):
        print(f"{vertex}: {distances}")

if __name__ == "__main__":
    main()