def map(vertex, edges, current_distance):
    data = {}
    data[vertex] = (vertex, current_distance[vertex])

    for neighbor, weight in edges.items():
        if current_distance != float('inf'):
            new_distance = current_distance[vertex] + weight
            data[neighbor] = (vertex, new_distance)
    return data

def reduce(key, values):
    min_distance = float('inf')
    predecessor = None

    for value in values:
        vertex, distance = value
        if distance < min_distance:
            min_distance = distance
            predecessor = vertex
    return (key, (predecessor, min_distance))


graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

current_distance = {
    'A': 0,
    'B': float('inf'),
    'C': float('inf'),
    'D': float('inf')
}

map_results = []
for vertex, edges in graph.items():
    map_results.append(map(vertex, edges, current_distance))


shuffled_data = {}
for result in map_results:
    for key, value in result.items():
        if key not in shuffled_data:
            shuffled_data[key] = []
        shuffled_data[key].append(value)

new_distances = {}
for key, values in shuffled_data.items():
    new_distances[key] = reduce(key, values)

for key, (predecessor, min_distance) in new_distances.items():
    current_distance[key] = min_distance

print("当前最短路径信息:")
for node, distance in current_distance.items():
    print(f"节点 {node}: 距离 {distance}")

        
            