import heapq

def dijkstra(graph, start):
    shortest_distance = {node: float('inf') for node in graph}
    shortest_distance[start] = 0
    visited = set()   # all visited node
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < shortest_distance[neighbor]:
                shortest_distance[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
        
    return shortest_distance

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
start_node = 'A'
shortest_paths = dijkstra(graph, start_node)
print("从起点 A 到其他节点的最短路径：", shortest_paths)