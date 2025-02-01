import heapq
def dijkstra(graph, start):
    res = {node: float('inf') for node in graph}
    res[start] = 0
    priority = [(0, start)]
    while priority:
        curr_dist, curr_node = heapq.heappop(priority)
        if curr_dist > res[curr_node]:
            continue
        for neighbor, weight in graph[curr_node].items():
            new_dist = res[curr_node] + weight
            if new_dist < res[neighbor]:
                res[neighbor] = new_dist
                heapq.heappush(priority, (new_dist, neighbor))
    return res

graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'C': 1, 'D': 7},
    'C': {'E': 3},
    'D': {'F': 1},
    'E': {'D': 2, 'F': 5},
    'F': {}
}
start_node = 'A'
shortest_paths = dijkstra(graph, start_node)
print("Shortest route from A to other route：", shortest_paths)