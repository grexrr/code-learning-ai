from collections import defaultdict
import random
import math

def euclidean_distance(point1, point2):
    """计算两个点之间的欧几里得距离"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def emit(key, value):
    """模拟MapReduce中的emit函数"""
    if key not in emit.output:
        emit.output[key] = []
    emit.output[key].append(value)

emit.output = defaultdict(list)

def _map(data_point, centroids):
    """计算数据点到所有簇中心的距离，并分配到最近的簇"""
    best_cluster = None
    min_distance = float('inf')

    for cluster_id, centroid in centroids.items():
        distance = euclidean_distance(data_point, centroid)
        if distance < min_distance:
            min_distance = distance
            best_cluster = cluster_id

    # 输出 (簇ID, (数据点, 1))，用于后续求均值
    emit(best_cluster, (data_point, 1))


def _reduce(cluster_id, cluster_data):
    """计算新簇中心（簇内所有点的均值）"""
    sum_points = [0] * len(cluster_data[0][0])  # 维度
    total_points = 0

    for point, count in cluster_data:
        sum_points = [sum_points[i] + point[i] for i in range(len(point))]
        total_points += count

    new_centroid = [sum_points[i] / total_points for i in range(len(sum_points))]
    return new_centroid


def initialize_k_centroids(data, k):
    """随机初始化k个簇中心"""
    return {i: random.choice(data) for i in range(k)}


def iterative_mapreduce_kmeans(data, k, max_iterations=100, tolerance=1e-6):
    """K-Means MapReduce 迭代计算"""
    centroids = initialize_k_centroids(data, k)

    iteration = 0
    while iteration < max_iterations:
        old_centroids = centroids.copy()

        # Map 阶段：分配数据点到簇
        emit.output.clear()
        for point in data:
            _map(point, centroids)

        # Reduce 阶段：计算新簇中心
        centroids = {}
        for cluster_id, cluster_data in emit.output.items():
            centroids[cluster_id] = _reduce(cluster_id, cluster_data)

        # 判断收敛
        mse = sum(euclidean_distance(centroids[c], old_centroids[c]) for c in centroids)
        if mse < tolerance:
            break

        iteration += 1

    return centroids, iteration


def main():
    # 示例数据
    data = [
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ]
    k = 2
    centroids, iterations = iterative_mapreduce_kmeans(data, k)
    print(f"Centroids: {centroids}")
    print(f"Iterations: {iterations}")


if __name__ == "__main__":
    main()
