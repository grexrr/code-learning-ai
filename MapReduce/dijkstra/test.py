import heapq

# 创建一个优先队列（实际上是一个列表）
priority_queue = [(0, 'A'), (5, 'B'), (2, 'C')]
heapq.heapify(priority_queue)  # 将列表转换为堆

# # heappop() 会返回并移除堆中最小的元素
first_item = heapq.heappop(priority_queue)
print(first_item)  # 输出: (0, 'A')