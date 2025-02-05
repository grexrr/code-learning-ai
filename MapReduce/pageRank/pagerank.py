# # Mapper
# def mapper(node_id, node_data):
#     rank = node_data.rank
#     links = node_data.out_links
#     num_links = len(links)
    
#     # 如果该网页有出链，则均分 PageRank 值
#     if num_links > 0:
#         rank_share = rank / num_links
#         for link in links:
#             emit(link, rank_share)  # 发送 PageRank 值
#     # 传递网页的出链信息，以便在 Reduce 阶段重构图结构
#     emit(node_id, links)  

# # Reducer
# def reducer(node_id, values):
#     damping_factor = 0.85
#     new_rank = (1 - damping_factor)  # 初始基本值
#     out_links = None
    
#     for value in values:
#         if is_list(value):  # 如果是出链信息
#             out_links = value
#         else:  # 否则是传递来的 PageRank 贡献
#             new_rank += damping_factor * value

#     emit(node_id, (new_rank, out_links))  # 输出新的 PageRank 值和结构信息
