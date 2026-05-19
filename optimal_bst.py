def optimal_bst(keys, freq):
    """使用动态规划解决最优二叉搜索树问题
    
    参数:
        keys: 键值列表
        freq: 对应的搜索频率列表
        
    返回:
        (min_cost, root_table): 最小搜索代价和根节点表
    """
    n = len(keys)
    
    # 创建DP表存储子树的代价
    cost = [[0] * n for _ in range(n)]
    # 创建根表记录最优根节点
    root = [[0] * n for _ in range(n)]
    
    # 初始化单个节点的子树
    for i in range(n):
        cost[i][i] = freq[i]
        root[i][i] = i
    
    # 考虑长度为L的子树
    for L in range(2, n+1):
        # 考虑所有起始点为i的子树
        for i in range(n-L+1):
            j = i + L - 1
            cost[i][j] = float('inf')
            total_freq = sum(freq[i:j+1])
            
            # 尝试所有可能的根节点r
            for r in range(i, j+1):
                # 计算左右子树的代价
                c = (cost[i][r-1] if r > i else 0) + \
                    (cost[r+1][j] if r < j else 0) + \
                    total_freq
                
                # 更新最小代价和根节点
                if c < cost[i][j]:
                    cost[i][j] = c
                    root[i][j] = r
    
    return cost[0][n-1], root

def construct_bst(keys, root, i, j):
    """根据根表构造最优BST的结构"""
    if i > j:
        return None
    if i == j:
        return keys[i]
    
    r = root[i][j]
    node = { 'key': keys[r] }
    node['left'] = construct_bst(keys, root, i, r-1)
    node['right'] = construct_bst(keys, root, r+1, j)
    return node

# 示例使用
if __name__ == "__main__":
    keys = [10, 12, 20]
    freq = [34, 8, 50]
    
    min_cost, root_table = optimal_bst(keys, freq)
    bst = construct_bst(keys, root_table, 0, len(keys)-1)
    
    print(f"最小搜索代价: {min_cost}")
    print("最优BST结构:")
    print(bst)
