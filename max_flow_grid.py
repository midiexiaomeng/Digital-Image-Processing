import sys
from collections import deque

class Edge:
    def __init__(self, to, rev, capacity):
        self.to = to
        self.rev = rev
        self.capacity = capacity
3
class MaxFlow:
    def __init__(self, n):
        self.size = n
        self.graph = [[] for _ in range(n)]
    
    def add_edge(self, fr, to, cap):
        forward = Edge(to, len(self.graph[to]), cap)
        backward = Edge(fr, len(self.graph[fr]), 0)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)
    
    def bfs_level(self, s, t, level):
        q = deque()
        level[:] = [-1]*self.size
        level[s] = 0
        q.append(s)
        while q:
            v = q.popleft()
            for edge in self.graph[v]:
                if edge.capacity > 0 and level[edge.to] < 0:
                    level[edge.to] = level[v] + 1
                    q.append(edge.to)
    
    def dfs_flow(self, v, t, upTo, iter_, level):
        if v == t:
            return upTo
        for i in range(iter_[v], len(self.graph[v])):
            edge = self.graph[v][i]
            if edge.capacity > 0 and level[v] < level[edge.to]:
                d = self.dfs_flow(edge.to, t, min(upTo, edge.capacity), iter_, level)
                if d > 0:
                    edge.capacity -= d
                    self.graph[edge.to][edge.rev].capacity += d
                    return d
            iter_[v] += 1
        return 0
    
    def max_flow(self, s, t):
        flow = 0
        level = [-1]*self.size
        while True:
            self.bfs_level(s, t, level)
            if level[t] < 0:
                return flow
            iter_ = [0]*self.size
            while True:
                f = self.dfs_flow(s, t, float('inf'), iter_, level)
                if f == 0:
                    break
                flow += f
            level = [-1]*self.size

def solve():
    try:
        # 读取第一行获取m和n
        first_line = sys.stdin.readline()
        if not first_line:
            print("错误：未提供输入数据", file=sys.stderr)
            print("使用方法：python max_flow_grid.py < input.txt", file=sys.stderr)
            return
        
        try:
            m, n = map(int, first_line.split())
            if m <= 0 or n <= 0:
                print("错误：行数和列数必须为正整数", file=sys.stderr)
                return
        except ValueError:
            print("错误：第一行必须包含两个正整数m和n", file=sys.stderr)
            return
        
        grid = []
        total = 0
        for i in range(m):
            line = sys.stdin.readline()
            if not line:
                print(f"错误：第{i+1}行数据缺失", file=sys.stderr)
                return
            
            try:
                row = list(map(int, line.split()))
                if len(row) != n:
                    print(f"错误：第{i+1}行应有{n}个数", file=sys.stderr)
                    return
                grid.append(row)
                total += sum(row)
            except ValueError:
                print(f"错误：第{i+1}行包含非数字内容", file=sys.stderr)
                return
    
    except Exception as e:
        print(f"程序运行时发生错误: {str(e)}", file=sys.stderr)
        return

    # 构建网络流图
    # 节点编号: 0=源点, 1=汇点, 2~m*n+1=方格(按行优先)
    size = m * n + 2
    mf = MaxFlow(size)
    S = 0
    T = 1
    
    # 定义方向: 上、下、左、右
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    for i in range(m):
        for j in range(n):
            node = 2 + i * n + j
            if (i + j) % 2 == 0:  # 黑色方格
                mf.add_edge(S, node, grid[i][j])
                # 连接相邻的白色方格
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        neighbor = 2 + ni * n + nj
                        mf.add_edge(node, neighbor, float('inf'))
            else:  # 白色方格
                mf.add_edge(node, T, grid[i][j])
    
    max_flow = mf.max_flow(S, T)
    print(total - max_flow)

if __name__ == "__main__":
    print("方格取数问题求解器")
    print("使用方法：")
    print("1. 创建一个输入文件input.txt，格式如下：")
    print("   第一行：m n (棋盘行数和列数)")
    print("   接下来m行：每行n个正整数)")
    print("2. 运行程序：python max_flow_grid.py < input.txt")
    print()
    print("示例输入：")
    print("3 3")
    print("1 2 3")
    print("3 2 3")
    print("2 3 1")
    print()
    solve()
